"""Exercise the v22 configuration through the real Parkour AMP runner."""

from dataclasses import asdict
from math import isfinite
from pathlib import Path

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.parkour_vision.rl.grouped_ppo import ParkourGroupedAMPPPO
from mjlab.tasks.parkour_vision.rl.runner import ParkourAmpOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

BASE = "Mjlab-ParkourVisionAMP-Unitree-Go1"
GROUPED = "Mjlab-ParkourVisionAMP-GroupedPPO-Unitree-Go1"


def test_grouped_experiment_keeps_the_baseline_environment():
  env_cfg = load_env_cfg(GROUPED)
  assert env_cfg.scene.terrain.spawn_terrain_probs == (0.75, 0.25)
  assert env_cfg.scene.terrain.terrain_generator.sub_terrains[
    "crawl"
  ].clearance_range == (0.45, 0.25)
  grouped = asdict(load_rl_cfg(GROUPED))
  base = asdict(load_rl_cfg(BASE))
  assert grouped["run_name"] == "v22_grouped_ppo_grad_spawn75_25_loss50_50"
  assert grouped["algorithm"]["terrain_group_weights"] == {
    "platforms": 0.5,
    "crawl": 0.5,
  }
  assert grouped["actor"] == base["actor"]
  assert grouped["critic"] == base["critic"]
  assert grouped["amp_task_reward_lerp"] == base["amp_task_reward_lerp"]
  assert "terrain_group_weights" not in base["algorithm"]


def test_grouped_ppo_weights_the_actual_surrogate(monkeypatch):
  cfg = load_env_cfg(GROUPED)
  cfg.scene.num_envs = 16
  cfg.seed = 7
  agent = asdict(load_rl_cfg(GROUPED))
  agent["amp_motion_files"] = [
    str(Path(__file__).resolve().parents[1] / "dataset/*.json")
  ]
  agent["amp_num_preload_transitions"] = 128
  agent["num_steps_per_env"] = 2
  agent["algorithm"]["num_learning_epochs"] = 1
  agent["algorithm"]["num_mini_batches"] = 1
  agent["algorithm"]["terrain_grad_samples"] = 8
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, "cpu"))
  try:
    runner = ParkourAmpOnPolicyRunner(env, agent, device="cpu")
    alg = runner.alg
    assert isinstance(alg, ParkourGroupedAMPPPO)
    logs = []
    monkeypatch.setattr(runner.logger, "log", lambda **kw: logs.append(kw["loss_dict"]))
    original_probe = alg._actor_gradient_metrics

    def check_probe():
      state = torch.random.get_rng_state()
      metrics = original_probe()
      torch.testing.assert_close(torch.random.get_rng_state(), state)
      assert all(param.grad is None for param in alg.actor.parameters())
      return metrics

    monkeypatch.setattr(alg, "_actor_gradient_metrics", check_probe)
    runner.learn(1)

    result = logs[0]
    platform_share = result["terrain/rollout_share/platforms"]
    assert 0.5 < platform_share < 1.0
    assert result["terrain/rollout_share/crawl"] == pytest.approx(1 - platform_share)
    assert result["terrain/policy_weight/platforms"] == 0.5
    assert result["terrain/policy_weight/crawl"] == 0.5
    assert result["terrain_grad/norm/platforms"] > 0
    assert result["terrain_grad/norm/crawl"] > 0
    cosine = result["terrain_grad/cosine/platforms__crawl"]
    assert isfinite(cosine) and -1 <= cosine <= 1

    # With one minibatch and one epoch, ratios at the first update are 1.
    # The optimized surrogate should match the mean of the two terrain losses.
    advantages = alg.storage.advantages.flatten()
    expected = -torch.stack(
      [advantages[ids].mean() for ids in alg._group_indices]
    ).mean()
    assert result["surrogate"] == pytest.approx(expected.item(), abs=1e-5)
  finally:
    env.close()
