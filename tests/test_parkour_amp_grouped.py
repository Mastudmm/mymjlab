"""Exercise the grouped CNN policy through the real Parkour AMP runner."""

from dataclasses import asdict
from math import isfinite
from pathlib import Path

import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator
from torch import nn

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.parkour_vision.rl.actor import ParkourVisionActor
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
  pure = asdict(load_rl_cfg("Mjlab-ParkourVision-Unitree-Go1"))
  assert grouped["run_name"] == "v23_cnn_grouped_ppo_grad_spawn75_25_loss50_50"
  assert grouped["algorithm"]["terrain_group_weights"] == {
    "platforms": 0.5,
    "crawl": 0.5,
  }
  assert grouped["actor"] == base["actor"] == pure["actor"]
  assert grouped["critic"] == base["critic"]
  assert grouped["amp_task_reward_lerp"] == base["amp_task_reward_lerp"]
  assert "terrain_group_weights" not in base["algorithm"]


def test_grouped_ppo_weights_the_actual_surrogate(monkeypatch, tmp_path):
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
    actor = alg.actor
    assert isinstance(actor, ParkourVisionActor)
    conv = actor.encoder.cnn[0]
    assert isinstance(conv, nn.Conv2d)
    weight_before = conv.weight.detach().clone()
    forward_calls = []

    def record_conv_inputs(_module, inputs, _output):
      forward_calls.append((inputs[0].shape[1:], torch.is_grad_enabled()))

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
    with conv.register_forward_hook(record_conv_inputs):
      runner.learn(1)

    # A registered layer can be saved and logged without ever training.
    assert ((3, 80, 80), False) in forward_calls  # Rollout
    assert ((3, 80, 80), True) in forward_calls  # PPO update
    assert not torch.equal(weight_before, conv.weight)
    adam_update = alg.optimizer.state[conv.weight]["exp_avg"]
    assert torch.isfinite(adam_update).all() and adam_update.abs().max() > 0

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

    policy = runner.get_inference_policy()
    obs = env.get_observations()[:1]
    sentinel = torch.arange(actor.obs_dim, dtype=torch.float32)[None, :]
    latest = torch.cat(
      [
        sentinel[:, end - width : end]
        for end, width in ((30, 3), (60, 3), (180, 12), (300, 12), (420, 12), (450, 3))
      ],
      dim=-1,
    )
    different_depth = obs.clone()
    different_depth["actor"][:, 450:] = 0.5
    with torch.inference_mode():
      torch.testing.assert_close(actor.encoder(sentinel)[:, :45], latest)
      actions = policy(obs)
      assert (actions - policy(different_depth)).abs().max() > 1e-6
      script = torch.jit.script(actor.as_jit()).eval()
      torch.testing.assert_close(script(obs["actor"]), actions)

    runner.export_policy_to_onnx(str(tmp_path))
    model = onnx.load(tmp_path / "policy.onnx")
    onnx.checker.check_model(model)
    assert sum(node.op_type == "Conv" for node in model.graph.node) == 3
    arrays = ReferenceEvaluator(model).run(None, {"obs": obs["actor"].numpy()})
    torch.testing.assert_close(torch.from_numpy(arrays[0]), actions)
  finally:
    env.close()
