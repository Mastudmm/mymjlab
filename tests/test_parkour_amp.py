"""Regressions for Parkour AMP rollouts, checkpoints, and stair boundaries."""

from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.parkour_vision.mdp.curriculums import terrain_levels_by_pass_rate
from mjlab.tasks.parkour_vision.mdp.terminations import out_of_bounds
from mjlab.tasks.parkour_vision.rl.runner import ParkourAmpOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

TASK = "Mjlab-ParkourVisionAMP-Unitree-Go1"


@pytest.fixture(scope="module")
def parkour_runner():
  env_cfg = load_env_cfg(TASK)
  env_cfg.seed = 7
  env_cfg.scene.num_envs = 5
  assert env_cfg.scene.terrain is not None
  # Assign one env per column to include both stairs in the same simulation.
  env_cfg.scene.terrain.spawn_terrain_names = None
  env_cfg.scene.terrain.spawn_terrain_probs = None
  cfg = asdict(load_rl_cfg(TASK))
  cfg["amp_motion_files"] = [
    str(Path(__file__).resolve().parents[1] / "dataset/*.json")
  ]
  cfg["amp_num_preload_transitions"] = 128
  cfg["num_steps_per_env"] = 2
  cfg["algorithm"]["num_learning_epochs"] = 1
  cfg["algorithm"]["num_mini_batches"] = 1
  env = ManagerBasedRlEnv(env_cfg, device="cpu")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg["clip_actions"])
  try:
    yield ParkourAmpOnPolicyRunner(wrapped, cfg, device="cpu")
  finally:
    wrapped.close()


def test_amp_rollout_uses_terminal_then_reset_state(parkour_runner, monkeypatch):
  runner = parkour_runner
  wrapped = runner.env
  env = wrapped.unwrapped
  wrapped.reset()
  env.episode_length_buf[0] = env.max_episode_length - 1
  first_replay_index = runner.alg.amp_storage.step
  original_reset = env._reset_idx
  original_predict = runner.alg.discriminator.predict_amp_reward
  terminal_states = []
  scored_states = []

  def record_before_reset(env_ids):
    terminal_states.append(
      (env_ids.clone(), wrapped.get_amp_obs_for_expert_trans()[env_ids])
    )
    original_reset(env_ids)

  def record_scored_states(state, next_state, task_reward, normalizer=None):
    scored_states.append(
      (state.clone(), next_state.clone(), wrapped.get_amp_obs_for_expert_trans())
    )
    return original_predict(state, next_state, task_reward, normalizer=normalizer)

  monkeypatch.setattr(env, "_reset_idx", record_before_reset)
  monkeypatch.setattr(
    runner.alg.discriminator, "predict_amp_reward", record_scored_states
  )
  runner.learn(1)

  reset_ids, terminal = terminal_states[0]
  assert reset_ids.tolist() == [0]
  _, scored_end, next_start = scored_states[0]
  torch.testing.assert_close(scored_end[reset_ids], terminal)
  assert not torch.allclose(terminal, next_start[reset_ids])
  torch.testing.assert_close(scored_end[1:], next_start[1:])
  torch.testing.assert_close(scored_states[1][0], next_start)
  torch.testing.assert_close(
    runner.alg.amp_storage.next_states[first_replay_index], terminal[0]
  )
  assert env.pre_reset_observer is None


def test_amp_normalizer_checkpoint_round_trip(parkour_runner):
  alg = parkour_runner.alg
  norm = alg.amp_normalizer
  obs = parkour_runner.env.get_amp_obs_for_expert_trans()
  norm.update(obs.cpu().numpy())
  mean, var, count = norm.mean.copy(), norm.var.copy(), norm.count
  task_reward = obs.new_zeros(obs.shape[0])
  before, _ = alg.discriminator.predict_amp_reward(obs, obs, task_reward, norm)

  file = BytesIO()
  torch.save(alg.save(), file)
  file.seek(0)
  checkpoint = torch.load(file, weights_only=True)
  norm.update(np.full((8, obs.shape[1]), 12.0))
  alg.load(checkpoint, {"discriminator": True}, strict=True)

  np.testing.assert_array_equal(norm.mean, mean)
  np.testing.assert_array_equal(norm.var, var)
  assert norm.count == count
  after, _ = alg.discriminator.predict_amp_reward(obs, obs, task_reward, norm)
  torch.testing.assert_close(after, before, rtol=0, atol=0)

  del checkpoint["amp_normalizer_state"]
  alg.load(checkpoint, {"actor": True}, strict=True)
  with pytest.warns(RuntimeWarning, match="no input normalization statistics"):
    alg.load(checkpoint, {"discriminator": True}, strict=True)
  assert norm.count == count


@torch.inference_mode()
def test_stair_bounds_follow_the_assigned_patch(parkour_runner):
  env = parkour_runner.env.unwrapped
  env.reset()
  terrain = env.scene.terrain
  assert terrain is not None
  assert terrain.terrain_types.tolist() == [0, 1, 2, 3, 4]
  robot = env.scene["robot"]
  pose = robot.data.root_link_pose_w.clone()
  origins = terrain.env_origins[:, :2]

  def place_robots(stair_offset):
    # An equally large +x offset on a waypoint route keeps its old bounds.
    pose[:, :2] = origins
    pose[0, 0] += 4.01
    pose[3:, :2] += pose.new_tensor(stair_offset)
    robot.write_root_link_pose_to_sim(pose)
    env.sim.forward()

  for axis in range(2):
    for sign in (-1, 1):
      offset = [0.0, 0.0]
      offset[axis] = sign * 3.99
      place_robots(offset)
      assert not out_of_bounds(env).any()
      offset[axis] = sign * 4.01
      place_robots(offset)
      assert out_of_bounds(env).tolist() == [False, False, False, True, True]

  # Both rays first meet the edge within 4.5 m of spawn.
  env.episode_length_buf[3:] = 1
  levels_before = terrain.terrain_levels.clone()
  for offset in ((11.5, 0.0), (11.5, 3.0)):
    place_robots(offset)
    terrain_levels_by_pass_rate(env, torch.tensor([3, 4]), stairs_promote_dist=4.5)
    torch.testing.assert_close(terrain.terrain_levels, levels_before)

  # The first step past the boundary also ends both real stair episodes.
  _, _, terminated, _, _ = env.step(torch.zeros(env.action_space.shape))
  assert terminated[3:].all()
  assert env.termination_manager.get_term("out_of_bounds")[3:].all()
