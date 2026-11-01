"""Parkour-specific curriculum terms.

Command curriculum follows wmp_go1's ``update_command_curriculum``: the
forward speed cap rises only when the group's normalized velocity-tracking
score exceeds a threshold — the robot earns faster commands by tracking well,
instead of unlocking speed at fixed iteration counts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.tasks.parkour_vision.mdp.waypoint_command import (
  WaypointVelocityCommand,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Module-level accumulator state (one training env per process, so a single
# entry keyed by the env object is sufficient; wiped if the env is rebuilt).
_STATE: dict[int, dict[str, float]] = {}

# Separate accumulator for waypoint_timeout_schedule (shared keys with _STATE
# would collide with commands_vel_score on the same env).
_TIMEOUT_STATE: dict[int, dict[str, float]] = {}


def _waypoint_env_mask(env: ManagerBasedRlEnv) -> torch.Tensor | None:
  """Boolean mask [num_envs]: True on envs whose current terrain patch has
  reward points (the waypoint/command curriculum group).

  Point-less columns (the native stairs) run the heading command mode with a
  STATIC speed range: they must neither contribute to nor dilute the
  waypoint-group curricula (wmp_go1 keeps per-terrain-group curricula for
  exactly this reason). Returns None when the terrain has no reward-point
  array at all (no mixed-terrain setup; historical behavior).
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.reward_counts is None:
    return None
  counts = terrain.reward_counts[terrain.terrain_levels, terrain.terrain_types]
  return counts > 0


def _waypoint_fraction(env: ManagerBasedRlEnv) -> float:
  """Fraction of terrain columns that carry reward points.

  Scales the curricula's "at least num_envs episodes" evaluation threshold
  down to the waypoint group's share of the envs, so a mixed 3/5 waypoint
  grid reaches its evaluation just as often per wall-clock time as the old
  all-waypoint grid.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.reward_counts is None:
    return 1.0
  num_cols = terrain.reward_counts.shape[1]
  # Columns with ANY nonzero count across rows carry reward points. (np.any
  # with an axis trips pyright's numpy-stub overloads; sum>0 is equivalent.)
  # Tensor must hop to CPU before numpy — on GPU runs reward_counts lives on
  # cuda:0 and torch.Tensor.numpy() raises TypeError (crashed full_v1 @14:34).
  counts = terrain.reward_counts
  if counts.device.type != "cpu":
    counts = counts.cpu()
  waypoint_cols = int((np.asarray(counts > 0).sum(axis=0) > 0).sum())
  return waypoint_cols / max(num_cols, 1)


def commands_vel_score(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str = "twist",
  reward_term_name: str = "track_linear_velocity",
  score_threshold: float = 0.8,
  increment: float = 0.2,
  vel_x_min: float = 0.2,
  vel_x_cap: float = 1.0,
  zero_step_reset_guard: bool = True,
) -> dict[str, torch.Tensor]:
  """Score-driven forward-speed curriculum (wmp_go1 forward group).

  On every reset batch, accumulates the resetting envs' normalized
  velocity-tracking episode scores (episode sum / full-episode maximum, so
  early termination lowers the score, exactly like wmp_go1). Once at least
  ``env.num_envs`` episodes have accumulated, the mean score is evaluated:

  - score > ``score_threshold``: raise the command's ``lin_vel_x`` upper
    bound by ``increment`` (never above ``vel_x_cap``, never below
    ``vel_x_min``; only up, never down);
  - otherwise: keep the current range.

  Stats returned for TensorBoard: current bounds and last evaluated score.
  """
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  ranges = command_term.cfg.ranges

  state = _STATE.setdefault(id(env), {"sum": 0.0, "count": 0.0, "last_score": 0.0})

  env_ids = (
    env_ids[env.episode_length_buf[env_ids] > 0] if zero_step_reset_guard else env_ids
  )
  if len(env_ids) > 0:
    # Waypoint group only (2026-11-01): heading-mode columns (stairs) run a
    # static speed range — mixing their tracking scores into this accumulator
    # would couple/dilute the waypoint group's speed curriculum.
    wp_mask = _waypoint_env_mask(env)
    if wp_mask is not None:
      env_ids = env_ids[wp_mask[env_ids]]

  if len(env_ids) > 0:
    reward_manager = env.reward_manager
    # Episode sums are already weight- and dt-scaled by the reward manager.
    episode_sums = reward_manager._episode_sums[reward_term_name][env_ids]
    term_cfg = reward_manager.get_term_cfg(reward_term_name)
    per_episode_max = env.max_episode_length * env.step_dt * term_cfg.weight
    # 2026-10-26: 跟踪项的实际上界是 weight * 指令速度，不是 weight * 1。
    # track_goal_vel 被 clamp 在 v_cmd，所以满速跟踪一回合的 episode_sum 只有
    # weight * speed * episode_s。用 weight * 1 归一化时 score 的上限就是
    # speed(0.4~0.8)，永远到不了 0.8 阈值——实测 4700 轮 lin_vel_x_max 恒为
    # 0.8、last_score 恒在 0.12 附近，这个课程实际上是死的。
    # 这里按每个 env 本回合实际采样的速度归一化（curriculum 在 command.reset
    # 之前计算，此时 speed 仍是刚结束那一回合的值），score 才是真正的
    # "跟踪达成率" [0, 1]。
    episode_speed = getattr(command_term, "speed", None)
    if episode_speed is not None:
      per_episode_max = per_episode_max * episode_speed[env_ids].clamp(min=0.05)
    normalized = episode_sums / per_episode_max
    state["sum"] += float(normalized.sum().item())
    state["count"] += float(len(env_ids))

    if state["count"] >= env.num_envs * _waypoint_fraction(env):
      score = state["sum"] / state["count"]
      state["last_score"] = score
      state["sum"] = 0.0
      state["count"] = 0.0
      if score > score_threshold:
        new_cap = min(ranges.lin_vel_x[1] + increment, vel_x_cap)
        ranges.lin_vel_x = (max(vel_x_min, ranges.lin_vel_x[0]), new_cap)

  return {
    "lin_vel_x_min": torch.tensor(ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(ranges.lin_vel_x[1]),
    "last_score": torch.tensor(state["last_score"]),
  }


def waypoint_timeout_schedule(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str = "twist",
  bonus_term_name: str = "reward_point_bonus",
  score_threshold: float = 0.8,
  decrement: float = 1.0,
  timeout_min: float = 3.0,
  zero_step_reset_guard: bool = True,
) -> dict[str, torch.Tensor]:
  """Pass-rate-driven waypoint-timeout curriculum.

  On every reset batch, accumulates the resetting envs' reward-point pass
  rate (claimed points / valid points of the env's current terrain, read from
  ``reward_point_bonus``'s ``claimed`` buffer — safe because the curriculum
  manager computes BEFORE the reward manager resets that buffer). Once at
  least ``env.num_envs`` episodes have accumulated, the mean pass rate is
  evaluated:

  - pass rate > ``score_threshold``: shrink the waypoint command's
    ``waypoint_timeout`` by ``decrement`` seconds (never below
    ``timeout_min``, only down, never up);
  - otherwise: keep the current budget.

  Infeasibility guard: ``WaypointVelocityCommand._effective_timeout`` floors
  the budget at ``dist / speed * timeout_speed_factor``, so even at
  ``timeout_min`` a robot moving at the commanded speed can always reach the
  waypoint in time — the curriculum only removes the surplus waiting budget.

  Stats returned for TensorBoard: current budget and last evaluated pass rate.
  """
  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, WaypointVelocityCommand)

  state = _TIMEOUT_STATE.setdefault(
    id(env), {"sum": 0.0, "count": 0.0, "last_rate": 0.0}
  )

  env_ids = (
    env_ids[env.episode_length_buf[env_ids] > 0] if zero_step_reset_guard else env_ids
  )
  if len(env_ids) > 0:
    # Waypoint group only (2026-11-01): heading-mode columns have no reward
    # points; their (always-zero) pass rate would dilute the accumulator and
    # the timeout budget could never shrink.
    wp_mask = _waypoint_env_mask(env)
    if wp_mask is not None:
      env_ids = env_ids[wp_mask[env_ids]]

  if len(env_ids) > 0:
    reward_manager = env.reward_manager
    bonus_cfg = reward_manager.get_term_cfg(bonus_term_name)
    claimed = bonus_cfg.func.claimed[env_ids]  # [n, max_points]
    terrain = env.scene.terrain
    assert terrain is not None
    counts_t = terrain.reward_counts
    assert counts_t is not None, (
      "waypoint_timeout_schedule requires terrain with reward points."
    )
    counts = counts_t[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]
    idx = torch.arange(claimed.shape[1], device=env.device)
    valid = idx[None, :] < counts[:, None]
    n_claimed = (claimed & valid).sum(dim=1).float()
    rate = n_claimed / valid.sum(dim=1).clamp(min=1).float()
    state["sum"] += float(rate.sum().item())
    state["count"] += float(len(env_ids))

    if state["count"] >= env.num_envs * _waypoint_fraction(env):
      pass_rate = state["sum"] / state["count"]
      state["last_rate"] = pass_rate
      state["sum"] = 0.0
      state["count"] = 0.0
      if pass_rate > score_threshold:
        command_term.cfg.waypoint_timeout = max(
          timeout_min, command_term.cfg.waypoint_timeout - decrement
        )

  return {
    "waypoint_timeout": torch.tensor(command_term.cfg.waypoint_timeout),
    "last_pass_rate": torch.tensor(state["last_rate"]),
  }


def terrain_levels_by_pass_rate(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  bonus_term_name: str = "reward_point_bonus",
  promote_rate: float = 0.5,
  demote_rate: float = 0.0,
  stairs_promote_dist: float = 2.5,
  zero_step_reset_guard: bool = True,
) -> torch.Tensor:
  """Terrain-difficulty curriculum driven by the reward-point pass rate.

  Replaces the velocity task's ``terrain_levels_vel`` for parkour. That one
  promotes on ``distance walked > patch/2``, which is a bad proxy once the
  patch contains walls: a robot can walk the full distance along flat ground
  (or circle in front of the wall) and never touch an obstacle, yet be sent to
  the next difficulty row — and, conversely, a robot that climbs slowly is
  demoted for "walking too little". The logs showed exactly this failure: rows
  decayed to 0 while the robots were walking around.

  Here the SAME quantity the milestone reward pays for drives the ladder, so
  difficulty only rises when the robot actually clears obstacles:

  - pass rate >= ``promote_rate``: move the env up one difficulty row;
  - pass rate <= ``demote_rate``: move it down one row;
  - in between: stay.

  Point-less terrain columns (the native stairs, 2026-11-01) have NO pass
  rate — left here they would read 0 every episode and be demoted to row 0
  forever. They run a separate DISTANCE ladder instead: an env whose base
  moved more than ``stairs_promote_dist`` (2.5 m) from its spawn origin over
  the episode is promoted; never demoted (slow stair climbing is legitimate
  progress, not failure). Spawn is at the patch center for the pyramid
  stairs, so 2.5 m crosses the full stair ring.

  The pass rate is read from ``reward_point_bonus``'s per-env ``claimed``
  buffer (claimed points / valid points of the env's current terrain patch).
  This is only valid because the curriculum manager computes BEFORE the reward
  manager resets that buffer inside ``_reset_idx`` (same contract as
  :func:`waypoint_timeout_schedule`); the base position read for the distance
  ladder is likewise the END-of-episode pose (pre-reset).

  Note the ordering with resets: ``terrain.update_env_origins`` changes the
  env's terrain (row, col) and its origin, so the NEXT episode starts on the
  new patch — no current-episode state is touched.

  Returns the mean terrain level for TensorBoard.
  """
  terrain = env.scene.terrain
  assert terrain is not None, "terrain_levels_by_pass_rate needs a terrain."
  if terrain.terrain_origins is None:
    # Flat-plane tasks have no difficulty rows to move between.
    return torch.tensor(0.0)

  # The construction-time reset has episode_length_buf == 0 (nothing claimed
  # yet, origin still the default pose); skipping it avoids dividing by zero
  # and wrongly demoting fresh envs.
  env_ids = (
    env_ids[env.episode_length_buf[env_ids] > 0] if zero_step_reset_guard else env_ids
  )
  if len(env_ids) == 0:
    return torch.mean(terrain.terrain_levels.float())

  counts_t = terrain.reward_counts
  assert counts_t is not None, (
    "terrain_levels_by_pass_rate requires terrain with reward points."
  )
  counts = counts_t[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]

  # -- Group split (2026-11-01): waypoint columns vs point-less columns. --
  wp_ids = env_ids[counts > 0]
  stairs_ids = env_ids[counts == 0]

  if len(stairs_ids) > 0:
    # Distance ladder for point-less columns: promote on displacement from
    # the spawn origin (read pre-reset, i.e. the ended episode's pose);
    # never demote.
    robot = env.scene["robot"]
    base_pos = robot.data.root_link_pos_w[stairs_ids][:, :2]
    spawn = env.scene.env_origins[stairs_ids][:, :2]
    dist = torch.norm(base_pos - spawn, dim=-1)
    move_up = dist > stairs_promote_dist
    terrain.update_env_origins(stairs_ids, move_up, torch.zeros_like(move_up))

  if len(wp_ids) == 0:
    return torch.mean(terrain.terrain_levels.float())

  reward_manager = env.reward_manager
  bonus_cfg = reward_manager.get_term_cfg(bonus_term_name)
  claimed = bonus_cfg.func.claimed[wp_ids]  # [n, max_points]
  idx = torch.arange(claimed.shape[1], device=env.device)
  valid = idx[None, :] < counts[counts > 0][:, None]
  n_claimed = (claimed & valid).sum(dim=1).float()
  pass_rate = n_claimed / valid.sum(dim=1).clamp(min=1).float()

  move_up = pass_rate >= promote_rate
  move_down = (pass_rate <= demote_rate) & (~move_up)
  terrain.update_env_origins(wp_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())
