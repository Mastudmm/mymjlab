"""Parkour-specific curriculum terms.

Command curriculum follows wmp_go1's ``update_command_curriculum``: the
forward speed cap rises only when the group's normalized velocity-tracking
score exceeds a threshold — the robot earns faster commands by tracking well,
instead of unlocking speed at fixed iteration counts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    reward_manager = env.reward_manager
    # Episode sums are already weight- and dt-scaled by the reward manager.
    episode_sums = reward_manager._episode_sums[reward_term_name][env_ids]
    term_cfg = reward_manager.get_term_cfg(reward_term_name)
    per_episode_max = env.max_episode_length * env.step_dt * term_cfg.weight
    normalized = episode_sums / per_episode_max
    state["sum"] += float(normalized.sum().item())
    state["count"] += float(len(env_ids))

    if state["count"] >= env.num_envs:
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

    if state["count"] >= env.num_envs:
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

  The pass rate is read from ``reward_point_bonus``'s per-env ``claimed``
  buffer (claimed points / valid points of the env's current terrain patch).
  This is only valid because the curriculum manager computes BEFORE the reward
  manager resets that buffer inside ``_reset_idx`` (same contract as
  :func:`waypoint_timeout_schedule`).

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

  reward_manager = env.reward_manager
  bonus_cfg = reward_manager.get_term_cfg(bonus_term_name)
  claimed = bonus_cfg.func.claimed[env_ids]  # [n, max_points]
  counts_t = terrain.reward_counts
  assert counts_t is not None, (
    "terrain_levels_by_pass_rate requires terrain with reward points."
  )
  counts = counts_t[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]
  idx = torch.arange(claimed.shape[1], device=env.device)
  valid = idx[None, :] < counts[:, None]
  n_claimed = (claimed & valid).sum(dim=1).float()
  pass_rate = n_claimed / valid.sum(dim=1).clamp(min=1).float()

  move_up = pass_rate >= promote_rate
  move_down = (pass_rate <= demote_rate) & (~move_up)
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())
