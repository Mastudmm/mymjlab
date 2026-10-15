"""Parkour-specific curriculum terms.

Command curriculum follows wmp_go1's ``update_command_curriculum``: the
forward speed cap rises only when the group's normalized velocity-tracking
score exceeds a threshold — the robot earns faster commands by tracking well,
instead of unlocking speed at fixed iteration counts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Module-level accumulator state (one training env per process, so a single
# entry keyed by the env object is sufficient; wiped if the env is rebuilt).
_STATE: dict[int, dict[str, float]] = {}


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

  state = _STATE.setdefault(
    id(env), {"sum": 0.0, "count": 0.0, "last_score": 0.0}
  )

  env_ids = env_ids[env.episode_length_buf[env_ids] > 0] if zero_step_reset_guard else env_ids
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
