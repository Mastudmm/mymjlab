"""Waypoint-directed velocity command for the parkour task.

Extreme-Parkour-style adaptive command: instead of a fixed pure-x velocity,
the command is recomputed EVERY step to point at the next terrain reward
point. The body-frame linear velocity is the world-frame direction to the
waypoint rotated into the base frame and scaled by the (curriculum-controlled)
commanded speed; the yaw rate comes from a P controller on the heading error
toward the waypoint. Following the command therefore walks the robot through
the obstacle chain by construction.

Waypoint advancement per env (re-checked in a loop so skipped points collapse
in a single step):

- reached: xy distance < ``reach_radius`` AND base z above the point z +
  ``z_margin`` (same height gate as ``reward_point_bonus`` — a robot still on
  the ground in front of a wall does not switch to the far side);
- passed: base x beyond the point x + ``pass_margin`` (a target behind the
  robot can never be reached by moving forward);
- timed out: time spent on the current waypoint exceeds the effective budget

  ``max(waypoint_timeout, dist / speed * timeout_speed_factor)``

  The second term is the feasibility floor: a robot moving AT the commanded
  speed never times out (its arrival time times the factor stays below the
  budget), so shrinking ``waypoint_timeout`` through
  :func:`mjlab.tasks.parkour_vision.mdp.curriculums.waypoint_timeout_schedule`
  can never make the command infeasible — only slower-than-commanded motion
  (slow climbing, circling to farm the tracking reward) trips it.

The speed magnitude is sampled once per episode from ``ranges.lin_vel_x``
(the score curriculum raises its upper bound); the waypoint chain restarts
from the first reward point on every episode reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity_vision.mdp.velocity_command import (
  UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class WaypointVelocityCommand(CommandTerm):
  cfg: WaypointVelocityCommandCfg

  def __init__(self, cfg: WaypointVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot = env.scene[cfg.entity_name]

    terrain = getattr(env.scene, "terrain", None)
    points = None if terrain is None else terrain.reward_points
    if points is not None:
      self._max_points = int(points.shape[2])
      self._has_points = True
    else:
      self._max_points = 0
      self._has_points = False

    # Body-frame command [vx, vy, wz], same layout as UniformVelocityCommand
    # so the observation pipeline (generated_commands) is unchanged.
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    # Commanded speed magnitude, sampled once per episode from ranges.lin_vel_x.
    self.speed = torch.zeros(self.num_envs, device=self.device)
    # Per-env waypoint chain state.
    self.waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.waypoint_time = torch.zeros(self.num_envs, device=self.device)
    self.is_standing_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    self.metrics["dist_to_waypoint"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["waypoint_progress"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """Lazily (re)compute the direction command from the CURRENT base pose.

    Consumers (observations, rewards, terminations) may read the command at
    points where no ``compute()`` has run yet — notably right after
    ``env.reset()``, whose observations are taken before the first
    ``command_manager.compute()``. Derived pose quantities (``xpos``/
    ``xquat``) are stale inside the reset block (forward kinematics has not
    run yet), so the command MUST NOT be baked into a buffer during
    ``_resample_command``; recomputing at read time always sees a fresh pose
    (observations are computed after ``sim.forward()``). The math is a few
    elementwise ops per env, so recomputing on every access is cheap.
    """
    self._refresh_command_buffer()
    return self.vel_command_b

  def _refresh_command_buffer(self) -> None:
    if self._has_points:
      self._compute_direction_command()
    else:
      self.vel_command_b.zero_()
    # Standing envs get a zero command (highest priority).
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0

  def _update_metrics(self) -> None:
    if not self._has_points:
      return
    points, counts = self._waypoint_batch()
    batch = torch.arange(self.num_envs, device=self.device)
    idx = torch.minimum(self.waypoint_idx, (counts - 1).clamp(min=0))
    wp = points[batch, idx]
    base_pos = self.robot.data.root_link_pos_w
    dist = torch.norm(wp[:, :2] - base_pos[:, :2], dim=-1)
    self.metrics["dist_to_waypoint"] += dist * (counts > 0).float()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    r = torch.empty(len(env_ids), device=self.device)
    self.speed[env_ids] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
    # resampling_time_range is (1e9, 1e9), so this only fires on episode
    # reset: restart the waypoint chain from the first reward point.
    # NOTE: do NOT touch the command buffer here — the base pose (xpos/xquat)
    # is stale inside the reset block; the command property recomputes lazily
    # from the fresh pose at read time (see the property docstring).
    self.waypoint_idx[env_ids] = 0
    self.waypoint_time[env_ids] = 0.0

  def _update_command(self) -> None:
    # Called from compute() after sim.forward(): base pose is fresh, so this
    # is where per-step waypoint advancement is evaluated.
    self.waypoint_time += self._env.step_dt
    if self._has_points:
      self._advance_waypoints()
    self._refresh_command_buffer()

  def _advance_waypoints(self) -> None:
    points, counts = self._waypoint_batch()
    batch = torch.arange(self.num_envs, device=self.device)
    idx_max = (counts - 1).clamp(min=0)
    base_pos = self.robot.data.root_link_pos_w

    # -- Waypoint advancement: reached | passed | timed out. Loop so that a
    # robot beyond several points at once collapses them in one step.
    advanced = torch.zeros(self.num_envs, device=self.device)
    for _ in range(self._max_points):
      idx = torch.minimum(self.waypoint_idx, idx_max)
      wp = points[batch, idx]
      dist_xy = torch.norm(wp[:, :2] - base_pos[:, :2], dim=-1)
      reached = (dist_xy < self.cfg.reach_radius) & (
        base_pos[:, 2] > wp[:, 2] + self.cfg.z_margin
      )
      passed = base_pos[:, 0] > wp[:, 0] + self.cfg.pass_margin
      timeout = self.waypoint_time > self._effective_timeout(dist_xy)
      advance = (reached | passed | timeout) & (self.waypoint_idx < idx_max)
      if not bool(advance.any()):
        break
      self.waypoint_idx = torch.where(advance, self.waypoint_idx + 1, self.waypoint_idx)
      self.waypoint_time = torch.where(advance, 0.0, self.waypoint_time)
      advanced += advance.float()
    self.metrics["waypoint_progress"] += advanced

  def _compute_direction_command(self) -> None:
    """Fill ``vel_command_b`` from the current pose and waypoint state."""
    points, counts = self._waypoint_batch()
    batch = torch.arange(self.num_envs, device=self.device)
    idx_max = (counts - 1).clamp(min=0)
    base_pos = self.robot.data.root_link_pos_w

    idx = torch.minimum(self.waypoint_idx, idx_max)
    wp = points[batch, idx]
    delta = wp[:, :2] - base_pos[:, :2]
    dist_xy = torch.norm(delta, dim=-1)
    dir_w = delta / dist_xy.clamp(min=1e-6).unsqueeze(-1)

    # Exhausted-chain override (user fix 2026-10-31): once the LAST waypoint
    # is behind the robot, pointing the command back at it creates a limit
    # cycle around the finish point (the direction flips each time the robot
    # crosses the point), so a slow policy never crosses the success line
    # (last_point + margin) and idles there forever in play (infinite episode
    # length). Freeze the direction to the chain axis (+x) instead, so the
    # robot keeps walking forward and parkour_completed fires. Measured
    # before the fix (AMP lerp=0.8 checkpoint, 2500 play steps): 4/4 envs
    # spent ~2000 steps circling the finish with ZERO resets.
    exhausted = (self.waypoint_idx >= idx_max) & (base_pos[:, 0] > wp[:, 0])
    if bool(exhausted.any()):
      forward = torch.zeros_like(dir_w)
      forward[:, 0] = 1.0
      dir_w = torch.where(exhausted.unsqueeze(-1), forward, dir_w)

    # Yaw rate: P controller on the heading error toward the waypoint
    # (same pattern as UniformVelocityCommand's heading_command mode).
    target_heading = torch.atan2(dir_w[:, 1], dir_w[:, 0])
    heading_err = wrap_to_pi(target_heading - self.robot.data.heading_w)
    wz = torch.clip(
      self.cfg.heading_control_stiffness * heading_err,
      min=self.cfg.ranges.ang_vel_z[0],
      max=self.cfg.ranges.ang_vel_z[1],
    )

    # Body-frame linear velocity: world direction rotated into the base
    # frame, scaled by the commanded speed.
    dir3_w = torch.cat(
      [dir_w, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1
    )
    dir_b = quat_apply_inverse(self.robot.data.root_link_quat_w, dir3_w)
    self.vel_command_b[:, :2] = dir_b[:, :2] * self.speed.unsqueeze(-1)
    self.vel_command_b[:, 2] = wz
    # Envs without reward points get no command.
    self.vel_command_b[counts == 0] = 0.0

  def _effective_timeout(self, dist_xy: torch.Tensor) -> torch.Tensor:
    """Effective per-waypoint time budget (seconds).

    ``max(waypoint_timeout, dist / speed * timeout_speed_factor)``: the second
    term guarantees the budget always exceeds the commanded-speed arrival time
    by the safety factor, so the curriculum can shorten ``waypoint_timeout``
    without ever making the command infeasible.
    """
    feasible = dist_xy / self.speed.clamp(min=0.1) * self.cfg.timeout_speed_factor
    return torch.maximum(feasible, torch.full_like(feasible, self.cfg.waypoint_timeout))

  def _waypoint_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env reward points [B, max, 3] and counts [B] (world frame)."""
    terrain = self._env.scene.terrain
    assert terrain is not None
    points_t = terrain.reward_points
    counts_t = terrain.reward_counts
    assert points_t is not None and counts_t is not None, (
      "WaypointVelocityCommand requires terrain with reward points."
    )
    return points_t[terrain.terrain_levels, terrain.terrain_types], counts_t[
      terrain.terrain_levels, terrain.terrain_types
    ]

  # Visualization.

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw the active waypoint (red sphere) and the command ray to it."""
    if not self._has_points:
      return
    terrain = self._env.scene.terrain
    assert terrain is not None
    points_t = terrain.reward_points
    counts_t = terrain.reward_counts
    assert points_t is not None and counts_t is not None
    base_pos_ws = self.robot.data.root_link_pos_w

    for batch in visualizer.get_env_indices(self.num_envs):
      row = int(terrain.terrain_levels[batch].item())
      col = int(terrain.terrain_types[batch].item())
      count = int(counts_t[row, col].item())
      if count == 0:
        continue
      idx = min(int(self.waypoint_idx[batch].item()), count - 1)
      wp = points_t[row, col, idx].detach().cpu().numpy()
      base = base_pos_ws[batch].detach().cpu().numpy()
      visualizer.add_sphere(wp, 0.12, (0.9, 0.25, 0.25, 0.9))
      visualizer.add_arrow(
        base + np.array([0.0, 0.0, 0.1]),
        wp + np.array([0.0, 0.0, 0.1]),
        color=(0.9, 0.25, 0.25, 0.6),
        width=0.02,
      )


@dataclass(kw_only=True)
class WaypointVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  ranges: UniformVelocityCommandCfg.Ranges
  # Waypoint chain state must survive within an episode; the magnitude is
  # only resampled on episode reset, hence the near-infinite interval.
  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  heading_control_stiffness: float = 1.5
  reach_radius: float = 0.3
  z_margin: float = 0.05
  pass_margin: float = 0.2
  # Base per-waypoint time budget (s); shrunk by the timeout curriculum as
  # the reward-point pass rate improves.
  waypoint_timeout: float = 10.0
  timeout_speed_factor: float = 1.3
  rel_standing_envs: float = 0.05

  def __post_init__(self):
    if self.reach_radius <= 0.0:
      raise ValueError("reach_radius must be positive.")
    if self.timeout_speed_factor <= 1.0:
      raise ValueError(
        "timeout_speed_factor must exceed 1.0 so the effective timeout stays "
        "slightly longer than the commanded-speed arrival time."
      )
    if self.ranges.ang_vel_z[0] >= self.ranges.ang_vel_z[1]:
      raise ValueError("ranges.ang_vel_z must be an increasing pair.")

  def build(self, env: ManagerBasedRlEnv) -> WaypointVelocityCommand:
    return WaypointVelocityCommand(self, env)
