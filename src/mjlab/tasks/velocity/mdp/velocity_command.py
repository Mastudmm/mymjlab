from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.asset_name]

    # 命令缓冲区（每个 env 的命令，机器人基座坐标系）：[线速度_x, 线速度_y, 角速度_z]
    # 说明：第三个分量（角速度_z）有两种可能来源：
    #  1) 在 `_resample_command` 中随机采样得到的 vel_yaw（默认/备选值）
    #  2) 在 `_update_command` 中基于 heading_target 的 P 控制器计算得到的角速度（覆盖采样值）
    # 当 `heading_command=True` 且该 env 被选为 heading 环境时，`_update_command` 会覆盖
    # 对应 env 的 ang_z 值。
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)

    # 若启用朝向命令，保存每个 env 的目标朝向（世界坐标）。
    # heading_target 在重采样（resample）时采样，并在 `_update_command` 中用于
    # 计算替代的角速度。
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)

    # 标记哪些 env 在最近一次重采样时被选为使用 heading 控制（布尔掩码）。
    # 选择在 `_resample_command` 中以概率 `cfg.rel_heading_envs` 进行。
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)

    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    # ---- 关于 heading 的选择与 ang_vel_z 的交互 ----
    # 这里先对 ang_vel_z 进行采样（作为默认角速度）。若启用了 heading_command，
    # 在重采样阶段会为被重采样的 env 随机采样 heading_target，并以概率
    # `rel_heading_envs` 决定哪些 env 使用 heading 控制。对于被选中的 env，
    # 后续 `_update_command` 会用 P 控制计算的角速度覆盖当前采样值。
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      # 在重采样时采样目标朝向（世界坐标系）
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      # 以概率 rel_heading_envs 决定每个 env 是否为 heading 环境
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    # 以概率 rel_standing_envs 决定是否将该 env 设为站立（命令置零）
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    if len(init_vel_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:

    # 若启用了 heading_control，则对在最近一次重采样中被选中的 env 计算
    # heading 误差并通过 P 控制器得到角速度（ang_z），覆盖原有采样值。
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    # 对被选为站立的 env，将所有命令分量置零（优先级高）
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows.

    Note: Only visualizes the selected environment (visualizer.env_idx).
    """
    batch = visualizer.env_idx

    if batch >= self.num_envs:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    base_pos_w = base_pos_ws[batch]
    base_mat_w = base_mat_ws[batch]
    cmd = cmds[batch]
    lin_vel_b = lin_vel_bs[batch]
    ang_vel_b = ang_vel_bs[batch]

    # Skip if robot appears uninitialized (at origin).
    if np.linalg.norm(base_pos_w) < 1e-6:
      return

    # Helper to transform local to world coordinates.
    def local_to_world(
      vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
    ) -> np.ndarray:
      return pos + mat @ vec

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    # Command linear velocity arrow (blue).
    cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
    cmd_lin_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
    )
    visualizer.add_arrow(
      cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
    )

    # Command angular velocity arrow (green).
    cmd_ang_from = cmd_lin_from
    cmd_ang_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
    )
    visualizer.add_arrow(
      cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
    )

    # Actual linear velocity arrow (cyan).
    act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
    act_lin_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
    )
    visualizer.add_arrow(
      act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
    )

    # Actual angular velocity arrow (light green).
    act_ang_from = act_lin_from
    act_ang_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
    )
    visualizer.add_arrow(
      act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
    )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  asset_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  init_velocity_prob: float = 0.0
  class_type: type[CommandTerm] = UniformVelocityCommand

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
