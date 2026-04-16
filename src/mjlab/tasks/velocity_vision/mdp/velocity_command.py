from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

if TYPE_CHECKING:
  import viser

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

    self.robot: Entity = env.scene[cfg.entity_name]

    # 课程地形下，每个列（terrain_types）映射到一个地形名称。
    # 用于在命令重采样时按地形类型应用不同速度范围。
    self._terrain_name_per_col: list[str] | None = None
    terrain = getattr(env.scene, "terrain", None)
    terrain_generator_cfg = None if terrain is None else terrain.cfg.terrain_generator
    if terrain_generator_cfg is not None and bool(getattr(terrain_generator_cfg, "curriculum", False)):
      self._terrain_name_per_col = self._resolve_curriculum_column_terrain_names(terrain_generator_cfg)
    elif self.cfg.terrain_command_ranges:
      print(
        "[UniformVelocityCommand] terrain_command_ranges is set but terrain curriculum mapping is unavailable "
        "(requires terrain_type='generator' and terrain_generator.curriculum=True)."
      )

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
    # 每个 env 当前有效角速度范围（用于 heading 控制时按 env 裁剪）。
    self._ang_vel_min = torch.full(
      (self.num_envs,), float(self.cfg.ranges.ang_vel_z[0]), device=self.device
    )
    self._ang_vel_max = torch.full(
      (self.num_envs,), float(self.cfg.ranges.ang_vel_z[1]), device=self.device
    )

    # 标记哪些 env 在最近一次重采样时被选为使用 heading 控制（布尔掩码）。
    # 选择在 `_resample_command` 中以概率 `cfg.rel_heading_envs` 进行。
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

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
    # 1) 先按全局范围采样（默认命令范围）
    self._sample_velocity_range(env_ids, self.cfg.ranges)

    # 2) 若配置了地形专属范围，则按 env 所在地形覆盖重采样
    #    仅在 terrain generator curriculum 下可可靠按“地形类型”区分。
    if self.cfg.terrain_command_ranges and self._terrain_name_per_col is not None:
      terrain = self._env.scene.terrain
      assert terrain is not None
      terrain_types = terrain.terrain_types[env_ids]

      for terrain_name, terrain_ranges in self.cfg.terrain_command_ranges.items():
        mask = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        for col, col_name in enumerate(self._terrain_name_per_col):
          if col_name == terrain_name:
            mask |= terrain_types == col
        matched_env_ids = env_ids[mask]
        if len(matched_env_ids) > 0:
          self._sample_velocity_range(matched_env_ids, terrain_ranges)

    r = torch.empty(len(env_ids), device=self.device)
    # ---- 关于 heading 的选择与 ang_vel_z 的交互 ----
    # 这里先对 ang_vel_z 进行采样（作为默认角速度）。若启用了 heading_command，
    # 在重采样阶段会为被重采样的 env 随机采样 heading_target，并以概率
    # `rel_heading_envs` 决定哪些 env 使用 heading 控制。对于被选中的 env，
    # 后续 `_update_command` 会用 P 控制计算的角速度覆盖当前采样值。
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      # 在重采样时采样目标朝向（世界坐标系）
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      # 以概率 rel_cardinal_heading_envs 将部分 env 的 heading_target
      # 覆盖为离散主方向（0/90/180/270 度）
      if self.cfg.rel_cardinal_heading_envs > 0.0:
        cardinal_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_cardinal_heading_envs
        #掩码，确定谁被抽到 TorF
        num_cardinal = int(cardinal_mask.sum().item())
        if num_cardinal > 0:
          cardinal_values = torch.tensor(
            self.cfg.cardinal_heading_values,
            dtype=self.heading_target.dtype,
            device=self.device,
          )
          choice_idx = torch.randint(0, cardinal_values.numel(), (num_cardinal,), device=self.device)
          self.heading_target[env_ids[cardinal_mask]] = cardinal_values[choice_idx]
      # 以概率 rel_heading_envs 决定每个 env 是否为 heading 环境
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    # 以概率 rel_standing_envs 决定是否将该 env 设为站立（命令置零）
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    # 处理纯 X 或纯 Y 方向的命令
    # 生成一个随机数用于决定是否应用纯方向约束
    # 0 ~ rel_pure_x_envs: 纯 X (vy=0)
    # rel_pure_x_envs ~ rel_pure_x_envs + rel_pure_y_envs: 纯 Y (vx=0)
    # 其他: 混合
    if self.cfg.rel_pure_x_envs > 0 or self.cfg.rel_pure_y_envs > 0:
      pure_probs = torch.rand(len(env_ids), device=self.device)
      
      # 纯 X 方向：将 y 方向速度置为 0
      pure_x_mask = pure_probs < self.cfg.rel_pure_x_envs
      self.vel_command_b[env_ids[pure_x_mask], 1] = 0.0
      
      # 纯 Y 方向：将 x 方向速度置为 0
      pure_y_mask = (pure_probs >= self.cfg.rel_pure_x_envs) & (
        pure_probs < (self.cfg.rel_pure_x_envs + self.cfg.rel_pure_y_envs)
      )
      self.vel_command_b[env_ids[pure_y_mask], 0] = 0.0

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

  def _sample_velocity_range(
    self,
    env_ids: torch.Tensor,
    ranges: "UniformVelocityCommandCfg.Ranges",
  ) -> None:
    """Sample vx, vy, wz commands for selected envs from the provided range config."""
    if len(env_ids) == 0:
      return
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*ranges.ang_vel_z)
    self._ang_vel_min[env_ids] = float(ranges.ang_vel_z[0])
    self._ang_vel_max[env_ids] = float(ranges.ang_vel_z[1])

  def _resolve_curriculum_column_terrain_names(self, terrain_generator_cfg) -> list[str]:
    """Build column->terrain_name mapping using the same allocation logic as TerrainGenerator curriculum mode."""
    sub_terrain_names = list(terrain_generator_cfg.sub_terrains.keys())
    proportions = np.array(
      [sub_cfg.proportion for sub_cfg in terrain_generator_cfg.sub_terrains.values()],
      dtype=np.float64,
    )
    if proportions.sum() <= 0.0:
      raise ValueError("Terrain proportions sum to zero; cannot resolve curriculum terrain columns.")
    proportions = proportions / proportions.sum()
    cumulative = np.cumsum(proportions)

    num_cols = int(terrain_generator_cfg.num_cols)
    terrain_name_per_col: list[str] = []
    for col in range(num_cols):
      sub_index = int(np.min(np.where(col / num_cols + 0.001 < cumulative)[0]))
      terrain_name_per_col.append(sub_terrain_names[sub_index])
    return terrain_name_per_col

  def _update_command(self) -> None:

    # 若启用了 heading_control，则对在最近一次重采样中被选中的 env 计算
    # heading 误差并通过 P 控制器得到角速度（ang_z），覆盖原有采样值。
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      heading_wz = self.cfg.heading_control_stiffness * self.heading_error[env_ids]
      self.vel_command_b[env_ids, 2] = torch.clip(
        heading_wz,
        min=self._ang_vel_min[env_ids],
        max=self._ang_vel_max[env_ids],
      )
    # 对被选为站立的 env，将所有命令分量置零（优先级高）
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0

  # GUI.

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon

    ranges = self.cfg.ranges

    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max_val,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    # Store GUI state for compute() override.
    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # Helper to transform local to world coordinates.
      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

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
  entity_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  # heading_command 开启时，heading_target 采样中离散主方向（cardinal）所占概率
  # 例如 0.5 表示 50% 采样 cardinal_heading_values，50% 按 ranges.heading 均匀采样。
  rel_cardinal_heading_envs: float = 0.0
  # 默认主方向：0°, 90°, 180°, 270°（弧度）
  cardinal_heading_values: tuple[float, ...] = (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0)
  rel_pure_x_envs: float = 0.0
  rel_pure_y_envs: float = 0.0
  init_velocity_prob: float = 0.0

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges
  # 按地形类型覆写命令范围（仅在 terrain_generator.curriculum=True 且 terrain_type='generator' 时生效）。
  # key 为地形名称（例如 "flat"、"pyramid_stairs_inv"），value 为该地形采样范围。
  terrain_command_ranges: dict[str, Ranges] = field(default_factory=dict)

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
    if not (0.0 <= self.rel_cardinal_heading_envs <= 1.0):
      raise ValueError("rel_cardinal_heading_envs must be in [0, 1].")
    if len(self.cardinal_heading_values) == 0:
      raise ValueError("cardinal_heading_values must not be empty.")
