from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class TerrainVelocityStages(TypedDict):
  terrain_name: str
  stages: list[VelocityStage]


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
  terrain_velocity_stages: list[TerrainVelocityStages] | None = None,
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  _apply_velocity_stages_to_ranges(cfg.ranges, velocity_stages, env.common_step_counter)

  # 可选：按地形类型分别应用课程速度范围。
  # 若某个地形尚未在 terrain_command_ranges 中出现，则从当前全局 ranges 克隆初值。
  if terrain_velocity_stages is not None:
    for terrain_cfg in terrain_velocity_stages:
      terrain_name = terrain_cfg["terrain_name"]
      if terrain_name not in cfg.terrain_command_ranges:
        cfg.terrain_command_ranges[terrain_name] = UniformVelocityCommandCfg.Ranges(
          lin_vel_x=cfg.ranges.lin_vel_x,
          lin_vel_y=cfg.ranges.lin_vel_y,
          ang_vel_z=cfg.ranges.ang_vel_z,
          heading=cfg.ranges.heading,
        )
      _apply_velocity_stages_to_ranges(
        cfg.terrain_command_ranges[terrain_name],
        terrain_cfg["stages"],
        env.common_step_counter,
      )

  stats: dict[str, torch.Tensor] = {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }

  if terrain_velocity_stages is not None:
    for terrain_cfg in terrain_velocity_stages:
      terrain_name = terrain_cfg["terrain_name"]
      ranges = cfg.terrain_command_ranges[terrain_name]
      prefix = f"terrain_{terrain_name}"
      stats[f"{prefix}_lin_vel_x_min"] = torch.tensor(ranges.lin_vel_x[0])
      stats[f"{prefix}_lin_vel_x_max"] = torch.tensor(ranges.lin_vel_x[1])
      stats[f"{prefix}_lin_vel_y_min"] = torch.tensor(ranges.lin_vel_y[0])
      stats[f"{prefix}_lin_vel_y_max"] = torch.tensor(ranges.lin_vel_y[1])
      stats[f"{prefix}_ang_vel_z_min"] = torch.tensor(ranges.ang_vel_z[0])
      stats[f"{prefix}_ang_vel_z_max"] = torch.tensor(ranges.ang_vel_z[1])

  return stats


def _apply_velocity_stages_to_ranges(
  ranges: UniformVelocityCommandCfg.Ranges,
  velocity_stages: list[VelocityStage],
  step_counter: int,
) -> None:
  for stage in velocity_stages:
    if step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        ranges.ang_vel_z = stage["ang_vel_z"]


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])
