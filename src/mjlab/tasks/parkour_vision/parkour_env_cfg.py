"""Parkour vision env cfg factory (robot-agnostic overrides).

Reuses :func:`mjlab.tasks.velocity_vision.velocity_env_cfg.make_velocity_env_cfg`
with the parkour terrain set, adds the ``reward_point_bonus`` milestone reward,
and switches the velocity command to pure forward (x-only).
"""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.parkour_vision import mdp
from mjlab.tasks.velocity_vision.mdp import UniformVelocityCommandCfg
from mjlab.terrains.config import PARKOUR_TERRAINS_CFG


def apply_parkour_overrides(
  cfg: ManagerBasedRlEnvCfg,
  terrain_cfg=PARKOUR_TERRAINS_CFG,
) -> None:
  """在 velocity_vision env cfg 上叠加 parkour 任务专属配置（机器人无关）。

  - 地形替换为 ``PARKOUR_TERRAINS_CFG``（梅花桩 / 高平台 / 沟壑，各占一列）。
  - 加 ``reward_point_bonus`` 奖励（基座穿越奖励点 + 速度门控）。
  - command 切纯 x 前进（``rel_pure_x_envs=1.0``，lin_vel_y=ang_vel_z=0，关闭
    heading/standing），固定初始 yaw 朝世界 x。
  - command curriculum 改为 parkour 速度阶段（仅 x）。
  """
  # 地形：替换为 parkour 集合（replace 浅拷贝 cfg 本身，sub_terrains 共享）。
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = replace(terrain_cfg)

  # 地形列索引（与 PARKOUR_TERRAINS_CFG.sub_terrains 顺序一致）。
  # 0=梅花桩, 1=矮墙(平台), 2=沟壑。
  wall_column = 1

  # 奖励点 milestone bonus（robot parkour 风格）：逐点给奖，每点每回合一次。
  # z 判定防止在墙根/坑中蹭墙顶/桩顶分；前进门控允许攀爬慢速过点拿分。
  cfg.rewards["reward_point_bonus"] = RewardTermCfg(
    func=mdp.reward_point_bonus,
    weight=1.0,
    params={
      "command_name": "twist",
      "hit_radius": 0.4,
      "z_margin": 0.05,
      "min_forward_speed": 0.05,
      "speed_tol": 0.5,
      "command_threshold": 0.1,
      # 量级设计: reward manager 每步按 dt(0.02) 缩放——bonus=1 时单点实际
      # 入账仅 0.02, 比 tracking(w=4, 每秒~1.3) 弱 200 倍, 策略理性选择绕行。
      # 200 → 单点入账 4.0(≈1 秒满速跟踪), 翻墙 2 点=8.0 显著覆盖攀爬损失。
      "bonus_value": 200.0,
    },
  )

  # 成功终止：越过整条障碍链（最后一个奖励点 + margin）立即重置，避免通过后
  # 在墙前空转到超时。非 timeout 结束，PPO 正常 bootstrap。margin=0.4 使触发
  # 线落在梅花桩终点柱左缘上（柱心 = 末桩 + 0.4 + 0.5）。
  # 方向性奖励：治 y/yaw 漂移与绕行问题（2026-10-14 play 反馈）。
  # 航向保持：yaw 对齐世界 +x（链方向）。
  cfg.rewards["heading_alignment"] = RewardTermCfg(
    func=mdp.heading_alignment,
    weight=1.0,
    params={"std": 0.4},
  )
  # 轨道保持：惩罚偏离出生中线。
  cfg.rewards["lateral_track"] = RewardTermCfg(
    func=mdp.lateral_track,
    weight=1.0,
    params={"std": 0.5},
  )
  # 前进进度：沿世界 +x 的位移直接给分（爬墙慢速也有收益，绕行无收益）。
  cfg.rewards["forward_progress"] = RewardTermCfg(
    func=mdp.forward_progress,
    weight=3.0,
    params={},
  )

  # 成功终止：越过整条障碍链（最后一个奖励点 + margin）立即重置，避免通过后
  # 在墙前空转到超时。非 timeout 结束，PPO 正常 bootstrap。margin=0.4 使触发
  # 线落在梅花桩终点柱左缘上（柱心 = 末桩 + 0.4 + 0.5）。
  cfg.terminations["parkour_completed"] = TerminationTermCfg(
    func=mdp.parkour_completed,
    params={"asset_cfg": SceneEntityCfg("robot"), "margin": 0.4},
  )

  # 躯干接触地形终止：矮墙列豁免（攀爬贴墙不可避免，摔倒由放宽的倾角兜底），
  # 梅花桩/沟壑列保留（躯干碰桩/坑壁 = 失败）。
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact_excluding,
    params={
      "sensor_name": "nonfoot_ground_touch",
      "exempt_terrain_types": (wall_column,),
    },
  )

  # 倾角终止：矮墙列放宽至 85°（攀爬前倾姿态，翻倒仍兜底），其余列 75°。
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation_excluding,
    params={
      "limit_angle": math.radians(75.0),
      "relaxed_angle": math.radians(85.0),
      "exempt_terrain_types": (wall_column,),
    },
  )

  # 速度跟踪 z 分量豁免：track_linear_velocity 的 vz 惩罚让跳跃翻墙代价
  # 巨大（w=4 主奖励瞬间归零）——矮墙列去掉竖直速度惩罚，xy 跟踪保持。
  cfg.rewards["track_linear_velocity"].func = mdp.track_linear_velocity_excluding_z
  cfg.rewards["track_linear_velocity"].params = {
    **cfg.rewards["track_linear_velocity"].params,
    "exempt_terrain_types": (wall_column,),
    "stall_speed": 0.15,
  }

  # 接触惩罚豁免：矮墙列攀爬时头/大腿/小腿贴墙不可避免——这三项对地形的
  # 接触惩罚在墙列置零；梅花桩/沟壑列保持原惩罚。
  for name in ("head_collision", "thigh_collision", "calf_collision"):
    term = cfg.rewards[name]
    term.func = mdp.contact_penalty_excluding
    term.params = {
      **term.params,
      "exempt_terrain_types": (wall_column,),
    }

  # 姿态奖励豁免：upright(w=1.0) 前倾时从 1.0 跌至 ~0.13，攀爬全程损失超过
  # 奖励点收益，会抑制攀爬——矮墙列置中性 1.0（无姿态信号），翻倒由 85° 兜底。
  upright = cfg.rewards["upright"]
  upright.func = mdp.flat_orientation_excluding
  upright.params = {
    **upright.params,
    "exempt_terrain_types": (wall_column,),
  }

  # 越界终止：偏离训练区域（y 越过 patch 半宽、x 倒退过远）直接终止——
  # 绕路即失去剩余全部奖励，防止不按链方向学习。
  cfg.terminations["out_of_bounds"] = TerminationTermCfg(
    func=mdp.out_of_bounds,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "max_y_offset": 3.5,
      "max_back_x": 3.5,
    },
  )

  # 坠落终止：掉进坑/沟底（坑深随机 0.4-1.0m，坑底站立 base≈depth-0.28）的
  # 机器人可能不触发非足接触也不摔倒，用 base 高度兜底。0.15 可覆盖最浅
  # 0.4m 坑（坑底 base≈0.12），正常通行 base≥0.28 不误触发。
  cfg.terminations["fell_into_pit"] = TerminationTermCfg(
    func=mdp.base_height_below,
    params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 0.15},
  )

  # 纯 x 前进 command。
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.rel_standing_envs = 0.0
  twist.rel_heading_envs = 0.0
  twist.rel_pure_x_envs = 1.0
  twist.rel_pure_y_envs = 0.0
  twist.heading_command = False
  twist.ranges.heading = None
  twist.ranges.lin_vel_x = (0.3, 1.0)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.terrain_command_ranges = {}

  # 固定初始 yaw 朝世界 x，收紧出生随机范围（障碍链附近需留出自由半径）。
  cfg.events["reset_base"].params["pose_range"]["x"] = (-0.2, 0.2)
  cfg.events["reset_base"].params["pose_range"]["y"] = (-0.3, 0.3)
  cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)

  # Command curriculum：仅 x 方向速度阶段。
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (0.3, 0.8),
      "lin_vel_y": (0.0, 0.0),
      "ang_vel_z": (0.0, 0.0),
    },
    {
      "step": 2000 * 24,
      "lin_vel_x": (0.4, 1.0),
      "lin_vel_y": (0.0, 0.0),
      "ang_vel_z": (0.0, 0.0),
    },
  ]
  cfg.curriculum["command_vel"].params["terrain_velocity_stages"] = None
