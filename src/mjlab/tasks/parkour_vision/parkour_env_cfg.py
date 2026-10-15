"""Parkour vision env cfg factory (robot-agnostic overrides).

Reuses :func:`mjlab.tasks.velocity_vision.velocity_env_cfg.make_velocity_env_cfg`
with the parkour terrain set. Reward/curriculum structure follows wmp_go1
(doc/source/note/parkour_wmp_refactor.md):

- ONE uniform velocity-tracking reward with overspeed clipping
  (``lin_vel_clip``) — no per-column exemptions except true wall-contact terms;
- wmp_go1's anti-stall pair: ``stuck`` penalty (stalls are punished, not
  neutralized) + ``cheat`` penalty (large world-yaw deviation);
- reward-point milestone bonus kept as the primary task drive;
- score-driven forward-speed command curriculum ([0.2, 0.5] -> [0.2, 1.0]);
- depth observations normalized to wmp_go1's [-0.5, 0.5] range with a 3 m
  far clip, and camera pitch randomized 15-25 deg down per env.
"""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.parkour_vision import mdp
from mjlab.tasks.velocity_vision.mdp import UniformVelocityCommandCfg
from mjlab.terrains.config import PARKOUR_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def apply_parkour_overrides(
  cfg: ManagerBasedRlEnvCfg,
  terrain_cfg=PARKOUR_TERRAINS_CFG,
) -> None:
  """在 velocity_vision env cfg 上叠加 parkour 任务专属配置（机器人无关）。

  - 地形替换为 ``PARKOUR_TERRAINS_CFG``（梅花桩 / 矮墙 / 沟壑，各占一列）。
  - 奖励按 wmp_go1 对齐重构：裁剪跟踪 + stuck/cheat 惩罚 + 奖励点 bonus。
  - command 切纯 x 前进，范围 [0.2, 0.5] 起步，分数课程升至 [0.2, 1.0]。
  """
  # 地形：替换为 parkour 集合（replace 浅拷贝 cfg 本身，sub_terrains 共享）。
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = replace(terrain_cfg)

  # 地形列索引（与 PARKOUR_TERRAINS_CFG.sub_terrains 顺序一致）。
  # 0=梅花桩, 1=矮墙(平台), 2=沟壑。
  wall_column = 1

  # ---- 跟踪奖励：统一裁剪版，无列级豁免（wmp_go1 核心结构） ----
  # 权重 4.0 -> 1.0（对齐 wmp_go1）；超速裁剪让冲障碍/跳跃期不罚快。
  # vz 惩罚保持全域生效：跳跃代价由障碍上的奖励点支付。
  cfg.rewards["track_linear_velocity"].func = mdp.track_linear_velocity_clipped
  cfg.rewards["track_linear_velocity"].weight = 1.0
  cfg.rewards["track_linear_velocity"].params = {
    **cfg.rewards["track_linear_velocity"].params,
    "lin_vel_clip": 0.1,
  }

  # ---- 奖励点 milestone bonus（robot parkour 风格，主任务驱动） ----
  # 量级（w=1 体系）：单点入账 = bonus*dt = 2.0 ≈ 2 秒满速跟踪收益。
  # 梅花 9 点=18.0，矮墙 2 点/墙=4.0，沟壑 3 点=6.0——覆盖攀爬期
  # tracking(vz)+upright 损失(~2-3)，绕行则被 cheat/越界终止硬性掐断。
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
      "bonus_value": 100.0,
    },
  )

  # ---- wmp_go1 反停顿双件套 ----
  # 基座的 EMA 平滑版 stuck_penalty 删除，避免与即时版双重计罚。
  cfg.rewards.pop("stuck_penalty", None)
  # stuck：指令>0.1 而 |vx|<0.1 —— 停顿是负收益而非中性（替代旧 anti-stall）。
  cfg.rewards["stuck"] = RewardTermCfg(
    func=mdp.stuck_penalty,
    weight=-1.0,
    params={"command_name": "twist"},
  )
  # cheat：世界系朝向偏离链方向(+x) 超 ±1 rad —— 绕行直接受罚。
  cfg.rewards["cheat"] = RewardTermCfg(
    func=mdp.cheat_penalty,
    weight=-1.0,
    params={},
  )

  # 方向性奖励：治 y/yaw 漂移与绕行问题（2026-10-14 play 反馈）。
  cfg.rewards["heading_alignment"] = RewardTermCfg(
    func=mdp.heading_alignment,
    weight=1.0,
    params={"std": 0.4},
  )
  cfg.rewards["lateral_track"] = RewardTermCfg(
    func=mdp.lateral_track,
    weight=1.0,
    params={"std": 0.5},
  )
  # 前进进度权重 3.0 -> 1.0：与 tracking 同量纲，避免位移奖励压过一切。
  cfg.rewards["forward_progress"] = RewardTermCfg(
    func=mdp.forward_progress,
    weight=1.0,
    params={},
  )

  # ---- 成功终止：越过整条障碍链（最后奖励点 + margin）立即重置 ----
  cfg.terminations["parkour_completed"] = TerminationTermCfg(
    func=mdp.parkour_completed,
    params={"asset_cfg": SceneEntityCfg("robot"), "margin": 0.4},
  )

  # 躯干接触地形终止：矮墙列豁免（真垂直墙攀爬贴墙不可避免——唯一保留的
  # 接触豁免），梅花桩/沟壑列保留（躯干碰桩/坑壁 = 失败）。
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

  # 速度违反终止（wmp_go1 同款）：倒退/停滞超 1.5 m/s（高难度行）。
  cfg.terminations["vel_violate"] = TerminationTermCfg(
    func=mdp.vel_violate,
    params={
      "command_name": "twist",
      "threshold": 1.5,
      "min_level": 3,
    },
  )

  # 越界终止：偏离训练区域（y 越过 patch 半宽、x 倒退过远）直接终止。
  cfg.terminations["out_of_bounds"] = TerminationTermCfg(
    func=mdp.out_of_bounds,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "max_y_offset": 3.5,
      "max_back_x": 3.5,
    },
  )

  # 坠落终止：掉进坑/沟底（坑深随机 0.4-1.0m）的机器人可能不触发非足接触
  # 也不摔倒，用 base 高度兜底。
  cfg.terminations["fell_into_pit"] = TerminationTermCfg(
    func=mdp.base_height_below,
    params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 0.15},
  )

  # ---- 接触惩罚豁免：矮墙列攀爬贴墙不可避免（保留的唯一惩罚豁免） ----
  for name in ("head_collision", "thigh_collision", "calf_collision"):
    term = cfg.rewards[name]
    term.func = mdp.contact_penalty_excluding
    term.params = {
      **term.params,
      "exempt_terrain_types": (wall_column,),
    }
  # upright 豁免已按 wmp_go1 结构删除：攀爬期 upright 损失由奖励点覆盖，
  # 且防止"后仰刷 upright"的局部最优。

  # ---- 深度观测：wmp_go1 量纲（[-0.5, 0.5]，far 3m） ----
  cfg.observations["actor"].terms["depth"] = ObservationTermCfg(
    func=mdp.depth_camera_normalized,
    params={"sensor_cfg": SceneEntityCfg("depth_camera"), "far_clip": 3.0},
    history_length=1,
    flatten_history_dim=True,
    delay_min_lag=2,
    delay_max_lag=5,
    delay_hold_prob=0.1,
    delay_update_period=2,
    # ±5cm 原始深度噪声换算到归一化量纲（1/3 每 m）。
    noise=Unoise(n_min=-0.017, n_max=0.017),
  )
  # 深度传感器量程 4.0 -> 3.0（wmp_go1 近量程实机验证配置）。
  for sensor in cfg.scene.sensors:
    if sensor.name == "depth_camera":
      sensor.max_distance = 3.0

  # ---- 相机俯仰随机化（wmp_go1 实机配置）：XML head site 水平朝前，
  # 每 env 固定随机下俯 15-25° + 偏航 ±2°——同时见过"近处脚下"与
  # "远处障碍"，根治固定大俯角的视野盲区。 ----
  cfg.events["cam_quat"].params["pitch_range"] = (
    math.radians(-25.0),
    math.radians(-15.0),
  )
  cfg.events["cam_quat"].params["yaw_range"] = (
    math.radians(-2.0),
    math.radians(2.0),
  )

  # ---- 纯 x 前进 command：范围 [0.2, 0.5] 起步（wmp_go1 forward 组）， ----
  # 5% 概率零指令（zero_command_prob），<0.2 的碎步指令通过下界 0.2 等效消除。
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.rel_standing_envs = 0.05
  twist.rel_heading_envs = 0.0
  twist.rel_pure_x_envs = 1.0
  twist.rel_pure_y_envs = 0.0
  twist.heading_command = False
  twist.ranges.heading = None
  twist.ranges.lin_vel_x = (0.2, 0.5)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.terrain_command_ranges = {}

  # 固定初始 yaw 朝世界 x，收紧出生随机范围（障碍链附近需留出自由半径）。
  cfg.events["reset_base"].params["pose_range"]["x"] = (-0.2, 0.2)
  cfg.events["reset_base"].params["pose_range"]["y"] = (-0.3, 0.3)
  cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)

  # ---- 指令课程：静态迭代阶段 -> 分数驱动（wmp_go1 forward 组） ----
  # 组内归一化 tracking 分数 > 0.8 时 x 上限 +0.2，封顶 1.0。
  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.commands_vel_score,
    params={
      "command_name": "twist",
      "reward_term_name": "track_linear_velocity",
      "score_threshold": 0.8,
      "increment": 0.2,
      "vel_x_min": 0.2,
      "vel_x_cap": 1.0,
    },
  )
