"""Parkour vision env cfg (robot-agnostic task layer).

本文件是 parkour 任务的**唯一**任务层配置入口，原则：**一个文件看全**。
奖励表 / 终止表 / 课程表 / 命令 / 深度观测 / 出生与相机事件 全部在此显式
写出（func + weight + params 一体），不做"改一个 weight"式的局部覆盖——
任何旋钮都应该在这一个文件里可见、可改。

仍在别处的配置（有意保留，勿在此重复）：
- PPO/网络超参（lr、gamma、entropy、depth_history_num、scan_dim）:
    config/go1/rl_cfg.py
- 机器人本体、传感器、domain randomization、Go1 姿态 std:
    velocity_vision/config/go1/env_cfgs.py（机器人相关层）
- 控制频率与回合长度在此文件 misc 段显式写出（与 base 相同值）。
- 墙体几何（墙高阶梯/厚度/间距）: terrains/config.py 的
    PARKOUR_TERRAINS_CFG（2026-10-27 已全部显式暴露）。

继承结构：velocity_vision/velocity_env_cfg.py（骨架）
  -> velocity_vision/config/go1/env_cfgs.py（Go1 特化）
  -> 本文件 apply_parkour_overrides（最后执行，优先级最高）
"""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.parkour_vision import mdp
from mjlab.tasks.velocity_vision import mdp as vv_mdp
from mjlab.tasks.velocity_vision.mdp import UniformVelocityCommandCfg
from mjlab.terrains.config import PARKOUR_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# ---- 公共 SceneEntityCfg（集中定义，奖励表内引用）----
_ROBOT = SceneEntityCfg("robot")
_TRUNK = SceneEntityCfg("robot", body_names=("trunk",))
_ALL_JOINTS = SceneEntityCfg("robot", joint_names=(".*",))
_FEET_SITES = SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL"))
_FRONT_FEET = SceneEntityCfg("robot", site_names=("FR", "FL"))
# gait_force 两项的足部顺序与上面不同（历史约定）：对角/相邻配对按此顺序。
_GAIT_SITES = SceneEntityCfg("robot", site_names=("FL", "FR", "RL", "RR"))

# ---- 单地形列豁免（2026-10-26 修复）----
# 单地形化后 3 列是同一种墙的 3 套随机布局，terrain_types 仍取 0/1/2，
# 任何"只豁免一列"的写法都会漏掉 2/3 的 env（曾导致 illegal_contact 占
# 终止 85%）。接触类豁免一律用全部列。
_WALL_COLUMNS = tuple(range(3))


def apply_parkour_overrides(
  cfg: ManagerBasedRlEnvCfg,
  terrain_cfg=PARKOUR_TERRAINS_CFG,
) -> None:
  """在 velocity_vision env cfg 上叠加 parkour 任务层配置。

  整表替换 rewards/terminations/curriculum：本函数执行后，这三个表的内容
  **完全**由本文件决定，继承层不再有任何隐式残留项。
  """
  # ============================================================
  # 0. 地形与全局开关
  # ============================================================
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = replace(terrain_cfg)
  # 所有 env 从 row 0（墙高 ~0 m）起步，难度由课程按奖励点通过率推进。
  cfg.scene.terrain.max_init_terrain_level = 0

  # 控制频率 / 回合（与 base 相同，显式写出便于调整）。
  cfg.decimation = 4  # 50 Hz 控制（200 Hz 物理 / 4）
  cfg.episode_length_s = 20.0
  # 每步总奖励截零（EP only_positive_rewards）。2026-10-21 实证对本任务
  # 反效果（惩罚项压过正项时梯度全无 -> std 爆炸死锁），保持关闭。
  cfg.only_positive_rewards = False
  cfg.scale_rewards_by_dt = True

  # ============================================================
  # 1. 奖励表（整表替换，28 项全部显式）
  # ------------------------------------------------------------
  # 预算设计（2026-10-25 再平衡，rate/s @20s 回合）：
  #   站立 ≈ +0.49/s（10 分） << 行走 ≈ +1.3/s（26 分） << 通关 ≈ 66 分
  # 姿态类正奖励只是"稳定器"，主力是 tracking（稠密）+ 奖励点（稀疏）。
  # ============================================================
  cfg.rewards = {
    # ---- 主任务驱动 ----
    # 线性截断方向跟踪（EP 同款）：r = min(<v_xy_w, d_cmd_w>, v_cmd)，
    # 误差大时仍有稠密梯度；天然不罚 vz。上界 = weight * 指令速度。
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_goal_vel,
      weight=1.5,
      params={"command_name": "twist"},
    ),
    # 偏航角速度跟踪：站立白拿项，已降权为稳定器。
    "track_angular_velocity": RewardTermCfg(
      func=vv_mdp.track_angular_velocity,
      weight=0.3,  # 1.0 -> 0.3（站立白拿 ~0.95/s，2026-10-25）
      params={"command_name": "twist", "std": math.sqrt(0.2)},
    ),
    # 里程碑 bonus：主要任务报酬。单点入账 = bonus_value * dt = 8 分，
    # 两条链 16 分 >> 站立 10 分。v8（2026-10-27）：1000 -> 400——v7 实证
    # 40 分的通关尖峰让 value loss 在 0.001~11.6 间震荡、策略反复崩回；
    # 降到 16 分在保留"通关是站立 1.6 倍"的同时把回报方差降 2.5 倍。
    # 若仍震荡，下一档 200（单点 4 分，两链 8 分）。
    # hit_radius 必须 >= 航点 reach_radius(0.3)，否则航点先切走、奖励点
    # 永远进不了计奖区（旧值 0.2 的坑）。
    "reward_point_bonus": RewardTermCfg(
      func=mdp.reward_point_bonus,
      weight=1.0,
      params={
        "command_name": "twist",
        "hit_radius": 0.35,
        "z_margin": 0.05,
        "min_forward_speed": 0.05,
        "speed_tol": 0.5,
        "command_threshold": 0.1,
        "bonus_value": 400.0,
      },
    ),
    # 近墙区攀爬塑造：贴近(0.3)+昂首(0.3)+前爪上墙(0.5)，满值 ~1.1，
    # weight 0.5 -> 满值 0.55/s，低于 tracking，只负责搭梯子。
    "wall_climb_shaping": RewardTermCfg(
      func=mdp.wall_climb_shaping,
      weight=0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": _FRONT_FEET,
      },
    ),
    # ---- 姿态稳定器（站立白拿项，全部降权）----
    "upright": RewardTermCfg(
      func=vv_mdp.flat_orientation,
      weight=0.2,  # 1.0 -> 0.2（站立白拿 ~0.95/s）
      params={
        "std": math.sqrt(0.5),  # sqrt(0.2) 放宽：45° 前倾不重罚，攀爬姿态友好
        "asset_cfg": _TRUNK,
      },
    ),
    "pose": RewardTermCfg(
      func=vv_mdp.variable_posture,
      weight=0.1,  # 0.75(base) -> 0.25 -> 0.1（站立白拿 ~0.2/s）
      params={
        "asset_cfg": _ALL_JOINTS,
        "command_name": "twist",
        "std_standing": {
          r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
          r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
        },
        "std_walking": {
          r".*(FR|FL|RR|RL)_hip_joint.*": 0.315,
          r".*(FR|FL|RR|RL)_thigh_joint.*": 0.325,
          r".*(FR|FL|RR|RL)_calf_joint.*": 0.625,
        },
        "std_running": {
          r".*(FR|FL|RR|RL)_hip_joint.*": 0.375,
          r".*(FR|FL|RR|RL)_thigh_joint.*": 0.45,
          r".*(FR|FL|RR|RL)_calf_joint.*": 0.75,
        },
        "walking_threshold": 0.10,
        "running_threshold": 1.25,
      },
    ),
    # ---- 步态 shaping（走/跑质量）----
    "air_time": RewardTermCfg(
      func=vv_mdp.feet_air_time,
      weight=0.475,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.125,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.10,
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=vv_mdp.feet_clearance,
      weight=-1.25,
      params={
        "asset_cfg": _FEET_SITES,
        "command_name": "twist",
        "command_threshold": 0.05,
        "target_height": -0.145,  # 机体系目标摆动高度（Go1 ~10cm 离地）
      },
    ),
    # 摆动峰值欠抬惩罚（v5 抬脚包核心：健康步态抬脚仅 2cm 会卡任何 >=3cm 坎）。
    # 落地评估；欠抬重罚(1.0)/过抬轻罚(0.35)；直立门控，摔倒不罚。
    # 注意：翻 0.45m 墙时过抬一次性约 -1.4（target 0.15 + 0.35 系数），
    # 相对 20 分 bonus 可接受，但若攀爬期拉扯明显可调低过抬系数。
    "foot_swing_height": RewardTermCfg(
      func=vv_mdp.feet_swing_height,
      weight=-0.5,  # -0.125 -> -0.5（v5 抬脚包）
      params={
        "sensor_name": "feet_ground_contact",
        "target_height": 0.15,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": _FEET_SITES,
        "height_sensor_name": None,  # 用机体系相对高度，不用射线
      },
    ),
    "foot_slip": RewardTermCfg(
      func=vv_mdp.feet_slip,
      weight=-0.3,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": _FEET_SITES,
      },
    ),
    "feet_air_time_variance": RewardTermCfg(
      func=vv_mdp.feet_air_time_variance_penalty,
      weight=-0.1,
      params={"sensor_name": "feet_ground_contact", "asset_cfg": _ROBOT},
    ),
    "soft_landing": RewardTermCfg(
      func=vv_mdp.soft_landing,
      weight=-0.0001,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.10,
      },
    ),
    # 踝击拖步惩罚（EP feet_stumble）：脚尖横向冲击 > 垂直支撑时罚。
    "stumble": RewardTermCfg(
      func=vv_mdp.stumble_penalty,
      weight=-0.2,
      params={"sensor_names": ["feet_ground_contact"]},
    ),
    # ---- 步态对称（trot 先验）----
    # v8（2026-10-27）：-0.2/-0.1 -> -0.05/-0.02。v7 好阶段（iter 4400）里
    # action_mirror 是最重惩罚（-0.237/s），而攀爬本质不对称，这两项在
    # 攀爬期是反向约束。保留小权重维持 trot 先验，不再压制攀爬姿态。
    "action_mirror": RewardTermCfg(
      func=vv_mdp.action_mirror,
      weight=-0.05,
      params={
        "asset_cfg": _ROBOT,
        "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
      },
    ),
    "action_sync": RewardTermCfg(
      func=vv_mdp.action_sync,
      weight=-0.02,
      params={
        "asset_cfg": _ROBOT,
        "joint_groups": [(".*_hip_joint",), (".*_thigh_joint",), (".*_calf_joint",)],
      },
    ),
    "gait_force_sync": RewardTermCfg(
      func=vv_mdp.gait_force_sync,
      weight=2.5e-3,
      params={
        "std": 0.1,
        "max_err": 0.2,
        "sensor_name": "feet_ground_contact",
        "asset_cfg": _GAIT_SITES,
        "pairs": [["FL", "RR"], ["FR", "RL"]],  # trot 对角同步
      },
    ),
    "gait_force_async": RewardTermCfg(
      func=vv_mdp.gait_force_async,
      weight=2.5e-3,
      params={
        "std": 0.1,
        "max_err": 0.2,
        "sensor_name": "feet_ground_contact",
        "asset_cfg": _GAIT_SITES,
        "pairs": [["FL", "FR"], ["RL", "RR"]],  # 相邻反同步
      },
    ),
    # ---- 正则 / 平滑 ----
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.1),
    "j_vel_l2": RewardTermCfg(func=envs_mdp.joint_vel_l2, weight=-0.0013),
    "dof_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-1.0),
    "energy_save": RewardTermCfg(
      func=vv_mdp.energy_saving, weight=-1e-06, params={"asset_cfg": _ROBOT}
    ),
    # ---- 机体角动量 / 高度 / 进度（当前关闭，保留旋钮）----
    "body_ang_vel": RewardTermCfg(
      func=vv_mdp.body_angular_velocity_penalty,
      weight=0.0,  # 关闭；打开即启用躯干角速度惩罚
      params={"asset_cfg": _TRUNK},
    ),
    "angular_momentum": RewardTermCfg(
      func=vv_mdp.angular_momentum_penalty,
      weight=0.0,  # 关闭
      params={"sensor_name": "robot/root_angmom"},
    ),
    "base_height": RewardTermCfg(
      func=vv_mdp.track_base_height,
      weight=0.0,  # 关闭；打开用于惩罚蹲伏
      params={
        "target_height": 0.28,
        "asset_cfg": _ROBOT,
        "sensor_name": "ray_base",
      },
    ),
    "progress": RewardTermCfg(
      func=vv_mdp.progress_reward,
      weight=0.0,  # 关闭（方向信息已由航点命令携带）
      params={"command_name": "twist", "threshold": 0.1},
    ),
    # ---- 身体碰撞惩罚（2026-10-27 注意：当前结构性为 0）----
    # 单地形下 terrain 涵盖"墙"与"地面"，无法用列区分"攀爬贴墙（必需）"与
    # "撞地（坏步态）"，故豁免全部列 => TB 中恒为 0（用户已知，去留待定：
    #  A 保持 0 / B 改 exempt=() 激活 / C 删除本三项）。
    "head_collision": RewardTermCfg(
      func=mdp.contact_penalty_excluding,
      weight=-1.0,
      params={
        "sensor_name": "head_ground_contact",
        "force_threshold": 0.2,
        "exempt_terrain_types": _WALL_COLUMNS,
      },
    ),
    "thigh_collision": RewardTermCfg(
      func=mdp.contact_penalty_excluding,
      weight=-1.2,
      params={
        "sensor_name": "thigh_ground_contact",
        "force_threshold": 0.2,
        "exempt_terrain_types": _WALL_COLUMNS,
      },
    ),
    "calf_collision": RewardTermCfg(
      func=mdp.contact_penalty_excluding,
      weight=-0.5,
      params={
        "sensor_name": "calf_ground_contact",
        "force_threshold": 0.2,
        "exempt_terrain_types": _WALL_COLUMNS,
      },
    ),
  }

  # ============================================================
  # 2. 终止表（整表替换，7 项全部显式）
  # ============================================================
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    # 倾角终止 85°：60-85° 攀爬俯仰放行，真翻倒立即重置（v7 恢复；
    # 不做列豁免——单地形下列豁免=全环境豁免=没有终止）。
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": math.radians(85.0)},
    ),
    # 躯干接触终止：攀爬贴墙必需，单地形下全列豁免 = 等效停用。
    "illegal_contact": TerminationTermCfg(
      func=mdp.illegal_contact_excluding,
      params={
        "sensor_name": "nonfoot_ground_touch",
        "exempt_terrain_types": _WALL_COLUMNS,
      },
    ),
    # 通关：越过最后一个奖励点 + margin（且 y 偏差不大防蹭线）。
    "parkour_completed": TerminationTermCfg(
      func=mdp.parkour_completed,
      params={"asset_cfg": _ROBOT, "margin": 0.4},
    ),
    # 速度违反：倒退/停滞超阈值（高难度行才启用）。
    "vel_violate": TerminationTermCfg(
      func=mdp.vel_violate,
      params={"command_name": "twist", "threshold": 1.5, "min_level": 3},
    ),
    # 越界：横向偏离 / 倒退过远（绕行兜底）。
    "out_of_bounds": TerminationTermCfg(
      func=mdp.out_of_bounds,
      params={"asset_cfg": _ROBOT, "max_y_offset": 3.5, "max_back_x": 3.5},
    ),
    # 坠坑：掉进沟底（z < 0.15）的兜底。
    "fell_into_pit": TerminationTermCfg(
      func=mdp.base_height_below,
      params={"asset_cfg": _ROBOT, "min_height": 0.15},
    ),
  }

  # ============================================================
  # 3. 课程表（整表替换，3 项全部显式）
  # ============================================================
  cfg.curriculum = {
    # 地形难度：奖励点通过率驱动（>=0.5 升行，<=0 降行）。
    # 旧 distance>4m 判据在有墙 patch 上失真（走平地也升行）。
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_by_pass_rate,
      params={
        "bonus_term_name": "reward_point_bonus",
        "promote_rate": 0.5,
        "demote_rate": 0.0,
      },
    ),
    # 前进速度上限：归一化跟踪分 > 0.8 时 +0.2，封顶 1.0。
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel_score,
      params={
        "command_name": "twist",
        "reward_term_name": "track_linear_velocity",
        "score_threshold": 0.8,
        "increment": 0.2,
        "vel_x_min": 0.2,
        "vel_x_cap": 1.0,
      },
    ),
    # 航点限时：通过率 > 0.8 时基础限时 -1s，下限 3s。
    "waypoint_timeout": CurriculumTermCfg(
      func=mdp.waypoint_timeout_schedule,
      params={
        "command_name": "twist",
        "bonus_term_name": "reward_point_bonus",
        "score_threshold": 0.8,
        "decrement": 1.0,
        "timeout_min": 3.0,
      },
    ),
  }

  # ============================================================
  # 4. 命令（航点制，全部字段显式）
  # ============================================================
  # 每步指向下一个奖励点：机体系 [vx, vy] = 指向航点的单位方向 × 幅值，
  # wz = 朝向航点的 P 控制。z 线速度不存在：方向在世界系强制 z=0 后转
  # 机体系并只取 xy（waypoint_command._compute_direction_command）。
  cfg.commands["twist"] = mdp.WaypointVelocityCommandCfg(
    entity_name="robot",
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.4, 0.8),  # 由 command_vel 课程逐步升到 (0.2, 1.0)
      lin_vel_y=(0.0, 0.0),  # 占位：方向来自航点，不独立采样
      ang_vel_z=(-1.0, 1.0),  # wz 裁剪范围
    ),
    resampling_time_range=(1e9, 1e9),  # 幅值每回合采样一次
    heading_control_stiffness=1.5,
    reach_radius=0.3,  # 航点到达半径（勿小于 reward_point_bonus.hit_radius）
    z_margin=0.05,
    pass_margin=0.2,
    waypoint_timeout=10.0,  # 基础限时，由 waypoint_timeout 课程收紧
    timeout_speed_factor=1.3,  # 可行性下限系数
    rel_standing_envs=0.05,  # 5% 概率零指令（站立）
    debug_vis=True,
  )

  # ============================================================
  # 5. 观测（parkour 仅覆盖深度项；其余继承 base/go1 层）
  # ============================================================
  # 继承的 actor 项（全部 10 帧历史）：base_ang_vel / projected_gravity /
  # joint_pos / joint_vel / actions / command；critic 额外有 base_lin_vel、
  # foot_* 特权项与 height_scan(289 点, scale 1/3)。
  # 深度：wmp_go1 量纲 [-0.5, 0.5]，3 帧历史 + 15% 整帧随机失效（模拟实机
  # 反光/丢帧，逼策略回退本体感知）；play 时 env_cfgs.py 将 dropout 置 0。
  cfg.observations["actor"].terms["depth"] = ObservationTermCfg(
    func=mdp.depth_camera_normalized,
    params={
      "sensor_cfg": SceneEntityCfg("depth_camera"),
      "far_clip": 3.0,
      "dropout_prob": 0.15,
    },
    history_length=3,  # 必须与 rl_cfg 的 depth_history_num 一致
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

  # ============================================================
  # 6. 事件（出生 / 相机；其余 DR 继承 go1 层）
  # ============================================================
  # 出生：固定 yaw 朝世界 +x，收紧 xy 随机（障碍链附近留自由半径）。
  # 想做"出生点课程"（缩短到首个奖励点的时间）就改这里的 x 范围。
  cfg.events["reset_base"] = EventTermCfg(
    func=envs_mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {
        "x": (-0.2, 0.2),
        "y": (-0.3, 0.3),
        "z": (0.01, 0.05),
        "yaw": (0.0, 0.0),
      },
      "velocity_range": {},
    },
  )
  # 相机俯仰随机化：每 env 固定随机下俯 15-25° + 偏航 ±2°（wmp_go1 实机
  # 配置）。符号：quat_from_euler_xyz 的 pitch 为正 = 下俯（2026-10-15 实证）。
  cfg.events["cam_quat"] = EventTermCfg(
    func=envs_mdp.dr.site_quat,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names="head"),
      "pitch_range": (math.radians(15.0), math.radians(25.0)),
      "yaw_range": (math.radians(-2.0), math.radians(2.0)),
    },
  )
  # 继承自 velocity_vision/config/go1/env_cfgs.py 的事件（勿在此重复）：
  # reset_robot_joints / push_robot / foot_friction / encoder_bias /
  # body_inertia_mass / cam_pos / dof_friction / dof_damping /
  # actuator_gains / joint_armature / effort_limits
