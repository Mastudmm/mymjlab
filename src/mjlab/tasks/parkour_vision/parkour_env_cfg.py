"""Parkour vision env cfg factory (robot-agnostic overrides).

Reuses :func:`mjlab.tasks.velocity_vision.velocity_env_cfg.make_velocity_env_cfg`
with the parkour terrain set. Reward/curriculum structure follows wmp_go1
(doc/source/note/parkour_wmp_refactor.md):

- ONE uniform velocity-tracking reward with overspeed clipping
  (``lin_vel_clip``) — no per-column exemptions except true wall-contact terms;
- waypoint-directed command (Extreme-Parkour style, user decision 2026-10-19):
  every step the command points at the next terrain reward point, replacing
  the fixed pure-x command AND the handcrafted direction rewards
  (forward_progress / lateral_track / heading_alignment / cheat are no
  longer registered — the command itself carries the direction);
- reward-point milestone bonus kept as the primary task drive;
- score-driven forward-speed command curriculum ([0.2, 0.5] -> [0.2, 1.0])
  plus a pass-rate-driven waypoint-timeout curriculum (10 s -> 3 s);
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

  - 地形替换为 ``PARKOUR_TERRAINS_CFG``：单地形任务，仅翻越高墙（矮墙/平台，
    3 列独立随机布局；梅花桩/沟壑已移除，用户决定 2026-10-19）。
  - 奖励按 wmp_go1 对齐重构：裁剪跟踪 + 奖励点 bonus（方向类手工奖励已随
    航点命令的引入移除——方向信息由命令本身携带）。
  - command 切航点制（WaypointVelocityCommand）：每步指向下一个奖励点，
    幅值范围 [0.4, 0.8] 起步，分数课程升至 [0.2, 1.0]；限时课程随通过率
    从 10 s 收紧至 3 s。
  """
  # 地形：替换为 parkour 集合（replace 浅拷贝 cfg 本身，sub_terrains 共享）。
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = replace(terrain_cfg)

  # 地形列索引（与 PARKOUR_TERRAINS_CFG.sub_terrains 顺序一致）。
  # 单地形任务：唯一列 0 = 矮墙(平台)（梅花桩/沟壑已移除，2026-10-19）。
  wall_column = 0

  # ---- 跟踪奖励：B2 实验组 = EP 线性截断版（w=1.5，EP 同款）----
  # r = min(⟨v_xy_w, d̂_cmd_w⟩, v_cmd)：误差大时仍有稠密梯度，翻墙期持续
  # 付钱；天然不罚 vz。
  # A2 对照组（另一 worktree/commit）：mdp.track_linear_velocity_clipped
  # （exp 版 wmp_go1 结构）w=1.0，params 加回 std/lin_vel_clip。
  cfg.rewards["track_linear_velocity"].func = mdp.track_goal_vel
  cfg.rewards["track_linear_velocity"].weight = 1.5
  cfg.rewards["track_linear_velocity"].params = {"command_name": "twist"}

  # ---- 抬脚包（v5 根因修复：健康 run 步态抬脚仅 2cm，任何 ≥3cm 的坎都
  # 撞墙卡死；训练中没有任何奖励抬脚）----
  # 1) 摆动峰值欠抬惩罚（velocity_vision 现成 class term，落地评估，
  #    命令门控 + 直立门控，欠抬重罚/过抬轻罚），目标 0.15m 已在 params；
  cfg.rewards["foot_swing_height"].weight = -0.5
  # 2) 踝击拖步惩罚（EP feet_stumble -1.0，取半起步）：脚尖横向冲击 >
  #    垂直支撑时罚，直立门控。
  cfg.rewards["stumble"].weight = -0.5
  # 3) 削弱静态姿态正奖励：默认关节角奖励与抬脚/攀爬姿态直接冲突。
  cfg.rewards["pose"].weight = 0.25

  # ---- upright 放宽（用户决定 2026-10-20）----
  # std sqrt(0.2) -> sqrt(0.5)：45° 前倾姿态奖励 0.08 -> 0.37，攀爬前倾
  # 不再重罚；深度翻倒仍由 bad_orientation 85° 终止兜底。
  cfg.rewards["upright"].params["std"] = math.sqrt(0.5)

  # ---- 初始难度最低（用户决定 2026-10-20）----
  # 所有 env 从 row 0（墙高 ~0-8 cm，近平障碍）开始学攀爬；
  # terrain_levels_vel 的升行条件（走出 4 m）在学会后自然推升难度。
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.max_init_terrain_level = 0

  # ---- 攀爬塑造奖励（用户决定 2026-10-20）----
  # 近墙区门控的连续梯度：贴近墙(0.3) + 前倾昂首(0.3) + 前爪上墙(0.5)。
  # 奖励点 bonus（登顶 2.0/点）仍是攀爬的主要报酬，塑造只负责搭梯子。
  cfg.rewards["wall_climb_shaping"] = RewardTermCfg(
    func=mdp.wall_climb_shaping,
    weight=1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", site_names=("FR", "FL")),
    },
  )

  # ---- 奖励点 milestone bonus（robot parkour 风格，主任务驱动） ----
  # 量级（w=1 体系）：单点入账 = bonus*dt = 2.0 ≈ 2 秒满速跟踪收益。
  # 梅花 9 点=18.0，矮墙 2 点/墙=4.0，沟壑 3 点=6.0——覆盖攀爬期
  # tracking(vz)+upright 损失(~2-3)，绕行则被航点超时推进/越界终止掐断。
  cfg.rewards["reward_point_bonus"] = RewardTermCfg(
    func=mdp.reward_point_bonus,
    weight=1.0,
    params={
      "command_name": "twist",
      "hit_radius": 0.2,
      "z_margin": 0.05,
      "min_forward_speed": 0.05,
      "speed_tol": 0.5,
      "command_threshold": 0.1,
      "bonus_value": 100.0,
    },
  )

  # ---- 方向类手工奖励全部移除（用户决定 2026-10-19）----
  # 航点命令本身携带方向信息（指向下一奖励点），与 forward_progress /
  # lateral_track / heading_alignment / cheat 重叠。绕行由航点超时推进与
  # out_of_bounds 兜底。函数定义保留在 mdp.rewards 中（旧实验 env.yaml
  # 转储以 !!python/name 引用它们，删除会破坏历史配置回放）。
  cfg.rewards.pop("stuck_penalty", None)
  for name in ("forward_progress", "lateral_track", "heading_alignment", "cheat"):
    cfg.rewards.pop(name, None)

  # ---- 成功终止：越过整条障碍链（最后奖励点 + margin）立即重置 ----
  cfg.terminations["parkour_completed"] = TerminationTermCfg(
    func=mdp.parkour_completed,
    params={"asset_cfg": SceneEntityCfg("robot"), "margin": 0.4},
  )

  # 躯干接触地形终止：墙列豁免（真垂直墙攀爬贴墙不可避免）。单地形任务下
  # 所有 env 都在墙列——等效于全局停用躯干接触终止（攀爬贴墙属正常姿态）。
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact_excluding,
    params={
      "sensor_name": "nonfoot_ground_touch",
      "exempt_terrain_types": (wall_column,),
    },
  )

  # ---- 移除倾角终止（v5，EP 对齐：EP 无任何倾角终止，摔倒自行翻回是
  # emergent re-trying 的前提）——翻 0.4-0.5m 墙时躯干俯仰可达 60-90°，
  # 75°/85° 终止在攀爬最关键的时刻杀掉尝试，攀爬永远得不到强化。 ----
  # 摔倒兜底：fell_into_pit（z<0.15）+ time_out（趴地无 upright/pose/
  # tracking 收益，自然抑制）。回退：恢复 bad_orientation_excluding 注册。
  del cfg.terminations["fell_over"]

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

  # ---- 接触惩罚豁免：墙列攀爬贴墙不可避免。单地形下等效于全局豁免
  # （头/大腿/小腿碰撞惩罚归零，攀爬贴墙不罚）。 ----
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
  # "远处障碍"，根治固定大俯角的视野盲区。
  # 注意符号：quat_from_euler_xyz 的 pitch 为正 = 下俯（wmp_go1 同款
  # "positive pitch down"）；负值会抬头看天（2026-10-15 play 实证）。----
  cfg.events["cam_quat"].params["pitch_range"] = (
    math.radians(15.0),
    math.radians(25.0),
  )
  cfg.events["cam_quat"].params["yaw_range"] = (
    math.radians(-2.0),
    math.radians(2.0),
  )

  # ---- 航点命令（Extreme-Parkour 风格，用户决定 2026-10-19）----
  # 每步动态指向下一个奖励点：机体系 [vx, vy] = 指向航点的单位方向 × 幅值，
  # wz = 朝向航点的 P 控制。幅值范围 [0.4, 0.8]（wmp_go1 forward 组起步值），
  # 分数课程升至 [0.2, 1.0]；ang_vel_z 为 wz 裁剪范围。lin_vel_y 不参与
  # （方向来自航点），仅为 Ranges 结构占位。5% 概率零指令（站立）。
  cfg.commands["twist"] = mdp.WaypointVelocityCommandCfg(
    entity_name="robot",
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.4, 0.8),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(-1.0, 1.0),
    ),
    debug_vis=True,
  )

  # 固定初始 yaw 朝世界 x，收紧出生随机范围（障碍链附近需留出自由半径）。
  cfg.events["reset_base"].params["pose_range"]["x"] = (-0.2, 0.2)
  cfg.events["reset_base"].params["pose_range"]["y"] = (-0.3, 0.3)
  cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)

  # ---- 指令课程：静态迭代阶段 -> 分数驱动（wmp_go1 forward 组） ----
  # 组内归一化 tracking 分数 > 0.8 时速度幅值上限 +0.2，封顶 1.0。
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

  # ---- 航点限时课程：通过率 > 0.8 时基础限时 -1 s，下限 3 s ----
  # 初期给足前往目标点的时间；可行性下限（dist/speed × 1.3）由命令侧公式
  # 硬性保证，课程收紧不会造成"命令速度下不可达"的死锁。
  cfg.curriculum["waypoint_timeout"] = CurriculumTermCfg(
    func=mdp.waypoint_timeout_schedule,
    params={
      "command_name": "twist",
      "bonus_term_name": "reward_point_bonus",
      "score_threshold": 0.8,
      "decrement": 1.0,
      "timeout_min": 3.0,
    },
  )
