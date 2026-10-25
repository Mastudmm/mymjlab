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
from mjlab.envs import mdp as envs_mdp
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

  # ======== 奖励尺度再平衡（用户决定 2026-10-25）========
  # 实证（HEAD 配置、8 env、200 步零动作 rollout）：
  #   站立不动 = +2.05 /s ≈ 41 分/20 s 回合，其中 track_angular_velocity
  #   +0.95/s、upright +0.95/s、pose +0.20/s 全部"免费"拿满；而整条跑酷链
  #   2 个奖励点合计只有 4 分。于是 PPO 收敛到"站着"退化解——日志佐证：
  #   2236 次迭代 reward_point_bonus 恒为 0、terrain_levels 衰减到 0、
  #   mean_reward≈48 ≈ 站立收益。
  #
  # 再平衡原则：只让「跟踪速度（稠密）」与「奖励点（稀疏里程碑）」当主力，
  # 姿态类正奖励降级为"稳定器"（防后仰刷分、保持可站立姿态），不再是收益源。
  # 目标预算（/s 速率，weight×raw 后）：
  #   站立: upright 0.19 + pose 0.08 + angular 0.29            ≈ 0.56/s (11 分)
  #   行走: tracking 0.9 + upright 0.18 + angular 0.25
  #         + pose 0.08 + air_time 0.4 - 动作类惩罚 ≈ 0.5       ≈ 1.3/s (26 分)
  #   通关: 行走 26 + 2 点 × 20 分 = 66 分                     → 站立的 6 倍
  # 三处都是单行可回退。
  cfg.rewards["track_angular_velocity"].weight = 0.3  # 1.0 -> 0.3（站立白拿 ~0.95/s）
  cfg.rewards["upright"].weight = 0.2  # 1.0 -> 0.2（站立白拿 ~0.95/s）

  # ---- 抬脚包（v5 根因修复：健康 run 步态抬脚仅 2cm，任何 ≥3cm 的坎都
  # 撞墙卡死；训练中没有任何奖励抬脚）----
  # 1) 摆动峰值欠抬惩罚（velocity_vision 现成 class term，落地评估，
  #    命令门控 + 直立门控，欠抬重罚/过抬轻罚），目标 0.15m 已在 params；
  cfg.rewards["foot_swing_height"].weight = -0.5
  # 2) 踝击拖步惩罚（EP feet_stumble -1.0，取半起步）：脚尖横向冲击 >
  #    垂直支撑时罚，直立门控。
  cfg.rewards["stumble"].weight = -0.5
  # 3) 削弱静态姿态正奖励：默认关节角奖励与抬脚/攀爬姿态直接冲突。
  #    0.25 -> 0.1（再平衡）：站立时 pose 白拿 ~0.2/s，必须压到稳定器量级。
  cfg.rewards["pose"].weight = 0.1

  # ---- upright 放宽（用户决定 2026-10-20）----
  # std sqrt(0.2) -> sqrt(0.5)：45° 前倾姿态奖励 0.08 -> 0.37，攀爬前倾
  # 不再重罚；深度翻倒仍由 bad_orientation 85° 终止兜底。
  cfg.rewards["upright"].params["std"] = math.sqrt(0.5)

  # ---- 初始难度最低（用户决定 2026-10-20）----
  # 所有 env 从 row 0（墙高 ~0-8 cm，近平障碍）开始学攀爬；
  # terrain_levels_vel 的升行条件（走出 4 m）在学会后自然推升难度。
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.max_init_terrain_level = 0

  # ---- 攀爬塑造奖励（用户决定 2026-10-20，2026-10-25 降权）----
  # 近墙区门控的连续梯度：贴近墙(0.3) + 前倾昂首(0.3) + 前爪上墙(0.5)，
  # 原件满值 ≈ 1.1 —— 会高于跟踪奖励(0.9)，且"贴墙抬头不动"就能白拿
  # approach+pitch ≈ 0.6/s，又变成一个局部最优。权重 1.0 -> 0.5 后满值
  # ≈ 0.55/s，低于 tracking(0.9) 与单点 bonus(20)，只负责"搭梯子"。
  # 主报酬仍是奖励点 bonus（单点 20 分）。
  cfg.rewards["wall_climb_shaping"] = RewardTermCfg(
    func=mdp.wall_climb_shaping,
    weight=0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", site_names=("FR", "FL")),
    },
  )

  # ---- 奖励点 milestone bonus（主任务驱动，2026-10-25 再平衡后为收益主项） ----
  # 量级（w=1 体系）：单点入账 = bonus_value * dt。dt = 0.02 s，因此
  # bonus_value 1000 -> 单点 20 分，两点共 40 分 ≈ 站立收益(11 分)的 3.6 倍、
  # 使"打通障碍链"成为全局最优，而不是"站着不动"。
  # 历史：bonus_value=100（单点 2 分、两点 4 分）时，跑酷总收益只有站立的
  # 1/10，是训练收敛到站立退化解的直接原因之一。
  # 回退：把 bonus_value 改回 100.0 即恢复到旧量级。
  cfg.rewards["reward_point_bonus"] = RewardTermCfg(
    func=mdp.reward_point_bonus,
    weight=1.0,
    params={
      "command_name": "twist",
      # 0.2 -> 0.35：必须 ≥ 航点命令的 reach_radius(0.3)。否则机器人在
      # 0.25-0.3 m 处就被判定"到达"并切走下一个航点，永远不会进入 0.2 m
      # 计奖区 → 白跑一趟（这是整程 bonus 恒为 0 的坑之一）。
      "hit_radius": 0.35,
      "z_margin": 0.05,
      "min_forward_speed": 0.05,
      "speed_tol": 0.5,
      "command_threshold": 0.1,
      "bonus_value": 1000.0,
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

  # ---- 恢复倾角终止，放宽到 85°（用户决定 2026-10-25）----
  # v5 曾整体删除倾角终止（EP 对齐），但实证代价是：摔倒的机器人不会被
  # 重置，白白烧掉整个 20 s 回合（单地形下 illegal_contact 的墙列豁免
  # 覆盖全部 env，也没有躯干接触兜底），只产出噪声。
  # 85° 的含义：0.4-0.5 m 墙攀爬时躯干俯仰 60-85° 全部放行（这是攀爬
  # 必需姿态），只有真正翻过去/侧翻（>85°）才终止——既保住"攀爬不被误
  # 杀"，又不让倒地机器人占着回合。
  # 注意：这里用 envs_mdp.bad_orientation 而**不做地形列豁免**。单地形
  # 任务下 terrain_types 恒为 0，任何 exempt_terrain_types=(0,) 都会变成
  # 全环境豁免 = 等于没有终止（v5 删除它的真实原因）。
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=envs_mdp.bad_orientation,
    params={"limit_angle": math.radians(85.0)},
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
  # 2026-10-25：history_length 1 -> 3（3 帧深度历史）+ 15% 整帧随机失效。
  #  - 多帧：单帧看不出"障碍边缘在靠近"的运动趋势；DepthActorCritic 已支持
  #    多帧（共用 CNN 逐帧提特征，见 rl/modules.py:depth_history_num），
  #    只需这里给 3 帧、并在 rl_cfg 里把 depth_history_num 同步为 3。
  #  - 随机失效（dropout_prob）：模拟实机反光/过曝/丢帧，逼策略学会回退到
  #    本体感知；配合 3 帧历史，失效帧会落在历史窗口内（正是部署时的表现）。
  # play 时在 unitree_go1_parkour_env_cfg 里把 dropout_prob 置 0。
  cfg.observations["actor"].terms["depth"] = ObservationTermCfg(
    func=mdp.depth_camera_normalized,
    params={
      "sensor_cfg": SceneEntityCfg("depth_camera"),
      "far_clip": 3.0,
      "dropout_prob": 0.15,
    },
    history_length=3,
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

  # ---- 地形难度课程：改用奖励点通过率驱动（用户决定 2026-10-25）----
  # 原 terrain_levels_vel 的升级判据是"走出 patch/2 (=4 m)"，在有墙的 patch 上
  # 严重失真：机器人可以沿平地走满 4 m（或贴着墙绕圈）却一次没上墙，照样被
  # 送上更高难度；而爬得慢的反而被判"没走动"降级。历史日志正是如此：
  # terrain_levels 在训练中衰减到 0，同时 reward_point_bonus 全程为 0。
  # 换成"本回合奖励点通过率 >= 0.5 升一行，<= 0 降一行"，让难度和真正的
  # 任务里程碑绑定；与 waypoint_timeout 课程读同一份 claimed 缓冲。
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=mdp.terrain_levels_by_pass_rate,
    params={
      "bonus_term_name": "reward_point_bonus",
      "promote_rate": 0.5,
      "demote_rate": 0.0,
    },
  )

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
