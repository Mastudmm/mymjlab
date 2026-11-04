from dataclasses import replace

import mujoco

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),  # 每个小地形的大小
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.0),
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.1,
      step_height_range=(0.0, 0.2),
      step_width_range=(0.25, 0.45),
      platform_width=3.0,
      border_width=1.0,
      stair_lip_enabled=False,
      stair_lip_outward=0.03,
      stair_lip_downward=0.04,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.8,
      step_height_range=(
        0.0,
        0.20,
      ),  # 台阶高度可以在play的时候自由修改 train的时候尽量高
      step_width_range=(0.25, 0.35),
      step_width_mode="random",
      platform_width=3.0,
      border_width=1.0,
      stair_lip_enabled=False,
      stair_lip_outward=0.04,
      stair_lip_downward=0.03,
    ),
    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.0,
      slope_range=(0.0, 1.0),
      platform_width=2.0,
      border_width=0.25,
    ),
    "hf_pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.0,
      slope_range=(0.0, 0.75),
      platform_width=2.0,
      border_width=0.25,
      inverted=True,
    ),
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.1,
      noise_range=(0.01, 0.1),
      noise_step=0.02,
      border_width=0.25,
    ),
    "wave_terrain": terrain_gen.HfWaveTerrainCfg(
      proportion=0.0,
      amplitude_range=(0.0, 0.2),
      num_waves=4,
      border_width=0.25,
    ),
  },
  add_lights=True,
)

ALL_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=16,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.0),
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.0,
      step_height_range=(0.0, 0.4),
      step_width_range=(0.3, 0.6),
      platform_width=3.0,
      border_width=1.0,
      stair_lip_enabled=True,
      stair_lip_outward=0.03,
      stair_lip_downward=0.04,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=10.5,
      step_height_range=(0.18, 0.2),
      step_width_range=(0.24, 0.3),
      platform_width=3.0,
      border_width=1.0,
      stair_lip_enabled=True,
      stair_lip_outward=0.01,
      stair_lip_downward=0.04,
    ),
    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.0,
      slope_range=(0.0, 0.7),
      platform_width=2.0,
      border_width=0.25,
    ),
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.0,
      noise_range=(0.02, 0.10),
      noise_step=0.02,
      border_width=0.25,
    ),
    "wave_terrain": terrain_gen.HfWaveTerrainCfg(
      proportion=0.0,
      amplitude_range=(0.0, 0.2),
      num_waves=4,
      border_width=0.25,
    ),
    "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
      proportion=0.0,
      obstacle_width_range=(0.3, 1.0),
      obstacle_height_range=(0.05, 0.3),
      num_obstacles=40,
      border_width=0.25,
    ),
    "perlin_noise": terrain_gen.HfPerlinNoiseTerrainCfg(
      proportion=0.0,
      height_range=(0.0, 1.0),
      octaves=4,
      persistence=0.3,
      lacunarity=2.0,
      scale=10.0,
      horizontal_scale=0.1,
      border_width=0.50,
    ),
    "box_random_grid": terrain_gen.BoxRandomGridTerrainCfg(
      proportion=0.0,
      grid_width=0.4,
      grid_height_range=(0.0, 0.3),
      platform_width=1.0,
    ),
    "random_spread_boxes": terrain_gen.BoxRandomSpreadTerrainCfg(
      proportion=0.0,
      num_boxes=80,
      box_width_range=(0.1, 1.0),
      box_length_range=(0.1, 2.0),
      box_height_range=(0.05, 0.3),
      platform_width=1.0,
      border_width=0.25,
    ),
    "open_stairs": terrain_gen.BoxOpenStairsTerrainCfg(
      proportion=0.0,
      step_height_range=(0.01, 0.2),
      step_width_range=(0.4, 0.8),
      platform_width=1.0,
      border_width=0.25,
    ),
    "random_stairs": terrain_gen.BoxRandomStairsTerrainCfg(
      proportion=0.0,
      step_width=0.8,
      step_height_range=(0.1, 0.3),
      platform_width=1.0,
      border_width=0.25,
    ),
    "stepping_stones": terrain_gen.BoxSteppingStonesTerrainCfg(
      proportion=0.0,
      stone_size_range=(0.3, 0.5),
      stone_distance_range=(0.0, 0.3),
      stone_height=0.2,
      stone_height_variation=0.05,
      stone_size_variation=0.02,
      displacement_range=0.02,
      floor_depth=2.0,
      platform_width=1.0,
      border_width=0.25,
    ),
    "narrow_beams": terrain_gen.BoxNarrowBeamsTerrainCfg(
      proportion=0.0,
      num_beams=12,
      beam_width_range=(0.2, 0.8),
      beam_height=0.2,
      spacing=0.8,
      platform_width=1.0,
      border_width=0.25,
      floor_depth=2.0,
    ),
    "nested_rings": terrain_gen.BoxNestedRingsTerrainCfg(
      proportion=0.5,
      num_rings=6,
      ring_width_range=(0.4, 0.4),
      gap_range=(0.25, 0.25),
      height_range=(0.01, 0.02),
      platform_width=1.25,
      border_width=0.25,
      floor_depth=0.2,
    ),
    "tilted_grid": terrain_gen.BoxTiltedGridTerrainCfg(
      proportion=0.0,
      grid_width=1.0,
      tilt_range_deg=20.0,
      height_range=0.1,
      platform_width=1.0,
      border_width=0.25,
      floor_depth=2.0,
    ),
  },
  add_lights=True,
)

PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  # 五列任务（用户定版 2026-11-01）：每列一种地形，等比例各占一列。
  # 列序由 sub_terrains 的书写顺序决定（cumulative proportion 分配，见
  # terrain_generator.resolve_curriculum_column_names）：
  #   col0=platforms  col1=platform_gap  col2=crawl
  #   col3=pyramid_stairs(逆台阶,中心高台)  col4=pyramid_stairs_inv(正台阶,坑底)
  # 出生在哪列可由 TerrainEntityCfg.spawn_terrain_names 过滤（与布局解耦）；
  # 2026-11-04 起训练用 ("platforms", "crawl") + probs (0.75, 0.25)（见
  # parkour_env_cfg.apply_parkour_overrides），gap/台阶列仍生成但零出生。
  num_cols=5,
  curriculum=True,
  sub_terrains={
    # 2026-10-27：几何参数全部显式写出（原先走 dataclass 默认值，改墙高必须
    # 翻 parkour_terrains.py 才知道有哪些旋钮）。这里是**唯一**该改墙的地方。
    "platforms": terrain_gen.BoxStraightPlatformTerrainCfg(
      proportion=1.0,
      # 墙高阶梯：difficulty 0 -> 0.0 m，difficulty 1 -> 0.5 m（10 行线性插值），
      # 再加 ±height_jitter 抖动。想要更高的墙就抬上限（上限会被
      # clip 到 0.55 m，见 parkour_terrains.BoxStraightPlatformTerrainCfg）。
      platform_height_range=(0.0, 0.5),
      height_jitter=0.05,
      # 墙沿 x 的厚度（可落脚顶面深度），每个 patch 随机采样。
      platform_depth_range=(0.4, 1.2),
      # 墙沿 y 的宽度：7.4-7.5 m 已接近 patch 全宽（8 m），机器人无法绕行。
      platform_width_range=(7.4, 7.5),
      # 第一面墙的左边缘 x（相对 patch 原点），随机采样。按左缘（而非中心）
      # 采样，使加速区与墙深采样解耦：出生点 0.75 m -> 加速区最差 1.35 m
      # （旧中心采样最差仅 0.45 m，机器人未加速即到墙，2026-10-30 修复）。
      wall1_left_x_range=(2.3, 2.7),
      # 两面墙之间的平地间距（边缘到边缘，自然值）。墙2 被下方上限夹紧时
      # 会压缩，最差 ~1.5 m，典型 2.5 m（wmp_go1 climb 的最小值）。
      wall_gap_range=(2.5, 2.9),
      # 第二面墙右边缘上限。wmp_go1 把墙2右缘钉在 8 m 单元边界上；这里同理
      # 把墙2推向 patch 边缘，保证终点引导点（右缘+finish_offset）与完成线
      # （引导点+0.4 margin）留在 patch 内，终止线 <= 8.0 m 恰好贴边
      # （旧值 6.8 在终点后留下 0.7-2.4 m 闲置地面，2026-10-30 修复）。
      wall2_right_max=6.6,
      # 墙2 后两点的偏移：落地里程碑（右缘+0.5，高价值，支付"翻越"）与
      # 终点引导点（右缘+1.0，低价值、链尾——parkour_completed 在链尾
      # +0.4 m 处结束回合，终点 bonus 距墙2 的距离因此翻倍，2026-10-30）。
      landing_offset=0.5,
      finish_offset=1.0,
      border_width=0.25,
      floor_thickness=0.5,
      # 每个 patch 的奖励点数量上限（低-高引导链：接近段 2 低值点 +
      # 墙1顶 + 下台里程碑(墙1后地面,高——2026-11-01 由低值升级，解决
      # "爬上墙1就趴在墙顶"的蹲营问题) + 墙2前低值点 + 落地里程碑 +
      # 终点引导点 = 7 点 = 3高4低）。
      max_reward_points=7,
      # 低价值引导点的奖励系数（相对里程碑 1.0 的比例）——低/高 bonus 点的
      # 价值比旋钮（2026-11-01 显式暴露）。单点入账 =
      # weight(1.25) × bonus_value(100) × 本系数 × dt(0.02)。
      low_bonus_ratio=0.2,
    ),
    # col1：高台+跨缝（用户定版 2026-11-01，同日修订）：两块**完全相同**的
    # 高台（高度/深度/宽度全共享一次采样），顶面之间 0->0.3 m 的真缝（挖到
    # 坑底，必须跳）。台1 左缘推远到 2.8-3.1 m（出生加速跑道 2.05-2.35 m），
    # 台深 1.2-1.6 m 保证上台后到缝前有足空间加速（用户 2026-11-01）。
    # bonus 链 5 点：3 低（上台前30cm / 下台后30cm / 终点100cm）+
    # 2 高（上台后 / 跨gap后）。
    "platform_gap": terrain_gen.BoxStraightPlatformGapTerrainCfg(
      proportion=1.0,
      platform_height_range=(0.0, 0.5),
      height_jitter=0.05,
      platform_depth_range=(1.2, 1.6),
      platform_width_range=(7.4, 7.5),
      wall1_left_x_range=(2.8, 3.1),
      gap_range=(0.0, 0.3),
      pit_depth_range=(0.5, 1.0),
      approach_offset=0.30,
      cross_offset=0.25,
      descend_offset=0.30,
      finish_offset=1.0,
      border_width=0.25,
      max_reward_points=5,
      # 低/高 bonus 点价值比（2026-11-01 显式暴露，见 platforms 注释）。
      low_bonus_ratio=0.2,
    ),
    # col2：低趴过横杆（wmp_go1 crawl 量纲）：2 根全宽悬空横杆，净高
    # 0.42->0.25 m 随难度收紧（2026-11-04 用户定版：不要压太低，横杆最低
    # 离地 25 cm；旧值 0.27），杆顶 >=1.25 m 跳不过去，只能趴低钻。
    # bonus 链 5 点：3 低（杆1前30cm / 两杆间 / 终点100cm）+
    # 2 高（过杆1后 / 过杆2后，各 30cm）。
    "crawl": terrain_gen.BoxStraightCrawlBarsTerrainCfg(
      proportion=1.0,
      clearance_range=(0.42, 0.25),
      bar_height=1.0,
      bar_thickness_range=(0.2, 0.5),
      bar_width=7.5,
      bar1_left_x_range=(3.2, 3.6),
      bar2_left_x_range=(5.8, 6.2),
      approach_offset=0.30,
      pass_offset=0.30,
      finish_offset=1.0,
      border_width=0.25,
      floor_thickness=0.5,
      max_reward_points=5,
      # 低/高 bonus 点价值比（2026-11-01 显式暴露，见 platforms 注释）。
      low_bonus_ratio=0.2,
    ),
    # col3/col4：mjlab 原生金字塔台阶（用户定版 2026-11-01：原生量纲，
    # 步高 1cm 起步 -> 21cm，步宽 0.3-0.34 m 每 patch 随机）。
    # 出生在 patch 中心 platform 上：
    #   pyramid_stairs      中心是顶台 -> 任意方向都是下台阶（逆台阶）；
    #   pyramid_stairs_inv  中心是坑底 -> 任意方向都是上台阶（正台阶）。
    # 两列无 reward_points -> 命令走 heading 模式（见 waypoint_command）。
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(0.01, 0.21),
      step_width_range=(0.3, 0.34),
      step_width_mode="random",
      platform_width=3.0,
      border_width=1.0,
      stair_lip_enabled=False,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(0.01, 0.21),
      step_width_range=(0.3, 0.34),
      step_width_mode="random",
      platform_width=3.0,
      border_width=1.0,
      stair_lip_enabled=False,
    ),
  },
  add_lights=True,
)


# ---- 全地形 play 预设（用户需求 2026-11-03；段序 2026-11-04 调整）----
# play 时不再按地形分类各自成列，而是把这几种地形**排成一条直线**：
# 机器人从最左端出生，依次穿过
#   crawl(低趴过横杆) -> platforms(翻越高台)
#   -> pyramid_stairs(中心高台台阶：从边缘进入 = 先上台阶，到顶台后再下台阶)
#   -> platform_gap(高台+跨缝，压轴收尾)
# 顺序即机器人前进方向（2026-11-04 用户定版：gap 挪到台阶之后，其余段
# 向前顺延一位）。
#
# 几何参数**唯一来源**仍是上面的 PARKOUR_TERRAINS_CFG：这里用 dataclasses.replace
# 派生新实例（不改动原对象），只为把每段 size 固定成原生 8 m patch 长度，避免
# 生成器在别处改 size 时让拼接总长与段内几何失配（拼接类在生成期会做
# 长度自检，失配直接报错）。每段沿用原生 8 m，因此整条线长 = 4×8 = 32 m、
# 宽 8 m；10 行难度阶梯原样保留：同一行内四段共用同一 difficulty，行号
# 递增 = 整条线难度递增。
#
# 只在 play 生效（play.py 的 --full-terrain + parkour_env_cfg 的
# apply_full_course_play_overrides，命令切换为「始终指向链尾终点」）；
# 训练仍用五列网格的 PARKOUR_TERRAINS_CFG，行为不变。
#
# 链尾（2026-11-04）：末段 platform_gap 自带终点引导点（段内 x = 墙2右缘
# +1.0 m，worst case 7.6 m），因此链条**不再**补合成终值点——无条件补点会
# 与该引导点撞碎「奖励点严格递增」自检（合成点固定在总长前 0.5 m = 段内
# 7.5 m < 7.6 m）。补点改为条件式：仅末段无奖励点（如原生台阶）时才补，
# 见 BoxStraightChainTerrainCfg。
_FULL_COURSE_SEGMENT_SIZE = (8.0, 8.0)
# 段名必须与 PARKOUR_TERRAINS_CFG 的 sub_terrains 键一致（写错在 import 期
# 就会 KeyError，不会静默拼错路线）。
_FULL_COURSE_SEGMENT_NAMES = ("crawl", "platforms", "pyramid_stairs", "platform_gap")

FULL_PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(
    _FULL_COURSE_SEGMENT_SIZE[0] * len(_FULL_COURSE_SEGMENT_NAMES),
    _FULL_COURSE_SEGMENT_SIZE[1],
  ),
  border_width=20.0,
  # 10 行 = 10 档整条线难度（--terrain-level 选行）；1 列 = 唯一的一条路线
  # （所有 env 都出生在同一条线上，行不同即难度不同）。
  num_rows=10,
  num_cols=1,
  curriculum=True,
  sub_terrains={
    "full_course": terrain_gen.BoxStraightChainTerrainCfg(
      proportion=1.0,
      segments=tuple(
        replace(
          PARKOUR_TERRAINS_CFG.sub_terrains[name],
          size=_FULL_COURSE_SEGMENT_SIZE,
        )
        for name in _FULL_COURSE_SEGMENT_NAMES
      ),
    ),
  },
  add_lights=True,
)


if __name__ == "__main__":
  import mujoco.viewer
  import torch

  device = "cuda" if torch.cuda.is_available() else "cpu"

  terrain_cfg = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
  )
  terrain = TerrainEntity(terrain_cfg, device=device)
  mujoco.viewer.launch(terrain.spec.compile())
