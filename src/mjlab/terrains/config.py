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
  # 单地形任务（用户决定 2026-10-19）：只保留翻越高墙（矮墙/平台），梅花桩
  # 与沟壑已移除——聚焦单一障碍类型更适配当前 MLP+CNN 网络结构。
  # 保留 3 列：同难度行有 3 个独立随机布局的墙 patch，增加 patch 多样性。
  num_cols=3,
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
      # 墙1顶 + 墙间 2 低值点 + 落地里程碑 + 终点引导点 = 7 点，
      # 2026-10-30 由 6 扩到 7）。
      max_reward_points=7,
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
