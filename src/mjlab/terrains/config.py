import mujoco

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg



ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),#每个小地形的大小
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=110.0),
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.1,
      step_height_range=(0.0, 0.4),
      step_width_range=(0.3,0.6),
      platform_width=3.0,
      border_width=1.0,
      stair_lip_enabled=True,
      stair_lip_outward=0.03,
      stair_lip_downward=0.04,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.9,
      step_height_range=(0.15,0.15), #台阶高度可以在play的时候自由修改 train的时候尽量高
      step_width_range=(0.25,0.25),
      platform_width=3.0,
      border_width=1.0,
      stair_lip_enabled=True,
      stair_lip_outward=0.03,
      stair_lip_downward=0.04,
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
      proportion=0.0,
      noise_range=(0.02, 0.15),
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
      step_width_range=(0.3,0.6),
      platform_width=3.0,
      border_width=1.0,
      stair_lip_enabled=True,
      stair_lip_outward=0.03,
      stair_lip_downward=0.04,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=10.5,
      step_height_range=(0.18, 0.2),
      step_width_range=(0.24,0.3),
      platform_width=3.0,
      border_width=1.0,
      stair_lip_enabled=True,
      stair_lip_outward=0.03,
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
