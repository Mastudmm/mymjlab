"""Tests for the straight-line parkour terrains."""

import mujoco
import numpy as np
import pytest

from mjlab.terrains.config import PARKOUR_TERRAINS_CFG
from mjlab.terrains.parkour_terrains import (
  BoxStraightGapTerrainCfg,
  BoxStraightPlatformTerrainCfg,
  BoxStraightStepStonesTerrainCfg,
)
from mjlab.terrains.terrain_generator import TerrainGenerator

SIZE = (8.0, 8.0)

ALL_PARKOUR_CFGS = [
  BoxStraightStepStonesTerrainCfg,
  BoxStraightPlatformTerrainCfg,
  BoxStraightGapTerrainCfg,
]


def _generate(cfg_cls, difficulty: float):
  cfg = cfg_cls(size=SIZE)
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  rng = np.random.default_rng(0)
  return cfg.function(difficulty, spec, rng)


@pytest.mark.parametrize("cfg_cls", ALL_PARKOUR_CFGS)
def test_reward_point_count_scales_with_difficulty(cfg_cls):
  """Low difficulty rows place fewer obstacles (and reward points)."""
  low = _generate(cfg_cls, 0.1)
  high = _generate(cfg_cls, 0.9)
  assert low.reward_points is not None and high.reward_points is not None
  assert len(low.reward_points) == 2
  assert len(high.reward_points) == 3


@pytest.mark.parametrize("cfg_cls", ALL_PARKOUR_CFGS)
def test_reward_points_inside_patch_and_spawning_free_radius(cfg_cls):
  """Reward points lie inside the patch and clear of the spawn zone."""
  output = _generate(cfg_cls, 0.9)
  assert output.reward_points is not None
  cfg = cfg_cls(size=SIZE)
  spawn_x = cfg.border_width + 0.5
  for point in output.reward_points:
    assert cfg.border_width < point[0] < SIZE[0] - cfg.border_width
    assert cfg.border_width < point[1] < SIZE[1] - cfg.border_width
    # No obstacle (hence reward point) within the spawn free radius.
    assert point[0] >= spawn_x + cfg.spawn_free_radius - 1e-6


def test_stone_top_height_increases_with_difficulty():
  """The stone top (reward point z) rises from 0.2 m toward 0.35 m."""
  low = _generate(BoxStraightStepStonesTerrainCfg, 0.0)
  high = _generate(BoxStraightStepStonesTerrainCfg, 1.0)
  assert low.reward_points is not None and high.reward_points is not None
  assert low.reward_points[0][2] == pytest.approx(0.2)
  assert high.reward_points[0][2] > low.reward_points[0][2]


def test_platform_height_increases_with_difficulty():
  """The platform top (reward point z) rises from 0.15 m toward 0.35 m."""
  low = _generate(BoxStraightPlatformTerrainCfg, 0.0)
  high = _generate(BoxStraightPlatformTerrainCfg, 1.0)
  assert low.reward_points is not None and high.reward_points is not None
  assert low.reward_points[0][2] == pytest.approx(0.15)
  assert high.reward_points[0][2] == pytest.approx(0.35)


def test_gap_reward_point_past_far_edge():
  """Gap reward points sit on the walkable ground past each gap's far edge."""
  cfg = BoxStraightGapTerrainCfg(size=SIZE)
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  rng = np.random.default_rng(0)
  output = cfg.function(1.0, spec, rng)

  assert output.reward_points is not None
  # Every reward point lies on the landing-side ground (z=0), inside the patch.
  for point in output.reward_points:
    assert point[2] == pytest.approx(0.0)
    assert point[0] < SIZE[0] - cfg.border_width


def test_spawn_origin_near_minus_x_edge():
  """All parkour terrains spawn near the -x edge, facing the +x chain."""
  for cfg_cls in ALL_PARKOUR_CFGS:
    output = _generate(cfg_cls, 0.5)
    cfg = cfg_cls(size=SIZE)
    expected_x = cfg.border_width + 0.5
    assert output.origin[0] == pytest.approx(expected_x)
    assert output.origin[1] == pytest.approx(SIZE[1] / 2)


def test_parkour_generator_collects_reward_points():
  """The generator fills per-(row, col) reward-point arrays for the preset."""
  cfg = PARKOUR_TERRAINS_CFG
  spec = mujoco.MjSpec()
  generator = TerrainGenerator(cfg)
  generator.compile(spec)

  assert generator.reward_points is not None
  assert generator.reward_counts is not None
  assert generator.reward_points.shape == (cfg.num_rows, cfg.num_cols, 3, 3)
  # Every curriculum patch places at least one obstacle (and reward point).
  assert (generator.reward_counts > 0).all()
