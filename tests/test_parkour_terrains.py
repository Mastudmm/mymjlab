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


@pytest.mark.parametrize(
  "cfg_cls,low_n,high_n",
  [
    (BoxStraightStepStonesTerrainCfg, 9, 9),
    (BoxStraightPlatformTerrainCfg, 2, 2),
    (BoxStraightGapTerrainCfg, 2, 3),
  ],
)
def test_reward_point_count_scales_with_difficulty(cfg_cls, low_n, high_n):
  """Low difficulty rows place fewer obstacles (and reward points)."""
  low = _generate(cfg_cls, 0.1)
  high = _generate(cfg_cls, 0.9)
  assert low.reward_points is not None and high.reward_points is not None
  assert len(low.reward_points) == low_n
  assert len(high.reward_points) == high_n


def test_stone_gap_increases_with_difficulty():
  """Stones keep a fixed width; the edge gap grows 0 -> 0.2 m, uniformly for
  the spawn->first-pair gap, between pairs, and last pair -> end column."""
  cfg = BoxStraightStepStonesTerrainCfg(size=SIZE)
  low = _generate(BoxStraightStepStonesTerrainCfg, 0.0)
  high = _generate(BoxStraightStepStonesTerrainCfg, 1.0)
  assert low.reward_points is not None and high.reward_points is not None
  for out, gap in ((low, 0.0), (high, 0.2)):
    points = out.reward_points
    # Spawn column edge -> first pair edge uses the same gap.
    spawn_right = cfg.border_width + 0.5 + 0.5
    assert points[0][0] - cfg.stone_size / 2 - spawn_right == pytest.approx(gap)
    # Consecutive pair gaps use the same gap.
    pitch = points[1][0] - points[0][0]
    assert pitch == pytest.approx(cfg.stone_size + gap)
    # Last pair edge -> end column edge uses the same gap. The end column
    # carries the final reward point at its center (half width 0.5).
    last_pair_right = points[-2][0] + cfg.stone_size / 2
    end_col_center = points[-1][0]
    assert end_col_center - 0.5 - last_pair_right == pytest.approx(gap)


@pytest.mark.parametrize("cfg_cls", ALL_PARKOUR_CFGS)
def test_reward_points_inside_patch_and_clear_of_spawn(cfg_cls):
  """Reward points lie inside the patch and clear of the spawn column."""
  output = _generate(cfg_cls, 0.9)
  assert output.reward_points is not None
  cfg = cfg_cls(size=SIZE)
  spawn_x = cfg.border_width + 0.5
  for point in output.reward_points:
    assert cfg.border_width < point[0] < SIZE[0] - cfg.border_width
    assert cfg.border_width < point[1] < SIZE[1] - cfg.border_width
    # No obstacle (hence reward point) within the spawn column's footprint.
    assert point[0] >= spawn_x + 0.5 - 1e-6


def test_stone_top_height_increases_with_difficulty():
  """The stone top (reward point z) rises from 0.2 m toward 0.35 m."""
  low = _generate(BoxStraightStepStonesTerrainCfg, 0.0)
  high = _generate(BoxStraightStepStonesTerrainCfg, 1.0)
  assert low.reward_points is not None and high.reward_points is not None
  assert low.reward_points[0][2] == pytest.approx(0.2)
  assert high.reward_points[0][2] > low.reward_points[0][2]


def test_platform_height_increases_with_difficulty():
  """The platform top (reward point z) follows the 0.2 -> 0.5 m ladder with
  a ±0.05 m per-patch jitter."""
  low = _generate(BoxStraightPlatformTerrainCfg, 0.0)
  high = _generate(BoxStraightPlatformTerrainCfg, 1.0)
  assert low.reward_points is not None and high.reward_points is not None
  assert low.reward_points[0][2] == pytest.approx(0.2, abs=0.055)
  assert high.reward_points[0][2] == pytest.approx(0.5, abs=0.055)
  assert high.reward_points[0][2] > low.reward_points[0][2]


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


def _solids_xy(output):
  """Collect (x, y, half_x, half_y) of stone/column boxes (orange + green),
  excluding the border ring and pit floor which legitimately touch."""

  def is_a(geom, color):
    return geom.color is not None and tuple(geom.color) == tuple(color)

  boxes = []
  for g in output.geometries:
    if g.geom is None:
      continue
    if is_a(g, (0.85, 0.55, 0.15, 1.0)) or is_a(g, (0.3, 0.55, 0.3, 1.0)):
      pos, size = g.geom.pos, g.geom.size
      boxes.append((pos[0], pos[1], size[0], size[1]))
  return boxes


@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_step_stone_solids_do_not_overlap(difficulty):
  """Stones and solid columns must never overlap in the xy plane."""
  cfg_cls = BoxStraightStepStonesTerrainCfg
  output = _generate(cfg_cls, difficulty)
  boxes = _solids_xy(output)
  assert len(boxes) >= 2
  for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
      x1, y1, hx1, hy1 = boxes[i]
      x2, y2, hx2, hy2 = boxes[j]
      overlap_x = abs(x1 - x2) < (hx1 + hx2) - 1e-6
      overlap_y = abs(y1 - y2) < (hy1 + hy2) - 1e-6
      assert not (overlap_x and overlap_y), (
        f"solids overlap: {boxes[i]} vs {boxes[j]} at difficulty {difficulty}"
      )


def test_parkour_generator_collects_reward_points():
  """The generator fills per-(row, col) reward-point arrays for the preset."""
  cfg = PARKOUR_TERRAINS_CFG
  spec = mujoco.MjSpec()
  generator = TerrainGenerator(cfg)
  generator.compile(spec)

  assert generator.reward_points is not None
  assert generator.reward_counts is not None
  assert generator.reward_points.shape == (cfg.num_rows, cfg.num_cols, 9, 3)
  # Every curriculum patch places at least one obstacle (and reward point).
  assert (generator.reward_counts > 0).all()
