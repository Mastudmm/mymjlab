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
    (BoxStraightStepStonesTerrainCfg, 2, 2),
    (BoxStraightPlatformTerrainCfg, 7, 7),
    (BoxStraightGapTerrainCfg, 2, 2),
  ],
)
def test_reward_point_count_scales_with_difficulty(cfg_cls, low_n, high_n):
  """Low difficulty rows place fewer obstacles (and reward points).

  BoxStraightPlatformTerrainCfg emits the low-high guide chain (2026-10-29,
  extended 2026-10-30): 2 approach lows + wall-1 top + 2 gap lows + landing
  milestone + finish guide = exactly 7 — wall 1 is sampled by left edge
  (>= 2.3 m), so both approach slots always fit their standoff/spacing
  constraints and the count no longer varies with wall geometry."""
  low = _generate(cfg_cls, 0.1)
  high = _generate(cfg_cls, 0.9)
  assert low.reward_points is not None and high.reward_points is not None
  assert low_n <= len(low.reward_points) <= high_n
  assert low_n <= len(high.reward_points) <= high_n


def test_platform_chain_structure():
  """The platform guide chain is x-ordered with exactly TWO high-value
  milestones (wall-1 top, landing) and a LOW-value finish guide as the
  chain tail (2026-10-30): parkour_completed reads the tail's position, so
  the ending bonus sits finish_offset past wall 2 without inflating the
  milestone payout."""
  output = _generate(BoxStraightPlatformTerrainCfg, 0.5)
  assert output.reward_points is not None
  assert output.reward_point_factors is not None
  pts = np.stack(output.reward_points)
  fac = np.stack(output.reward_point_factors)
  assert len(pts) == len(fac) == 7
  assert int((fac == 1.0).sum()) == 2
  assert int((fac == 0.2).sum()) == len(pts) - 2
  # Strictly x-ordered (chain semantics: spawn -> ... -> finish guide).
  assert np.all(np.diff(pts[:, 0]) > 0)
  # The two milestones are the wall-1 top (z == wall height) and the landing
  # (second to last); the finish guide tail sits on the ground (z == 0).
  assert fac[0] == 0.2  # first approach point is a guide
  assert fac[-2] == 1.0  # landing milestone pays for crossing wall 2
  assert fac[-1] == 0.2  # finish guide tail (low value)
  assert pts[-2][2] == pytest.approx(0.0)  # landing is on the ground
  assert pts[-1][2] == pytest.approx(0.0)  # finish guide is on the ground
  guide_z = pts[fac < 0.9][:, 2]
  assert np.all(guide_z == pytest.approx(0.0))


def test_stone_gap_increases_with_difficulty():
  """Stones keep a fixed width; the edge gap grows 0 -> 0.2 m, uniformly for
  the spawn->first-pair gap, between pairs, and last pair -> end column.

  Positions are read from the stone geometries: the chain carries a SINGLE
  reward point on the end column only (user decision 2026-10-16)."""
  cfg = BoxStraightStepStonesTerrainCfg(size=SIZE)
  half = cfg.stone_size / 2
  spawn_right = cfg.border_width + 1.0  # green spawn column right edge

  def orange_x(output):
    xs = sorted(
      {
        round(g.geom.pos[0], 6)
        for g in output.geometries
        if g.geom is not None
        and g.color is not None
        and tuple(g.color) == (0.85, 0.55, 0.15, 1.0)
      }
    )
    return xs

  for difficulty, gap in ((0.0, 0.0), (1.0, 0.2)):
    output = _generate(BoxStraightStepStonesTerrainCfg, difficulty)
    assert output.reward_points is not None and len(output.reward_points) == 2
    xs = orange_x(output)
    assert len(xs) == cfg.num_pairs
    # Spawn column edge -> first pair edge uses the same gap.
    assert xs[0] - half - spawn_right == pytest.approx(gap)
    # Consecutive pair gaps use the same gap.
    for a, b in zip(xs[:-1], xs[1:], strict=True):
      assert b - a == pytest.approx(cfg.stone_size + gap)
    # Last pair edge -> end column edge uses the same gap; the end column
    # carries the terminal reward point at its center, the mid point sits
    # on the middle pair (4th of 8).
    end_col_center = output.reward_points[1][0]
    assert end_col_center - 0.5 - (xs[-1] + half) == pytest.approx(gap)
    assert output.reward_points[0][0] == pytest.approx(xs[3])


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
  """The wall-1 top milestone (the first factor-1.0 point in the chain)
  follows the 0.0 -> 0.5 m ladder with a ±0.05 m per-patch jitter (user spec
  2026-10-16: start from 0 cm). Guide points sit on the ground regardless of
  difficulty."""
  low = _generate(BoxStraightPlatformTerrainCfg, 0.0)
  high = _generate(BoxStraightPlatformTerrainCfg, 1.0)
  assert low.reward_points is not None and high.reward_points is not None
  assert low.reward_point_factors is not None
  assert high.reward_point_factors is not None
  # Wall-1 top milestone = the first factor-1.0 point in the chain.
  low_pts = np.stack(low.reward_points)
  low_top = low_pts[np.stack(low.reward_point_factors) > 0.9][0]
  high_pts = np.stack(high.reward_points)
  high_top = high_pts[np.stack(high.reward_point_factors) > 0.9][0]
  assert low_top[2] == pytest.approx(0.0, abs=0.055)
  assert high_top[2] == pytest.approx(0.5, abs=0.055)
  assert high_top[2] > low_top[2]


def test_gap_reward_point_past_far_edge():
  """Gap reward points sit on the walkable ground past each gap's far edge.

  All segments are flush with the spawn plane at z=0 (a micro-elevation
  experiment created a climb lip after every jump — reverted), so every
  point lies exactly on the landing plane."""
  cfg = BoxStraightGapTerrainCfg(size=SIZE)
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  rng = np.random.default_rng(0)
  output = cfg.function(1.0, spec, rng)

  assert output.reward_points is not None
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
  assert generator.reward_point_factors is not None
  assert generator.reward_points.shape == (cfg.num_rows, cfg.num_cols, 7, 3)
  assert generator.reward_point_factors.shape == (
    cfg.num_rows,
    cfg.num_cols,
    7,
  )
  # Every curriculum patch places at least one obstacle (and reward point).
  assert (generator.reward_counts > 0).all()
  # Every valid slot carries a value factor; each patch has exactly two
  # milestone (1.0) points — the wall-1 top and the landing (the finish
  # guide tail is a 0.2 low-value point, 2026-10-30).
  for row in range(cfg.num_rows):
    for col in range(cfg.num_cols):
      n = int(generator.reward_counts[row, col])
      factors = generator.reward_point_factors[row, col, :n]
      assert (factors == 1.0).sum() == 2
      assert (factors > 0).all()


def _platform_walls(output):
  """Collect the two wall boxes' (center_x, depth) from the orange geoms.

  The platform terrain colors its walls _ORANGE and its floor _GRAY, so the
  orange boxes are exactly the two walls (difficulty >= 0.5 guarantees both
  heights survive the +/-0.05 m jitter above the 1e-3 skip threshold)."""
  walls = []
  for g in output.geometries:
    if (
      g.geom is not None
      and g.color is not None
      and tuple(g.color) == (0.85, 0.55, 0.15, 1.0)
    ):
      walls.append((g.geom.pos[0], g.geom.size[0] * 2))
  assert len(walls) == 2, f"expected 2 walls, got {walls}"
  walls.sort(key=lambda w: w[0])
  return walls


@pytest.mark.parametrize("difficulty", [0.5, 0.9, 1.0])
def test_platform_layout_budget(difficulty):
  """Layout budget invariants over many random draws (2026-10-30 redesign).

  Verifies the three user goals hold across the sampling space:
  - spawn acceleration runway >= 1.55 m (wall 1 left edge >= 2.3 m);
  - wall 2's right edge <= wall2_right_max, so the finish guide
    (edge + finish_offset) and the success line (guide + 0.4 m margin)
    stay inside the patch (termination line <= 8.0 m);
  - the realized wall gap never drops below ~1.5 m (guide-point spacing
    stays >= 0.45 m > 0.35 m minimum);
  - the chain tail is the finish guide at exactly finish_offset past wall 2
    and the landing milestone at landing_offset (ending bonus is now twice
    as far from wall 2 as the old landing-only chain)."""
  cfg = BoxStraightPlatformTerrainCfg(size=SIZE)
  spawn_x = cfg.border_width + 0.5
  for seed in range(30):
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    rng = np.random.default_rng(seed)
    output = cfg.function(difficulty, spec, rng)
    assert output.reward_points is not None
    assert output.reward_point_factors is not None
    pts = np.stack(output.reward_points)
    fac = np.stack(output.reward_point_factors)

    (cx1, d1), (cx2, d2) = _platform_walls(output)
    left1 = cx1 - d1 / 2
    right_edge2 = cx2 + d2 / 2
    gap = (cx2 - d2 / 2) - (cx1 + d1 / 2)

    # Q3 fix: guaranteed acceleration runway, independent of wall depth.
    assert left1 >= cfg.wall1_left_x_range[0] - 1e-6
    assert left1 - spawn_x >= 1.55
    # Q2 fix: wall 2 pinned near the patch edge, course spans the patch.
    assert right_edge2 <= cfg.wall2_right_max + 1e-6
    # The realized gap only compresses to the 1.5 m worst case.
    assert gap >= 1.5 - 1e-6
    # Q4 fix: landing milestone then finish guide tail past wall 2.
    assert pts[-2][0] == pytest.approx(right_edge2 + cfg.landing_offset)
    assert pts[-1][0] == pytest.approx(right_edge2 + cfg.finish_offset)
    assert pts[-1][0] - right_edge2 >= 2 * cfg.landing_offset - 1e-6
    assert fac[-2] == 1.0 and fac[-1] == 0.2
    # Success line (parkour_completed margin 0.4) stays inside the patch.
    assert pts[-1][0] + 0.4 <= SIZE[0] + 1e-6
    # The whole chain remains x-ordered and inside the patch.
    assert np.all(np.diff(pts[:, 0]) > 0)
    assert pts[:, 0].max() < SIZE[0] - cfg.border_width
