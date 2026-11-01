"""Tests for the straight-line parkour terrains."""

import mujoco
import numpy as np
import pytest
import torch

from mjlab.terrains.config import PARKOUR_TERRAINS_CFG
from mjlab.terrains.parkour_terrains import (
  BoxStraightCrawlBarsTerrainCfg,
  BoxStraightGapTerrainCfg,
  BoxStraightPlatformGapTerrainCfg,
  BoxStraightPlatformTerrainCfg,
  BoxStraightStepStonesTerrainCfg,
)
from mjlab.terrains.terrain_generator import (
  TerrainGenerator,
  TerrainGeneratorCfg,
  resolve_curriculum_column_names,
)

SIZE = (8.0, 8.0)

ALL_PARKOUR_CFGS = [
  BoxStraightStepStonesTerrainCfg,
  BoxStraightPlatformTerrainCfg,
  BoxStraightGapTerrainCfg,
  BoxStraightPlatformGapTerrainCfg,
  BoxStraightCrawlBarsTerrainCfg,
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
  """The platform guide chain is x-ordered with exactly THREE high-value
  milestones (wall-1 top, descent landing past wall 1, landing past wall 2)
  and a LOW-value finish guide as the chain tail (2026-10-30; descent point
  promoted 2026-11-01 to pay for JUMPING DOWN — the old low-value point let
  the robot camp on the wall top after banking milestone 1)."""
  output = _generate(BoxStraightPlatformTerrainCfg, 0.5)
  assert output.reward_points is not None
  assert output.reward_point_factors is not None
  pts = np.stack(output.reward_points)
  fac = np.stack(output.reward_point_factors)
  assert len(pts) == len(fac) == 7
  assert int((fac == 1.0).sum()) == 3
  assert int((fac == 0.2).sum()) == len(pts) - 3
  # Strictly x-ordered (chain semantics: spawn -> ... -> finish guide).
  assert np.all(np.diff(pts[:, 0]) > 0)
  # The milestones are the wall-1 top (z == wall height) and the descent
  # landing + landing (both on the ground); the finish guide tail sits on
  # the ground (z == 0).
  assert fac[0] == 0.2  # first approach point is a guide
  assert fac[2] == 1.0  # wall-1 top milestone
  assert fac[3] == 1.0  # descent milestone past wall 1 (anti wall-top camping)
  assert pts[3][2] == pytest.approx(0.0)  # descent milestone is on the ground
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
  """The generator fills per-(row, col) reward-point arrays for the preset.

  Five-column layout (2026-11-01): the first three columns are bonus terrains
  (platforms=7, platform_gap=5, crawl=5 points; each with exactly TWO
  factor-1.0 milestones), the last two are the point-less native stairs
  (heading-mode columns)."""
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
  # Column -> terrain-name mapping matches the written sub_terrains order.
  names = resolve_curriculum_column_names(cfg)
  assert names == [
    "platforms",
    "platform_gap",
    "crawl",
    "pyramid_stairs",
    "pyramid_stairs_inv",
  ]
  expected_counts = {
    "platforms": 7,
    "platform_gap": 5,
    "crawl": 5,
    "pyramid_stairs": 0,
    "pyramid_stairs_inv": 0,
  }
  # Milestones (factor 1.0) per terrain: platforms has THREE since the
  # descent point past wall 1 was promoted (user decision 2026-11-01);
  # gap/crawl keep two.
  expected_milestones = {
    "platforms": 3,
    "platform_gap": 2,
    "crawl": 2,
  }
  for row in range(cfg.num_rows):
    for col, name in enumerate(names):
      n = int(generator.reward_counts[row, col])
      assert n == expected_counts[name], (row, col, name, n)
      if n > 0:
        factors = generator.reward_point_factors[row, col, :n]
        assert (factors == 1.0).sum() == expected_milestones[name]
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


# ---------------------------------------------------------------------------
# 2026-11-01 terrain expansion: platform_gap, crawl bars, native stairs.
# ---------------------------------------------------------------------------


def test_platform_gap_chain_structure():
  """The gap-terrain guide chain (user spec 2026-11-01): exactly 5 points —
  3 low (approach / descend / finish) + 2 high (platform-1 top / past-gap) —
  x-ordered, with the z pattern [ground, top, top, ground, ground]."""
  output = _generate(BoxStraightPlatformGapTerrainCfg, 0.5)
  assert output.reward_points is not None
  assert output.reward_point_factors is not None
  pts = np.stack(output.reward_points)
  fac = np.stack(output.reward_point_factors)
  assert len(pts) == 5
  assert int((fac == 1.0).sum()) == 2
  assert int((fac == 0.2).sum()) == 3
  assert np.all(np.diff(pts[:, 0]) > 0)
  # z pattern: approach ground, both platform-top milestones, then ground.
  assert pts[0][2] == pytest.approx(0.0)
  assert pts[1][2] == pytest.approx(pts[2][2])
  assert pts[1][2] > 0.1
  assert pts[3][2] == pytest.approx(0.0)
  assert pts[4][2] == pytest.approx(0.0)
  # Cadence: low, high, high, low, low.
  assert fac[0] == 0.2 and fac[1] == 1.0 and fac[2] == 1.0
  assert fac[3] == 0.2 and fac[4] == 0.2


def test_platform_gap_equal_heights_and_gap_ladder():
  """Both platforms are IDENTICAL (one shared height AND depth sample — user
  request 2026-11-01), and the edge-to-edge gap interpolates 0 -> 0.3 m with
  difficulty (user spec: 0 -> 30 cm)."""
  for difficulty, expected_gap in ((0.5, 0.15), (0.9, 0.27)):
    output = _generate(BoxStraightPlatformGapTerrainCfg, difficulty)
    (cx1, d1), (cx2, d2) = _platform_walls(output)
    # Identical platforms: same depth (and same full width).
    assert d1 == pytest.approx(d2)
    tops = sorted(
      round(g.geom.pos[2] * 2, 6)
      for g in output.geometries
      if g.geom is not None
      and g.color is not None
      and tuple(g.color) == (0.85, 0.55, 0.15, 1.0)
    )
    assert len(tops) == 2 and tops[0] == tops[1]
    gap = (cx2 - d2 / 2) - (cx1 + d1 / 2)
    assert gap == pytest.approx(expected_gap)
    # Farther spawn runway (user request 2026-11-01): left edge >= 2.8 m.
    assert cx1 - d1 / 2 >= 2.8 - 1e-6
    # On-top acceleration runway = the shared platform depth (>= 1.2 m).
    assert d1 >= 1.2 - 1e-6
    # The past-gap milestone sits ON platform 2's top (inside its span).
    pts = np.stack(output.reward_points)
    assert cx2 - d2 / 2 - 1e-6 <= pts[2][0] <= cx2 + d2 / 2 + 1e-6
    assert pts[2][2] == pytest.approx(tops[0])
    # Descend low + finish tail are past platform 2's right edge.
    right2 = cx2 + d2 / 2
    assert pts[3][0] == pytest.approx(right2 + 0.30)
    assert pts[4][0] == pytest.approx(right2 + 1.00)
    # Success line (finish + parkour_completed margin 0.4) stays inside.
    assert pts[4][0] + 0.4 <= SIZE[0] + 1e-6


def test_platform_gap_layout_budget():
  """Layout budget over many random draws (2026-11-01 revision): the farther
  wall position + deeper identical platforms must still keep the whole chain
  and the success line inside the patch for EVERY sampling draw."""
  cfg = BoxStraightPlatformGapTerrainCfg(size=SIZE)
  for seed in range(30):
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    rng = np.random.default_rng(seed)
    output = cfg.function(1.0, spec, rng)  # difficulty 1.0 = worst case
    assert output.reward_points is not None
    pts = np.stack(output.reward_points)
    (cx1, d1), (cx2, d2) = _platform_walls(output)
    # Identical platforms at max difficulty too.
    assert d1 == pytest.approx(d2)
    right2 = cx2 + d2 / 2
    assert right2 <= 6.6 + 1e-6
    assert pts[-1][0] + 0.4 <= SIZE[0] + 1e-6
    assert np.all(np.diff(pts[:, 0]) > 0)


def test_platform_gap_pit_has_no_cave():
  """The gap is a true hole down to a deep pit floor; the ground slabs reach
  BELOW the pit floor so a fallen robot cannot walk under the ground."""
  output = _generate(BoxStraightPlatformGapTerrainCfg, 0.7)
  gray = [
    g.geom
    for g in output.geometries
    if g.geom is not None
    and g.color is not None
    and tuple(g.color) == (0.1, 0.1, 0.1, 1.0)
  ]
  # Pit floor box only (dark _PIT_FLOOR): top sits in [-1.0, -0.5] m.
  assert len(gray) == 1
  pit_top = gray[0].pos[2] + gray[0].size[2]
  assert -1.0 <= pit_top <= -0.5
  # Slabs (gray _GRAY, deeper than the border ring): bottom == pit bottom.
  slabs = [
    g.geom
    for g in output.geometries
    if g.geom is not None
    and g.color is not None
    and tuple(g.color) == (0.35, 0.35, 0.35, 1.0)
    and g.geom.size[2] * 2 > 0.7  # deeper than the border ring boxes
  ]
  assert len(slabs) == 2
  for slab in slabs:
    assert slab.pos[2] - slab.size[2] == pytest.approx(pit_top - 0.1)


def test_crawl_chain_structure():
  """The crawl-terrain guide chain (user cadence 2026-11-01): exactly 5 points
  on the ground — low, HIGH, low, HIGH, low — x-ordered around the two bars."""
  output = _generate(BoxStraightCrawlBarsTerrainCfg, 0.5)
  assert output.reward_points is not None
  assert output.reward_point_factors is not None
  pts = np.stack(output.reward_points)
  fac = np.stack(output.reward_point_factors)
  assert len(pts) == 5
  assert np.all(np.diff(pts[:, 0]) > 0)
  assert np.all(pts[:, 2] == pytest.approx(0.0))
  assert list(fac) == [0.2, 1.0, 0.2, 1.0, 0.2]

  bars = sorted(
    (
      g.geom
      for g in output.geometries
      if g.geom is not None
      and g.color is not None
      and tuple(g.color) == (0.55, 0.30, 0.75, 1.0)
    ),
    key=lambda geom: geom.pos[0],
  )
  assert len(bars) == 2
  left1 = bars[0].pos[0] - bars[0].size[0]
  right1 = bars[0].pos[0] + bars[0].size[0]
  left2 = bars[1].pos[0] - bars[1].size[0]
  right2 = bars[1].pos[0] + bars[1].size[0]
  assert pts[0][0] == pytest.approx(left1 - 0.30)
  assert pts[1][0] == pytest.approx(right1 + 0.30)
  assert pts[2][0] == pytest.approx((right1 + left2) / 2)
  assert pts[3][0] == pytest.approx(right2 + 0.30)
  assert pts[4][0] == pytest.approx(right2 + 1.00)
  # Success line stays inside the patch.
  assert pts[4][0] + 0.4 <= SIZE[0] + 1e-6


def test_crawl_clearance_ladder():
  """Bar underside (clearance) falls 0.42 -> 0.27 m with difficulty (wmp_go1
  quantities); the 1.0 m bar body keeps the top >= 1.27 m (no jump-over)."""
  for difficulty, clearance in ((0.0, 0.42), (0.5, 0.345), (1.0, 0.27)):
    output = _generate(BoxStraightCrawlBarsTerrainCfg, difficulty)
    bars = [
      g.geom
      for g in output.geometries
      if g.geom is not None
      and g.color is not None
      and tuple(g.color) == (0.55, 0.30, 0.75, 1.0)
    ]
    assert len(bars) == 2
    for bar in bars:
      bottom = bar.pos[2] - bar.size[2]
      assert bottom == pytest.approx(clearance)
      top = bar.pos[2] + bar.size[2]
      assert top >= 1.27
      # Full-width bars: no side bypass.
      assert bar.size[1] * 2 == pytest.approx(7.5)


def test_parkour_columns_resolve_in_written_order():
  """The shared column resolver maps the five preset columns in the
  sub_terrains' written order (the command's and entity's terrain dispatch
  depend on this exact mapping)."""
  names = resolve_curriculum_column_names(PARKOUR_TERRAINS_CFG)
  assert names == [
    "platforms",
    "platform_gap",
    "crawl",
    "pyramid_stairs",
    "pyramid_stairs_inv",
  ]


def test_spawn_terrain_names_filter():
  """``spawn_terrain_names`` restricts env terrain_types to the matching
  columns (sampling decoupled from the grid layout); ``None`` keeps the
  historical index round-robin."""
  from mjlab.terrains.primitive_terrains import (
    BoxFlatTerrainCfg,
    BoxPyramidStairsTerrainCfg,
  )
  from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg

  gen_cfg = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    border_width=1.0,
    num_rows=2,
    num_cols=2,
    curriculum=True,
    add_lights=False,
    sub_terrains={
      "flat": BoxFlatTerrainCfg(proportion=1.0),
      "stairs": BoxPyramidStairsTerrainCfg(
        proportion=1.0,
        step_height_range=(0.05, 0.1),
        step_width_range=(0.3, 0.4),
        platform_width=1.5,
        border_width=0.5,
        stair_lip_enabled=False,
      ),
    },
  )

  filtered = TerrainEntity(
    TerrainEntityCfg(
      terrain_type="generator",
      terrain_generator=gen_cfg,
      num_envs=32,
      spawn_terrain_names=("stairs",),
    ),
    device="cpu",
  )
  # Every env spawns on the stairs column (col 1); levels stay random rows.
  assert torch.all(filtered.terrain_types == 1)
  assert filtered.terrain_levels.shape == (32,)

  historical = TerrainEntity(
    TerrainEntityCfg(
      terrain_type="generator",
      terrain_generator=gen_cfg,
      num_envs=32,
    ),
    device="cpu",
  )
  assert torch.equal(
    historical.terrain_types,
    torch.div(torch.arange(32), 32 / 2, rounding_mode="floor").to(torch.long),
  )

  with pytest.raises(ValueError, match="matched no terrain column"):
    TerrainEntity(
      TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=gen_cfg,
        num_envs=8,
        spawn_terrain_names=("nonexistent",),
      ),
      device="cpu",
    )


def test_init_terrain_level_pin():
  """``init_terrain_level`` pins EVERY env to exactly that difficulty row
  (clamped to the last row); ``None`` keeps uniform sampling within
  ``[0, max_init_terrain_level]`` (2026-11-01 play/eval feature)."""
  import torch

  from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg
  from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg

  gen_cfg = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    border_width=1.0,
    num_rows=4,
    num_cols=1,
    curriculum=True,
    add_lights=False,
    sub_terrains={"flat": BoxFlatTerrainCfg(proportion=1.0)},
  )

  def build(**kwargs):
    return TerrainEntity(
      TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=gen_cfg,
        num_envs=16,
        **kwargs,
      ),
      device="cpu",
    )

  # Pinned row: every env starts on exactly row 2 (medium of 4).
  pinned = build(init_terrain_level=2)
  assert torch.all(pinned.terrain_levels == 2)

  # Clamped to the last row when out of range.
  clamped = build(init_terrain_level=99)
  assert torch.all(clamped.terrain_levels == 3)

  # None: historical behavior — levels inside [0, max_init_terrain_level]
  # (with max_init_terrain_level=1: only rows 0/1 may appear).
  sampled = build(max_init_terrain_level=1)
  assert sampled.terrain_levels.min() >= 0
  assert sampled.terrain_levels.max() <= 1
