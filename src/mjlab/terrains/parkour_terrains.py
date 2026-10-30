"""Straight-line parkour terrains (box-geom style).

Three standard parkour obstacle courses laid out as a chain along the patch's
local +x axis (matching a pure forward-x velocity command with fixed spawn
yaw). Each obstacle contributes one milestone reward point (recorded via
``TerrainOutput.reward_points``) so reward functions can detect traversal.

All three terrains spawn the robot near the -x edge of the patch and keep a
``spawn_free_radius``-clear zone around the spawn point.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.terrains.utils import make_border

_GRAY = (0.35, 0.35, 0.35, 1.0)
_ORANGE = (0.85, 0.55, 0.15, 1.0)
_GREEN = (0.3, 0.55, 0.3, 1.0)
_PIT_FLOOR = (0.1, 0.1, 0.1, 1.0)


def _lerp_range(rng_or_difficulty, value_range: tuple[float, float]) -> float:
  """Linear interpolation of a (min, max) range by difficulty in [0, 1]."""
  low, high = value_range
  return low + (high - low) * rng_or_difficulty


@dataclass(kw_only=True)
class BoxStraightStepStonesTerrainCfg(SubTerrainCfg):
  """Plum-blossom stepping stones rising from a pit, chained along +x.

  **Two parallel, aligned columns of stone pairs** along x: at every pair
  position one stone sits left and one right of the centerline, so the Go1
  plants its left/right feet on the two columns naturally. The gap between
  the two columns matches the robot's stance width; stone width is sized so
  the robot can plant its feet flat (no contorted gaits).

  ALL edge-to-edge gaps in the chain use the same difficulty-interpolated
  ``gap`` (0 -> 0.2 m): spawn column -> first pair, between consecutive
  pairs, and last pair -> end column. The chain therefore never asks for a
  first/last jump larger than the trained inter-stone gap. The chain starts
  at a solid spawn column and ends at a solid end column (the final reward
  point), so crossing it always starts and ends on firm ground. Pit depth is
  randomized per patch to diversify depth-camera signals.
  """

  stone_size: float = 0.35
  """Stone side length (square columns), in meters. Sized for the Go1 stance."""
  stone_height_range: tuple[float, float] = (0.2, 0.35)
  """Stone top height above z=0. Lower bound at difficulty 0, upper at 1."""
  stone_height_variation: float = 0.05
  """Per-stone top-height jitter (scaled by difficulty), in meters."""
  gap_range: tuple[float, float] = (0.0, 0.2)
  """Edge-to-edge gap used uniformly across the chain (spawn -> first pair,
  between pairs, last pair -> end column). Lower bound at difficulty 0,
  upper at 1 (user spec: 0 cm -> 20 cm)."""
  column_gap_range: tuple[float, float] = (0.0, 0.2)
  """Edge-to-edge gap between the two stone columns, difficulty-interpolated
  (user spec: 0 cm -> 20 cm). Column center distance = stone_size + gap,
  matching the Go1 stance width."""
  num_pairs: int = 8
  """Number of stone pairs along the chain; each column has ``num_pairs``
  stones. Exactly TWO reward points (uniform with the other columns: mid +
  terminal, user decision 2026-10-16): one on the middle pair, one on the
  end column — the whole chain must be crossed to score both."""
  border_width: float = 0.25
  """Solid ring width around the patch closing the pit, in meters."""
  floor_depth_range: tuple[float, float] = (0.4, 1.0)
  """Pit depth below z=0, randomized per patch to diversify depth-camera
  signals."""
  max_reward_points: int = 2
  """Pre-allocation cap for per-(row, col) reward-point arrays (middle pair
  + end column)."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []
    reward_points: list[np.ndarray] = []

    top = _lerp_range(difficulty, self.stone_height_range)
    gap = _lerp_range(difficulty, self.gap_range)
    column_gap = _lerp_range(difficulty, self.column_gap_range)
    variation = self.stone_height_variation * difficulty
    pitch = self.stone_size + gap
    col_half_dist = (self.stone_size + column_gap) / 2
    floor_depth = rng.uniform(*self.floor_depth_range)

    spawn_x = self.border_width + 0.5
    spawn_right = spawn_x + 0.5
    cy = self.size[1] / 2

    # Border ring closing the pit walls.
    if self.border_width > 0.0:
      border_boxes = make_border(
        body,
        self.size,
        (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width),
        top + floor_depth,
        (self.size[0] / 2, self.size[1] / 2, (top - floor_depth) / 2),
      )
      for box in border_boxes:
        geometries.append(TerrainGeometry(geom=box, color=_GRAY))

    # Pit floor.
    floor_h = 0.1
    floor_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, floor_h / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -floor_depth - floor_h / 2),
    )
    geometries.append(TerrainGeometry(geom=floor_geom, color=_PIT_FLOOR))

    def add_column(
      px: float, py: float, half: float, col_top: float, color: tuple
    ) -> None:
      h = floor_depth + col_top
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(half, half, h / 2),
        pos=(px, py, -floor_depth + h / 2),
      )
      geometries.append(TerrainGeometry(geom=geom, color=color))

    # Spawn column (solid, top flush with the stones).
    add_column(spawn_x, cy, 0.5, top, _GREEN)

    # Aligned stone pairs: one stone left + one right of the centerline at
    # every pair position. The first pair starts gap away from the spawn
    # column edge (same gap as between pairs). Reward points: the MIDDLE
    # pair (on the centerline between its two stones) and the end column.
    first_x = spawn_right + gap + self.stone_size / 2
    mid_pair = self.num_pairs // 2 - 1
    for i in range(self.num_pairs):
      pair_x = first_x + i * pitch
      top_left = top + rng.uniform(-variation, variation)
      top_right = top + rng.uniform(-variation, variation)
      add_column(pair_x, cy + col_half_dist, self.stone_size / 2, top_left, _ORANGE)
      add_column(pair_x, cy - col_half_dist, self.stone_size / 2, top_right, _ORANGE)
      if i == mid_pair:
        reward_points.append(np.array([pair_x, cy, 0.5 * (top_left + top_right)]))

    # End column (1x1 m), gap away from the last pair's edge (same gap as
    # the chain). It carries the final reward point; stepping onto it ends
    # the episode (parkour_completed triggers at last point + margin 0.4,
    # which lies on the column surface).
    last_pair_right = first_x + (self.num_pairs - 1) * pitch + self.stone_size / 2
    end_x = last_pair_right + gap + 0.5
    add_column(end_x, cy, 0.5, top, _GREEN)
    reward_points.append(np.array([end_x, cy, top]))

    origin = np.array([spawn_x, cy, top])
    return TerrainOutput(
      origin=origin, geometries=geometries, reward_points=reward_points
    )


@dataclass(kw_only=True)
class BoxStraightPlatformTerrainCfg(SubTerrainCfg):
  """TWO raised platforms ("low walls") on flat ground, chained along +x.

  Layout mirrors wmp_go1's ``climb`` terrain: two near-full-width wall groups
  per patch with a flat run between them, and the second wall pushed as far
  right as the patch allows (wmp_go1 pins its right edge AT the 8 m unit
  boundary; here ``wall2_right_max`` keeps the finish guide and the success
  line inside the patch, termination line <= 8.0 m). Wall-1 position is
  sampled by LEFT edge so the spawn acceleration runway is guaranteed
  regardless of the sampled depth. Heights follow the difficulty ladder
  (0.0 -> 0.5 m over the 10 rows) with per-wall jitter; thicknesses are
  randomized per patch (thin walls get hopped over, thick walls get climbed).

  SEVEN reward points form a low-high guide chain (user decisions
  2026-10-29/30, anti "flea-jump" economics): two low-value points on the
  approach before wall 1, the wall-1 top (high), two low-value points in the
  gap run before wall 2, the landing ground just past wall 2 (high — pays
  for CROSSING), and a finish guide one meter farther out (low) as the chain
  TAIL: ``parkour_completed`` fires margin (0.4 m) past the tail, ending the
  episode near the patch edge. Low points pay ``low_bonus_ratio`` of the
  milestone value so that walking the chain pays throughout, while a single
  leap can at most clip one bubble. Approach points are ADAPTIVE: a patch
  may emit 1 or 0 approach points when the spawn-to-wall1 span cannot fit
  them with 0.35 m spacing and 0.45 m wall standoff (counts handle the
  variable length).
  """

  platform_height_range: tuple[float, float] = (0.0, 0.5)
  """Platform height above ground. Lower bound at difficulty 0, upper at 1
  (user spec 2026-10-16: start from 0 cm -> 50 cm over the 10 difficulty
  rows, so the reward/behavior signal ramps up smoothly for the policy)."""
  height_jitter: float = 0.05
  """Per-wall uniform jitter added on top of the difficulty ladder height,
  in meters (clamped to [0.0, 0.55])."""
  platform_depth_range: tuple[float, float] = (0.4, 1.2)
  """Platform extent along x, randomized per patch (landable top surface)."""
  platform_width_range: tuple[float, float] = (7.4, 7.5)
  """Platform extent along y. Patch-full-width (no side gaps) so the robot
  physically cannot walk around the platform; it must climb or jump over."""
  wall1_left_x_range: tuple[float, float] = (2.3, 2.7)
  """First wall LEFT-edge x within the patch, randomized per patch.

  Sampling the left edge (not the center) makes the spawn acceleration
  runway independent of the sampled depth: with spawn at border_width + 0.5
  (0.75 m) the runway is 1.55-1.95 m (1.35 m worst case with the +/-0.2 m
  reset jitter). The old center-based sampling could put the wall face
  0.45 m from spawn — the robot hit the first wall before building speed
  (user fix 2026-10-30)."""
  wall_gap_range: tuple[float, float] = (2.5, 2.9)
  """Edge-to-edge flat gap between the two walls (wmp_go1 climb: 2.5-2.9 m).
  This is the NATURAL gap used to place wall 2; when the wall-2 clamp below
  binds (fat wall draws), the realized gap compresses down to ~1.5 m."""
  wall2_right_max: float = 6.6
  """Upper bound for the second wall's right edge. wmp_go1 pins wall 2's
  right edge AT the 8 m unit boundary; here the bound keeps the finish guide
  (edge + finish_offset) and the success-termination line (guide + 0.4 m
  margin) inside the patch, so the course spans ~95% of the map and the
  termination line stays <= 8.0 m (user fix 2026-10-30: the old 6.8 cap left
  0.7-2.4 m of unused ground past the finish)."""
  landing_offset: float = 0.5
  """Landing milestone x-offset past the second wall's right edge (HIGH
  value: pays for CROSSING the wall; user decision 2026-10-30)."""
  finish_offset: float = 1.0
  """Finish guide x-offset past the second wall's right edge. Chain TAIL:
  ``parkour_completed`` fires margin (0.4 m) past it, so the wall-to-finish
  distance doubles vs the old landing-only chain (user decision 2026-10-30).
  LOW value (0.2) keeps the two-milestones-per-patch economics intact."""
  border_width: float = 0.25
  """Flat border width around the patch, in meters."""
  floor_thickness: float = 0.5
  """Thickness of the ground slab below z=0, in meters."""
  max_reward_points: int = 7
  """Pre-allocation cap for per-(row, col) reward-point arrays — low-high
  guide chain: up to 2 adaptive approach lows + wall-1 top + 2 gap lows +
  landing milestone + finish guide (approach lows drop adaptively when the
  spawn-to-wall span is too short)."""
  low_bonus_ratio: float = 0.2
  """Reward factor for low-value guide points, relative to a milestone
  point's 1.0 (i.e. a guide point pays 20% of the milestone bonus)."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []
    reward_points: list[np.ndarray] = []
    reward_point_factors: list[np.ndarray] = []
    # 0-d arrays (not np.float32 scalars): TerrainOutput declares the factor
    # list as list[np.ndarray], and the generator assigns these into a float
    # array — a 0-d ndarray type-checks and behaves identically at runtime.
    low = np.asarray(self.low_bonus_ratio, dtype=np.float32)
    high = np.asarray(1.0, dtype=np.float32)

    def sample_height() -> float:
      h = _lerp_range(difficulty, self.platform_height_range)
      h += rng.uniform(-self.height_jitter, self.height_jitter)
      return float(np.clip(h, 0.0, 0.55))

    height1 = sample_height()
    height2 = sample_height()
    depth1 = rng.uniform(*self.platform_depth_range)
    depth2 = rng.uniform(*self.platform_depth_range)
    width = _lerp_range(difficulty, self.platform_width_range)
    # Wall 1 is placed by LEFT edge so the spawn runway holds for any depth.
    left1 = rng.uniform(*self.wall1_left_x_range)
    px1 = left1 + depth1 / 2
    gap = rng.uniform(*self.wall_gap_range)
    # Natural wall-2 position from the sampled gap, then clamped so its RIGHT
    # edge (and the finish guide + termination line past it) stays on the
    # patch. When the clamp binds it compresses the WALL GAP: worst case
    # 1.5 m (fattest wall draws), typical 2.5 m (= wmp_go1's minimum).
    px2 = min(
      left1 + depth1 + gap + depth2 / 2,
      self.wall2_right_max - depth2 / 2,
    )

    spawn_x = self.border_width + 0.5
    cy = self.size[1] / 2

    # -- Low-high guide reward chain (x-ordered; anti "flea-jump" design) --
    # Walking the chain pays at every guide point; a leap can at most clip
    # ONE capture bubble (and only if it lands in contact, see
    # reward_point_bonus's feet-on-ground gate).
    left_edge1 = left1
    right_edge1 = left1 + depth1
    left_edge2 = px2 - depth2 / 2

    # Approach guide points before wall 1 (adaptive). Two hard constraints:
    # every point keeps >= 0.45 m standoff from the wall face (so the robot
    # can stand on it without its ~0.26 m half-length body clipping the
    # wall) and >= 0.35 m spacing between points; the first point also
    # stays >= 0.5 m clear of the spawn column (same clearance convention
    # as the other reward points, spawn jitter is +/-0.2 m). With the
    # current wall1_left_x_range (>= 2.3 m) both points always fit; the
    # adaptive filter stays as a guard for future range edits.
    approach: list[float] = []
    prev_x = spawn_x + 0.5
    for x in (spawn_x + 0.5, left_edge1 - 0.45):
      min_x = prev_x if not approach else prev_x + 0.35
      if x >= min_x and x <= left_edge1 - 0.45:
        approach.append(x)
        prev_x = x
    for x in approach:
      reward_points.append(np.array([x, cy, 0.0]))
      reward_point_factors.append(low)
    # Milestone 1: wall-1 top center (high value).
    reward_points.append(np.array([px1, cy, height1]))
    reward_point_factors.append(high)
    # Gap guide points in the run between the walls. The realized gap is
    # >= 1.5 m even in the worst clamped draw, so the two points keep
    # >= 0.45 m spacing (> 0.35 m minimum).
    for x in (right_edge1 + 0.6, left_edge2 - 0.45):
      reward_points.append(np.array([x, cy, 0.0]))
      reward_point_factors.append(low)
    # Milestone 2: landing ground just past wall 2 (HIGH value, pays for
    # CROSSING; user decision 2026-10-30 keeps this milestone at 1.0).
    right_edge2 = px2 + depth2 / 2
    reward_points.append(np.array([right_edge2 + self.landing_offset, cy, 0.0]))
    reward_point_factors.append(high)
    # Finish guide (chain TAIL, user decision 2026-10-30): one more meter
    # out, so the ending bonus sits twice as far from wall 2 as before and
    # the success line (tail + parkour_completed margin 0.4) lands right at
    # the patch edge (<= 8.0 m). LOW value keeps the two-milestones-per-patch
    # economics; the termination reads the tail's POSITION only, so its 0.2
    # factor does not affect when the episode ends.
    reward_points.append(np.array([right_edge2 + self.finish_offset, cy, 0.0]))
    reward_point_factors.append(low)

    # Ground slab (top at z=0).
    floor_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, self.floor_thickness / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -self.floor_thickness / 2),
    )
    geometries.append(TerrainGeometry(geom=floor_geom, color=_GRAY))

    for px, height, depth in ((px1, height1, depth1), (px2, height2, depth2)):
      # A zero-height wall (row 0 of the 0->0.5 ladder + negative jitter
      # clamps to 0) would be a zero-thickness box, which MuJoCo rejects at
      # compile time — skip the geom entirely, the floor slab IS the ground.
      if height > 1e-3:
        platform_geom = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(depth / 2, width / 2, height / 2),
          pos=(px, cy, height / 2),
        )
        geometries.append(TerrainGeometry(geom=platform_geom, color=_ORANGE))

    origin = np.array([spawn_x, cy, 0.0])
    return TerrainOutput(
      origin=origin,
      geometries=geometries,
      reward_points=reward_points,
      reward_point_factors=reward_point_factors,
    )


@dataclass(kw_only=True)
class BoxStraightGapTerrainCfg(SubTerrainCfg):
  """Narrow trenches cutting across the patch, chained along +x.

  The ground is built as box segments along x spanning the full inner width,
  with true gaps (width up to 0.35 m at difficulty 1) between them over a deep
  pit floor. One reward point sits just past the far edge of each gap.
  """

  gap_width_range: tuple[float, float] = (0.0, 0.5)
  """Gap extent along x. Lower bound at difficulty 0 (flat ground), upper at
  1 (user spec: 0 cm -> 50 cm over the 10 difficulty rows)."""
  gap_depth_range: tuple[float, float] = (0.4, 1.0)
  """Trench depth below z=0, randomized per patch to diversify depth-camera
  signals."""
  segment_range: tuple[float, float] = (1.2, 1.8)
  """Walkable segment length between gaps, in meters."""
  max_gaps: int = 3
  """Maximum number of gaps along the chain."""
  spawn_free_radius: float = 1.0
  """Radius around the spawn point kept free of gaps, in meters."""
  border_width: float = 0.25
  """Solid ring width around the patch closing the trench sides, in meters."""
  max_reward_points: int = 2
  """Pre-allocation cap for per-(row, col) reward-point arrays — uniform
  mid + terminal scheme (after the middle gap + after the last gap)."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []
    reward_points: list[np.ndarray] = []

    gap_width = _lerp_range(difficulty, self.gap_width_range)
    num_gaps = 2 if difficulty < 0.5 else min(3, self.max_gaps)
    gap_depth = rng.uniform(*self.gap_depth_range)

    spawn_x = self.border_width + 0.5
    cy = self.size[1] / 2
    inner_len_y = self.size[1] - 2 * self.border_width

    # Border ring closing the trench sides.
    if self.border_width > 0.0:
      border_boxes = make_border(
        body,
        self.size,
        (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width),
        gap_depth,
        (self.size[0] / 2, self.size[1] / 2, -gap_depth / 2),
      )
      for box in border_boxes:
        geometries.append(TerrainGeometry(geom=box, color=_GRAY))

    # Pit floor.
    floor_h = 0.1
    floor_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, floor_h / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -gap_depth - floor_h / 2),
    )
    geometries.append(TerrainGeometry(geom=floor_geom, color=_PIT_FLOOR))

    def add_segment(x0: float, x1: float) -> None:
      length = x1 - x0
      if length <= 1e-3:
        return
      # Box spans [x0, x1] x [-gap_depth, 0] — ALL segments flush with the
      # spawn plane (a 2026-10-16 attempt at random micro-elevation put a
      # vertical lip at the far edge of every gap, forcing a climb right
      # after each jump — reverted).
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(length / 2, inner_len_y / 2, gap_depth / 2),
        pos=((x0 + x1) / 2, cy, -gap_depth / 2),
      )
      geometries.append(TerrainGeometry(geom=geom, color=_GRAY))

    # First walkable segment covers the spawn zone and everything behind it
    # (from the border, so there is no pit gap under the spawn area).
    # Reward points: uniform mid + terminal scheme — after the middle gap
    # and after the last gap only.
    mid_gap = (num_gaps - 1) // 2
    cursor = self.border_width
    for i in range(num_gaps):
      seg_len = rng.uniform(*self.segment_range)
      add_segment(cursor, cursor + seg_len)
      cursor += seg_len
      # The gap itself is left empty (true hole over the pit floor). The
      # reward point sits just past the gap's far edge (landing side).
      if i == mid_gap or i == num_gaps - 1:
        reward_points.append(np.array([cursor + gap_width + 0.15, cy, 0.0]))
      cursor += gap_width

    # Close the patch with a final walkable segment up to the border.
    add_segment(cursor, self.size[0] - self.border_width)

    origin = np.array([spawn_x, cy, 0.0])
    return TerrainOutput(
      origin=origin, geometries=geometries, reward_points=reward_points
    )
