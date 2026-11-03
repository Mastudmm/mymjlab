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
  2026-10-29/30, anti "flea-jump" economics; descent point promoted
  2026-11-01): two low-value points on the approach before wall 1, the wall-1
  top (high), a HIGH descent milestone on the ground just past wall 1 (pays
  for JUMPING DOWN — as a low point the robot camped on the wall top after
  banking milestone 1), a low-value guide point before wall 2, the landing
  ground just past wall 2 (high — pays for CROSSING), and a finish guide one
  meter farther out (low) as the chain TAIL: ``parkour_completed`` fires
  margin (0.4 m) past the tail, ending the episode near the patch edge. Low
  points pay ``low_bonus_ratio`` of the milestone value so that walking the
  chain pays throughout, while a single leap can at most clip one bubble.
  Approach points are ADAPTIVE: a patch may emit 1 or 0 approach points when
  the spawn-to-wall1 span cannot fit them with 0.35 m spacing and 0.45 m wall
  standoff (counts handle the variable length).
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
    # Descent milestone (HIGH value, user decision 2026-11-01): the ground
    # just past wall 1 pays for JUMPING DOWN and continuing. As a low-value
    # guide point it paid only 0.2 — after banking the wall-1-top milestone
    # the robot could camp/lie on the wall top with no downside; a milestone
    # here prices the descent like the wall-2 landing. Walking on toward
    # wall 2 sweeps through the capture bubble (collinear chain), so a long
    # jump that overshoots the landing spot still claims it.
    reward_points.append(np.array([right_edge1 + 0.6, cy, 0.0]))
    reward_point_factors.append(high)
    # Guide point in the run before wall 2 (LOW). The realized gap is
    # >= 1.5 m even in the worst clamped draw, so the spacing to the descent
    # milestone stays >= 0.45 m (> 0.35 m minimum).
    reward_points.append(np.array([left_edge2 - 0.45, cy, 0.0]))
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


@dataclass(kw_only=True)
class BoxStraightPlatformGapTerrainCfg(SubTerrainCfg):
  """Two EQUAL-height platforms chained along +x with a true gap between them.

  Layout (user spec 2026-11-01): spawn runway -> platform 1 (climb up) -> a
  full-width PIT gap (0 -> 0.3 m, difficulty-interpolated) -> platform 2 at
  the SAME height -> ground -> finish. Both platforms share one height sample
  from the same 0 -> 0.5 m ladder as ``BoxStraightPlatformTerrainCfg`` so the
  climb skill transfers; the gap exposes a deep pit floor, so crossing REQUIRES
  a jump/stride and falling in is a hard fail (body-contact / orientation
  terminations catch the fall — the task registers no pit-height termination).

  FIVE reward points form the low-high guide chain (3 low @ 0.2 + 2 high @
  1.0): 0.3 m before the climb, platform-1 top (milestone), just past the gap
  on platform 2 (milestone — pays for CROSSING), 0.3 m after the descent, and
  the finish guide 1.0 m out as the chain TAIL (``parkour_completed`` fires
  margin past it). Full-width platforms (no side bypass) and a guaranteed
  spawn runway mirror the platform terrain conventions.
  """

  platform_height_range: tuple[float, float] = (0.0, 0.5)
  """Shared platform height above ground (both platforms EQUAL). Lower bound
  at difficulty 0, upper at 1 — the SAME ladder as the platform terrain."""
  height_jitter: float = 0.05
  """Per-patch uniform jitter added on top of the ladder height, in meters
  (clamped to [0.0, 0.55])."""
  platform_depth_range: tuple[float, float] = (1.2, 1.6)
  """Platform extent along x (landable top surface). ONE shared sample for
  BOTH platforms — identical size and shape (user request 2026-11-01) — and
  deep enough that the robot has room to climb up, stabilize and ACCELERATE
  on the top before reaching the gap (user request 2026-11-01)."""
  platform_width_range: tuple[float, float] = (7.4, 7.5)
  """Platform extent along y — patch-full-width so the gap cannot be bypassed
  sideways."""
  wall1_left_x_range: tuple[float, float] = (2.8, 3.1)
  """Platform 1 LEFT-edge x, sampled per patch. Pushed farther from the spawn
  than the platform terrain's (2.3, 2.7) (user request 2026-11-01: the flat
  acceleration runway before the climb is 2.05-2.35 m, worst 1.85 m with the
  +/-0.2 m reset jitter). Budget check: worst case left1=3.1 + depth=1.6 +
  gap=0.3 + depth=1.6 = right2=6.6 -> finish 7.6, success line 8.0 (same
  bound as the platform terrain)."""
  gap_range: tuple[float, float] = (0.0, 0.3)
  """Edge-to-edge gap between the two platform tops (user spec: 0 cm -> 30 cm
  with difficulty). The gap is a TRUE hole down to the pit floor."""
  pit_depth_range: tuple[float, float] = (0.5, 1.0)
  """Pit floor depth below z=0 under the gap, randomized per patch — deep
  enough that falling in cannot be walked off (0.5 m walls on both sides)."""
  approach_offset: float = 0.30
  """Low guide point x-offset BEFORE platform 1's left edge (user: 上台前
  30 cm)."""
  cross_offset: float = 0.25
  """High milestone x-offset past the gap onto platform 2 (clamped to the
  platform top so the capture bubble always lands on solid surface)."""
  descend_offset: float = 0.30
  """Low guide point x-offset AFTER platform 2's right edge (user: 下台后
  30 cm)."""
  finish_offset: float = 1.0
  """Finish guide x-offset past platform 2 (user: 终点 100 cm); chain TAIL —
  ``parkour_completed`` fires 0.4 m past it."""
  border_width: float = 0.25
  """Solid ring width around the patch closing the pit sides, in meters."""
  max_reward_points: int = 5
  """Pre-allocation cap: approach low + platform-1 top + past-gap high +
  descend low + finish tail."""
  low_bonus_ratio: float = 0.2
  """Reward factor for low-value guide points (milestones pay 1.0)."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []
    reward_points: list[np.ndarray] = []
    reward_point_factors: list[np.ndarray] = []
    low = np.asarray(self.low_bonus_ratio, dtype=np.float32)
    high = np.asarray(1.0, dtype=np.float32)

    height = _lerp_range(difficulty, self.platform_height_range)
    height = float(
      np.clip(height + rng.uniform(-self.height_jitter, self.height_jitter), 0.0, 0.55)
    )
    gap = _lerp_range(difficulty, self.gap_range)
    pit_depth = rng.uniform(*self.pit_depth_range)

    spawn_x = self.border_width + 0.5
    cy = self.size[1] / 2

    left1 = rng.uniform(*self.wall1_left_x_range)
    # ONE shared depth sample: the two platforms are IDENTICAL in size and
    # shape (user request 2026-11-01) — same depth, same full width, and the
    # same height sample above.
    depth = rng.uniform(*self.platform_depth_range)
    right1 = left1 + depth
    left2 = right1 + gap
    right2 = left2 + depth

    # -- Low-high guide reward chain (x-ordered; 3 low + 2 high) --
    reward_points.append(np.array([left1 - self.approach_offset, cy, 0.0]))
    reward_point_factors.append(low)
    reward_points.append(np.array([(left1 + right1) / 2, cy, height]))
    reward_point_factors.append(high)
    # Past-gap milestone: clamp the offset to half the platform depth so the
    # capture bubble (hit_radius <= 0.125) always sits ON platform 2's top.
    cross_x = left2 + min(self.cross_offset, depth / 2)
    reward_points.append(np.array([cross_x, cy, height]))
    reward_point_factors.append(high)
    reward_points.append(np.array([right2 + self.descend_offset, cy, 0.0]))
    reward_point_factors.append(low)
    reward_points.append(np.array([right2 + self.finish_offset, cy, 0.0]))
    reward_point_factors.append(low)

    # -- Geometry --
    # Pit floor: ONLY under the gap strip (slabs elsewhere reach below it, so
    # there is no crawlspace under the ground boxes).
    floor_h = 0.1
    pit_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(
        (left2 - right1) / 2,
        (self.size[1] - 2 * self.border_width) / 2,
        floor_h / 2,
      ),
      pos=((right1 + left2) / 2, cy, -pit_depth - floor_h / 2),
    )
    geometries.append(TerrainGeometry(geom=pit_geom, color=_PIT_FLOOR))

    # Border ring closing the pit sides (down to the pit floor, top at z=0).
    if self.border_width > 0.0:
      border_boxes = make_border(
        body,
        self.size,
        (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width),
        pit_depth,
        (self.size[0] / 2, self.size[1] / 2, -pit_depth / 2),
      )
      for box in border_boxes:
        geometries.append(TerrainGeometry(geom=box, color=_GRAY))

    # Ground slabs: spawn-side ground up to platform 1, landing-side ground
    # after platform 2 (top at z=0). The strip between them is the pit. Each
    # slab reaches DOWN to the pit floor's bottom (pit_depth + floor_h), so
    # the pit walls are the slab sides all the way down — no hidden cave
    # under the ground for a fallen robot to walk into. Inner width only:
    # the border ring already occupies the patch perimeter.
    slab_depth = pit_depth + floor_h
    inner_len_y = self.size[1] - 2 * self.border_width
    inner_right = self.size[0] - self.border_width
    slab_spans = ((self.border_width, right1), (left2, inner_right))
    for x0, x1 in slab_spans:
      length = x1 - x0
      if length <= 1e-3:
        continue
      slab = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(length / 2, inner_len_y / 2, slab_depth / 2),
        pos=((x0 + x1) / 2, cy, -slab_depth / 2),
      )
      geometries.append(TerrainGeometry(geom=slab, color=_GRAY))

    # The two equal-height platforms (a near-zero height would be a
    # zero-thickness box — MuJoCo rejects that; the slab IS the surface).
    if height > 1e-3:
      # Identical platforms: one shared (depth, height) sample, two centers.
      for px in ((left1 + right1) / 2, (left2 + right2) / 2):
        platform_geom = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(
            depth / 2,
            _lerp_range(difficulty, self.platform_width_range) / 2,
            height / 2,
          ),
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
class BoxStraightCrawlBarsTerrainCfg(SubTerrainCfg):
  """Two full-width overhead bars the robot must crawl UNDER, chained along +x.

  Layout mirrors wmp_go1's ``crawl`` terrain (terrain.py): each patch carries
  two floating bars spanning the full inner width. A bar's underside sits at
  the ``clearance`` (0.42 -> 0.27 m, difficulty-interpolated — wmp_go1's
  validated quantities) and extends 1.0 m upward, so jumping over is
  impossible and passing REQUIRES a low-profile crouch. Bar thickness along x
  is randomized per bar; the robot walks on flat ground the whole way.

  FIVE reward points (3 low @ 0.2 + 2 high @ 1.0, same cadence as the gap
  terrain): 0.3 m before bar 1, 0.3 m past bar 1 (milestone), mid-run, 0.3 m
  past bar 2 (milestone), and the finish guide 1.0 m out as the chain TAIL.
  All points sit on the ground (z=0) — the z-margin gate of
  ``reward_point_bonus`` is trivially satisfied there.
  """

  clearance_range: tuple[float, float] = (0.42, 0.27)
  """Bar underside height above ground. Difficulty 0 -> 0.42 m (mild crouch),
  difficulty 1 -> 0.27 m (real low crawl) — wmp_go1 quantities."""
  bar_height: float = 1.0
  """Bar vertical extent (from the underside upward, in meters). With the
  worst-case clearance the bar top is >= 1.27 m — no jump-over."""
  bar_thickness_range: tuple[float, float] = (0.2, 0.5)
  """Bar extent along x, randomized per bar (thin bars are hopped under,
  thick bars force a sustained crouch)."""
  bar_width: float = 7.5
  """Bar extent along y — patch-full-width (8 m patch minus borders), so the
  bars cannot be bypassed sideways."""
  bar1_left_x_range: tuple[float, float] = (3.2, 3.6)
  """Bar 1 LEFT-edge x, sampled per patch (spawn runway >= 2.45 m)."""
  bar2_left_x_range: tuple[float, float] = (5.8, 6.2)
  """Bar 2 LEFT-edge x, sampled per patch. Worst case (left 6.2 + thickness
  0.5) keeps the finish guide (right + 1.0) and the success line (guide +
  0.4 m margin) inside the patch (<= 7.75 m inner edge)."""
  approach_offset: float = 0.30
  """Low guide point x-offset before bar 1."""
  pass_offset: float = 0.30
  """High milestone x-offset past each bar (user cadence: 过杆后 30 cm)."""
  finish_offset: float = 1.0
  """Finish guide x-offset past bar 2; chain TAIL."""
  border_width: float = 0.25
  """Flat border width around the patch, in meters."""
  floor_thickness: float = 0.5
  """Thickness of the ground slab below z=0, in meters."""
  max_reward_points: int = 5
  """Pre-allocation cap: approach low + bar-1 pass + mid-run low + bar-2 pass
  + finish tail."""
  low_bonus_ratio: float = 0.2
  """Reward factor for low-value guide points (milestones pay 1.0)."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []
    reward_points: list[np.ndarray] = []
    reward_point_factors: list[np.ndarray] = []
    low = np.asarray(self.low_bonus_ratio, dtype=np.float32)
    high = np.asarray(1.0, dtype=np.float32)

    clearance = _lerp_range(difficulty, self.clearance_range)
    spawn_x = self.border_width + 0.5
    cy = self.size[1] / 2

    # Two bars, each sampled independently (left edge + x thickness).
    left1 = rng.uniform(*self.bar1_left_x_range)
    thick1 = rng.uniform(*self.bar_thickness_range)
    left2 = rng.uniform(*self.bar2_left_x_range)
    thick2 = rng.uniform(*self.bar_thickness_range)
    right1 = left1 + thick1
    right2 = left2 + thick2

    # -- Low-high guide reward chain (x-ordered; 3 low + 2 high, all z=0) --
    reward_points.append(np.array([left1 - self.approach_offset, cy, 0.0]))
    reward_point_factors.append(low)
    reward_points.append(np.array([right1 + self.pass_offset, cy, 0.0]))
    reward_point_factors.append(high)
    reward_points.append(np.array([(right1 + left2) / 2, cy, 0.0]))
    reward_point_factors.append(low)
    reward_points.append(np.array([right2 + self.pass_offset, cy, 0.0]))
    reward_point_factors.append(high)
    reward_points.append(np.array([right2 + self.finish_offset, cy, 0.0]))
    reward_point_factors.append(low)

    # -- Geometry --
    # Ground slab (top at z=0) across the whole patch.
    floor_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, self.floor_thickness / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -self.floor_thickness / 2),
    )
    geometries.append(TerrainGeometry(geom=floor_geom, color=_GRAY))

    # The two floating bars: static boxes from `clearance` up (MuJoCo static
    # geoms need no support structure). Distinct color: NOT the climb walls'
    # orange — the depth camera must tell "go under" from "go over".
    bar_color = (0.55, 0.30, 0.75, 1.0)
    for px, thick in (((left1 + right1) / 2, thick1), ((left2 + right2) / 2, thick2)):
      bar_geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(thick / 2, self.bar_width / 2, self.bar_height / 2),
        pos=(px, cy, clearance + self.bar_height / 2),
      )
      geometries.append(TerrainGeometry(geom=bar_geom, color=bar_color))

    origin = np.array([spawn_x, cy, 0.0])
    return TerrainOutput(
      origin=origin,
      geometries=geometries,
      reward_points=reward_points,
      reward_point_factors=reward_point_factors,
    )


@dataclass(kw_only=True)
class BoxStraightChainTerrainCfg(SubTerrainCfg):
  """把若干「直线单地形」首尾相接拼成**一条** +x 方向的完整跑酷路线。

  用途（用户需求 2026-11-03）：play 时不再按地形分类各自成列，而是把若
  干种地形排成一条直线，机器人从最左端出生、依次穿过所有地形，命令方向
  始终指向整条路线终点。训练仍用 `PARKOUR_TERRAINS_CFG` 的「每列一种
  地形」网格，本类只服务于 play（见 `FULL_PARKOUR_TERRAINS_CFG`）。

  拼接方式：**几何平移 + 奖励点平移**，不做任何几何重写。每个子地形都
  是在自己的 patch 局部坐标系里沿 +x 长 0..size[0] 布障碍、出生点位于
  `border_width + 0.5` 的直线地形；因此把第 i 段的局部坐标整体平移到
  `offset_x = sum(前 i 段的 size[0])` 即可无缝对接（相邻两段的边框/地面
  板在接缝处是同一 z=0 平面，不产生台阶）。

  段顺序与难度：`segments` 的顺序 = 机器人经过的顺序；同一条线内**所有
  段共用同一个 `difficulty`**（= 地形行号，见
  `TerrainGenerator._generate_curriculum_terrains`），相邻行整条线难度递增。

  终点：最后一段（台阶）自身不产奖励点，因此这里在整条线末端**补一个
  终值点**（`final_point_offset` 反向偏移、低价值 `final_point_factor`），
  使航点链尾 / `parkour_completed` / 「命令始终指向的终点」三者的位置
  一致地落在最后一段的末端平地上。
  """

  segments: tuple[SubTerrainCfg, ...] = ()
  """按机器人经过顺序排列的子地形配置。每段**保留自己的 `size`**（x 长度
  与 y 宽度），拼接总长 = 各段 `size[0]` 之和，宽度必须一致。"""

  final_point_offset: float = 0.5
  """终点奖励点在整条线末端之前的 x 偏移（米）：终值点 x =
  总长 - final_point_offset。

  **耦合约束（复核 D-5，改动时请看这里）**：终值点既是命令的链尾目标，
  也是 `parkour_completed` 的判定基准（成功线 = 终值点 + margin，当前
  `parkour_env_cfg.py` 的 `margin=0.4`）。因此要求
  ``final_point_offset >= margin + 最后一段末端的平地余量``，成功线才会
  落在实地上：当前最后一段（原生台阶，border_width=1.0）末端有 1 m 平地，
  终值点取在末端前 0.5 m，成功线在末端前 0.1 m（31.9 m ≤ 32 m 线内）。
  若把本参数调到 < 0.4 m 或把台阶段的 border_width 收窄，必须同步检查
  成功线是否仍在线内、且脚下有地面。"""

  final_point_factor: float = 0.2
  """终点奖励点的价值系数（低值 0.2，与各段引导点同一量纲）：终点只作为
  链尾位置标识，不给通关额外大额 bonus（play 时回合无限，通关不重置）。"""

  max_reward_points: int = 0
  """奖励点前置分配上限 = 各段 `max_reward_points` 之和 + 1（终点）。在
  `__post_init__` 里按 `segments` 自动算出，勿手工填写。"""

  def __post_init__(self) -> None:
    if len(self.segments) < 1:
      raise ValueError("BoxStraightChainTerrainCfg needs at least one segment.")
    # 生成器在建网格时会扫描该属性做前置分配（terrain_generator.py 的
    # max_reward_points 汇总），因此必须在构造时就算好：各段上限之和
    # （没有奖励点的段按 0 计）+ 本类补的终点。
    self.max_reward_points = (
      sum(int(getattr(seg, "max_reward_points", 0)) for seg in self.segments) + 1
    )

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    # NOTE: no direct geom creation here — each segment appends its own
    # geometries to the shared spec body, then we translate them by +x.
    geometries: list[TerrainGeometry] = []
    reward_points: list[np.ndarray] = []
    reward_point_factors: list[np.ndarray] = []

    # -- 布局自检（失败要响亮：拼接错位会让障碍物叠在一起，且奖励点/命令
    # 终点会落到错误位置，属于静默错乱，必须在生成期直接报错）--
    total_len = float(sum(float(seg.size[0]) for seg in self.segments))
    if abs(total_len - float(self.size[0])) > 1e-6:
      raise ValueError(
        "BoxStraightChainTerrainCfg: the chained segments' x lengths sum to "
        f"{total_len:.3f} m but the composite patch x size is "
        f"{float(self.size[0]):.3f} m. Set the terrain generator size to the "
        f"segment sum ({total_len:.1f}, {float(self.size[1]):.1f})."
      )
    widths = {round(float(seg.size[1]), 6) for seg in self.segments}
    if len(widths) > 1 or abs(next(iter(widths)) - float(self.size[1])) > 1e-6:
      raise ValueError(
        "BoxStraightChainTerrainCfg: every segment must share ONE patch "
        f"width (segments: {sorted(widths)}, composite: {float(self.size[1])})."
      )

    offset_x = 0.0
    origin = np.zeros(3)
    for seg_index, seg_cfg in enumerate(self.segments):
      output = seg_cfg.function(difficulty, spec, rng)
      # 第一段的出生点就是整条线的出生点（后续段只平移几何，不改出生）。
      if seg_index == 0:
        origin = np.asarray(output.origin, dtype=np.float64).copy()
      # 几何：把该段的局部坐标平移到本段在整条线上的起点。
      shift = np.array([offset_x, 0.0, 0.0])
      for terrain_geom in output.geometries:
        if terrain_geom.geom is not None:
          terrain_geom.geom.pos = np.asarray(terrain_geom.geom.pos) + shift
        geometries.append(terrain_geom)
      # 奖励点：同样平移，按段顺序天然保持 x 递增（跨段即机器人前进方向）。
      if output.reward_points:
        seg_factors = output.reward_point_factors
        seg_end_x = offset_x + float(seg_cfg.size[0])
        for point_index, point in enumerate(output.reward_points):
          shifted = np.asarray(point, dtype=np.float64) + shift
          # 逐段区间自检（复核建议 D-3）：点必须落在**本段**的 x 跨度内。只查
          # 全局递增抓不到"点串到相邻段"这类错（例如将来调大 finish_offset
          # 时点会越过本段边界，却仍可能满足全局递增）。
          if not (offset_x - 1e-6 <= shifted[0] <= seg_end_x + 1e-6):
            raise ValueError(
              "BoxStraightChainTerrainCfg: segment "
              f"{self.segments[seg_index]!r} emits a reward point at global "
              f"x={shifted[0]:.3f} m, outside its own span [{offset_x:.3f}, "
              f"{seg_end_x:.3f}] m — check the segment's offset fields."
            )
          reward_points.append(shifted)
          if seg_factors is None:
            # 未声明系数的段（如原生台阶/步石类地形）默认全值 1.0，与生成器
            # 预填 1.0 的默认语义一致（terrain_generator 的 factors 数组）。
            reward_point_factors.append(np.asarray(1.0, dtype=np.float32))
          else:
            reward_point_factors.append(
              np.asarray(seg_factors[point_index], dtype=np.float32)
            )
      offset_x += float(seg_cfg.size[0])

    # -- 路线终点：最后一段（台阶）没有自己的奖励点，这里补链尾终值点 --
    finish_x = total_len - float(self.final_point_offset)
    reward_points.append(np.array([finish_x, float(self.size[1]) / 2, 0.0]))
    reward_point_factors.append(np.asarray(self.final_point_factor, dtype=np.float32))

    # 航点链、终点朝向与 `parkour_completed` 都依赖「点按 x 递增」这一不变量。
    xs = np.array([float(p[0]) for p in reward_points])
    if np.any(np.diff(xs) <= 1e-3):
      raise ValueError(
        "BoxStraightChainTerrainCfg: reward points must be strictly "
        f"increasing along +x (got x={np.round(xs, 3).tolist()})."
      )

    return TerrainOutput(
      origin=origin,
      geometries=geometries,
      reward_points=reward_points,
      reward_point_factors=reward_point_factors,
    )
