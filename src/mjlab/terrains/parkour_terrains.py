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
  stones. NO intermediate reward points (user decision 2026-10-16: dense
  mid-chain points let the policy farm easy low rows by suiciding down the
  curriculum ladder) — the ONLY reward point sits on the end column, so the
  whole chain must be crossed to score."""
  border_width: float = 0.25
  """Solid ring width around the patch closing the pit, in meters."""
  floor_depth_range: tuple[float, float] = (0.4, 1.0)
  """Pit depth below z=0, randomized per patch to diversify depth-camera
  signals."""
  max_reward_points: int = 1
  """Pre-allocation cap for per-(row, col) reward-point arrays (end column
  only)."""

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
    # column edge (same gap as between pairs). No mid-chain reward points:
    # the end column carries the single point for the whole chain.
    first_x = spawn_right + gap + self.stone_size / 2
    for i in range(self.num_pairs):
      pair_x = first_x + i * pitch
      top_left = top + rng.uniform(-variation, variation)
      top_right = top + rng.uniform(-variation, variation)
      add_column(pair_x, cy + col_half_dist, self.stone_size / 2, top_left, _ORANGE)
      add_column(pair_x, cy - col_half_dist, self.stone_size / 2, top_right, _ORANGE)

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
  per patch with a 2.5-2.9 m flat run between them. Heights follow the
  difficulty ladder (0.2 -> 0.5 m over the 10 rows) with per-wall jitter;
  thicknesses are randomized per patch (thin walls get hopped over, thick
  walls get climbed). Each wall carries two reward points (top + landing),
  so the full chain pays 4 points.
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
  wall1_x_range: tuple[float, float] = (2.0, 2.5)
  """First wall center x within the patch, randomized per patch."""
  wall_gap_range: tuple[float, float] = (2.5, 2.9)
  """Edge-to-edge flat gap between the two walls (wmp_go1 climb: 2.5-2.9 m).
  The second wall is clamped so its landing reward point stays reachable
  before the patch border."""
  wall2_right_max: float = 6.8
  """Upper bound for the second wall's right edge; keeps the landing point
  (edge + 0.5 m) and the success-termination line (landing + 0.4 m margin)
  inside the walkable patch."""
  border_width: float = 0.25
  """Flat border width around the patch, in meters."""
  floor_thickness: float = 0.5
  """Thickness of the ground slab below z=0, in meters."""
  max_reward_points: int = 4
  """Pre-allocation cap for per-(row, col) reward-point arrays (2 walls x
  top + landing point)."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []
    reward_points: list[np.ndarray] = []

    def sample_height() -> float:
      h = _lerp_range(difficulty, self.platform_height_range)
      h += rng.uniform(-self.height_jitter, self.height_jitter)
      return float(np.clip(h, 0.0, 0.55))

    height1 = sample_height()
    height2 = sample_height()
    depth1 = rng.uniform(*self.platform_depth_range)
    depth2 = rng.uniform(*self.platform_depth_range)
    width = _lerp_range(difficulty, self.platform_width_range)
    px1 = rng.uniform(*self.wall1_x_range)
    gap = rng.uniform(*self.wall_gap_range)
    # Clamp the second wall so landing point + success margin stay on the patch.
    px2 = min(
      px1 + depth1 / 2 + gap + depth2 / 2,
      self.wall2_right_max - depth2 / 2,
    )

    spawn_x = self.border_width + 0.5
    cy = self.size[1] / 2

    # Ground slab (top at z=0).
    floor_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, self.floor_thickness / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -self.floor_thickness / 2),
    )
    geometries.append(TerrainGeometry(geom=floor_geom, color=_GRAY))

    for px, height, depth in ((px1, height1, depth1), (px2, height2, depth2)):
      platform_geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(depth / 2, width / 2, height / 2),
        pos=(px, cy, height / 2),
      )
      geometries.append(TerrainGeometry(geom=platform_geom, color=_ORANGE))
      reward_points.append(np.array([px, cy, height]))
      # Landing point on the ground just past the wall: pays for CROSSING it,
      # not merely touching the top.
      reward_points.append(np.array([px + depth / 2 + 0.5, cy, 0.0]))

    origin = np.array([spawn_x, cy, 0.0])
    return TerrainOutput(
      origin=origin, geometries=geometries, reward_points=reward_points
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
  max_reward_points: int = 3
  """Pre-allocation cap for per-(row, col) reward-point arrays."""

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
    cursor = self.border_width
    for _ in range(num_gaps):
      seg_len = rng.uniform(*self.segment_range)
      add_segment(cursor, cursor + seg_len)
      cursor += seg_len
      # The gap itself is left empty (true hole over the pit floor). The
      # reward point sits just past the gap's far edge (landing side).
      reward_points.append(np.array([cursor + gap_width + 0.15, cy, 0.0]))
      cursor += gap_width

    # Close the patch with a final walkable segment up to the border.
    add_segment(cursor, self.size[0] - self.border_width)

    origin = np.array([spawn_x, cy, 0.0])
    return TerrainOutput(
      origin=origin, geometries=geometries, reward_points=reward_points
    )
