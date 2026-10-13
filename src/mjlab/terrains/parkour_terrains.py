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
  """Plum-blossom stepping stones rising from a deep pit, chained along +x.

  Each cluster is a triangle of three stone columns (one front, two rear-side).
  The robot spawns on a solid spawn column and hops cluster to cluster; the
  pit floor (``floor_depth`` below) makes any fall a terminating event.
  """

  stone_size_range: tuple[float, float] = (0.25, 0.4)
  """Stone side length (square columns), in meters."""
  stone_height_range: tuple[float, float] = (0.2, 0.35)
  """Stone top height above z=0. Lower bound at difficulty 0, upper at 1."""
  stone_height_variation: float = 0.05
  """Per-stone top-height jitter (scaled by difficulty), in meters."""
  spacing_range: tuple[float, float] = (1.2, 2.0)
  """Center-to-center distance between consecutive clusters, in meters."""
  max_clusters: int = 3
  """Maximum number of clusters along the chain."""
  spawn_free_radius: float = 1.0
  """Radius around the spawn point kept free of obstacles, in meters."""
  border_width: float = 0.25
  """Solid ring width around the patch closing the pit, in meters."""
  floor_depth: float = 2.0
  """Pit depth below z=0, in meters."""
  max_reward_points: int = 3
  """Pre-allocation cap for per-(row, col) reward-point arrays."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []
    reward_points: list[np.ndarray] = []

    top = _lerp_range(difficulty, self.stone_height_range)
    spacing = _lerp_range(difficulty, self.spacing_range)
    variation = self.stone_height_variation * difficulty
    num_clusters = 2 if difficulty < 0.5 else min(3, self.max_clusters)

    spawn_x = self.border_width + 0.5
    cy = self.size[1] / 2

    # Border ring closing the pit walls.
    if self.border_width > 0.0:
      border_boxes = make_border(
        body,
        self.size,
        (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width),
        top + self.floor_depth,
        (self.size[0] / 2, self.size[1] / 2, (top - self.floor_depth) / 2),
      )
      for box in border_boxes:
        geometries.append(TerrainGeometry(geom=box, color=_GRAY))

    # Pit floor.
    floor_h = 0.1
    floor_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, floor_h / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -self.floor_depth - floor_h / 2),
    )
    geometries.append(TerrainGeometry(geom=floor_geom, color=_PIT_FLOOR))

    # Spawn column (solid, top flush with the stones).
    spawn_half = 0.5
    spawn_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(spawn_half, spawn_half, (top + self.floor_depth) / 2),
      pos=(spawn_x, cy, (top - self.floor_depth) / 2),
    )
    geometries.append(TerrainGeometry(geom=spawn_geom, color=_GREEN))

    def add_stone(px: float, py: float, stone_top: float) -> None:
      h = self.floor_depth + stone_top
      size = rng.uniform(*self.stone_size_range)
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(size / 2, size / 2, h / 2),
        pos=(px, py, -self.floor_depth + h / 2),
      )
      geometries.append(TerrainGeometry(geom=geom, color=_ORANGE))

    for i in range(num_clusters):
      cx = spawn_x + self.spawn_free_radius + i * spacing
      stone_top = top + rng.uniform(-variation, variation)
      # Plum-blossom cluster: one front stone, two rear-side stones.
      front_size = rng.uniform(*self.stone_size_range)
      side_dx = front_size * 0.6
      side_dy = front_size * 0.9
      add_stone(cx, cy, stone_top)
      add_stone(cx - side_dx, cy - side_dy, stone_top)
      add_stone(cx - side_dx, cy + side_dy, stone_top)
      reward_points.append(np.array([cx, cy, stone_top]))

    origin = np.array([spawn_x, cy, top])
    return TerrainOutput(
      origin=origin, geometries=geometries, reward_points=reward_points
    )


@dataclass(kw_only=True)
class BoxStraightPlatformTerrainCfg(SubTerrainCfg):
  """Raised platform boxes chained along +x on flat ground.

  Each platform is tall (up to 0.35 m) and wide enough across y to block
  detours, forcing the robot to jump onto/over it. One reward point sits on
  each platform's top center.
  """

  platform_height_range: tuple[float, float] = (0.15, 0.35)
  """Platform height above ground. Lower bound at difficulty 0, upper at 1."""
  platform_depth: float = 0.8
  """Platform extent along x (landable top surface), in meters."""
  platform_width_range: tuple[float, float] = (3.0, 4.5)
  """Platform extent along y (blocks detours). Lower bound at difficulty 0."""
  spacing_range: tuple[float, float] = (1.5, 2.5)
  """Center-to-center distance between consecutive platforms, in meters."""
  max_platforms: int = 3
  """Maximum number of platforms along the chain."""
  spawn_free_radius: float = 1.0
  """Radius around the spawn point kept free of obstacles, in meters."""
  border_width: float = 0.25
  """Flat border width around the patch, in meters."""
  floor_thickness: float = 0.5
  """Thickness of the ground slab below z=0, in meters."""
  max_reward_points: int = 3
  """Pre-allocation cap for per-(row, col) reward-point arrays."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []
    reward_points: list[np.ndarray] = []

    height = _lerp_range(difficulty, self.platform_height_range)
    width = _lerp_range(difficulty, self.platform_width_range)
    spacing = _lerp_range(difficulty, self.spacing_range)
    num_platforms = 2 if difficulty < 0.5 else min(3, self.max_platforms)

    spawn_x = self.border_width + 0.5
    cy = self.size[1] / 2

    # Ground slab (top at z=0).
    floor_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, self.floor_thickness / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -self.floor_thickness / 2),
    )
    geometries.append(TerrainGeometry(geom=floor_geom, color=_GRAY))

    for i in range(num_platforms):
      px = spawn_x + self.spawn_free_radius + i * spacing
      platform_geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(self.platform_depth / 2, width / 2, height / 2),
        pos=(px, cy, height / 2),
      )
      geometries.append(TerrainGeometry(geom=platform_geom, color=_ORANGE))
      reward_points.append(np.array([px, cy, height]))

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

  gap_width_range: tuple[float, float] = (0.2, 0.35)
  """Gap extent along x. Lower bound at difficulty 0, upper at 1."""
  gap_depth: float = 1.5
  """Trench depth below z=0, in meters."""
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

    spawn_x = self.border_width + 0.5
    cy = self.size[1] / 2
    inner_len_y = self.size[1] - 2 * self.border_width

    # Border ring closing the trench sides.
    if self.border_width > 0.0:
      border_boxes = make_border(
        body,
        self.size,
        (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width),
        self.gap_depth,
        (self.size[0] / 2, self.size[1] / 2, -self.gap_depth / 2),
      )
      for box in border_boxes:
        geometries.append(TerrainGeometry(geom=box, color=_GRAY))

    # Pit floor.
    floor_h = 0.1
    floor_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, floor_h / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -self.gap_depth - floor_h / 2),
    )
    geometries.append(TerrainGeometry(geom=floor_geom, color=_PIT_FLOOR))

    def add_segment(x0: float, x1: float) -> None:
      length = x1 - x0
      if length <= 1e-3:
        return
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(length / 2, inner_len_y / 2, self.gap_depth / 2),
        pos=((x0 + x1) / 2, cy, -self.gap_depth / 2),
      )
      geometries.append(TerrainGeometry(geom=geom, color=_GRAY))

    # First walkable segment covers the spawn zone.
    cursor = spawn_x
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
