"""Render top-down layout images of the parkour platform (low-wall) terrain.

Verifies the 2026-10-30 layout redesign: wall-1 acceleration runway, wall-2
pinned near the patch edge, and the 7-point reward chain (landing milestone
+ finish guide tail past wall 2). Each render is accompanied by the numeric
layout (wall edges, gap, reward points, success line) so the picture can be
checked against the budget numbers.

Drawing is pure matplotlib (no OpenGL), so it works headless. The MuJoCo
spec is still compiled per case as a construction-chain smoke check.

Marker colors: RED = milestone (factor 1.0), BLUE = low-value guide (0.2),
GREEN = spawn origin. The success-termination line (finish guide + 0.4 m
parkour_completed margin) is a dashed yellow line; the patch edge is at
x = 8.0.

Run with:
  uv run python scripts/tools/render_parkour_platform.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import Rectangle

from mjlab.terrains.parkour_terrains import BoxStraightPlatformTerrainCfg
from mjlab.terrains.terrain_generator import TerrainOutput

OUTPUT_DIR = Path("docs/source/_static/terrains")
SIZE = (8.0, 8.0)

# (difficulty, seed) cases: mid difficulty with two random draws plus a hard
# row. difficulty >= 0.5 guarantees both walls survive the height-jitter
# skip threshold and appear as geometry.
CASES = [(0.5, 0), (0.5, 1), (0.9, 2)]


def build_patch(
  cfg: BoxStraightPlatformTerrainCfg, difficulty: float, seed: int
) -> TerrainOutput:
  """Generate one patch and compile its spec as a smoke check."""
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  rng = np.random.default_rng(seed)
  output = cfg.function(difficulty, spec, rng)
  spec.compile()  # rejects invalid geometry (zero-thickness boxes etc.)
  return output


def print_layout(output: TerrainOutput, difficulty: float, seed: int) -> None:
  """Print the numeric layout so the render can be checked by the numbers."""
  walls = sorted(
    (g.geom.pos[0], g.geom.size[0] * 2)
    for g in output.geometries
    if g.geom is not None
    and g.color is not None
    and tuple(g.color) == (0.85, 0.55, 0.15, 1.0)
  )
  (cx1, d1), (cx2, d2) = walls
  left1, re1 = cx1 - d1 / 2, cx1 + d1 / 2
  left2, re2 = cx2 - d2 / 2, cx2 + d2 / 2
  pts = np.stack(output.reward_points)
  # Label from the chain TAIL (fixed slots); approach points take whatever
  # prefix remains (adaptive in principle, fixed at 2 with the current
  # wall1_left_x_range).
  tail_labels = ["finish", "landing", "gap*", "gap*", "wall1_top"]
  labels = ["approach*"] * (len(pts) - len(tail_labels)) + tail_labels[::-1]
  print(f"\n== difficulty={difficulty} seed={seed} ==")
  print(f"spawn x={output.origin[0]:.2f}  runway={left1 - output.origin[0]:.2f} m")
  print(
    f"wall1 left={left1:.2f} depth={d1:.2f} right={re1:.2f} | "
    f"gap={left2 - re1:.2f} m | wall2 left={left2:.2f} depth={d2:.2f} "
    f"right={re2:.2f}"
  )
  for label, p in zip(labels, pts, strict=True):
    print(f"  {label:>9}: x={p[0]:.2f} z={p[2]:.2f}")
  print(
    f"success line x={pts[-1][0] + 0.4:.2f} "
    f"(patch edge {SIZE[0]:.1f}, utilization to line = "
    f"{(pts[-1][0] + 0.4) / SIZE[0]:.0%})"
  )


def draw_patch(output: TerrainOutput, difficulty: float, seed: int, path: Path) -> None:
  """Draw one top-down patch view with walls, reward points, success line."""
  fig, ax = plt.subplots(figsize=(11, 4))
  ax.set_xlim(-0.2, SIZE[0] + 0.6)
  ax.set_ylim(SIZE[1] / 2 - 1.2, SIZE[1] / 2 + 1.2)
  ax.set_aspect("equal")
  ax.set_xlabel("patch-local x (m)")
  ax.set_ylabel("y (m)")
  ax.set_title(
    f"parkour platform d={difficulty} seed={seed} — "
    "red=milestone, blue=guide, green=spawn, yellow=success line",
    fontsize=11,
  )

  # Floor slab and walls from the generated geometry (top-down rectangles).
  for g in output.geometries:
    if g.geom is None:
      continue
    cx, cy, hx, hy = (
      g.geom.pos[0],
      g.geom.pos[1],
      g.geom.size[0],
      g.geom.size[1],
    )
    is_wall = g.color is not None and tuple(g.color) == (0.85, 0.55, 0.15, 1.0)
    # The floor slab is the full-width gray box; draw it lightly.
    if not is_wall and hx > SIZE[0] / 2 - 1e-6:
      ax.add_patch(
        Rectangle(
          (cx - hx, cy - hy), 2 * hx, 2 * hy, facecolor="#eeeeee", edgecolor="none"
        )
      )
    elif is_wall:
      ax.add_patch(
        Rectangle(
          (cx - hx, cy - hy),
          2 * hx,
          2 * hy,
          facecolor="#f0a860",
          edgecolor="#a06010",
          label="low wall",
        )
      )

  # Reward points: red = milestone, blue = low-value guide.
  assert output.reward_points is not None
  assert output.reward_point_factors is not None
  for point, factor in zip(
    output.reward_points, output.reward_point_factors, strict=True
  ):
    milestone = factor >= 0.9
    ax.plot(
      point[0],
      point[1],
      marker="o",
      markersize=9,
      color="#d02020" if milestone else "#3050d0",
      markeredgecolor="black",
      markeredgewidth=0.5,
      zorder=3,
    )
  # Spawn origin (green) and the patch edge + success line.
  ax.plot(
    output.origin[0],
    output.origin[1],
    marker="s",
    markersize=9,
    color="#20a040",
    markeredgecolor="black",
    markeredgewidth=0.5,
    zorder=3,
  )
  finish_x = float(output.reward_points[-1][0])
  ax.axvline(SIZE[0], color="black", linewidth=1.0, linestyle=":")
  ax.axvline(finish_x + 0.4, color="#e0c020", linewidth=2.0, linestyle="--")

  fig.tight_layout()
  fig.savefig(path, dpi=150)
  plt.close(fig)


def main() -> None:
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  cfg = BoxStraightPlatformTerrainCfg(size=SIZE)
  for difficulty, seed in CASES:
    output = build_patch(cfg, difficulty, seed)
    print_layout(output, difficulty, seed)
    path = OUTPUT_DIR / f"parkour_platform_d{difficulty:.2f}_s{seed}.png"
    draw_patch(output, difficulty, seed, path)
    print(f"  saved {path}")
  print("\nDone.")


if __name__ == "__main__":
  main()
