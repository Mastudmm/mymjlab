"""可视化 AME 训练/play 地形（不训练，只看地形 mesh）。

从 AME env_cfg 读取 ``terrain_generator``（反映你在 env_cfg 里的实际配置，包括
改成 ``ALL_TERRAINS_CFG`` 或自定义 set），生成地形 mesh 用 MuJoCo viewer 显示。

训练地形与 play 地形不同（train 用课程模式，play 用随机 5×5），所以 play 时
看到的地形与训练时不一致。本工具可分别查看两种地形。

用法::

  # 看训练地形（课程模式，反映 env_cfg 的 terrain_generator）
  uv run ame-viz-terrain Mjlab-VelocityAme-Base-Unitree-Go1 --mode train

  # 看 play 地形（随机 5×5）
  uv run ame-viz-terrain Mjlab-VelocityAme-Base-Unitree-Go1 --mode play

注意：需本地图形显示（与 ame-play 的 native viewer 同要求）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

import mujoco.viewer
import torch
import tyro

from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.terrains import TerrainEntity, TerrainEntityCfg


@dataclass(frozen=True)
class VizTerrainConfig:
  """地形可视化配置。"""

  mode: Literal["train", "play"] = "train"
  """看训练地形（课程模式）还是 play 地形（随机 5×5）。"""


def main() -> None:
  maybe_print_top_level_help("ame-viz-terrain")
  import mjlab.tasks  # noqa: F401  触发任务注册

  ame_tasks = [t for t in list_tasks() if "VelocityAme" in t]
  if not ame_tasks:
    raise RuntimeError("未找到 AME 任务，请确认 velocity_ame 已注册。")

  task_id, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(ame_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  cfg = tyro.cli(
    VizTerrainConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {task_id}",
    config=mjlab.TYRO_FLAGS,
  )

  # 从 AME env_cfg 读取 terrain_generator（反映 env_cfg 里的实际配置，包括
  # 改成 ALL_TERRAINS_CFG 或自定义 set）。
  env_cfg = load_env_cfg(task_id, play=(cfg.mode == "play"))
  if env_cfg.scene.terrain is None or env_cfg.scene.terrain.terrain_generator is None:
    raise RuntimeError("AME env_cfg 未配置 terrain_generator。")
  terrain_generator = env_cfg.scene.terrain.terrain_generator

  terrain_cfg = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=terrain_generator,
  )
  device = "cuda" if torch.cuda.is_available() else "cpu"
  terrain = TerrainEntity(terrain_cfg, device=device)

  # 课程模式下列数 = 地形种类数；随机模式下用配置的 num_cols。
  num_types = len(terrain_generator.sub_terrains)
  cols = num_types if terrain_generator.curriculum else terrain_generator.num_cols
  print(
    f"[INFO] 可视化 {cfg.mode} 地形（task={task_id}）: {num_types} 种地形类型, "
    f"{terrain_generator.num_rows} 行 × {cols} 列, "
    f"curriculum={terrain_generator.curriculum}"
  )
  print("[INFO] 关闭 viewer 窗口退出。")
  mujoco.viewer.launch(terrain.spec.compile())


if __name__ == "__main__":
  main()
