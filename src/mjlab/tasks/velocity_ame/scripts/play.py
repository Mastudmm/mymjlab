"""AME play 入口：加载训练好的 AME 策略并在 viewer 中回放。

支持两种模式：

- **正常 play** (``--no-attention``)：标准 viewer，只显示机器人和地形，
  与 ``mjlab.scripts.play`` 行为一致。
- **注意力可视化** (``--attention``，默认)：用 :class:`AmeAttentionViewer`
  把策略关注的地形位置画成彩色球体(蓝->红)叠加在 viewer 中，实时展示
  AME 注意力编码器正在"看"地形的哪些位置。

用法示例::

  # 注意力可视化(默认)
  uv run python -m mjlab.tasks.velocity_ame.scripts.play \\
    Mjlab-VelocityAme-Base-Unitree-Go1 \\
    --checkpoint-file /abs/path/to/model_15000.pt

  # 正常 play(不画注意力)
  uv run python -m mjlab.tasks.velocity_ame.scripts.play \\
    Mjlab-VelocityAme-Base-Unitree-Go1 --no-attention

注意：注意力可视化目前只支持 native viewer(需本地图形显示)；
在无显示的服务器上请用 Xvfb 或切换到 ``--no-attention`` + ``--viewer viser``。
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity_ame.rl.attention_viewer import AmeAttentionViewer
from mjlab.utils.os import get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class AmePlayConfig:
  """AME play 配置。"""

  checkpoint_file: str | None = None
  """已训练 checkpoint 的绝对路径(优先于 load_run)。"""
  load_run: str = ".*"
  """从 log_root 下匹配的运行目录加载(正则，取最新)。"""
  load_checkpoint: str = "model_.*.pt"
  """匹配的 checkpoint 文件名(正则，取最新)。"""
  attention: bool = True
  """是否在 viewer 中叠加注意力可视化(关掉则正常 play)。"""
  viewer: Literal["auto", "native", "viser"] = "auto"
  """viewer 后端。auto 按是否有显示选 native/viser。"""
  num_envs: int = 1
  """play 的并行环境数(注意力可视化只看 env 0)。"""
  device: str | None = None
  """计算设备(默认自动选 cuda:0 或 cpu)。"""
  log_root: str = "logs/rsl_rl"
  """实验日志根目录，用于查找 checkpoint。"""


def _resolve_viewer_backend(requested: str) -> str:
  """auto 模式下按是否有图形显示选 native(有)或 viser(无)。"""
  if requested != "auto":
    return requested
  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  return "native" if has_display else "viser"


def main() -> None:
  maybe_print_top_level_help("ame-play")
  import mjlab.tasks  # noqa: F401  触发任务注册

  # 只列出 AME 任务供选择。
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
    AmePlayConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {task_id}",
    config=mjlab.TYRO_FLAGS,
  )

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # 加载 play 环境配置与 RL 配置。
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # 定位 checkpoint：优先 checkpoint_file，否则从 log_root 按 load_run 找最新。
  log_root_path = Path(cfg.log_root) / agent_cfg.experiment_name
  if cfg.checkpoint_file is not None:
    resume_path = Path(cfg.checkpoint_file).expanduser().resolve()
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint 文件不存在: {resume_path}")
  else:
    resume_path = get_checkpoint_path(log_root_path, cfg.load_run, cfg.load_checkpoint)
  print(f"[INFO] 加载 checkpoint: {resume_path}")

  # 构造 runner 并加载 actor 权重(get_inference_policy 返回 AME actor 本身，
  # 其 forward 会缓存 last_attention_weights/last_attention_points)。
  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    raise RuntimeError(f"任务 {task_id} 未注册 runner_cls。")
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  # 选 viewer：attention=True 用 AmeAttentionViewer(仅 native)，否则标准 viewer。
  viewer_backend = _resolve_viewer_backend(cfg.viewer)
  if cfg.attention and viewer_backend != "native":
    print("[WARN] 注意力可视化只支持 native viewer，已切到 native(需图形显示)。")
    viewer_backend = "native"

  if viewer_backend == "native":
    viewer_cls = AmeAttentionViewer if cfg.attention else NativeMujocoViewer
    viewer_cls(env, policy).run()
  elif viewer_backend == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"不支持的 viewer 后端: {viewer_backend}")

  env.close()


if __name__ == "__main__":
  main()
