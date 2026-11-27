"""Export ONNX from an existing checkpoint without requiring Weights & Biases.

Usage:
  python -m mjlab.scripts.export_onnx --task-id <task_id> --checkpoint /path/to/model_XXXX.pt

Optional:
  --run-dir /path/to/run_dir     # override output dir; default = checkpoint's parent dir
  --device cpu|cuda:0            # default inferred from CUDA_VISIBLE_DEVICES
  --run-id <string>              # set metadata run_id to match prior W&B exports
  --filename <name.onnx>         # override output filename; default = <run_dir>.onnx

This script reconstructs the env and runner for the given task, loads the
checkpoint weights, and exports the ONNX actor (with normalizer if present).
It also attaches metadata best-effort. No WandB is required.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


@dataclass
class Args:
  task_id: str
  checkpoint: Path
  run_dir: Path | None = None
  device: str | None = None
  run_id: str | None = None
  filename: str | None = None


def main(args: Args):
  # Prepare device
  if args.device:
    device = args.device
  else:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device = "cpu" if cuda_visible == "" else "cuda:0"

  # Load configs and env
  env_cfg = load_env_cfg(args.task_id)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(args.task_id)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  try:
    runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    runner.load(
      str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )

    run_dir = args.run_dir or args.checkpoint.parent
    filename = args.filename or f"{run_dir.name}.onnx"
    onnx_path = run_dir / filename
    runner.export_policy_to_onnx(str(run_dir), filename)
    print(f"[INFO] Exported ONNX: {onnx_path}")

    try:
      metadata = get_base_metadata(env.unwrapped, args.run_id or run_dir.name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      print("[INFO] Attached ONNX metadata.")
    except Exception as e:
      print(f"[WARN] Attaching ONNX metadata failed: {e}")
  finally:
    env.close()


if __name__ == "__main__":
  args = tyro.cli(Args)
  main(args)
