"""Export ONNX from an existing checkpoint without requiring Weights & Biases.

Usage:
  python -m mjlab.scripts.export_onnx <task_id> --checkpoint /path/to/model_XXXX.pt

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
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import tyro

import mjlab.tasks  # ensure task registry is populated via module side-effects
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.rl.exporter import (
  attach_onnx_metadata,
  export_velocity_policy_as_onnx,
)


@dataclass
class Args:
  task_id: str
  checkpoint: Path
  run_dir: Optional[Path] = None
  device: Optional[str] = None
  run_id: Optional[str] = None  # Optional: set metadata run_id to match prior exports
  filename: Optional[str] = None  # Optional: override output filename


def main(args: Args):
  # Prepare device
  if args.device:
    device = args.device
  else:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device = "cpu" if cuda_visible == "" else "cuda:0"

  # Load configs and env
  env_cfg = load_env_cfg(args.task_id)
  agent_cfg = load_rl_cfg(args.task_id)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # Runner class and initialization
  runner_cls = load_runner_cls(args.task_id)
  if runner_cls is None:
    from rsl_rl.runners import OnPolicyRunner
    runner_cls = OnPolicyRunner

  # Infer run_dir
  run_dir = args.run_dir or args.checkpoint.parent
  run_dir.mkdir(parents=True, exist_ok=True)

  runner = runner_cls(env, asdict(agent_cfg), str(run_dir), device)

  # Load checkpoint
  print(f"[INFO] Loading checkpoint: {args.checkpoint}")
  runner.load(str(args.checkpoint))

  # Compute export path and filename
  policy_path = str(run_dir) + os.sep
  filename = args.filename or (run_dir.name + ".onnx")

  # Determine normalizer if present
  if getattr(runner.alg.policy, "actor_obs_normalization", False):
    normalizer = runner.alg.policy.actor_obs_normalizer
  else:
    normalizer = None

  # Export with guards (uses the exact same exporter as training-time path)
  try:
    export_velocity_policy_as_onnx(
      runner.alg.policy,
      normalizer=normalizer,
      path=policy_path,
      filename=filename,
    )
    print(f"[INFO] Exported ONNX: {Path(policy_path) / filename}")
  except Exception as e:
    print(f"[ERROR] ONNX export failed: {e}")
    sys.exit(1)

  # Attach metadata (best-effort). If you know the original W&B run.name, pass --run-id to match.
  try:
    run_id = args.run_id or run_dir.name
    attach_onnx_metadata(
      env.unwrapped,
      run_id,
      path=policy_path,
      filename=filename,
    )
    print("[INFO] Attached ONNX metadata.")
  except Exception as e:
    print(f"[WARN] Attaching ONNX metadata failed: {e}")


if __name__ == "__main__":
  args = tyro.cli(Args)
  main(args)
