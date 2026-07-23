"""AMP Reference State Initialization (RSI) events."""

from __future__ import annotations

import glob
import json
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Cache: motion pattern key -> {joint_pos, joint_vel} on device.
_MOTION_CACHE: dict[str, dict[str, torch.Tensor]] = {}


def _load_motion_frames(patterns: list[str], device: torch.device) -> dict:
  """Load expert motion frames joint_pos[0:12] / joint_vel[12:24] from JSON files."""
  key = ",".join(sorted(patterns))
  if key not in _MOTION_CACHE:
    files: list[str] = []
    for p in patterns:
      matched = glob.glob(p)
      files.extend(matched if matched else [p])
    arrays = [np.array(json.load(open(f))["Frames"]) for f in files]
    frames = np.concatenate(arrays, axis=0)
    _MOTION_CACHE[key] = {
      "joint_pos": torch.tensor(frames[:, :12], dtype=torch.float32, device=device),
      "joint_vel": torch.tensor(frames[:, 12:24], dtype=torch.float32, device=device),
    }
  return _MOTION_CACHE[key]


def reset_joints_from_motion(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  motion_files: list[str],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """AMP Reference State Initialization.

  Reset joint pos/vel from a random expert motion frame so each env starts
  inside the expert gait cycle (matches amp_go2 ``reference_state_initialization``).
  Expert joint_pos is absolute (URDF-zeroed) and written directly as dof_pos.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  data = _load_motion_frames(motion_files, env.device)
  n = data["joint_pos"].shape[0]
  frame_ids = torch.randint(0, n, (len(env_ids),), device=env.device)

  asset = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  asset.write_joint_state_to_sim(
    data["joint_pos"][frame_ids],
    data["joint_vel"][frame_ids],
    env_ids=env_ids,
    joint_ids=joint_ids,
  )
