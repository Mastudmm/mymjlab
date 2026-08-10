"""Phase registry for AME Go1 tasks.

A single Go1 variant is trained in two phases:

* ``base``: rough-terrain curriculum training from scratch.
* ``finetune``: domain randomization + map-scan drift on top of a base
  checkpoint, to harden the terrain encoder for deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AmePhase = Literal["base", "finetune"]

TRAIN_NUM_ENVS = 4096
PLAY_NUM_ENVS = 1


@dataclass(frozen=True)
class AmeTaskSpec:
  """Static task identifiers for an AME Go1 phase."""

  phase: AmePhase
  task_id: str
  experiment_name: str
  train_num_envs: int = TRAIN_NUM_ENVS
  play_num_envs: int = PLAY_NUM_ENVS


_TASK_SPECS: dict[AmePhase, AmeTaskSpec] = {
  "base": AmeTaskSpec(
    phase="base",
    task_id="Mjlab-VelocityAme-Base-Unitree-Go1",
    experiment_name="go1_velocity_ame_base",
  ),
  "finetune": AmeTaskSpec(
    phase="finetune",
    task_id="Mjlab-VelocityAme-Finetune-Unitree-Go1",
    experiment_name="go1_velocity_ame_finetune",
  ),
}


def iter_task_specs() -> tuple[AmeTaskSpec, ...]:
  """Return all registered task specs in stable (base, finetune) order."""

  return tuple(_TASK_SPECS[phase] for phase in ("base", "finetune"))


def resolve_task_spec(phase: AmePhase = "base") -> AmeTaskSpec:
  """Return the task spec for the requested phase."""

  try:
    return _TASK_SPECS[phase]
  except KeyError as error:
    raise ValueError(f"Unsupported AME phase: {phase!r}") from error
