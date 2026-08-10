"""AME on-policy runner for Go1 velocity tasks.

Extends :class:`VelocityOnPolicyRunner` to:

* Export AME attention metadata (per-token layout, encoder variant, etc.)
  alongside each checkpoint, so downstream attention visualization works.
* Skip the legacy MLPModel key migration on load. The AME model stores
  submodules directly (``proprio_normalizer``, ``local_encoder``,
  ``attention``, ``mlp``, ``distribution``) without the ``actor.``/``mlp.``
  prefixes that :class:`MjlabOnPolicyRunner` migrates, so only the
  rsl-rl 4.x -> 5.x distribution key migration is kept.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner


class AmeOnPolicyRunner(VelocityOnPolicyRunner):
  """Runner that persists AME attention metadata with each checkpoint."""

  def save(self, path: str, infos=None) -> None:
    # Persist checkpoint + env_state via MjlabOnPolicyRunner directly, skipping
    # VelocityOnPolicyRunner.save's base-metadata step. That step reads the
    # "actor" observation group, which AME does not have (it uses four groups:
    # actor_proprio/actor_terrain/critic_proprio/critic_terrain), so
    # get_base_metadata would raise KeyError("actor").
    MjlabOnPolicyRunner.save(self, path, infos)
    policy_path = Path(path).resolve().parent
    filename = f"{policy_path.name}.onnx"
    try:
      self.export_policy_to_onnx(str(policy_path), filename)
    except Exception as error:  # noqa: BLE001
      print(f"[WARN] AME ONNX export failed (training continues): {error}")
    actor = getattr(self.alg, "actor", None)
    critic = getattr(self.alg, "critic", None)
    try:
      if actor is not None and hasattr(actor, "export_attention_metadata"):
        actor.export_attention_metadata(str(policy_path))
      if critic is not None and hasattr(critic, "export_attention_metadata"):
        critic.export_attention_metadata(
          str(policy_path), filename="critic_attention_metadata.json"
        )
      metadata = {
        "runner_cls": type(self).__name__,
        "actor_cls": type(actor).__name__ if actor is not None else None,
        "critic_cls": type(critic).__name__ if critic is not None else None,
      }
      with open(
        policy_path / "ame_runner_metadata.json", "w", encoding="utf-8"
      ) as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
    except Exception as error:  # noqa: BLE001
      print(
        f"[WARN] AME attention metadata export failed (training continues): {error}"
      )

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    # AME models store submodules directly (proprio_normalizer, local_encoder,
    # attention, mlp, distribution) without the legacy "actor."/"mlp." prefixes,
    # so the MLPModel key migration in MjlabOnPolicyRunner does not apply.
    # Keep only the rsl-rl 4.x -> 5.x distribution key migration.
    actor_sd = loaded_dict.get("actor_state_dict", {})
    if "std" in actor_sd:
      actor_sd["distribution.std_param"] = actor_sd.pop("std")
    if "log_std" in actor_sd:
      actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict.get("infos")
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos
