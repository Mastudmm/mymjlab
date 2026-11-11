"""Terrain-balanced actor updates and gradient diagnostics for Parkour AMP."""

from itertools import combinations
from math import isfinite

import torch
from torch.nn.functional import cosine_similarity

from mjlab.tasks.parkour_vision.rl.runner import ParkourAMPPPO
from mjlab.terrains.terrain_generator import resolve_curriculum_column_names


class ParkourGroupedAMPPPO(ParkourAMPPPO):
  """Weight the PPO surrogate by terrain, preserving the upstream AMP update."""

  def __init__(
    self,
    *args,
    terrain_group_weights: dict[str, float],
    terrain_grad_interval: int,
    terrain_grad_samples: int,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    total = sum(terrain_group_weights.values())
    if (
      not isfinite(total)
      or not terrain_group_weights
      or any(weight <= 0 for weight in terrain_group_weights.values())
    ):
      raise ValueError("Terrain group weights must be finite and positive.")
    if terrain_grad_interval < 1 or terrain_grad_samples < 1:
      raise ValueError("Gradient interval and sample count must be positive.")

    self._group_names = tuple(terrain_group_weights)
    self._group_weights = tuple(
      weight / total for weight in terrain_group_weights.values()
    )
    self._grad_interval = terrain_grad_interval
    self._grad_samples = terrain_grad_samples
    self._group_indices: tuple[torch.Tensor, ...] = ()
    self._sample_scales: torch.Tensor
    self._terrain_group_ids: torch.Tensor
    self._group_metrics: dict[str, float]
    self._updates = 0

  @classmethod
  def construct_algorithm(
    cls, obs, env, cfg, device, discriminator=None, amp_data=None, amp_normalizer=None
  ):
    alg = super().construct_algorithm(
      obs, env, cfg, device, discriminator, amp_data, amp_normalizer
    )
    assert isinstance(alg, cls)
    terrain = env.unwrapped.scene.terrain
    assert terrain is not None and terrain.cfg.terrain_generator is not None
    columns = resolve_curriculum_column_names(terrain.cfg.terrain_generator)
    groups = {name: i for i, name in enumerate(alg._group_names)}
    by_column = torch.tensor([groups.get(name, -1) for name in columns], device=device)
    # Training keeps each environment in the same column across resets.
    alg._terrain_group_ids = by_column[terrain.terrain_types.to(device)]
    return alg

  def _prepare_groups(self) -> None:
    if self.actor.is_recurrent or self.critic.is_recurrent:
      raise ValueError("Grouped Parkour PPO uses feed-forward rollout batches.")
    if self.normalize_advantage_per_mini_batch:
      raise ValueError(
        "Use rollout-wide advantage normalization with grouped Parkour PPO."
      )

    storage = self.storage
    total = storage.num_envs * storage.num_transitions_per_env
    # Match the sample pool used by RSL-RL's flat mini-batch generator.
    used = total - total % self.num_mini_batches
    group_ids = self._terrain_group_ids.repeat(storage.num_transitions_per_env)[:used]
    self._group_indices = tuple(
      (group_ids == group_id).nonzero(as_tuple=True)[0]
      for group_id in range(len(self._group_names))
    )
    count_tensor = torch.tensor(
      [ids.numel() for ids in self._group_indices], device=self.device
    )
    if self.is_multi_gpu:
      torch.distributed.all_reduce(count_tensor)
      used *= self.gpu_world_size
    counts = count_tensor.tolist()
    if sum(counts) != used or any(count == 0 for count in counts):
      raise ValueError(
        f"Each sampled terrain needs a group and every group needs samples. "
        f"Counts for {self._group_names}: {counts}, total samples: {used}."
      )

    scales = torch.tensor(
      [
        weight * used / count
        for weight, count in zip(self._group_weights, counts, strict=True)
      ],
      device=self.device,
    )
    self._sample_scales = scales[self._terrain_group_ids][None, :, None]
    self._group_metrics = {}
    for name, weight, count in zip(
      self._group_names, self._group_weights, counts, strict=True
    ):
      self._group_metrics[f"terrain/rollout_share/{name}"] = count / used
      self._group_metrics[f"terrain/policy_weight/{name}"] = weight

  def _actor_gradient_metrics(self) -> dict[str, float]:
    storage = self.storage
    observations = storage.observations.flatten(0, 1)
    actions = storage.actions.flatten(0, 1)
    advantages = storage.advantages.flatten()
    old_log_prob = storage.actions_log_prob.flatten()
    params = tuple(param for param in self.actor.parameters() if param.requires_grad)
    device = torch.device(self.device)
    gradients = {}
    # Drawing diagnostics must not advance the training sampler's RNG.
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
      for name, indices in zip(self._group_names, self._group_indices, strict=True):
        count = min(indices.numel(), self._grad_samples)
        if count:
          chosen = indices[torch.randperm(indices.numel(), device=device)[:count]]
          self.actor(observations[chosen], stochastic_output=True)
          log_prob = self.actor.get_output_log_prob(actions[chosen]).reshape(-1)
          ratio = (log_prob - old_log_prob[chosen]).exp()
          advantage = advantages[chosen]
          clipped = ratio.clamp(1 - self.clip_param, 1 + self.clip_param)
          loss = torch.maximum(-advantage * ratio, -advantage * clipped).mean()
          grad = torch.autograd.grad(loss, params, allow_unused=True)
          flat = torch.cat(
            [
              g.reshape(-1) if g is not None else p.new_zeros(p.numel())
              for p, g in zip(params, grad, strict=True)
            ]
          )
        else:
          flat = params[0].new_zeros(sum(p.numel() for p in params))
        if self.is_multi_gpu:
          # Average the probe gradients by actual sample count across ranks.
          count_tensor = flat.new_tensor(count)
          flat *= count_tensor
          torch.distributed.all_reduce(flat)
          torch.distributed.all_reduce(count_tensor)
          flat /= count_tensor
        gradients[name] = flat

    metrics = {
      f"terrain_grad/norm/{name}": g.norm().item() for name, g in gradients.items()
    }
    for left, right in combinations(gradients, 2):
      metrics[f"terrain_grad/cosine/{left}__{right}"] = cosine_similarity(
        gradients[left], gradients[right], dim=0
      ).item()
    return metrics

  def update(self) -> dict[str, float]:
    if not self._group_indices:
      self._prepare_groups()
    metrics = dict(self._group_metrics)
    if self._updates % self._grad_interval == 0:
      metrics.update(self._actor_gradient_metrics())

    # Both branches of the clipped surrogate are linear in a positive
    # advantage multiplier. Scaling by (target share / observed share) makes
    # its rollout mean exactly the weighted mean of the terrain losses.
    advantages = self.storage.advantages
    self.storage.advantages = advantages * self._sample_scales
    try:
      losses = super().update()
    finally:
      self.storage.advantages = advantages
    self._updates += 1
    return losses | metrics
