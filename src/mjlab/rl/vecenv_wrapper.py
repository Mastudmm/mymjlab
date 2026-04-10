import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.utils.spaces import Space


class RslRlVecEnvWrapper(VecEnv):
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    clip_actions: float | None = None,
    amp_enabled: bool = False,
  ):
    self.env = env
    self.clip_actions = clip_actions
    self.amp_enabled = amp_enabled

    self.num_envs = self.unwrapped.num_envs
    self.device = torch.device(self.unwrapped.device)
    self.max_episode_length = self.unwrapped.max_episode_length
    self.num_actions = self.unwrapped.action_manager.total_action_dim
    self._modify_action_space()
    self.reset_env_ids = torch.empty(0, dtype=torch.long, device=self.device)
    self._amp_joint_ids: torch.Tensor | None = None
    self._amp_site_ids: torch.Tensor | None = None

    # Reset at the start since rsl_rl does not call reset.
    self.env.reset()

  @property
  def cfg(self) -> ManagerBasedRlEnvCfg:
    return self.unwrapped.cfg

  @property
  def render_mode(self) -> str | None:
    return self.env.render_mode

  @property
  def observation_space(self) -> Space:
    return self.env.observation_space

  @property
  def action_space(self) -> Space:
    return self.env.action_space

  @classmethod
  def class_name(cls) -> str:
    return cls.__name__

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self.env.unwrapped

  # Properties.

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return self.unwrapped.episode_length_buf

  @property
  def step_dt(self) -> float:
    """Environment control step (seconds) expected by AMP motion sampling."""
    return self.unwrapped.step_dt

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]
    self.unwrapped.episode_length_buf = value

  def seed(self, seed: int = -1) -> int:
    return self.unwrapped.seed(seed)

  def get_observations(self) -> TensorDict:
    obs_dict = self.unwrapped.observation_manager.compute()
    return TensorDict(obs_dict, batch_size=[self.num_envs])

  def reset(self) -> tuple[TensorDict, dict]:
    obs_dict, extras = self.env.reset()
    return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
    obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
    term_or_trunc = terminated | truncated
    assert isinstance(rew, torch.Tensor)
    assert isinstance(term_or_trunc, torch.Tensor)
    dones = term_or_trunc.to(dtype=torch.long)
    self.reset_env_ids = term_or_trunc.nonzero(as_tuple=False).squeeze(-1)
    if self.amp_enabled:
      extras["amp_obs"] = self.get_amp_obs_for_expert_trans()
    if not self.cfg.is_finite_horizon:
      extras["time_outs"] = truncated
    return (
      TensorDict(obs_dict, batch_size=[self.num_envs]),
      rew,
      dones,
      extras,
    )

  def close(self) -> None:
    return self.env.close()

  def get_amp_obs_for_expert_trans(
    self, env_ids: torch.Tensor | None = None
  ) -> torch.Tensor:
    """Return AMP observation as [joint_pos_rel(12), joint_vel_rel(12), foot_pos_b(12)].

    If ``env_ids`` is provided, only those environments are processed.
    """
    asset = self.unwrapped.scene["robot"]

    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)

    joint_ids = self._get_amp_joint_ids(asset)
    site_ids = self._get_amp_site_ids(asset)

    joint_pos_rel = (
      asset.data.joint_pos[env_ids][:, joint_ids]
      - asset.data.default_joint_pos[env_ids][:, joint_ids]
    )
    joint_vel_rel = (
      asset.data.joint_vel[env_ids][:, joint_ids]
      - asset.data.default_joint_vel[env_ids][:, joint_ids]
    )

    feet_pos_w = asset.data.site_pos_w[env_ids][:, site_ids, :]
    root_pos_w = asset.data.root_link_pos_w[env_ids].unsqueeze(1)
    feet_pos_root_w = feet_pos_w - root_pos_w

    root_quat = asset.data.root_link_quat_w[env_ids]
    feet_pos_b = quat_apply_inverse(
      root_quat.unsqueeze(1).expand(-1, feet_pos_root_w.shape[1], -1).reshape(-1, 4),
      feet_pos_root_w.reshape(-1, 3),
    ).reshape(env_ids.shape[0], len(site_ids), 3)

    return torch.cat([joint_pos_rel, joint_vel_rel, feet_pos_b.flatten(start_dim=1)], dim=-1)

  # Private methods.

  def _get_amp_joint_ids(self, asset) -> torch.Tensor:
    if self._amp_joint_ids is not None:
      return self._amp_joint_ids

    joint_names = list(asset.joint_names)
    preferred_patterns = [
      r"^FR_.*hip.*",
      r"^FR_.*thigh.*",
      r"^FR_.*calf.*",
      r"^FL_.*hip.*",
      r"^FL_.*thigh.*",
      r"^FL_.*calf.*",
      r"^RR_.*hip.*",
      r"^RR_.*thigh.*",
      r"^RR_.*calf.*",
      r"^RL_.*hip.*",
      r"^RL_.*thigh.*",
      r"^RL_.*calf.*",
    ]

    import re

    selected: list[int] = []
    for pattern in preferred_patterns:
      for idx, name in enumerate(joint_names):
        if re.search(pattern, name):
          selected.append(idx)
          break

    if len(selected) != 12:
      if len(joint_names) == 12:
        selected = list(range(12))
      else:
        raise ValueError(
          "AMP requires exactly 12 leg joints (FR/FL/RR/RL x hip/thigh/calf). "
          f"Resolved {len(selected)} AMP joints from robot joints: {joint_names}"
        )

    self._amp_joint_ids = torch.tensor(selected, dtype=torch.long, device=self.device)
    return self._amp_joint_ids

  def _get_amp_site_ids(self, asset) -> torch.Tensor:
    if self._amp_site_ids is not None:
      return self._amp_site_ids

    candidate_site_names = None
    reward_cfg = self.cfg.rewards.get("foot_clearance") if hasattr(self.cfg, "rewards") else None
    if reward_cfg is not None:
      asset_cfg = reward_cfg.params.get("asset_cfg")
      if asset_cfg is not None:
        candidate_site_names = getattr(asset_cfg, "site_names", None)

    site_names = list(asset.site_names)
    if candidate_site_names:
      name_to_id = {name: i for i, name in enumerate(site_names)}
      selected = [name_to_id[name] for name in candidate_site_names if name in name_to_id]
    else:
      selected = []

    if len(selected) != 4:
      import re

      selected = [
        idx
        for idx, name in enumerate(site_names)
        if re.search(r"FR|FL|RR|RL|front|rear|foot", name, re.IGNORECASE)
      ][:4]

    if len(selected) != 4:
      raise ValueError(
        "AMP requires exactly 4 foot sites for end-effector positions. "
        f"Resolved {len(selected)} from site names: {site_names}"
      )

    self._amp_site_ids = torch.tensor(selected, dtype=torch.long, device=self.device)
    return self._amp_site_ids

  def _modify_action_space(self) -> None:
    if self.clip_actions is None:
      return

    from mjlab.utils.spaces import Box, batch_space

    self.unwrapped.single_action_space = Box(
      shape=(self.num_actions,), low=-self.clip_actions, high=self.clip_actions
    )
    self.unwrapped.action_space = batch_space(
      self.unwrapped.single_action_space, self.num_envs
    )
