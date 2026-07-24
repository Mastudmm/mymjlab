"""Live matplotlib plot of base velocity + 12 leg joints during play/eval.

Opens a real-time window with 4 stacked subplots: base linear velocity
(vx, vy, vz), then hip / thigh / calf joint angles (4 legs per subplot).
The selected env (dog) is plotted. Uses ``constrained_layout`` so subplots
reflow on resize/fullscreen, with auto-scaling y-axes. Disabled by default —
only activates when the wrapper is applied (e.g. via ``--live-plot``).
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv

_LEG_ORDER = ("FR", "FL", "RR", "RL")
_PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
_VEL_LABELS = ("vx", "vy", "vz")
_VEL_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")


class LiveJointPlot(ManagerBasedRlEnv):
  """Wraps an env to plot base velocity + leg joints in real time.

  Four stacked subplots: base lin vel, then hip / thigh / calf (4 legs each).
  ``constrained_layout`` adapts subplot geometry on window resize/fullscreen;
  y-axes auto-rescale every refresh while x stays a fixed rolling time window.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    env_idx: int = 0,
    history_len: int = 300,
    refresh_every: int = 4,
  ) -> None:
    self._wrapped_env = env
    self._env_idx = env_idx
    self._history_len = history_len
    self._refresh_every = refresh_every
    self._step_count = 0
    self._step_dt = float(env.step_dt)

    asset = env.scene["robot"]
    names = list(asset.joint_names)
    leg_rank = {leg: k for k, leg in enumerate(_LEG_ORDER)}

    def _joints_of(jtype: str) -> list[int]:
      hits = [(i, n) for i, n in enumerate(names) if re.search(rf"{jtype}_joint$", n)]
      hits.sort(key=lambda in_: next((leg_rank[leg] for leg in _LEG_ORDER if leg in in_[1]), 99))
      return [i for i, _ in hits[:4]]

    self._joint_groups = {jt: _joints_of(jt) for jt in ("hip", "thigh", "calf")}

    self._vel_hist = [deque(maxlen=history_len) for _ in _VEL_LABELS]
    self._joint_hist = {
      jt: [deque(maxlen=history_len) for _ in self._joint_groups[jt]]
      for jt in ("hip", "thigh", "calf")
    }

    import matplotlib.pyplot as plt

    self._plt = plt
    plt.ion()
    self.fig, self._axs = plt.subplots(
      4, 1, figsize=(9, 10), constrained_layout=True, num="Live Joint Plot"
    )
    self._x_lim = (-history_len * self._step_dt, 0.0)

    # Subplot 0: base linear velocity (base frame).
    self._vel_lines = [
      self._axs[0].plot([], [], label=vl, color=vc)[0]
      for vl, vc in zip(_VEL_LABELS, _VEL_COLORS, strict=True)
    ]
    self._axs[0].set_ylabel("base lin vel [m/s]")
    self._axs[0].set_title(f"env {env_idx}  (step_dt={self._step_dt:.3f}s)")
    self._axs[0].legend(loc="upper right", fontsize=7, ncol=3)
    self._axs[0].grid(True, alpha=0.3)
    self._axs[0].set_xlim(*self._x_lim)

    # Subplots 1-3: hip / thigh / calf (4 legs each).
    self._joint_lines: dict[str, list] = {}
    for k, jtype in enumerate(("hip", "thigh", "calf")):
      ax = self._axs[k + 1]
      self._joint_lines[jtype] = [
        ax.plot([], [], label=leg, color=_PALETTE[i])[0]
        for i, leg in enumerate(_LEG_ORDER)
      ]
      ax.set_ylabel(f"{jtype} [rad]")
      if k == 2:
        ax.set_xlabel("time [s]")
      ax.legend(loc="upper right", fontsize=7, ncol=4)
      ax.grid(True, alpha=0.3)
      ax.set_xlim(*self._x_lim)

  def __getattr__(self, name: str) -> Any:
    return getattr(self._wrapped_env, name)

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self._wrapped_env.unwrapped

  def reset(self, **kwargs: Any) -> Any:
    return self._wrapped_env.reset(**kwargs)

  def step(self, action: torch.Tensor) -> Any:
    result = self._wrapped_env.step(action)
    asset = self.unwrapped.scene["robot"]
    ei = self._env_idx
    lin_vel = asset.data.root_link_lin_vel_b[ei].cpu().numpy()
    for h, v in zip(self._vel_hist, lin_vel, strict=True):
      h.append(float(v))
    joint_pos = asset.data.joint_pos
    for jtype in ("hip", "thigh", "calf"):
      for h, idx in zip(
        self._joint_hist[jtype], self._joint_groups[jtype], strict=True
      ):
        h.append(float(joint_pos[ei, idx].item()))
    self._step_count += 1
    if self._step_count % self._refresh_every == 0:
      self._refresh()
    return result

  def _draw_ax(self, ax, lines, hists) -> None:
    for line, h in zip(lines, hists, strict=True):
      y = np.fromiter(h, dtype=np.float64, count=len(h))
      x = np.arange(-len(y), 0, dtype=np.float64) * self._step_dt
      line.set_data(x, y)
    ax.relim()
    ax.autoscale_view(scalex=False)
    ax.set_xlim(*self._x_lim)

  def _refresh(self) -> None:
    self._draw_ax(self._axs[0], self._vel_lines, self._vel_hist)
    for k, jtype in enumerate(("hip", "thigh", "calf")):
      self._draw_ax(self._axs[k + 1], self._joint_lines[jtype], self._joint_hist[jtype])
    self.fig.canvas.draw_idle()
    try:
      self.fig.canvas.flush_events()
    except Exception:
      self._plt.pause(1e-3)
