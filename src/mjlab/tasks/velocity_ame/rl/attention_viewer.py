"""AME 注意力可视化 viewer。

在 ``NativeMujocoViewer`` 基础上，每帧把 AME actor 缓存的注意力点
(``last_attention_points``) 与权重 (``last_attention_weights``) 画成彩色
球体叠加到 viewer 中：颜色从蓝(低注意力)到红(高注意力)，半径随权重增大，
让用户在 play 时直观看到策略正在关注地形的哪些位置。

坐标转换：``last_attention_points`` 是机器人中心化的 yaw 系坐标(相对 sensor
原点)，需用 terrain_scan sensor 的世界位姿旋转并平移回世界坐标，才能画到
viewer 里与机器人对齐。
"""

from __future__ import annotations

import torch

from mjlab.utils.lab_api.math import quat_apply, yaw_quat
from mjlab.viewer.native.viewer import NativeMujocoViewer
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer


def _weight_to_rgba(weight: float) -> tuple[float, float, float, float]:
  """把归一化权重 [0,1] 映射成 RGBA 颜色。

  0(低注意力)->蓝 (0.2, 0.4, 1.0)，1(高注意力)->红 (1.0, 0.2, 0.2)。
  """
  r = 0.2 + 0.8 * weight
  g = 0.4 - 0.2 * weight
  b = 1.0 - 0.8 * weight
  return (r, g, b, 0.85)


class AmeAttentionViewer(NativeMujocoViewer):
  """Native viewer 叠加 AME 注意力点为彩色球体。"""

  def __init__(
    self,
    env,
    policy,
    sensor_name: str = "terrain_scan",
    frame_rate: float = 60.0,
    **kwargs,
  ):
    super().__init__(env, policy, frame_rate=frame_rate, **kwargs)
    # terrain_scan sensor 名字，用于取其世界位姿做坐标转换。
    self._sensor_name = sensor_name
    # 球体半径范围：基址 + 权重缩放。低权重球小、高权重球大。
    self._base_radius = 0.02
    self._weight_scale = 0.06

  def _update_debug_visualizers(self, viewer) -> None:
    # 先跑 env 原有的 debug 可视化(如 terrain_scan 射线)，再叠加注意力球体。
    super()._update_debug_visualizers(viewer)
    if not self._show_debug_vis:
      return
    self._draw_attention(viewer)

  def _draw_attention(self, viewer) -> None:
    # get_inference_policy 返回 AME actor 本身，其 forward 已缓存上一步的注意力。
    actor = self.policy
    weights = getattr(actor, "last_attention_weights", None)
    points = getattr(actor, "last_attention_points", None)
    if weights is None or points is None:
      # 首步之前还没有注意力数据，跳过。
      return

    idx = self.env_idx
    # 取 terrain_scan sensor 的世界位姿，用于把机器人中心化点转回世界坐标。
    sensor = self.env.unwrapped.scene[self._sensor_name]
    pos_w = sensor.data.pos_w[idx]  # [3] sensor 原点世界位置
    quat_w = sensor.data.quat_w[idx]  # [4] sensor 世界姿态四元数
    yaw_q = yaw_quat(quat_w)  # 仅保留 yaw 分量(丢弃 roll/pitch)

    # attention_points: [H', W', 3] 机器人中心化(yaw 系，相对 sensor 原点)。
    pts_local = points[idx]  # [H', W', 3]
    pts_flat = pts_local.reshape(-1, 3)  # [N, 3]
    num_tokens = pts_flat.shape[0]

    # 旋转回世界系：用 yaw 四元数把 body 系向量转到世界系(仅 yaw，z 不变)，
    # 再加 sensor 世界位置得到命中点世界坐标。
    yaw_q_expanded = yaw_q.unsqueeze(0).expand(num_tokens, 4)
    pts_world = pos_w + quat_apply(yaw_q_expanded, pts_flat)  # [N, 3]
    pts_world_np = pts_world.detach().cpu().numpy()

    # weights: [N]，归一化到 [0,1] 便于颜色/大小映射。
    w = weights[idx].reshape(-1).float().detach().cpu()
    w_min, w_max = float(w.min()), float(w.max())
    if w_max - w_min < 1e-8:
      w_norm = torch.zeros_like(w)
    else:
      w_norm = (w - w_min) / (w_max - w_min)

    # 用同一个 user_scn 追加注意力球体(在 env debug viz 之后)。
    assert self.mjm is not None
    visualizer = MujocoNativeDebugVisualizer(
      viewer.user_scn, self.mjm, idx, show_all_envs=False
    )
    for i in range(num_tokens):
      weight = float(w_norm[i])
      radius = self._base_radius + self._weight_scale * weight
      color = _weight_to_rgba(weight)
      visualizer.add_sphere(pts_world_np[i], radius, color)
