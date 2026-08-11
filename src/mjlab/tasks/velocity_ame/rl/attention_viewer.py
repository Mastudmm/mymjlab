"""AME 注意力可视化 viewer。

在 ``NativeMujocoViewer`` 基础上，每帧把 AME actor 缓存的注意力点
(``last_attention_points``) 与权重 (``last_attention_weights``) 画成彩色
球体叠加到 viewer 中，让用户在 play 时直观看到策略正在关注地形的哪些位置。

颜色与大小均基于**绝对注意力权重**（非每帧 min-max 相对值），以均匀分布
基线 ``1/N`` 为锚点（N 为 token 数），固定映射范围，跨帧/跨 iteration 可比：

- ``scale = w_i * N``：相对均匀基线的倍数。``1.0`` = 均匀（未学到聚焦），
  ``>1`` = 超基线被注意，``<1`` = 低于均匀。
- **颜色**：``scale`` 经固定范围 ``[0, scale_ref]`` 线性映射到蓝->红渐变，
  表示绝对注意力强度。``scale_ref`` 取 6（6 倍基线即满红），基线 ``1.0``
  显示淡蓝。
- **大小**：``scale`` 经 ``sqrt`` 阈值化映射到半径。``scale <= 1``（低于/等于
  均匀）-> 最小半径（视觉上"不在场"），``scale >= scale_ref`` -> 最大半径。
  用作显著性粗筛，让真正超基线的球在视觉上跳出。

这样训练初期（attention 均匀）全场淡蓝小球，训练后期少数红大球跳出，可一眼
判断 attention 是否从均匀演化到聚焦。

坐标转换：``last_attention_points`` 是机器人中心化的 yaw 系坐标(相对 sensor
原点)，需用 terrain_scan sensor 的世界位姿旋转并平移回世界坐标，才能画到
viewer 里与机器人对齐。
"""

from __future__ import annotations

from mjlab.utils.lab_api.math import quat_apply, yaw_quat
from mjlab.viewer.native.viewer import NativeMujocoViewer
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer


def _intensity_to_rgba(t: float) -> tuple[float, float, float, float]:
  """把归一化强度 ``t∈[0,1]`` 映射成 RGBA。

  ``t`` 由绝对权重相对均匀基线的倍数 ``scale`` 经固定范围 ``[0, scale_ref]``
  线性归一化得到（非每帧 min-max），故跨帧可比。0(低/低于基线)->蓝
  ``(0.2,0.4,1.0)``，1(高/显著超基线)->红 ``(1.0,0.2,0.2)``。
  """
  r = 0.2 + 0.8 * t
  g = 0.4 - 0.2 * t
  b = 1.0 - 0.8 * t
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
    # 满映射阈值：6 倍均匀基线 (scale=6) 即满色满大。固定范围保证跨帧可比。
    self._scale_ref = 6.0
    # 球半径范围：scale<=1(低于/等于均匀)->最小，scale>=scale_ref->最大。
    self._min_radius = 0.01
    self._max_radius = 0.08

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

    # 绝对权重 w_i (softmax，全帧和=1)。均匀基线 = 1/N。
    # scale = w_i * N 为"相对均匀基线的倍数"：1.0=均匀，>1=超基线被注意。
    # 用绝对值(固定基准 N，非每帧 min-max)，跨帧可比，能反映真实集中程度。
    w = weights[idx].reshape(-1).float().detach().cpu()
    n = int(w.shape[0])
    scale = w * n  # [N]

    # 用同一个 user_scn 追加注意力球体(在 env debug viz 之后)。
    assert self.mjm is not None
    visualizer = MujocoNativeDebugVisualizer(
      viewer.user_scn, self.mjm, idx, show_all_envs=False
    )
    for i in range(num_tokens):
      s = float(scale[i])
      # 颜色：绝对强度，固定范围 [0, scale_ref] 线性映射到蓝->红。
      # 基线 scale=1.0 -> 淡蓝；scale>=scale_ref -> 满红。
      t_color = min(max(s / self._scale_ref, 0.0), 1.0)
      color = _intensity_to_rgba(t_color)
      # 大小：显著性粗筛。scale<=1 -> 最小半径(低于均匀，视觉不在场)；
      # scale>=scale_ref -> 最大半径。sqrt 压缩高值，让超基线的球更突出。
      t_size = min(max((s - 1.0) / (self._scale_ref - 1.0), 0.0), 1.0)
      t_size = t_size**0.5
      radius = self._min_radius + (self._max_radius - self._min_radius) * t_size
      visualizer.add_sphere(pts_world_np[i], radius, color)
