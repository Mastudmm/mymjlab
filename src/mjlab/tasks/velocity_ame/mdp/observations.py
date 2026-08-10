"""AME-specific observation helpers.

``terrain_points`` produces a robot-centric terrain map with shape
``[B, num_x, num_y, 3]`` from a ``RayCastSensor`` that uses a
``GridPatternCfg`` and ``ray_alignment="yaw"``. The three channels are the
hit-point displacement relative to the sensor origin expressed in the sensor's
yaw frame (x forward, y lateral, z height). Misses fall back to the nominal
ray offset in x/y and ``-max_distance`` in z.
"""

from __future__ import annotations

import torch

from mjlab.sensor import GridPatternCfg, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat


def terrain_points(
  env,
  sensor_name: str,
  height_noise_range: tuple[float, float] | None = None,
  apply_drift: bool = False,
  drift_attr: str = "_ame_map_scan_drift_xy",
  clip_height_range: tuple[float, float] | None = None,
) -> torch.Tensor:
  """Return a robot-centric terrain map with shape [B, L, W, 3]."""

  sensor: RayCastSensor = env.scene[sensor_name]
  if not isinstance(sensor.cfg.pattern, GridPatternCfg):
    raise TypeError("AME terrain_points currently supports GridPatternCfg only.")

  # 在 env 上缓存网格形状与标称射线偏移（只计算一次，后续 step 直接复用）。
  # GridPatternCfg 的射线顺序为 x 最快、y 最慢（meshgrid "xy" 生成），故先
  # reshape 成 (num_y, num_x, 3) 再 permute 到 (num_x, num_y, 3)，得到
  # 前向 x / 侧向 y 的布局，与 CNN 输入的 H/W 维度对齐。
  shape_attr = f"_ame_{sensor_name}_grid_shape"
  offsets_attr = f"_ame_{sensor_name}_grid_offsets"
  if not hasattr(env, shape_attr) or not hasattr(env, offsets_attr):
    pattern = sensor.cfg.pattern
    num_x = int(round(pattern.size[0] / pattern.resolution)) + 1
    num_y = int(round(pattern.size[1] / pattern.resolution)) + 1
    local_offsets, _ = pattern.generate_rays(None, env.device)
    local_offsets = local_offsets.reshape(num_y, num_x, 3).permute(1, 0, 2).contiguous()
    setattr(env, shape_attr, (num_x, num_y))
    setattr(env, offsets_attr, local_offsets)

  num_x, num_y = getattr(env, shape_attr)
  local_offsets = getattr(env, offsets_attr)

  # 命中点相对 sensor 原点的位移（世界系）。hit_pos_w 形状 [B, N, 3]，
  # pos_w 是 sensor 第 0 个 frame 的世界位置 [B, 3]。
  relative_pos_w = sensor.data.hit_pos_w - sensor.data.pos_w.unsqueeze(1)
  # 旋转到 sensor 的 yaw 系（用 yaw_quat 提取纯 yaw 分量、丢弃 roll/pitch），
  # 使输出地图机器人中心化：x = 前向, y = 侧向, z = 高度。
  # 这样无论机器人朝哪，地形图都以前方为 +x，注意力才能学到统一语义。
  sensor_quat = yaw_quat(sensor.data.quat_w)
  batch_size, num_points, _ = relative_pos_w.shape
  local_points = quat_apply_inverse(
    sensor_quat.unsqueeze(1).expand(batch_size, num_points, 4).reshape(-1, 4),
    relative_pos_w.reshape(-1, 3),
  ).reshape(batch_size, num_points, 3)
  local_points = (
    local_points.reshape(env.num_envs, num_y, num_x, 3).permute(0, 2, 1, 3).contiguous()
  )

  # 处理未命中射线（射线打到天空/超出 max_distance）：x/y 回退到标称射线
  # 偏移（保持网格位置），z 设为大负值哨兵 -max_distance 表示"此处无地面"。
  # 用大负值而非 0，避免与真实地面高度混淆，让 CNN/注意力能识别为"未知"。
  miss_mask = sensor.data.distances < 0
  miss_mask = (
    miss_mask.reshape(env.num_envs, num_y, num_x).permute(0, 2, 1).contiguous()
  )
  points = local_points
  points[..., 0:2] = torch.where(
    miss_mask.unsqueeze(-1),
    local_offsets.unsqueeze(0).expand(env.num_envs, -1, -1, -1)[..., 0:2],
    points[..., 0:2],
  )
  points[..., 2] = torch.where(
    miss_mask,
    torch.full_like(points[..., 2], -sensor.cfg.max_distance),
    points[..., 2],
  )
  # 以下为 finetune 阶段可选的感知扰动（base 阶段不启用）：
  # - height_noise_range：给 z 加均匀噪声，模拟高度测量误差；
  # - apply_drift：加 XY 漂移（从 env._ame_map_scan_drift_xy 读取的系统性
  #   定位偏差），模拟传感器-机体标定误差，硬化编码器鲁棒性；
  # - clip_height_range：裁剪 z 到合理范围，避免极端值干扰训练。
  if height_noise_range is not None:
    points[..., 2] = points[..., 2] + torch.empty_like(points[..., 2]).uniform_(
      height_noise_range[0],
      height_noise_range[1],
    )
  if apply_drift and hasattr(env, drift_attr):
    drift_xy = getattr(env, drift_attr)
    points[..., 0] += drift_xy[:, None, None, 0]
    points[..., 1] += drift_xy[:, None, None, 1]
  if clip_height_range is not None:
    points[..., 2] = torch.clamp(
      points[..., 2], min=clip_height_range[0], max=clip_height_range[1]
    )
  return points
