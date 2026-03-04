import os

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.utils.lab_api.rl.exporter import _OnnxPolicyExporter


def export_velocity_policy_as_onnx(
  actor_critic: object,
  path: str,
  normalizer: object | None = None,
  filename="policy.onnx",
  verbose=False,
):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
  policy_exporter = _OnnxPolicyExporter(actor_critic, normalizer, verbose)
  policy_exporter.export(path, filename)



def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
  fmt = f"{{:.{decimals}f}}"
  return delimiter.join(
    fmt.format(x)
    if isinstance(x, (int, float))
    else str(x)  # numbers → format, strings → as-is
    for x in arr
  )


def attach_onnx_metadata(   #ONNX metadata 的作用：让你不需要重建环境就能知道准确的观测项顺序、关节顺序、缩放等。使用 .pt 则没有这些附加键值。
  env: ManagerBasedRlEnv, run_path: str, path: str, filename="policy.onnx"
) -> None:
  """Attach velocity-specific metadata to ONNX model.

  Args:
    env: The RL environment.
    run_path: W&B run path or other identifier.
    path: Directory containing the ONNX file.
    filename: Name of the ONNX file.
  """
  onnx_path = os.path.join(path, filename)
  metadata = get_base_metadata(env, run_path)  # Velocity has no extra metadata.
  attach_metadata_to_onnx(onnx_path, metadata)
