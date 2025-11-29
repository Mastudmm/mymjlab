"""Print ONNX model metadata and a concise graph summary.

Usage:
    python3 src/mjlab/scripts/show_onnx_metadata.py --path <policy.onnx> [--graph]

Shows:
  - run_path
  - joint_names (ordered)
  - default_joint_pos
  - action_scale
  - joint_stiffness
  - joint_damping
  - command_names
  - observation_names (policy)

Notes:
- Metadata values were stored as CSV strings; this script parses them back.
- If onnx is not installed, install: `pip install onnx` (or add to pyproject).
- Use --graph to also print model inputs/outputs, initializers (params) and a node op summary.
"""
from __future__ import annotations

import argparse
import onnx


def parse_csv(s: str):
    # Split by comma, try float conversion, fallback to original.
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            out.append(p)
    return out


def _print_graph_summary(model: onnx.ModelProto):
    print("\n[GRAPH] IR version:", model.ir_version)
    graph = model.graph
    print("[GRAPH] Producer:", f"{model.producer_name} {model.producer_version}".strip())
    print("[GRAPH] Opset imports:", [(imp.domain or "ai.onnx", imp.version) for imp in model.opset_import])

    # Inputs / Outputs
    print("\n[GRAPH] Inputs (name:type[shape]):")
    for vi in graph.input:
        t = vi.type.tensor_type
        dtype = t.elem_type
        dims = [d.dim_value if d.dim_value != 0 else d.dim_param or "?" for d in t.shape.dim]
        print(f"  - {vi.name}: elem={dtype} shape={dims}")

    print("\n[GRAPH] Outputs (name:type[shape]):")
    for vo in graph.output:
        t = vo.type.tensor_type
        dtype = t.elem_type
        dims = [d.dim_value if d.dim_value != 0 else d.dim_param or "?" for d in t.shape.dim]
        print(f"  - {vo.name}: elem={dtype} shape={dims}")

    # Initializers (parameters)
    print("\n[GRAPH] Initializers (params):")
    total_params = 0
    for init in graph.initializer:
        shape = list(init.dims)
        numel = 1
        for d in shape:
            numel *= int(d or 1)
        total_params += numel
        print(f"  - {init.name}: dtype={init.data_type} shape={shape} numel={numel}")
    print(f"[GRAPH] Total parameters: {total_params}")

    # Nodes summary by op_type
    print("\n[GRAPH] Nodes summary:")
    from collections import Counter
    counts = Counter(node.op_type for node in graph.node)
    for op, cnt in counts.most_common():
        print(f"  - {op}: {cnt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to ONNX policy file")
    ap.add_argument("--graph", action="store_true", help="Print graph inputs/outputs, params and node ops summary")
    args = ap.parse_args()

    model = onnx.load(args.path)
    meta = {m.key: m.value for m in model.metadata_props}

    # Keys we expect.
    expected = [
        "run_path",
        "joint_names",
        "default_joint_pos",
        "action_scale",
        "joint_stiffness",
        "joint_damping",
        "command_names",
        "observation_names",
    ]

    print(f"[INFO] Metadata keys present: {sorted(meta.keys())}")

    for k in expected:
        if k not in meta:
            print(f"[WARN] Missing key: {k}")
            continue
        val = meta[k]
        if k in {"joint_names", "command_names", "observation_names", "run_path"}:
            # These are strings possibly containing commas but represent lists.
            parsed = [p for p in val.split(",") if p]
            print(f"\n{k} ({len(parsed)}):")
            for i, item in enumerate(parsed):
                print(f"  [{i:02d}] {item}")
        else:
            parsed = parse_csv(val)
            print(f"\n{k} ({len(parsed)}):")
            for i, item in enumerate(parsed):
                print(f"  [{i:02d}] {item}")

    # Convenience: joint name → action_scale mapping (same ordering index alignment assumed).
    if "joint_names" in meta and "action_scale" in meta:
        joint_names = [p for p in meta["joint_names"].split(",") if p]
        scales = parse_csv(meta["action_scale"])
        print("\nJoint → action_scale mapping:")
        for j, sc in zip(joint_names, scales):
            print(f"  {j}: {sc}")

    if args.graph:
        _print_graph_summary(model)


if __name__ == "__main__":
    main()
