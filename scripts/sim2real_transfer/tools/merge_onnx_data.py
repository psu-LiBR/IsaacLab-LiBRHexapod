# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Merge an ONNX graph and its external weights file into one self-contained ``.onnx``.

Newer PyTorch exporters can write the weights to a separate ``<name>.onnx.data`` file. The ``.onnx`` records that
file by *name*, so the pair breaks when the files are renamed, downloaded under another name, or copied without the
data file (onnxruntime then fails to load the graph). This tool reads the name the graph expects, finds your data
file, and writes a single file that needs nothing else. Nothing is run from the input files; they are only parsed.

Examples:
  python tools/merge_onnx_data.py Hexapod_Sideways.onnx                      # data file auto-detected next to it
  python tools/merge_onnx_data.py Hexapod_Sideways.onnx --data Hexapod_Sideways.onnx.data
  python tools/merge_onnx_data.py policy.onnx --out policy_single.onnx

Needs ``onnx`` (``pip install onnx``); the final load check uses ``onnxruntime`` when it is installed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile


def referenced_locations(model) -> set[str]:
    """File names the graph's initializers expect their external weights to live in."""
    return {
        entry.value for tensor in model.graph.initializer for entry in tensor.external_data if entry.key == "location"
    }


def find_data_file(onnx_path: str, explicit: str | None) -> str | None:
    """The user's data file: ``--data`` if given, else ``<onnx>.data`` or the only ``*.data`` next to the graph."""
    if explicit:
        return explicit
    candidate = onnx_path + ".data"
    if os.path.isfile(candidate):
        return candidate
    siblings = [f for f in os.listdir(os.path.dirname(os.path.abspath(onnx_path))) if f.endswith(".data")]
    if len(siblings) == 1:
        return os.path.join(os.path.dirname(os.path.abspath(onnx_path)), siblings[0])
    return None


def merge(onnx_path: str, out_path: str, data_path: str | None = None) -> None:
    import onnx

    model_stub = onnx.load(onnx_path, load_external_data=False)
    locations = referenced_locations(model_stub)

    if not locations:
        print("[merge_onnx_data] the graph has no external data; copying it unchanged.")
        if os.path.abspath(onnx_path) != os.path.abspath(out_path):
            shutil.copyfile(onnx_path, out_path)
        return

    data_path = find_data_file(onnx_path, data_path)
    if data_path is None or not os.path.isfile(data_path):
        raise SystemExit(
            f"the graph expects external data {sorted(locations)} but no data file was found; "
            "pass it with --data <file>."
        )
    if len(locations) != 1:
        raise SystemExit(f"the graph references several external data files {sorted(locations)}; not supported.")
    (expected_name,) = locations
    if os.path.basename(expected_name) != expected_name:
        raise SystemExit(f"refusing external data location with a path component: {expected_name!r}")
    print(f"[merge_onnx_data] graph expects '{expected_name}'; using data file '{data_path}'.")

    # Stage both files in a scratch folder with the name the graph expects, so renamed copies still work.
    with tempfile.TemporaryDirectory() as tmp:
        staged_onnx = os.path.join(tmp, "model.onnx")
        shutil.copyfile(onnx_path, staged_onnx)
        shutil.copyfile(data_path, os.path.join(tmp, expected_name))
        model = onnx.load(staged_onnx)  # resolves the external weights from the staging folder
        onnx.save_model(model, out_path, save_as_external_data=False)

    print(f"[merge_onnx_data] wrote {out_path} ({os.path.getsize(out_path)} bytes, no external data).")

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("[merge_onnx_data] onnxruntime not installed; skipped the load check.")
        return
    session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    inp, out = session.get_inputs()[0], session.get_outputs()[0]
    dummy = np.zeros([d if isinstance(d, int) else 1 for d in inp.shape], dtype=np.float32)
    result = session.run(None, {inp.name: dummy})[0]
    ok = bool(np.isfinite(result).all())
    print(f"[merge_onnx_data] load check: input {inp.name}{inp.shape} -> output {out.name}{out.shape}, finite={ok}")
    if not ok:
        raise SystemExit("the merged model produced non-finite output on a zero input.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("onnx", help="The .onnx graph.")
    parser.add_argument("--data", default=None, help="The external weights file (default: auto-detect).")
    parser.add_argument("--out", default=None, help="Output path (default: <name>_single.onnx next to the input).")
    args = parser.parse_args()

    out_path = args.out or os.path.splitext(args.onnx)[0] + "_single.onnx"
    merge(args.onnx, out_path, args.data)


if __name__ == "__main__":
    sys.exit(main())
