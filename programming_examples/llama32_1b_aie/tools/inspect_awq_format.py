#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Inspect and assert the Llama-3.2 AWQ checkpoint tensor format.

This is the Stage-1 format gate for the AWQ decode implementation. It trusts
``config.json`` and actual safetensors metadata rather than README prose.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple

# Allow running as ``python3 tools/inspect_awq_format.py`` from the example dir.
sys.path.insert(0, os.path.dirname(__file__))

from awq_format_reference import (  # noqa: E402
    load_awq_tensors,
    unpack_int4_reorder,
    unpack_qzeros_reorder,
    validate_quant_config,
)

EXPECTED_SHAPES: Dict[str, Tuple[int, ...]] = {
    "model.layers.0.self_attn.q_proj.qweight": (2048, 256),
    "model.layers.0.self_attn.q_proj.qzeros": (16, 256),
    "model.layers.0.self_attn.q_proj.scales": (16, 2048),
    "model.layers.0.self_attn.k_proj.qweight": (2048, 64),
    "model.layers.0.mlp.down_proj.qweight": (8192, 256),
    "lm_head.qweight": (2048, 16032),
}


def _collect_shapes(model_path: str) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    from safetensors import safe_open

    shapes: Dict[str, Tuple[Tuple[int, ...], str]] = {}
    files = sorted(
        os.path.join(model_path, name)
        for name in os.listdir(model_path)
        if name.endswith(".safetensors")
    )
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in EXPECTED_SHAPES:
                    tensor = handle.get_tensor(key)
                    shapes[key] = (tuple(tensor.shape), str(tensor.dtype))
    return shapes


def main() -> int:
    parser = argparse.ArgumentParser(description="Assert AWQ config and representative tensor shapes")
    parser.add_argument("--model", required=True, help="Local AWQ model directory")
    args = parser.parse_args()

    config = validate_quant_config(args.model)
    qc = config["quantization_config"]
    print("quantization_config:")
    for key in ["bits", "group_size", "quant_method", "version", "pack_method", "zero_point"]:
        print(f"  {key}: {qc[key]!r}")

    shapes = _collect_shapes(args.model)
    print("representative_shapes:")
    for key, expected in EXPECTED_SHAPES.items():
        if key not in shapes:
            raise AssertionError(f"missing tensor {key}")
        actual, dtype = shapes[key]
        print(f"  {key}: {actual} {dtype}")
        if actual != expected:
            raise AssertionError(f"{key}: expected {expected}, got {actual}")

    # Exercise unsigned unpack paths and qzero +1 rule on a representative module.
    q_proj = load_awq_tensors(args.model, "model.layers.0.self_attn.q_proj")
    logical = unpack_int4_reorder(q_proj.qweight, q_proj.scales.shape[1])
    zeros = unpack_qzeros_reorder(q_proj.qzeros, q_proj.scales.shape[1])
    if logical.shape != (2048, 2048):
        raise AssertionError(f"q_proj logical qweight shape mismatch: {logical.shape}")
    if zeros.shape != (16, 2048):
        raise AssertionError(f"q_proj logical qzeros shape mismatch: {zeros.shape}")
    if logical.min() < 0 or logical.max() > 15:
        raise AssertionError("unpacked qweight is outside uint4 range")
    if zeros.min() < 0 or zeros.max() > 15:
        raise AssertionError("unpacked qzeros are outside expected uint4 range [0, 15]")
    print("unpack_smoke:")
    print(f"  q_proj.logical_qweight: {logical.shape} min={int(logical.min())} max={int(logical.max())}")
    print(f"  q_proj.zeros_raw: {zeros.shape} min={int(zeros.min())} max={int(zeros.max())}")
    print("PASS inspect_awq_format")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
