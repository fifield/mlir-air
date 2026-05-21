#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Offline AWQ tensor repacker for AIR Llama decode.

Converts source AWQ GEMM/reorder tensors:
  qweight: (K, N/8) int32
  qzeros:  (K/group_size, N/8) int32
  scales:  (K/group_size, N) bf16/f32

into decode-friendly row-major tensors:
  qweight_repacked:   (N, K/2) uint8, packed low-nibble-first along K
  params_interleaved: (N, 2*K/group_size) bf16, [scale, zero, ...]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, Iterable, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from awq_format_reference import (  # noqa: E402
    dequant_awq_weight,
    load_awq_tensors,
    unpack_int4_reorder,
    unpack_qzeros_reorder,
    validate_quant_config,
)

try:
    from ml_dtypes import bfloat16
except Exception:  # pragma: no cover - environment dependent
    bfloat16 = np.float32

LINEAR_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

COPY_TENSORS = {
    "model.embed_tokens.weight",
    "model.norm.weight",
}


def repack_logical_int4_to_row_major(qweight_u4: np.ndarray) -> np.ndarray:
    """Pack logical ``(K, N)`` uint4 values into ``(N, K/2)`` uint8 rows."""
    qweight_u4 = np.asarray(qweight_u4, dtype=np.uint8)
    if qweight_u4.ndim != 2:
        raise ValueError(f"qweight_u4 must be 2D, got {qweight_u4.shape}")
    if np.any(qweight_u4 > 0xF):
        raise ValueError("qweight_u4 contains values outside [0, 15]")
    k, _n = qweight_u4.shape
    if k % 2 != 0:
        raise ValueError(f"K must be even for uint4 pair packing, got {k}")

    rows_nk = np.ascontiguousarray(qweight_u4.T)
    low = rows_nk[:, 0::2]
    high = rows_nk[:, 1::2]
    return (low | (high << np.uint8(4))).astype(np.uint8, copy=False)


def _interleave_params(qzeros: np.ndarray, scales: np.ndarray, dtype=bfloat16) -> np.ndarray:
    qzeros = np.asarray(qzeros, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    if qzeros.shape != scales.shape:
        raise ValueError(f"qzeros shape {qzeros.shape} != scales shape {scales.shape}")
    params = np.empty((scales.shape[1], scales.shape[0] * 2), dtype=np.float32)
    params[:, 0::2] = scales.T
    params[:, 1::2] = qzeros.T
    return params.astype(dtype)


def repack_module_from_logical(
    qweight_u4: np.ndarray,
    qzeros: np.ndarray,
    scales: np.ndarray,
    group_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Repack already-unpacked logical AWQ tensors."""
    qweight_u4 = np.asarray(qweight_u4, dtype=np.uint8)
    qzeros = np.asarray(qzeros, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.float32)
    k, n = qweight_u4.shape
    groups = (k + group_size - 1) // group_size
    if qzeros.shape != (groups, n):
        raise ValueError(f"qzeros shape {qzeros.shape} does not match {(groups, n)}")
    if scales.shape != (groups, n):
        raise ValueError(f"scales shape {scales.shape} does not match {(groups, n)}")
    return repack_logical_int4_to_row_major(qweight_u4), _interleave_params(qzeros, scales)


def repack_module_from_awq_tensors(
    qweight: np.ndarray,
    qzeros: np.ndarray,
    scales: np.ndarray,
    group_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unpack source AWQ tensors and repack to row-major AIR decode layout."""
    n = np.asarray(scales).shape[1]
    qweight_u4 = unpack_int4_reorder(qweight, n)
    zeros = unpack_qzeros_reorder(qzeros, n)
    return repack_module_from_logical(qweight_u4, zeros, scales, group_size)


def _unpack_row_major_qweight(repacked_qweight: np.ndarray, k: int) -> np.ndarray:
    repacked_qweight = np.asarray(repacked_qweight, dtype=np.uint8)
    if repacked_qweight.ndim != 2:
        raise ValueError(f"repacked_qweight must be 2D, got {repacked_qweight.shape}")
    if repacked_qweight.shape[1] * 2 != k:
        raise ValueError(f"packed K mismatch: {repacked_qweight.shape[1] * 2} != {k}")
    out = np.empty((repacked_qweight.shape[0], k), dtype=np.uint8)
    out[:, 0::2] = repacked_qweight & np.uint8(0xF)
    out[:, 1::2] = (repacked_qweight >> np.uint8(4)) & np.uint8(0xF)
    return out


def dequant_repacked_awq(
    repacked_qweight: np.ndarray,
    params_interleaved: np.ndarray,
    *,
    k: int,
    group_size: int,
) -> np.ndarray:
    """Reference dequant for repacked layout, returning ``(N, K)`` rows."""
    qweight_nk = _unpack_row_major_qweight(repacked_qweight, k)
    params = np.asarray(params_interleaved, dtype=np.float32)
    n = qweight_nk.shape[0]
    groups = (k + group_size - 1) // group_size
    if params.shape != (n, groups * 2):
        raise ValueError(f"params shape {params.shape} does not match {(n, groups * 2)}")
    scales = params[:, 0::2]
    zeros = params[:, 1::2]
    group_ids = np.arange(k) // group_size
    return (qweight_nk.astype(np.float32) - zeros[:, group_ids]) * scales[:, group_ids]


def dequant_repacked_awq_rows(
    repacked_qweight: np.ndarray,
    params_interleaved: np.ndarray,
    *,
    k: int,
    group_size: int,
    row_start: int,
    row_stop: int,
) -> np.ndarray:
    """Reference dequant for a row slice of repacked layout.

    This is useful for very large outputs such as ``lm_head`` where full
    ``(vocab, K)`` dequantization would consume unnecessary memory.
    """
    return dequant_repacked_awq(
        repacked_qweight[row_start:row_stop],
        params_interleaved[row_start:row_stop],
        k=k,
        group_size=group_size,
    )


def linear_output_repacked_awq(
    x: np.ndarray,
    repacked_qweight: np.ndarray,
    params_interleaved: np.ndarray,
    *,
    k: int,
    group_size: int,
    chunk_rows: int = 4096,
) -> np.ndarray:
    """Compute ``x @ W`` from repacked rows without materializing all of W."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] != k:
        raise ValueError(f"input last dim {x.shape[-1]} does not match K={k}")
    n_rows = np.asarray(repacked_qweight).shape[0]
    out = np.empty(x.shape[:-1] + (n_rows,), dtype=np.float32)
    x2 = x.reshape(-1, k)
    out2 = out.reshape(-1, n_rows)
    for row_start in range(0, n_rows, chunk_rows):
        row_stop = min(row_start + chunk_rows, n_rows)
        weight_rows = dequant_repacked_awq_rows(
            repacked_qweight,
            params_interleaved,
            k=k,
            group_size=group_size,
            row_start=row_start,
            row_stop=row_stop,
        )
        out2[:, row_start:row_stop] = x2 @ weight_rows.T
    return out


def discover_awq_linears(model_path: str) -> list[str]:
    """Return module prefixes that have qweight/qzeros/scales tensors."""
    from safetensors import safe_open

    keys = set()
    for name in sorted(os.listdir(model_path)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model_path, name), framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    modules = []
    for key in sorted(keys):
        if key.endswith(".qweight"):
            prefix = key[: -len(".qweight")]
            if f"{prefix}.qzeros" in keys and f"{prefix}.scales" in keys:
                modules.append(prefix)
    return modules


def _load_copy_tensors(model_path: str) -> Dict[str, np.ndarray]:
    from safetensors import safe_open

    copied: Dict[str, np.ndarray] = {}
    layer_norm_suffixes = ("input_layernorm.weight", "post_attention_layernorm.weight")
    for name in sorted(os.listdir(model_path)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model_path, name), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in COPY_TENSORS or key.endswith(layer_norm_suffixes):
                    tensor = handle.get_tensor(key).detach().cpu()
                    if str(tensor.dtype) == "torch.bfloat16":
                        tensor = tensor.float()
                    copied[key] = tensor.numpy().astype(bfloat16)
    return copied


def repack_model(src: str, dst: str) -> str:
    """Repack all AWQ linears from ``src`` into ``dst/model.safetensors``."""
    from safetensors.numpy import save_file

    config = validate_quant_config(src)
    group_size = int(config["quantization_config"]["group_size"])
    os.makedirs(dst, exist_ok=True)

    tensors: Dict[str, np.ndarray] = {}
    tensors.update(_load_copy_tensors(src))
    modules = discover_awq_linears(src)
    if not modules:
        raise RuntimeError(f"No AWQ linear tensors found in {src}")
    for module in modules:
        awq = load_awq_tensors(src, module)
        qweight, params = repack_module_from_awq_tensors(
            awq.qweight, awq.qzeros, awq.scales, group_size
        )
        tensors[f"{module}.qweight_repacked"] = qweight
        tensors[f"{module}.params_interleaved"] = params

    out_file = os.path.join(dst, "model.safetensors")
    save_file(tensors, out_file)
    for name in ("config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
        src_file = os.path.join(src, name)
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join(dst, name))
    with open(os.path.join(dst, "awq_repack_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": os.path.abspath(src),
                "group_size": group_size,
                "qzero_rule": "unpack(qzeros)",
                "modules": modules,
            },
            handle,
            indent=2,
        )
    return out_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Repack AWQ tensors for AIR decode")
    parser.add_argument("--src", required=True, help="Source AWQ model directory")
    parser.add_argument("--dst", required=True, help="Destination repacked model directory")
    args = parser.parse_args()
    out_file = repack_model(args.src, args.dst)
    print(f"wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
