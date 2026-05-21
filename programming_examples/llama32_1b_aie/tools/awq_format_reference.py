#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reference helpers for the Llama-3.2 AWQ GEMM/reorder tensor format.

This module is intentionally CPU/NumPy-oriented. It is used to lock down the
checkpoint format before the AIR/AIE fused int4 GEMV path repacks or consumes
weights.

Chosen qzero rule for the current checkpoint family:
    zero_point = unpack(qzeros)

This is the rule used by the independent GPTQModel/Transformers AWQ GEMM
reference for this checkpoint. The CLI can compare local linear outputs against
that oracle when the local Python environment has the required quantized loader
support installed.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

AWQ_REORDER_SHUFFLE: Tuple[int, ...] = (0, 4, 1, 5, 2, 6, 3, 7)
QZERO_PLUS_ONE = False


@dataclass(frozen=True)
class AwqModuleTensors:
    qweight: np.ndarray
    qzeros: np.ndarray
    scales: np.ndarray


@dataclass(frozen=True)
class AwqDequantizedLinear:
    weight_kn: np.ndarray
    qweight_u4: np.ndarray
    qzeros: np.ndarray
    scales: np.ndarray


def _require_2d(name: str, array: np.ndarray) -> None:
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {array.shape}")


def _as_uint32(array: np.ndarray) -> np.ndarray:
    """View/cast signed int32 AWQ tensors as unsigned before shifting."""
    if array.dtype == np.uint32:
        return array
    if array.dtype == np.int32:
        return array.view(np.uint32)
    return array.astype(np.uint32, copy=False)


def pack_int4_reorder(logical: np.ndarray, shuffle: Iterable[int] = AWQ_REORDER_SHUFFLE) -> np.ndarray:
    """Pack a tiny logical uint4 matrix using AWQ GEMM reorder semantics.

    This is primarily a test helper. ``logical`` has shape ``(..., N)`` and N
    must be divisible by 8. For each group of 8 logical columns, values are
    stored into one int32 in the reordered nibble slots.
    """
    logical = np.asarray(logical, dtype=np.uint8)
    if logical.shape[-1] % 8 != 0:
        raise ValueError(f"last dimension must be divisible by 8, got {logical.shape[-1]}")
    if np.any(logical > 0xF):
        raise ValueError("int4 values must be in [0, 15]")

    shuffle = tuple(shuffle)
    out_shape = logical.shape[:-1] + (logical.shape[-1] // 8,)
    packed = np.zeros(out_shape, dtype=np.uint32)
    for logical_col, packed_slot in enumerate(shuffle):
        vals = logical[..., logical_col::8].astype(np.uint32)
        packed |= vals << np.uint32(4 * packed_slot)
    return packed.view(np.int32)


def unpack_int4_reorder(
    packed: np.ndarray,
    logical_cols: Optional[int] = None,
    shuffle: Iterable[int] = AWQ_REORDER_SHUFFLE,
) -> np.ndarray:
    """Unpack AWQ GEMM/reorder qweight or raw qzero nibbles.

    ``packed`` is shaped ``(..., N/8)`` int32. The returned uint8 tensor is
    shaped ``(..., N)`` in logical column order.
    """
    _require_2d("packed", np.asarray(packed))
    packed_u = _as_uint32(np.asarray(packed))
    max_cols = packed_u.shape[-1] * 8
    if logical_cols is None:
        logical_cols = max_cols
    if logical_cols < 0 or logical_cols > max_cols:
        raise ValueError(f"logical_cols must be in [0, {max_cols}], got {logical_cols}")

    shuffle = tuple(shuffle)
    unpacked = np.empty(packed_u.shape[:-1] + (max_cols,), dtype=np.uint8)
    for logical_offset, packed_slot in enumerate(shuffle):
        vals = ((packed_u >> np.uint32(4 * packed_slot)) & np.uint32(0xF)).astype(np.uint8)
        unpacked[..., logical_offset::8] = vals
    return unpacked[..., :logical_cols]


def unpack_qzeros_reorder(
    packed_qzeros: np.ndarray,
    logical_cols: Optional[int] = None,
    *,
    plus_one: bool = QZERO_PLUS_ONE,
) -> np.ndarray:
    """Unpack asymmetric AWQ qzeros into logical zero points.

    The current checkpoint matches the GPTQModel/Transformers AWQ GEMM oracle
    with raw unpacked qzeros (no +1 offset). Keep ``plus_one`` explicit so a
    future checkpoint family can test that variant without changing callers.
    """
    zeros = unpack_int4_reorder(packed_qzeros, logical_cols)
    if plus_one:
        zeros = zeros + np.uint8(1)
    return zeros.astype(np.uint8, copy=False)


def dequant_awq_weight(
    qweight_u4: np.ndarray,
    qzeros: np.ndarray,
    scales: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """Dequantize logical AWQ tensors to a float32 ``(K, N)`` matrix."""
    qweight_u4 = np.asarray(qweight_u4, dtype=np.uint8)
    qzeros = np.asarray(qzeros, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.float32)
    _require_2d("qweight_u4", qweight_u4)
    _require_2d("qzeros", qzeros)
    _require_2d("scales", scales)

    k, n = qweight_u4.shape
    expected_groups = (k + group_size - 1) // group_size
    if qzeros.shape != (expected_groups, n):
        raise ValueError(f"qzeros shape {qzeros.shape} does not match {(expected_groups, n)}")
    if scales.shape != (expected_groups, n):
        raise ValueError(f"scales shape {scales.shape} does not match {(expected_groups, n)}")

    group_ids = np.arange(k) // group_size
    return (qweight_u4.astype(np.float32) - qzeros[group_ids].astype(np.float32)) * scales[group_ids]


def dequantize_module_tensors(tensors: AwqModuleTensors, group_size: int) -> AwqDequantizedLinear:
    """Unpack and dequantize a source AWQ linear module."""
    qweight = np.asarray(tensors.qweight)
    qzeros = np.asarray(tensors.qzeros)
    scales = np.asarray(tensors.scales, dtype=np.float32)
    _require_2d("qweight", qweight)
    _require_2d("qzeros", qzeros)
    _require_2d("scales", scales)

    k = qweight.shape[0]
    n = scales.shape[1]
    qweight_u4 = unpack_int4_reorder(qweight, n)
    zeros = unpack_qzeros_reorder(qzeros, n)
    weight = dequant_awq_weight(qweight_u4, zeros, scales, group_size)
    return AwqDequantizedLinear(weight_kn=weight, qweight_u4=qweight_u4, qzeros=zeros, scales=scales)


def _resolve_safetensor_files(model_path: str) -> list[str]:
    files = sorted(
        os.path.join(model_path, name)
        for name in os.listdir(model_path)
        if name.endswith(".safetensors")
    )
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")
    return files


def load_awq_tensors(model_path: str, module: str) -> AwqModuleTensors:
    """Load qweight/qzeros/scales for ``module`` from a local safetensors model."""
    from safetensors import safe_open

    wanted = {
        "qweight": f"{module}.qweight",
        "qzeros": f"{module}.qzeros",
        "scales": f"{module}.scales",
    }
    found: Dict[str, np.ndarray] = {}
    for path in _resolve_safetensor_files(model_path):
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            for field, key in wanted.items():
                if field not in found and key in keys:
                    tensor = handle.get_tensor(key).detach().cpu()
                    # NumPy cannot directly ingest torch.bfloat16. Scales are
                    # reference-only here, so promote BF16 scales to float32.
                    if str(tensor.dtype) == "torch.bfloat16":
                        tensor = tensor.float()
                    found[field] = tensor.numpy()
    missing = [key for field, key in wanted.items() if field not in found]
    if missing:
        raise KeyError(f"Missing AWQ tensors for {module}: {missing}")
    return AwqModuleTensors(found["qweight"], found["qzeros"], found["scales"])


def validate_quant_config(model_path: str) -> dict:
    """Load and assert the AWQ quantization config required by this path."""
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    qc = config.get("quantization_config") or {}
    expected = {
        "bits": 4,
        "group_size": 128,
        "quant_method": "awq",
        "version": "gemm",
        "pack_method": "reorder",
        "zero_point": True,
    }
    for key, value in expected.items():
        if qc.get(key) != value:
            raise AssertionError(f"quantization_config.{key}: expected {value!r}, got {qc.get(key)!r}")
    return config


def local_linear_output(model_path: str, module: str, x: np.ndarray, group_size: int = 128) -> np.ndarray:
    """Compute ``x @ W`` with the local NumPy AWQ dequant reference."""
    deq = dequantize_module_tensors(load_awq_tensors(model_path, module), group_size)
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] != deq.weight_kn.shape[0]:
        raise ValueError(f"input last dim {x.shape[-1]} does not match K={deq.weight_kn.shape[0]}")
    return x @ deq.weight_kn


def _try_transformers_oracle(model_path: str, module: str, x: np.ndarray) -> Optional[np.ndarray]:
    """Best-effort independent oracle using installed Transformers quantized loader."""
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"ORACLE unavailable: cannot import transformers/torch: {exc}")
        return None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        model.to("cpu")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"ORACLE unavailable: Transformers could not load AWQ model: {exc}")
        return None

    obj = model
    try:
        for part in module.split("."):
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"ORACLE unavailable: could not resolve module {module}: {exc}")
        return None

    with torch.no_grad():
        inp = torch.tensor(x, dtype=torch.bfloat16)
        out = obj(inp).detach().float().cpu().numpy()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Reference AWQ GEMM/reorder unpack checker")
    parser.add_argument("--model", required=True, help="Local AWQ model directory")
    parser.add_argument("--module", required=True, help="Module prefix, e.g. model.layers.0.self_attn.q_proj")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oracle", action="store_true", help="Try an independent Transformers/AutoAWQ oracle")
    args = parser.parse_args()

    config = validate_quant_config(args.model)
    group_size = int(config["quantization_config"]["group_size"])
    tensors = load_awq_tensors(args.model, args.module)
    deq = dequantize_module_tensors(tensors, group_size)
    print(f"module={args.module}")
    print(f"qweight={tensors.qweight.shape} qzeros={tensors.qzeros.shape} scales={tensors.scales.shape}")
    print(f"logical={deq.qweight_u4.shape} zeros={deq.qzeros.shape} weight={deq.weight_kn.shape}")
    print(f"qzero_rule=unpack(qzeros){' + 1' if QZERO_PLUS_ONE else ''}")

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((1, deq.weight_kn.shape[0])).astype(np.float32) * 0.02
    local = x @ deq.weight_kn
    print(f"local_output: shape={local.shape} mean={float(local.mean()):.6g} std={float(local.std()):.6g}")

    if args.oracle:
        oracle = _try_transformers_oracle(args.model, args.module, x)
        if oracle is not None:
            np.testing.assert_allclose(local, oracle, rtol=3e-2, atol=3e-2)
            print("oracle=PASS")
        else:
            print("oracle=SKIP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
