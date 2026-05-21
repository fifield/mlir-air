# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Correctness-first AWQ GEMV helpers for LLAMA decode.

Stage 5 is intentionally split from AIR codegen while the decode path is being
validated end-to-end. The public helpers here consume the same packed tensor
layout that the future AIR submodule will consume:

  qweight: row-major ``(M, K/2)`` uint8, low nibble first along K
  params:  row-major ``(M, 2*K/group_size)`` bf16, ``[scale, zero]`` pairs
  input:   ``(K,)`` bf16
  output:  ``(M,)`` bf16

The CPU implementation is slow but exact enough to prove decoder wiring and
packed-weight correctness before investing in AIR generation/performance work.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ml_dtypes import bfloat16


def _as_awq_parts(awq_or_qweight: Any, params=None, *, k=None, group_size=None):
    """Normalize either an AwqLinear-like object or explicit arrays."""
    if hasattr(awq_or_qweight, "qweight") and hasattr(awq_or_qweight, "params"):
        awq = awq_or_qweight
        qweight = awq.qweight
        params = awq.params
        k = awq.k if k is None else k
        group_size = awq.group_size if group_size is None else group_size
    else:
        qweight = awq_or_qweight
        if params is None:
            raise TypeError("params is required when qweight is passed directly")
        if k is None:
            raise TypeError("k is required when qweight is passed directly")
        if group_size is None:
            raise TypeError("group_size is required when qweight is passed directly")

    qweight = np.asarray(qweight, dtype=np.uint8)
    params = np.asarray(params, dtype=np.float32)
    k = int(k)
    group_size = int(group_size)
    if k % 2 != 0:
        raise ValueError(f"AWQ K must be even for uint4 packing, got {k}")
    if qweight.ndim != 2 or qweight.shape[1] != k // 2:
        raise ValueError(f"qweight shape {qweight.shape} does not match (M, {k // 2})")
    groups = (k + group_size - 1) // group_size
    if params.ndim != 2 or params.shape != (qweight.shape[0], groups * 2):
        raise ValueError(
            f"params shape {params.shape} does not match {(qweight.shape[0], groups * 2)}"
        )
    return qweight, params, k, group_size


def dequant_awq_rows_cpu(awq_or_qweight: Any, params=None, *, k=None, group_size=None) -> np.ndarray:
    """Dequantize repacked AWQ rows to float32 ``(M, K)``.

    This mirrors the Stage 4 external-kernel semantics: each uint4 value is
    converted as ``(q - zero) * scale`` using one scale/zero pair per K group.
    """
    qweight, params, k, group_size = _as_awq_parts(
        awq_or_qweight, params, k=k, group_size=group_size
    )
    unpacked = np.empty((qweight.shape[0], k), dtype=np.uint8)
    unpacked[:, 0::2] = qweight & np.uint8(0xF)
    unpacked[:, 1::2] = (qweight >> np.uint8(4)) & np.uint8(0xF)
    group_ids = np.arange(k) // group_size
    scales = params[:, 0::2]
    zeros = params[:, 1::2]
    return (unpacked.astype(np.float32) - zeros[:, group_ids]) * scales[:, group_ids]


def awq_gemv_cpu(
    awq_or_qweight: Any,
    params_or_input,
    input_vec=None,
    *,
    k=None,
    group_size=None,
    out_dtype=bfloat16,
) -> np.ndarray:
    """Run packed AWQ GEMV on CPU and return ``(M,)``.

    Supports both call forms:

      awq_gemv_cpu(awq_linear, input_vec)
      awq_gemv_cpu(qweight, params, input_vec, k=..., group_size=...)
    """
    if hasattr(awq_or_qweight, "qweight") and hasattr(awq_or_qweight, "params"):
        awq = awq_or_qweight
        x = np.asarray(params_or_input, dtype=np.float32).reshape(-1)
        rows = dequant_awq_rows_cpu(awq, k=k, group_size=group_size)
    else:
        params = params_or_input
        if input_vec is None:
            raise TypeError("input_vec is required when qweight is passed directly")
        x = np.asarray(input_vec, dtype=np.float32).reshape(-1)
        rows = dequant_awq_rows_cpu(awq_or_qweight, params, k=k, group_size=group_size)

    if rows.shape[1] != x.shape[0]:
        raise ValueError(f"input length {x.shape[0]} does not match AWQ K {rows.shape[1]}")
    return (rows @ x).astype(out_dtype)


def build_matvec_awq_module(*args, **kwargs):  # noqa: D401, ANN002, ANN003
    """Placeholder for the future AIR AWQ GEMV builder.

    The decoder currently uses ``awq_gemv_cpu`` for correctness-first plumbing.
    Keeping this symbol makes Stage 5 imports explicit without pretending AIR
    generation exists yet.
    """
    raise NotImplementedError(
        "AWQ AIR GEMV generation is intentionally deferred; use awq_gemv_cpu "
        "for correctness-first decoder plumbing."
    )
