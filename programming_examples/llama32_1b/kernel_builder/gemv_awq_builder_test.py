#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage 5 correctness-first tests for AWQ GEMV decoder plumbing.

These tests deliberately avoid AIR generation and NPU execution. They validate
that the packed AWQ row-major tensors can drive the full single-token decode
block through a CPU fallback path before the AIR builder/integration is tuned.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_EXAMPLE_DIR = _THIS_DIR.parent
_TOOLS_DIR = _EXAMPLE_DIR / "tools"
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_EXAMPLE_DIR))
sys.path.insert(0, str(_TOOLS_DIR))

from gemv_awq_builder import awq_gemv_cpu  # noqa: E402
from llama32_1b_decode import run_decode_block  # noqa: E402
from llama32_1b_reference import transformer_block  # noqa: E402
from llama32_1b_weights import LlamaConfig, load_awq_weights  # noqa: E402
from repack_awq import dequant_repacked_awq, repack_module_from_logical  # noqa: E402
from test_awq_weight_loader import _write_tiny_repacked_awq_model  # noqa: E402


def test_awq_gemv_cpu_matches_repacked_dequant_reference():
    k = 8
    m = 6
    group_size = 2
    logical = (np.arange(k * m, dtype=np.uint8).reshape(k, m) * 3) % 16
    zeros = (np.arange((k // group_size) * m, dtype=np.uint8).reshape(k // group_size, m) % 5) + 2
    scales = (0.03 + 0.01 * np.arange((k // group_size) * m, dtype=np.float32).reshape(k // group_size, m)).astype(np.float32)
    qweight, params = repack_module_from_logical(logical, zeros, scales, group_size)
    x = (np.arange(k, dtype=np.float32) / 7 - 0.4).astype(bfloat16)

    got = awq_gemv_cpu(qweight, params, x, k=k, group_size=group_size)
    rows_mk = dequant_repacked_awq(qweight, params, k=k, group_size=group_size)
    expected = rows_mk.astype(np.float32) @ x.astype(np.float32)

    assert got.dtype == bfloat16
    np.testing.assert_allclose(got.astype(np.float32), expected.astype(bfloat16).astype(np.float32), rtol=0, atol=0)


def test_awq_cpu_decode_block_matches_dequantized_reference_for_one_token():
    config = LlamaConfig(
        n_layers=1,
        emb_dim=4,
        n_heads=2,
        head_dim=2,
        n_kv_heads=1,
        hidden_dim=8,
        vocab_size=6,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_tiny_repacked_awq_model(tmpdir, config)
        weights = load_awq_weights(tmpdir, config=config)

    x = (np.arange(config.emb_dim, dtype=np.float32) / 10).astype(bfloat16)
    rope_lut = np.ones((1, config.head_dim), dtype=bfloat16)
    k_cache = np.zeros((config.n_kv_heads, 1, config.head_dim), dtype=bfloat16)
    v_cache = np.zeros((config.n_kv_heads, 1, config.head_dim), dtype=bfloat16)

    got = run_decode_block(
        x,
        weights.layers[0],
        cache=None,
        config=config,
        k_cache_layer=k_cache,
        v_cache_layer=v_cache,
        current_pos=0,
        rope_lut_bf16=rope_lut,
        awq_layer=weights.awq_layers[0],
    )
    expected, _ = transformer_block(x.reshape(1, config.emb_dim), weights.layers[0], rope_lut, config)

    assert got.shape == (config.emb_dim,)
    np.testing.assert_allclose(got.astype(np.float32), expected.reshape(-1).astype(np.float32), rtol=0, atol=0.35)
    assert np.isfinite(k_cache.astype(np.float32)).all()
    assert np.isfinite(v_cache.astype(np.float32)).all()


def main() -> int:
    test_awq_gemv_cpu_matches_repacked_dequant_reference()
    print("PASS test_awq_gemv_cpu_matches_repacked_dequant_reference")
    test_awq_cpu_decode_block_matches_dequantized_reference_for_one_token()
    print("PASS test_awq_cpu_decode_block_matches_dequantized_reference_for_one_token")
    print("PASS gemv_awq_builder_test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
