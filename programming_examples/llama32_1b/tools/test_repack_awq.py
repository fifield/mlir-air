#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for offline AWQ repacking.

Without ``--model`` this runs small hand-authored tests. With ``--model`` it
also runs non-circular real-checkpoint checks:
  * source AWQ dequant reference vs repacked dequant reference
  * source linear output vs repacked linear output
  * saved safetensors key/shape checks after a full repack
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

from awq_format_reference import (
    dequant_awq_weight,
    dequantize_module_tensors,
    load_awq_tensors,
    pack_int4_reorder,
    validate_quant_config,
)
from repack_awq import (
    dequant_repacked_awq,
    dequant_repacked_awq_rows,
    linear_output_repacked_awq,
    repack_logical_int4_to_row_major,
    repack_model,
    repack_module_from_awq_tensors,
    repack_module_from_logical,
)

REAL_MODULES = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.mlp.down_proj",
    "lm_head",
)


def test_repack_logical_int4_to_row_major_packs_output_rows_along_k():
    # Logical source qweight is (K, N). Repacked output is (N, K/2), with low
    # nibble holding even K and high nibble holding odd K for each output row.
    logical_kn = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
        ],
        dtype=np.uint8,
    )

    packed = repack_logical_int4_to_row_major(logical_kn)

    expected = np.array(
        [
            [0 | (3 << 4), 6 | (9 << 4)],
            [1 | (4 << 4), 7 | (10 << 4)],
            [2 | (5 << 4), 8 | (11 << 4)],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(packed, expected)


def test_repack_module_from_logical_interleaves_params_per_output_row():
    qweight_u4 = np.array(
        [
            [1, 3],
            [5, 7],
            [9, 11],
            [13, 15],
        ],
        dtype=np.uint8,
    )
    qzeros = np.array([[1, 2], [4, 8]], dtype=np.uint8)
    scales = np.array([[0.5, 0.25], [0.125, 0.0625]], dtype=np.float32)

    repacked_qweight, params = repack_module_from_logical(
        qweight_u4, qzeros, scales, group_size=2
    )

    np.testing.assert_array_equal(
        repacked_qweight,
        np.array(
            [[1 | (5 << 4), 9 | (13 << 4)], [3 | (7 << 4), 11 | (15 << 4)]],
            dtype=np.uint8,
        ),
    )
    np.testing.assert_allclose(
        params.astype(np.float32),
        np.array(
            [[0.5, 1.0, 0.125, 4.0], [0.25, 2.0, 0.0625, 8.0]],
            dtype=np.float32,
        ),
    )

    dequant_from_repacked = dequant_repacked_awq(repacked_qweight, params, k=4, group_size=2)
    dequant_from_source = dequant_awq_weight(qweight_u4, qzeros, scales, group_size=2).T
    np.testing.assert_allclose(dequant_from_repacked, dequant_from_source)


def test_repack_from_packed_awq_round_trips_reference_dequant():
    qweight_u4 = np.array(
        [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        dtype=np.uint8,
    )
    qzeros = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.uint8)
    scales = np.array(
        [[0.5, 0.25, 0.125, 0.0625, 0.5, 0.25, 0.125, 0.0625]],
        dtype=np.float32,
    )
    source_qweight = pack_int4_reorder(qweight_u4)
    source_qzeros = pack_int4_reorder(qzeros)

    repacked_qweight, params = repack_module_from_awq_tensors(
        source_qweight, source_qzeros, scales, group_size=2
    )

    np.testing.assert_allclose(
        dequant_repacked_awq(repacked_qweight, params, k=2, group_size=2),
        dequant_awq_weight(qweight_u4, qzeros, scales, group_size=2).T,
    )


def test_linear_output_repacked_awq_matches_full_repacked_dequant():
    qweight_u4 = np.arange(64, dtype=np.uint8).reshape(8, 8) & np.uint8(0xF)
    qzeros = np.ones((2, 8), dtype=np.uint8)
    scales = np.full((2, 8), 0.25, dtype=np.float32)
    repacked_qweight, params = repack_module_from_logical(
        qweight_u4, qzeros, scales, group_size=4
    )
    x = np.arange(8, dtype=np.float32).reshape(1, 8) * 0.01

    chunked = linear_output_repacked_awq(
        x, repacked_qweight, params, k=8, group_size=4, chunk_rows=3
    )
    full = x @ dequant_repacked_awq(repacked_qweight, params, k=8, group_size=4).T

    np.testing.assert_allclose(chunked, full)


def _run_small_tests() -> None:
    test_repack_logical_int4_to_row_major_packs_output_rows_along_k()
    test_repack_module_from_logical_interleaves_params_per_output_row()
    test_repack_from_packed_awq_round_trips_reference_dequant()
    test_linear_output_repacked_awq_matches_full_repacked_dequant()


def _assert_real_module_round_trip(model: str, module: str, group_size: int) -> None:
    source = load_awq_tensors(model, module)
    deq_source = dequantize_module_tensors(source, group_size)
    repacked_qweight, params = repack_module_from_awq_tensors(
        source.qweight, source.qzeros, source.scales, group_size
    )
    k, n = deq_source.weight_kn.shape

    # q_proj/down_proj are modest enough to compare full matrices. lm_head is
    # huge, so compare deterministic row slices to avoid materializing N*K.
    if module == "lm_head":
        row_indices = [0, 1, 127, 1024, n // 2, n - 2, n - 1]
        for row in row_indices:
            repacked_row = dequant_repacked_awq_rows(
                repacked_qweight, params, k=k, group_size=group_size, row_start=row, row_stop=row + 1
            )[0]
            np.testing.assert_allclose(repacked_row, deq_source.weight_kn[:, row], rtol=0, atol=0)
    else:
        repacked_deq = dequant_repacked_awq(repacked_qweight, params, k=k, group_size=group_size)
        np.testing.assert_allclose(repacked_deq, deq_source.weight_kn.T, rtol=0, atol=0)

    rng = np.random.default_rng(17)
    x = (rng.standard_normal((1, k)).astype(np.float32) * 0.02).astype(np.float32)
    repacked_out = linear_output_repacked_awq(
        x, repacked_qweight, params, k=k, group_size=group_size, chunk_rows=4096
    )
    source_out = x @ deq_source.weight_kn
    np.testing.assert_allclose(repacked_out, source_out, rtol=1e-5, atol=1e-5)
    print(f"PASS real_module_round_trip {module}: K={k} N={n}")


def _assert_full_repack_file(model: str) -> None:
    from safetensors import safe_open

    with tempfile.TemporaryDirectory(prefix="llama_awq_repack_test_") as tmpdir:
        out_file = repack_model(model, tmpdir)
        expected_shapes = {
            "model.layers.0.self_attn.q_proj.qweight_repacked": (2048, 1024),
            "model.layers.0.self_attn.q_proj.params_interleaved": (2048, 32),
            "model.layers.0.mlp.down_proj.qweight_repacked": (2048, 4096),
            "lm_head.qweight_repacked": (128256, 1024),
            "lm_head.params_interleaved": (128256, 32),
            "model.embed_tokens.weight": (128256, 2048),
            "model.layers.0.input_layernorm.weight": (2048,),
        }
        with safe_open(out_file, framework="pt", device="cpu") as handle:
            for key, shape in expected_shapes.items():
                tensor = handle.get_tensor(key)
                if tuple(tensor.shape) != shape:
                    raise AssertionError(f"{key}: expected {shape}, got {tuple(tensor.shape)}")
        manifest = os.path.join(tmpdir, "awq_repack_manifest.json")
        if not os.path.exists(manifest):
            raise AssertionError("missing awq_repack_manifest.json")
        print(f"PASS full_repack_file {out_file}")


def _run_real_model_tests(model: str) -> None:
    config = validate_quant_config(model)
    group_size = int(config["quantization_config"]["group_size"])
    for module in REAL_MODULES:
        _assert_real_module_round_trip(model, module, group_size)
    _assert_full_repack_file(model)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test AWQ repack helpers")
    parser.add_argument("--model", help="Optional local AWQ model directory for real-checkpoint tests")
    args = parser.parse_args()

    _run_small_tests()
    if args.model:
        _run_real_model_tests(args.model)
    print("PASS test_repack_awq")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
