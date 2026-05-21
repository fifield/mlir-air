#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for offline AWQ repacking."""

import numpy as np

from awq_format_reference import dequant_awq_weight, pack_int4_reorder
from repack_awq import (
    dequant_repacked_awq,
    repack_logical_int4_to_row_major,
    repack_module_from_logical,
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
        np.array([[1 | (5 << 4), 9 | (13 << 4)], [3 | (7 << 4), 11 | (15 << 4)]], dtype=np.uint8),
    )
    np.testing.assert_allclose(
        params.astype(np.float32),
        np.array([[0.5, 1.0, 0.125, 4.0], [0.25, 2.0, 0.0625, 8.0]], dtype=np.float32),
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
    scales = np.array([[0.5, 0.25, 0.125, 0.0625, 0.5, 0.25, 0.125, 0.0625]], dtype=np.float32)
    source_qweight = pack_int4_reorder(qweight_u4)
    source_qzeros = pack_int4_reorder(qzeros)

    from repack_awq import repack_module_from_awq_tensors

    repacked_qweight, params = repack_module_from_awq_tensors(
        source_qweight, source_qzeros, scales, group_size=2
    )

    np.testing.assert_allclose(
        dequant_repacked_awq(repacked_qweight, params, k=2, group_size=2),
        dequant_awq_weight(qweight_u4, qzeros, scales, group_size=2).T,
    )


if __name__ == "__main__":
    test_repack_logical_int4_to_row_major_packs_output_rows_along_k()
    test_repack_module_from_logical_interleaves_params_per_output_row()
    test_repack_from_packed_awq_round_trips_reference_dequant()
    print("PASS test_repack_awq")
