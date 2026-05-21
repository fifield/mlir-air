#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for AWQ tensor unpack helpers.

These tests intentionally use tiny hand-authored tensors so the expected values
are independent of the implementation used for real checkpoints.
"""

import numpy as np

from awq_format_reference import (
    dequant_awq_weight,
    pack_int4_reorder,
    unpack_int4_reorder,
    unpack_qzeros_reorder,
)


def test_unpack_int4_reorder_recovers_logical_columns():
    logical = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
        ],
        dtype=np.uint8,
    )

    packed = pack_int4_reorder(logical)

    np.testing.assert_array_equal(unpack_int4_reorder(packed, logical.shape[1]), logical)


def test_unpack_qzeros_reorder_uses_raw_zero_point_rule():
    logical_zero = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.uint8)
    # The target checkpoint matches the independent GPTQModel/Transformers AWQ
    # GEMM oracle with raw unpacked qzeros (no +1 offset).
    packed = pack_int4_reorder(logical_zero)

    unpacked = unpack_qzeros_reorder(packed, logical_zero.shape[1])

    np.testing.assert_array_equal(unpacked, logical_zero)


def test_dequant_awq_weight_uses_grouped_scales_and_zero_points():
    qweight = np.array([[1, 3], [5, 7], [9, 11], [13, 15]], dtype=np.uint8)
    qzeros = np.array([[1, 2], [4, 8]], dtype=np.uint8)
    scales = np.array([[0.5, 0.25], [0.125, 0.0625]], dtype=np.float32)

    dequant = dequant_awq_weight(qweight, qzeros, scales, group_size=2)

    expected = np.array(
        [
            [(1 - 1) * 0.5, (3 - 2) * 0.25],
            [(5 - 1) * 0.5, (7 - 2) * 0.25],
            [(9 - 4) * 0.125, (11 - 8) * 0.0625],
            [(13 - 4) * 0.125, (15 - 8) * 0.0625],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(dequant, expected)


if __name__ == "__main__":
    test_unpack_int4_reorder_recovers_logical_columns()
    test_unpack_qzeros_reorder_uses_raw_zero_point_rule()
    test_dequant_awq_weight_uses_grouped_scales_and_zero_points()
    print("PASS test_awq_format_reference")
