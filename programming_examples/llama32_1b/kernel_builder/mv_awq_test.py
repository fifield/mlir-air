#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage 4 smoke tests for the fused AWQ GEMV external kernel.

These tests intentionally avoid requiring an NPU runtime. They validate the
standalone reference data layout used by the C++ kernel, the public C ABI, and
Peano compile hooks. When PEANO_INSTALL_DIR/aie-opt are available, they also
compile mv_awq.o and mv_awq_k8192.o.
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
sys.path.insert(0, str(_TOOLS_DIR))

from repack_awq import dequant_repacked_awq, repack_module_from_logical  # noqa: E402


def _make_case(k: int = 128, m: int = 8, group_size: int = 32):
    rng = np.random.default_rng(4)
    logical = rng.integers(0, 16, size=(k, m), dtype=np.uint8)
    zeros = rng.integers(0, 16, size=(k // group_size, m), dtype=np.uint8)
    scales = rng.uniform(0.005, 0.08, size=(k // group_size, m)).astype(np.float32)
    qweight, params = repack_module_from_logical(logical, zeros, scales, group_size)
    vector = rng.normal(size=(k,)).astype(np.float32).astype(bfloat16)
    return qweight, params, vector


def test_python_reference_uses_stage4_layout():
    qweight, params, vector = _make_case()
    rows_mk = dequant_repacked_awq(qweight, params, k=vector.shape[0], group_size=32)
    ref = rows_mk.astype(np.float32) @ vector.astype(np.float32)
    assert qweight.dtype == np.uint8
    assert qweight.shape == (8, 64)
    assert params.shape == (8, 8)
    assert ref.shape == (8,)
    assert np.isfinite(ref).all()


def test_mv_awq_source_exports_expected_c_abi():
    src_path = _THIS_DIR / "mv_awq.cc"
    text = src_path.read_text(encoding="utf-8")
    assert "void matvec_awq_bf16(uint32_t m, uint32_t k, uint32_t row_offset" in text
    assert "const uint8_t *__restrict qweights" in text
    assert "const bfloat16 *__restrict params" in text
    assert "const bfloat16 *__restrict b_in" in text
    assert "bfloat16 *__restrict c_out" in text
    assert "void linalg_fill_bf16(bfloat16 *c_out)" in text
    assert "c_out += row_offset" in text


def test_compile_hooks_exist_and_accept_force():
    import external_kernels

    assert hasattr(external_kernels, "compile_mv_awq")
    assert hasattr(external_kernels, "compile_mv_awq_k8192")

    with tempfile.TemporaryDirectory() as td:
        cwd = os.getcwd()
        os.chdir(td)
        try:
            external_kernels.compile_mv_awq(tile_m=8, force=True)
            external_kernels.compile_mv_awq_k8192(force=True)
            assert Path("mv_awq.o").exists()
            assert Path("mv_awq_k8192.o").exists()
        finally:
            os.chdir(cwd)


def main() -> int:
    test_python_reference_uses_stage4_layout()
    print("PASS test_python_reference_uses_stage4_layout")
    test_mv_awq_source_exports_expected_c_abi()
    print("PASS test_mv_awq_source_exports_expected_c_abi")
    test_compile_hooks_exist_and_accept_force()
    print("PASS test_compile_hooks_exist_and_accept_force")
    print("PASS mv_awq_test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
