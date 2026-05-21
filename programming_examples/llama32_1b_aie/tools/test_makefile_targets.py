#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke tests for llama32_1b_aie Makefile NPU run/profile/verify forwarding."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parent.parent


def _make_dry(target: str, **vars_: str) -> str:
    cmd = ["make", "-n", target]
    cmd.extend(f"{key}={value}" for key, value in vars_.items())
    env = os.environ.copy()
    env.setdefault("PEANO_INSTALL_DIR", "/tmp/peano-for-make-smoke")
    result = subprocess.run(
        cmd,
        cwd=EXAMPLE_DIR,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout


def _make(target: str, **vars_: str) -> subprocess.CompletedProcess[str]:
    cmd = ["make", target]
    cmd.extend(f"{key}={value}" for key, value in vars_.items())
    env = os.environ.copy()
    env.setdefault("PEANO_INSTALL_DIR", "/tmp/peano-for-make-smoke")
    return subprocess.run(
        cmd,
        cwd=EXAMPLE_DIR,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def test_run_profile_verify_forward_awq_to_aie_npu_pipeline():
    common = {
        "QUANT": "awq",
        "AWQ_WEIGHTS": "/tmp/awq model",
        "N_TOKENS": "3",
        "PROMPT": "Hello quoted world",
        "MODEL": "base",
    }
    for target in ("run", "profile", "verify"):
        out = _make_dry(target, **common)
        assert "llama32_1b_aie" in out
        assert "llama32_1b_inference.py" in out
        assert "--run-only" in out
        assert "--quant awq" in out
        assert "--awq-weights '/tmp/awq model'" in out
        assert "--awq-cpu-only" not in out
        assert "--n-tokens 3" in out
        assert '--prompt "Hello quoted world"' in out
        if target == "profile":
            assert "--profile" in out
        if target == "verify":
            assert "--verify" in out
            assert "--profile" in out


def test_compile_forwards_awq_to_aie_npu_compilation():
    out = _make_dry("compile", QUANT="awq", AWQ_WEIGHTS="/tmp/awq model")
    assert "llama32_1b_aie" in out
    assert "--compile-only" in out
    assert "--quant awq" in out
    assert "--awq-weights '/tmp/awq model'" in out
    assert "--awq-cpu-only" not in out


def test_awq_shortcuts_select_aie_npu_awq_defaults():
    out = _make_dry("profile-awq", AWQ_WEIGHTS="/tmp/awq", PROMPT="abc")
    assert "llama32_1b_aie" in out
    assert "--quant awq" in out
    assert "--awq-weights /tmp/awq" in out
    assert "--awq-cpu-only" not in out
    assert "--profile" in out
    assert '--prompt "abc"' in out


def test_awq_shortcuts_reject_missing_local_awq_dir_before_python():
    result = _make("run-awq", AWQ_WEIGHTS="/tmp/awq model")
    assert result.returncode != 0
    assert "AWQ_WEIGHTS must point to an existing repacked AWQ directory" in result.stdout
    assert "llama32_1b_inference.py" not in result.stdout


def main() -> int:
    test_run_profile_verify_forward_awq_to_aie_npu_pipeline()
    print("PASS test_run_profile_verify_forward_awq_to_aie_npu_pipeline")
    test_compile_forwards_awq_to_aie_npu_compilation()
    print("PASS test_compile_forwards_awq_to_aie_npu_compilation")
    test_awq_shortcuts_select_aie_npu_awq_defaults()
    print("PASS test_awq_shortcuts_select_aie_npu_awq_defaults")
    test_awq_shortcuts_reject_missing_local_awq_dir_before_python()
    print("PASS test_awq_shortcuts_reject_missing_local_awq_dir_before_python")
    print("PASS test_makefile_targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
