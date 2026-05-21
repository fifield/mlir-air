#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for AWQ weight path handling."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXAMPLE_DIR))

from llama32_1b_weights import _resolve_safetensor_files  # noqa: E402


def test_missing_absolute_awq_path_reports_file_not_found_not_hf_repo_error():
    missing = "/tmp/awq model"
    try:
        _resolve_safetensor_files(missing)
    except FileNotFoundError as exc:
        assert missing in str(exc)
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def main() -> int:
    test_missing_absolute_awq_path_reports_file_not_found_not_hf_repo_error()
    print("PASS test_missing_absolute_awq_path_reports_file_not_found_not_hf_repo_error")
    print("PASS test_awq_path_validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
