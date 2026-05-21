# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate post-stitched mlir-aie text (`npu.air.mlir`) for each unique kernel.

This is the seam where placed-iron python would slot in. Today the IR is
harvested by running aircc with `--output-format=none` on the AIR module
produced by the existing multi-launch builders -- aircc runs the AIR
passes (placement, air-to-aie, airrt-to-npu) but skips the aiecc backend.
The resulting `npu.air.mlir` is exactly the aie/aiex-dialect text that
aiecc accepts. Replacing the aircc shell-out below with python that emits
the same dialect by hand is a drop-in change.

Cached per kernel-name under <cache_dir>/<name>.npu.air.mlir; recompute
only when missing.
"""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional


def _run(cmd, cwd=None, verbose=False):
    if verbose:
        print(f"  [aircc] {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _resolve_peano_dir() -> str:
    p = os.environ.get("PEANO_INSTALL_DIR", "")
    if p:
        return p
    here = Path(__file__).resolve()
    install = here.parents[5] / "install" / "peano"
    if install.exists():
        return str(install)
    raise RuntimeError("PEANO_INSTALL_DIR is not set and install/peano not found")


_LINK_OBJS = [
    "silu_and_mul.o", "rope.o", "attn.o", "attn_npu2.o",
    "mv.o", "mv_k8192.o", "attn_decode_npu2.o",
]


def lower_air_to_npu_air_mlir(
    air_module_text: str,
    *,
    device: str = "npu2",
    num_cols: int = 8,
    omit_while_true_loop: bool = False,
    omit_pingpong: Optional[str] = None,  # "", "L1", "L2", "all", or None
    runtime_loop_tiling_sizes=(),
    use_lock_race_condition_fix: bool = False,
    workdir: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """Run AIR passes and return the `npu.air.mlir` text.

    Mirrors the knobs in `air.backend.xrt.XRTBackend.compile()` that the
    existing kernels rely on. We invoke aircc with `--output-format=elf`
    (matching XRTBackend) so that `airrt-to-npu` runs with `output-elf=true`
    -- the pass option that emits the unnamed top-level `aie.device` +
    `aie.runtime_sequence @<instance>` dispatcher block consumed by
    `xrt.ext.kernel("main:<instance>")`. Without it, single-launch kernels
    (e.g. flash_attn) produce an ELF that XRT can't dispatch.

    aircc shells out to aiecc internally; we discard the resulting `.elf`
    here (the steady-state aiecc-only path in `aie_compile.py` rebuilds it
    from the cached IR). External `.o` files are staged so aiecc succeeds.
    """
    aircc_exe = shutil.which("aircc")
    if not aircc_exe:
        raise RuntimeError("aircc not found on PATH")

    work = Path(workdir or tempfile.mkdtemp(prefix="air_lower_"))
    work.mkdir(parents=True, exist_ok=True)

    # Stage external .o files from cwd into the workdir so the aiecc
    # invocation inside aircc can resolve `link_with = "..."` references.
    cwd = Path.cwd()
    for obj_name in _LINK_OBJS:
        src = cwd / obj_name
        if src.exists():
            shutil.copy2(src, work / obj_name)

    air_path = work / "air.mlir"
    air_path.write_text(air_module_text)

    cmd = [
        aircc_exe,
        "--device",
        device,
        "--output-format",
        "elf",
        "--elf-name",
        "aie.elf",
        f"--tmpdir={work}",
        f"--peano={_resolve_peano_dir()}",
        "--no-xchesscc",
        "--no-xbridge",
    ]
    if num_cols:
        cmd += [f"--num-cols={num_cols}"]
    if omit_while_true_loop:
        cmd += ["--omit-while-true-loop"]
    if omit_pingpong is not None:
        pp = "all" if omit_pingpong is True else str(omit_pingpong)
        cmd += [f"--omit-ping-pong-transform={pp}"]
    for s in runtime_loop_tiling_sizes:
        cmd += [f"--air-runtime-loop-tiling-sizes={s}"]
    if use_lock_race_condition_fix:
        cmd += ["--use-lock-race-condition-fix"]
    if verbose:
        cmd += ["-v"]

    cmd.append(str(air_path))

    if verbose:
        print(f"  [aircc lowering] {' '.join(cmd)}")

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(work), capture_output=True, text=True)
    dt = time.time() - t0

    # npu.air.mlir is written to tmpdir *before* aircc shells out to aiecc.
    # If the IR landed on disk we accept it even when aircc returns nonzero
    # (typically: a missing .o in the workdir trips the backend link step
    # but the IR we want is already valid).
    npu_path = work / "npu.air.mlir"
    if not npu_path.exists():
        msg = proc.stderr or proc.stdout
        raise RuntimeError(
            f"aircc lowering produced no npu.air.mlir in {dt:.1f}s "
            f"(returncode={proc.returncode}):\n{msg}"
        )
    if verbose and proc.returncode != 0:
        print(f"  [aircc] backend step failed but IR was recovered; "
              f"returncode={proc.returncode}")
    return npu_path.read_text()


# ---------------------------------------------------------------------------
# Per-kernel IR builders. Each returns the npu.air.mlir text for a single
# logical kernel ELF (i.e. the post-stitched form that aiecc consumes).
#
# These currently call into the source project's multi_launch_builder / iron
# AIR builders to produce an MLIR-AIR module and then lower it via
# `lower_air_to_npu_air_mlir`. Hand-written placed-iron python that emits
# the same dialect text is a drop-in replacement for any of these.
# ---------------------------------------------------------------------------


def build_rms_gemms_rope_ir(seq_len, emb_dim, kv_dim, n_heads, n_kv_heads, head_dim,
                            *, verbose=False, omit_while_true_loop=False):
    from multi_launch_builder.rms_gemms_rope_multi import build_rms_gemms_rope_module
    mod = build_rms_gemms_rope_module(
        seq_len, emb_dim, kv_dim, n_heads, n_kv_heads, head_dim
    )
    return lower_air_to_npu_air_mlir(
        str(mod),
        device="npu2",
        num_cols=8,
        omit_while_true_loop=omit_while_true_loop,
        verbose=verbose,
    )


def build_o_ffn_ir(seq_len, emb_dim, hidden_dim, *, verbose=False,
                   omit_while_true_loop=False):
    from multi_launch_builder.o_ffn_multi import build_o_ffn_module
    mod = build_o_ffn_module(seq_len, emb_dim, hidden_dim)
    return lower_air_to_npu_air_mlir(
        str(mod),
        device="npu2",
        num_cols=8,
        omit_while_true_loop=omit_while_true_loop,
        verbose=verbose,
    )


def build_flash_attn_ir(seq_len, n_heads, n_kv_heads, head_dim, *,
                        verbose=False):
    from flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
        build_module as build_attn,
    )
    lkp = head_dim
    lqp = 256
    enable_shared_buffers = lkp == head_dim
    mod = build_attn(
        lk=seq_len, lkp=lkp, lq=seq_len, lqp=lqp,
        dk=head_dim, dv=head_dim,
        num_q_tiles=4, num_cascade_stages=4,
        num_heads=n_heads, num_kv_heads=n_kv_heads,
        causal=True,
    )
    return lower_air_to_npu_air_mlir(
        str(mod),
        device="npu2",
        num_cols=8,
        omit_while_true_loop=not enable_shared_buffers,
        omit_pingpong="all",
        runtime_loop_tiling_sizes=[1, 1],
        verbose=verbose,
    )


def build_rms_gemv_rope_ir(emb_dim, kv_dim, n_heads, n_kv_heads, head_dim,
                           *, verbose=False):
    from multi_launch_builder.rms_gemv_rope_multi import build_rms_gemv_rope_module
    mod = build_rms_gemv_rope_module(emb_dim, kv_dim, n_heads, n_kv_heads, head_dim)
    return lower_air_to_npu_air_mlir(
        str(mod),
        device="npu2",
        num_cols=8,
        omit_pingpong="",
        runtime_loop_tiling_sizes=[16, 16],
        use_lock_race_condition_fix=False,
        verbose=verbose,
    )


def build_o_gemv_ffn_ir(emb_dim, hidden_dim, *, verbose=False):
    from multi_launch_builder.o_gemv_ffn_multi import build_o_gemv_ffn_module
    mod = build_o_gemv_ffn_module(emb_dim, hidden_dim)
    return lower_air_to_npu_air_mlir(
        str(mod),
        device="npu2",
        num_cols=8,
        omit_pingpong="all",
        runtime_loop_tiling_sizes=[16, 16],
        use_lock_race_condition_fix=False,
        verbose=verbose,
    )


def build_lm_head_gemv_ir(emb_dim, *, verbose=False):
    from multi_launch_builder.lm_head_gemv_multi import build_lm_head_gemv_module
    mod = build_lm_head_gemv_module(emb_dim)
    return lower_air_to_npu_air_mlir(
        str(mod),
        device="npu2",
        num_cols=8,
        omit_pingpong="",
        runtime_loop_tiling_sizes=[16, 16],
        use_lock_race_condition_fix=False,
        verbose=verbose,
    )
