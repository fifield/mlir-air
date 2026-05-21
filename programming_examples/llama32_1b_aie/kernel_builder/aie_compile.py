# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compile post-stitched MLIR-AIE text (aie/aiex dialect) to an ELF via aiecc.

This bypasses MLIR-AIR. The input is expected to already contain one or more
`aie.device` blocks plus a top-level `aie.runtime_sequence` (the form aircc
emits internally as `npu.air.mlir`). Hand-written placed-iron python that
produces an equivalent module is a drop-in replacement.
"""

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AIECompileArtifact:
    """Drop-in for air.backend.xrt.XRTCompileArtifact (ELF mode)."""

    output_binary: str
    kernel: str  # "main:<instance_name>"
    insts: Optional[str] = None  # always None for ELF mode


def _resolve_peano_dir() -> str:
    p = os.environ.get("PEANO_INSTALL_DIR", "")
    if p:
        return p
    # Fall back to the location used by build.sh / env.sh
    here = Path(__file__).resolve()
    install = here.parents[5] / "install" / "peano"
    if install.exists():
        return str(install)
    raise RuntimeError("PEANO_INSTALL_DIR is not set and install/peano not found")


def compile_aie_to_elf(
    mlir_text: str,
    instance_name: str,
    output_elf: str,
    *,
    workdir: Optional[str] = None,
    verbose: bool = False,
    extra_object_files: Optional[list] = None,
    aiecc_path: Optional[str] = None,
    bf16_emulation: bool = False,
) -> AIECompileArtifact:
    """Run aiecc on `mlir_text` and place the resulting ELF at `output_elf`.

    The kernel symbol for XRT loading is `main:<instance_name>` -- matching
    the convention used by aircc/XRTBackend in ELF mode.

    `extra_object_files` are copied into the working directory so aiecc can
    resolve `link_with = "..."` references emitted by the kernels.
    """
    workdir = Path(workdir or "aiecc_project").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Stage external .o files alongside the input mlir.
    for obj in extra_object_files or []:
        src = Path(obj)
        if src.exists():
            shutil.copy2(src, workdir / src.name)

    mlir_path = workdir / "input.mlir"
    mlir_path.write_text(mlir_text)

    aiecc_exe = aiecc_path or shutil.which("aiecc") or shutil.which("aiecc.py")
    if not aiecc_exe:
        raise RuntimeError("aiecc / aiecc.py not found in PATH")

    output_elf_path = Path(output_elf).resolve()
    output_elf_path.parent.mkdir(parents=True, exist_ok=True)
    # aiecc writes the ELF into its tmpdir; we copy it out at the end.
    tmp_elf_name = "aie.elf"

    cmd = [
        aiecc_exe,
        "--no-aiesim",
        "--no-xchesscc",
        "--no-xbridge",
        "--no-compile-host",
        f"--tmpdir={workdir}",
        "--generate-full-elf",
        "--expand-load-pdis",
        f"--full-elf-name={tmp_elf_name}",
        f"--peano={_resolve_peano_dir()}",
        "-O",
        "3",
    ]
    if bf16_emulation:
        cmd.append("--bf16-emulation")
    if verbose:
        cmd.append("-v")
    cmd.append(str(mlir_path))

    if verbose:
        print(f"  [aiecc] {' '.join(cmd)}")

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        # Surface stderr+stdout for diagnostics.
        msg = proc.stderr or proc.stdout
        raise RuntimeError(f"aiecc failed in {dt:.1f}s:\n{msg}")

    produced_elf = workdir / tmp_elf_name
    if not produced_elf.exists():
        raise RuntimeError(
            f"aiecc reported success but {produced_elf} does not exist. "
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    shutil.copy2(produced_elf, output_elf_path)

    return AIECompileArtifact(
        output_binary=str(output_elf_path),
        kernel=f"main:{instance_name}",
        insts=None,
    )
