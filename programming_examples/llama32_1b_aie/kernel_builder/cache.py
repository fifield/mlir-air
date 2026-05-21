# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel compilation cache for the AIE-only port.

Same external interface as the MLIR-AIR llama32_1b cache.py so the rest of
the inference pipeline is unchanged. The differences are internal:

  * `compile_and_cache(name, ir_text, instance_name)` -- consumes
    mlir-aie text (aie/aiex dialect, already stitched) and runs **aiecc**
    directly. No aircc in the steady state. The IR text is sourced today
    by `aie_ir_gen.build_*_ir(...)` which shells through aircc once with
    `--output-format=none` to harvest the post-stitching IR; replacing
    those builders with hand-written placed-iron python is a drop-in.

  * Loading uses xrt's ELF path (`xrt.elf` + `xrt.hw_context` +
    `xrt.ext.kernel`), so no air.backend dependency at runtime.
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from .aie_compile import AIECompileArtifact, compile_aie_to_elf


# ---------------------------------------------------------------------------
# Stage external .o files alongside aiecc's tmpdir on each compile so that
# `link_with = "<obj>"` references in the kernels resolve.
# ---------------------------------------------------------------------------

_LINK_OBJS = [
    "silu_and_mul.o",
    "rope.o",
    "attn.o",
    "attn_npu2.o",
    "mv.o",
    "mv_k8192.o",
    "attn_decode_npu2.o",
]


def _link_obj_paths():
    """Return the subset of _LINK_OBJS that exist in the cwd."""
    return [str(Path(o).resolve()) for o in _LINK_OBJS if Path(o).exists()]


# ---------------------------------------------------------------------------
# Profiler -- identical to the AIR-side version (just records timings)
# ---------------------------------------------------------------------------


class Profiler:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.compile_times = {}
        self.kernel_times = {}
        self.layer_times = []
        self.kernel_breakdowns = {}

    def record_compile(self, name, duration):
        if self.enabled:
            self.compile_times[name] = duration

    def record_kernel(self, name, duration):
        if self.enabled:
            self.kernel_times.setdefault(name, []).append(duration)

    def record_breakdown(self, name, write_ms, kernel_ms, read_ms,
                         n_written, bytes_written, n_readback):
        if self.enabled:
            self.kernel_breakdowns.setdefault(name, []).append(
                {
                    "write_ms": write_ms,
                    "kernel_ms": kernel_ms,
                    "read_ms": read_ms,
                    "n_written": n_written,
                    "bytes_written": bytes_written,
                    "n_readback": n_readback,
                }
            )

    def start_layer(self):
        if self.enabled:
            return time.time()
        return None

    def end_layer(self, layer_idx, t0):
        if self.enabled and t0 is not None:
            self.layer_times.append((layer_idx, time.time() - t0))

    def report(self):
        if not self.enabled:
            return
        print(f"\n{'='*60}\nPROFILING REPORT\n{'='*60}")
        if self.compile_times:
            print("\n--- Compilation Phase ---")
            total = 0
            for name, t in sorted(self.compile_times.items()):
                print(f"  {name:40s} {t:8.1f}s")
                total += t
            print(f"  {'Total compilation':40s} {total:8.1f}s "
                  f"({len(self.compile_times)} kernels)")
        if self.layer_times:
            print("\n--- Per-Layer Execution ---")
            for idx, t in self.layer_times:
                print(f"  Layer {idx:3d}: {t:8.2f}s")
            total_layers = sum(t for _, t in self.layer_times)
            print(f"  {'Total prefill':40s} {total_layers:8.2f}s")
        if self.kernel_times:
            print("\n--- Kernel Breakdown (avg per invocation) ---")
            total_avg = 0
            for name, times in sorted(self.kernel_times.items()):
                avg = sum(times) / len(times)
                total_avg += avg * len(times)
                print(f"  {name:40s} avg={avg:6.3f}s  "
                      f"min={min(times):6.3f}s  max={max(times):6.3f}s  "
                      f"(x{len(times)})")
            if self.layer_times:
                n_layers = len(self.layer_times)
                print(f"  {'Total kernel time':40s} {total_avg:8.2f}s")
                print(f"  {'Avg per layer (kernel time)':40s} "
                      f"{total_avg/n_layers:8.2f}s")
        if self.kernel_breakdowns:
            print("\n--- Fine-Grained Breakdown (avg per invocation) ---")
            print(f"  {'Kernel':20s} {'BO Write':>10s} {'NPU Run':>10s} "
                  f"{'BO Read':>10s} {'Total':>10s}  {'Written':>8s} {'Read':>6s}")
            print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}  "
                  f"{'-'*8} {'-'*6}")
            total_w = total_k = total_r = 0
            for name in sorted(self.kernel_breakdowns.keys()):
                entries = self.kernel_breakdowns[name]
                n = len(entries)
                avg_w = sum(e["write_ms"] for e in entries) / n
                avg_k = sum(e["kernel_ms"] for e in entries) / n
                avg_r = sum(e["read_ms"] for e in entries) / n
                avg_total = avg_w + avg_k + avg_r
                avg_mb = sum(e["bytes_written"] for e in entries) / n / 1024 / 1024
                avg_nr = sum(e["n_readback"] for e in entries) / n
                total_w += avg_w * n
                total_k += avg_k * n
                total_r += avg_r * n
                print(f"  {name:20s} {avg_w:8.2f}ms {avg_k:8.2f}ms "
                      f"{avg_r:8.2f}ms {avg_total:8.2f}ms  "
                      f"{avg_mb:6.1f}MB {avg_nr:4.0f}bo  (x{n})")
            grand = total_w + total_k + total_r
            print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
            print(f"  {'TOTAL':20s} {total_w:8.1f}ms {total_k:8.1f}ms "
                  f"{total_r:8.1f}ms {grand:8.1f}ms")


# ---------------------------------------------------------------------------
# Minimal XRT loader (no air.backend dependency).
# ---------------------------------------------------------------------------


class _XRTRunner:
    """Holds the (device, hw_context, kernel) tuple for one cached ELF.

    Mirrors the bits of `air.backend.xrt.XRTBackend` that cache.load_and_run
    pokes at (`.device`, `.kernel`) -- nothing else.
    """

    def __init__(self):
        self.device = None
        self.hw_context = None
        self.elf = None
        self.kernel = None
        # Kept for cache.load_and_run's xclbin branch (always None for ELF).
        self.bo_instr = None
        self.instr_v = None

    def load(self, artifact: AIECompileArtifact):
        import pyxrt as xrt

        self.device = xrt.device(0)
        self.elf = xrt.elf(artifact.output_binary)
        self.hw_context = xrt.hw_context(self.device, self.elf)
        self.kernel = xrt.ext.kernel(self.hw_context, artifact.kernel)


# ---------------------------------------------------------------------------
# KernelCache -- same call-shape as the AIR-side version.
# ---------------------------------------------------------------------------


class KernelCache:
    MANIFEST_FILE = "manifest.json"

    def __init__(self, cache_dir=None, verbose=False, profiler=None):
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parent / "kernel_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.profiler = profiler or Profiler()
        self.artifacts = {}  # name -> AIECompileArtifact
        self._loaded = {}    # name -> _XRTRunner
        self._cached_bos = {}

    def _log(self, msg):
        if self.verbose:
            print(f"  [KernelCache] {msg}")

    # ----- Compile path -----------------------------------------------------

    def compile_and_cache(self, name, ir_text, instance_name):
        """Compile post-stitched mlir-aie text to an ELF and cache it.

        Args:
            name: Unique key for this kernel.
            ir_text: mlir-aie (aie/aiex) dialect text. Must contain a
                top-level `aie.runtime_sequence @<instance_name>(...)` --
                i.e. the form aircc emits internally before invoking aiecc.
            instance_name: function symbol that becomes the XRT kernel
                identifier (`main:<instance_name>`).
        """
        self._log(f"Compiling {name}...")
        t0 = time.time()

        # Save the IR text alongside the cached ELF for inspection / editing.
        mlir_cached = self.cache_dir / f"{name}.npu.air.mlir"
        mlir_cached.write_text(ir_text)

        elf_path = self.cache_dir / f"{name}.elf"
        artifact = compile_aie_to_elf(
            ir_text,
            instance_name=instance_name,
            output_elf=str(elf_path),
            workdir=str(self.cache_dir / f".{name}.work"),
            verbose=self.verbose,
            extra_object_files=_link_obj_paths(),
        )
        dt = time.time() - t0
        self.profiler.record_compile(name, dt)
        self.artifacts[name] = artifact
        print(f"  Compiled {name}: {dt:.1f}s -> {elf_path.name}")

    def compile_from_cached_ir(self, name, instance_name):
        """Re-compile from cache_dir/<name>.npu.air.mlir (skip IR gen).

        Useful after hand-editing the cached IR or when running with a
        purely AIE pipeline (no aircc on PATH).
        """
        mlir_cached = self.cache_dir / f"{name}.npu.air.mlir"
        if not mlir_cached.exists():
            raise FileNotFoundError(
                f"No cached IR for {name} at {mlir_cached}. "
                "Run `make compile` (with aircc available) first to seed it."
            )
        self.compile_and_cache(name, mlir_cached.read_text(), instance_name)

    # ----- Manifest persistence --------------------------------------------

    def _save_manifest(self):
        manifest = {}
        for name, art in self.artifacts.items():
            manifest[name] = {
                "output_binary": str(art.output_binary),
                "kernel": art.kernel,
                "insts": None,
            }
        (self.cache_dir / self.MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2)
        )
        self._log(f"Saved manifest with {len(manifest)} entries")

    def load_manifest(self):
        path = self.cache_dir / self.MANIFEST_FILE
        if not path.exists():
            return False
        manifest = json.loads(path.read_text())
        for name, info in manifest.items():
            binary = info["output_binary"]
            if not Path(binary).exists():
                print(f"  WARNING: cached binary not found: {binary}")
                return False
            self.artifacts[name] = AIECompileArtifact(
                output_binary=binary,
                kernel=info["kernel"],
                insts=None,
            )
        self._log(f"Loaded manifest with {len(self.artifacts)} entries")
        return True

    # ----- Run path --------------------------------------------------------

    def load_and_run(
        self,
        name,
        backend_kwargs,  # ignored -- kept for API compat with the AIR cache
        *inputs,
        output_indices=None,
        static_input_indices=None,
        intermediate_indices=None,
        bo_key=None,
    ):
        """Load cached ELF and execute with BO reuse (ELF path only).

        Three levels of caching match the AIR-side version:
          1. XRT context per kernel name
          2. Buffer Objects per `bo_key` (lets layers share an ELF but
             keep distinct weight BOs)
          3. Static / intermediate skipping on weight + scratch BOs
        """
        del backend_kwargs  # AIE port has no backend knobs at run time
        import filelock
        import pyxrt as xrt

        if name not in self.artifacts:
            raise RuntimeError(
                f"Kernel '{name}' not found in cache. "
                f"Available: {list(self.artifacts.keys())}"
            )

        if name not in self._loaded:
            runner = _XRTRunner()
            with filelock.FileLock("/tmp/npu.lock"):
                runner.load(self.artifacts[name])
            self._loaded[name] = runner
            self._log(f"Loaded {name} (XRT context cached)")

        runner = self._loaded[name]

        _bo_key = bo_key if bo_key is not None else name
        sizes_in_bytes = [a.size * a.itemsize for a in inputs]
        static_indices = set(static_input_indices or [])
        intermediate_set = set(intermediate_indices or [])

        first_call = _bo_key not in self._cached_bos
        if first_call:
            bos = [xrt.ext.bo(runner.device, s) for s in sizes_in_bytes]
            self._cached_bos[_bo_key] = bos
            self._log(f"Allocated {len(bos)} BOs for {_bo_key}")

        bos = self._cached_bos[_bo_key]

        t0 = time.time()
        with filelock.FileLock("/tmp/npu.lock"):
            # Phase 1: write inputs
            t_write = time.perf_counter()
            n_written = 0
            bytes_written = 0
            for i, a in enumerate(inputs):
                if i in static_indices and not first_call:
                    continue
                if i in intermediate_set and not first_call:
                    continue
                if a.dtype == bfloat16:
                    a = a.view(np.int16)
                mv = bos[i].map()
                src = np.frombuffer(a, dtype=np.uint8)
                dst = np.frombuffer(mv, dtype=np.uint8, count=len(src))
                np.copyto(dst, src, casting="no")
                bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                n_written += 1
                bytes_written += len(src)
            t_write_ms = (time.perf_counter() - t_write) * 1000

            # Phase 2: launch
            t_kernel = time.perf_counter()
            run = xrt.run(runner.kernel)
            for i, bo in enumerate(bos):
                run.set_arg(i, bo)
            run.start()
            run.wait2()
            t_kernel_ms = (time.perf_counter() - t_kernel) * 1000

            # Phase 3: read back
            t_read = time.perf_counter()
            if output_indices is None:
                readback_set = {len(inputs) - 1}
            else:
                readback_set = set(output_indices)
            for idx in readback_set:
                bos[idx].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            results = tuple(
                (
                    np.frombuffer(
                        bos[i].map(),
                        dtype=inputs[i].dtype,
                        count=inputs[i].size,
                    )
                    if i in readback_set
                    else np.empty(0, dtype=inputs[i].dtype)
                )
                for i, s in enumerate(sizes_in_bytes)
            )
            t_read_ms = (time.perf_counter() - t_read) * 1000

        duration = time.time() - t0
        self.profiler.record_kernel(name, duration)
        self.profiler.record_breakdown(
            name, t_write_ms, t_kernel_ms, t_read_ms,
            n_written, bytes_written, len(readback_set),
        )
        return results


# ---------------------------------------------------------------------------
# Compatibility helper: callers in this project pass `prepare_air_project()`
# from the AIR-side cache. Provide a no-op (and a stub) under the same name
# so prefill/decode helpers from the AIR project can be imported unchanged
# during the iterative port.
# ---------------------------------------------------------------------------


def prepare_air_project():  # pragma: no cover - vestigial entry point
    """No-op shim. The AIR-side `prepare_air_project` wiped & re-staged the
    aircc working directory; on the AIE side we stage object files per-call
    inside compile_aie_to_elf, so nothing to do here."""
