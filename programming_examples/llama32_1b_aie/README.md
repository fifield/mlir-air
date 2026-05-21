# LLAMA-3.2-1B BF16 Inference on AMD NPU2 — MLIR-AIE port

Low-level duplicate of [`../llama32_1b`](../llama32_1b) (the MLIR-AIR
implementation). The model definition, weight loading, prefill/decode
orchestration, CPU reference, and Makefile mirror the source 1:1. The
difference is where the seam to the hardware sits.

| Layer | `../llama32_1b` (MLIR-AIR) | This project (MLIR-AIE) |
|-------|---------------------------|--------------------------|
| Kernel IR builders | mlir-air dialect via python, stitched in-process | reuses the same mlir-air builders today, but the post-stitching aie/aiex IR is captured and cached as `<name>.npu.air.mlir` |
| Compile backend | `aircc` (runs AIR passes + invokes `aiecc`) | `aiecc` directly on the cached IR |
| ELF / XRT load | `xrt.elf` / `xrt.ext.kernel` | same |

The purpose of this fork is to let you tweak the design at the low level
— the aie/aiex dialect, the same input aiecc consumes — without
mlir-air's compiler in the loop. Each unique kernel's IR is saved
alongside its `.elf` and is regenerated only when missing.

## What's the seam?

`kernel_builder/aie_ir_gen.py` has one function per unique kernel:

```python
def build_rms_gemv_rope_ir(...) -> str: ...
def build_o_gemv_ffn_ir(...) -> str: ...
def build_lm_head_gemv_ir(...) -> str: ...
def build_rms_gemms_rope_ir(...) -> str: ...
def build_o_ffn_ir(...) -> str: ...
def build_flash_attn_ir(...) -> str: ...
```

Each returns post-stitched mlir-aie text (multiple `aie.device` blocks
plus a top-level unnamed `aie.device` containing an `aie.runtime_sequence
@<instance>` dispatcher — exactly the form aiecc accepts and XRT loads
via `main:<instance>`). Today they shell through the existing
multi-launch builders + `aircc --output-format=elf` to harvest the IR.
Replacing any one of them with hand-written placed-iron python that
emits the same dialect is a drop-in change — no other file needs to be
touched.

`kernel_builder/aie_compile.py` is the aiecc-only compile path; given
`(ir_text, instance_name, output_elf)` it produces an ELF that the
existing `KernelCache.load_and_run(...)` path can run.

## What changed vs `../llama32_1b`

The port keeps host code, kernel builders, and Makefile targets
identical. All edits are localized to the compile/load seam.

**Verbatim from the source (no edits):**

- `llama32_1b_weights.py`, `llama32_1b_reference.py`
- `requirements.txt`, `.gitignore`
- `multi_launch_builder/` (all 5 stitched-kernel builders)
- `kernel_builder/{backend_presets.py, external_kernels.py,
  gemm_builder.py, rope_halfsplit.cc, stitching.py, ffn_swiglu/}`

**Modified files (compile-path seam only):**

- `kernel_builder/cache.py` — rewritten. Compile path uses
  `aie_compile.py` (aiecc-only). Load path uses `xrt.elf` +
  `xrt.hw_context` + `xrt.ext.kernel` directly, dropping the
  `air.backend` runtime dependency and the xclbin code path.
  Same public API (`compile_and_cache`, `load_manifest`,
  `load_and_run`). Also writes `<name>.npu.air.mlir` next to each
  `.elf` so the IR is tweakable on disk.
- `llama32_1b_prefill.py` — only `compile_all_kernels` was rewritten
  to call `aie_ir_gen.build_*_ir` + the new
  `cache.compile_and_cache(name, ir_text, instance_name=...)`.
  `run_transformer_block` and `preload_prefill_weights` are
  unchanged.
- `llama32_1b_decode.py` — same kind of change to
  `compile_decode_kernels` only; `run_decode_block` unchanged.
- `llama32_1b_inference.py` — only the top docstring changed.
- `Makefile` — cosmetic differences (header text, section comments).
  Targets and flags match 1:1 with the source.
- `README.md` — this file.

**New files:**

- `kernel_builder/aie_compile.py` — `aiecc` invocation +
  minimal `_XRTRunner` (replaces `air.backend.xrt.XRTBackend` at
  load time).
- `kernel_builder/aie_ir_gen.py` — `build_*_ir(...)` per unique
  kernel; harvests the post-stitching aie/aiex IR from `aircc
  --output-format=elf` (matching what `XRTBackend` passes through, so
  single-launch kernels like `flash_attn` get a `main:<instance>`
  dispatcher).

**Not copied from the source:**

- `ARCHITECTURE.md`, `docs/`, `run_npu2_makefile_peano_synthetic_verify.lit`.
  None affect runtime behavior — recreate them only if needed.

**Behavior differences worth noting:**

- `backend_presets.py` is still imported by prefill/decode for
  `RGR_BACKEND` etc., but `cache.load_and_run` ignores its
  `backend_kwargs` argument (kept only for API parity). The
  kernel-tuning knobs (`omit_pingpong`, `runtime_loop_tiling_sizes`,
  `omit_while_true_loop`, etc.) move to kwargs on the `build_*_ir`
  calls and are applied during the one-shot IR-generation step.
- `prepare_air_project()` in the source wipes a single global
  `air_project/` between compiles; this port wipes per-kernel
  `.<name>.work/` workdirs inside `compile_aie_to_elf` and stages
  `.o` files per-call instead.

## Hand-editing the IR

```bash
make compile                                       # populates the *.npu.air.mlir cache
$EDITOR build_peano/decode_kernel_cache/lm_head_gemv.npu.air.mlir
rm build_peano/decode_kernel_cache/lm_head_gemv.elf
# next `make run` will rebuild lm_head_gemv.elf from your edited IR
# (or call cache.compile_from_cached_ir("lm_head_gemv", "lm_head_gemv"))
```

## Quick start

```bash
# Compile all unique kernels (~80 s; flash_attn IR-gen is the slow step)
make compile

# Run inference (instruct model, up to 1000 tokens, stops on EOT)
make run HF_MODEL_ID=unsloth/Llama-3.2-1B-Instruct \
    PROMPT="What is the capital of France?"
```

See [`../llama32_1b/README.md`](../llama32_1b/README.md) for prerequisites
(MLIR-AIR base env, HuggingFace setup, weight download, etc.) — the
project shares all of them.
