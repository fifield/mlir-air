# Plan — AWQ-quantized Decode for AIR LLAMA-3.2-1B (Revised)

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task. Do not modify the original plan at `~/.claude/plans/dapper-wiggling-starlight.md`.

**Goal:** Add an AWQ int4 decode path for LLAMA-3.2-1B on MLIR-AIR that coexists with the current BF16 path and reduces decode weight bandwidth enough to materially improve tokens/sec.

**Architecture:** Keep the existing BF16 prefill/decode path intact. Add a separate AWQ decode path built around offline-repacked row-major int4 weights and a fused int4-dequant + BF16 GEMV kernel. Treat the fused kernel microbenchmark as the hard gate: if it cannot beat BF16 GEMV by at least 2x on representative M/K shapes, stop integration and pivot.

**Tech Stack:** MLIR-AIR, MLIR-AIE/AIE2P Peano C++ kernels, XRT runner, safetensors, NumPy/ml_dtypes bfloat16, existing LLAMA-3.2-1B AIR example.

---

## Critical design decisions

### Decision 1 — AWQ is decode-only, but prefill still needs BF16-compatible weights

The AWQ checkpoint does not contain BF16 transformer linear weights. It contains packed AWQ tensors:

- `*.qweight`
- `*.qzeros`
- `*.scales`

Therefore, "decode-only AWQ, prefill BF16" requires one explicit mixed-weight strategy.

Implement **Strategy B** for first delivery:

- Load the AWQ checkpoint once.
- Keep packed AWQ tensors for decode.
- CPU-dequant AWQ linears to BF16 at load/preparation time for prefill.
- Norms, embeddings, tokenizer, and RoPE LUT are loaded unchanged from the AWQ checkpoint.
- LM head uses the AWQ lm_head tensors for decode logits unless explicitly disabled.

This means prefill is not original Meta BF16; it is BF16 reconstructed from AWQ. That is acceptable for this benchmark path and keeps memory/load complexity lower than loading both original BF16 and AWQ checkpoints.

Later optional strategy:

- Add `--bf16-weights <path>` to load original BF16 weights for prefill while using AWQ weights for decode.

### Decision 2 — Trust config/tensors, not README prose

The AWQ model README says "symmetric" in one place, but the actual config and tensors indicate asymmetric zero-point AWQ:

- `config.json:quantization_config.zero_point = true`
- `quantization_config.quant_method = awq`
- `quantization_config.version = gemm`
- `quantization_config.pack_method = reorder`
- tensor set includes `qzeros`

Use the actual config and safetensors layout as source of truth.

### Decision 3 — Fused kernel performance is not guaranteed

The existing AWQ example proves scalar dequant correctness only. It does not prove that fused unpack/dequant/MAC vectorizes or schedules well on AIE2P.

Treat 3.7x weight-byte reduction as a theoretical upper bound, not a target assumption.

Stage 4 success gate:

- Representative AWQ GEMV is at least 2x faster than BF16 GEMV for `(M=2048,K=2048)`.
- Stretch goal is within 1.5x of the theoretical weight-bandwidth speedup.

If Stage 4 misses that gate after one focused tuning pass, stop the full integration and pivot to W8A16 or another lower-risk quantization path.

---

## Known current code facts

Current BF16 decode GEMV:

- External kernel: `programming_examples/matrix_vector_multiplication/bf16/mv.cc`
- Builder: `programming_examples/matrix_vector_multiplication/bf16/matvec.py`
- GEMV computes `C[M] = A[M,K] @ B[K]`
- External public entry is:

```cpp
void matvec_vectorized_bf16_bf16(uint32_t m, uint32_t k, uint32_t row_offset,
                                 const bfloat16 *__restrict a_in,
                                 const bfloat16 *__restrict b_in,
                                 bfloat16 *__restrict c_out);
```

Current BF16 decode pre-transposes weights in:

- `programming_examples/llama32_1b/llama32_1b_inference.py:prepare_runtime`

Current multi-launch GEMV arg convention is:

- subkernel arg0 = weight matrix
- subkernel arg1 = input vector
- subkernel arg2 = output vector

For example in `rms_gemv_rope_multi.py`, Q GEMV maps:

```python
(q_ir, "q", {0: 3, 1: 2, 2: 4})  # weight=wq, input=normed, output=q
```

Current external-object copy list is fixed in:

- `programming_examples/llama32_1b/kernel_builder/cache.py:prepare_air_project`

Any new `link_with = "mv_awq*.o"` object must be compiled and copied there.

---

## Target AWQ decode tensor layout

For each quantized linear layer with logical BF16 weight shape `(K, N)` in HF/current-loader convention before decode transpose:

- Quark/AutoAWQ source tensors:
  - `qweight`: `(K, N/8)` int32
  - `qzeros`: `(K/group_size, N/8)` int32
  - `scales`: `(K/group_size, N)` bf16

For fused GEMV, store output rows contiguous, matching `A[M,K] @ B[K]` where `M=N` and rows are output channels:

- `qweight_repacked`: `(M, K/2)` uint8
  - two uint4 values per byte
  - natural low-nibble-first along K
- `params_interleaved`: `(M, 2 * K/group_size)` bf16
  - `[scale_0, zero_0_as_bf16, scale_1, zero_1_as_bf16, ...]`
  - one params row per output row

Important: qzeros semantics are format-specific and must be validated before repacking all tensors. Do not assume whether unpacked qzero is used directly or with an offset such as `+1` until Stage 1 passes against an independent AWQ oracle.

---

## Stage 0 — Environment cleanup (pre-req, small)

**Objective:** Make `source env.sh` sufficient for Python imports of `air` and `aie` from the local install tree.

**Why:** Current generated `env.sh` does not export the local `install/mlir-air/python` or `install/mlir-aie/python` paths. After sourcing it, `import air` currently fails unless PYTHONPATH is supplied elsewhere.

**Files:**

- Modify: `setup.py`
- Regenerate: `env.sh`

**Implementation notes:**

In `setup.py:generate_env_sh`, add PYTHONPATH entries if directories exist:

```bash
for d in "${INSTALL_DIR}/mlir-air/python" "${INSTALL_DIR}/mlir-aie/python"; do
  [ -d "${d}" ] && case ":${PYTHONPATH:-}:" in *:"${d}":*) ;; *) PYTHONPATH="${d}:${PYTHONPATH:-}";; esac
done
export PYTHONPATH
```

**Verification:**

```bash
cd /home/jfifield/npu-dev-air
python3 setup.py --defaults   # or the project's env-regeneration path if available
source env.sh
python3 - <<'PY'
import air
import aie
print('ok', air.__file__, aie.__file__)
PY
```

Expected: imports succeed without relying on `~/.bashrc` leakage.

---

## Stage 1 — Independent AWQ format validation (hard correctness gate)

**Objective:** Lock down qweight/qzeros unpack semantics for the actual checkpoint before writing the repacker.

**Files to create:**

- `programming_examples/llama32_1b/tools/inspect_awq_format.py`
- `programming_examples/llama32_1b/tools/awq_format_reference.py`

**Inputs:**

- AWQ model directory, e.g.
  `/home/jfifield/npu-dev-air/gpu-awq-onnx-bench/models/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead/`

**Validation tasks:**

1. Read `config.json` and assert:

```python
quantization_config.bits == 4
quantization_config.group_size == 128
quantization_config.quant_method == 'awq'
quantization_config.version == 'gemm'
quantization_config.pack_method == 'reorder'
quantization_config.zero_point is True
```

2. Print and assert representative shapes:

- `model.layers.0.self_attn.q_proj.qweight == (2048, 256)`
- `model.layers.0.self_attn.q_proj.qzeros == (16, 256)`
- `model.layers.0.self_attn.q_proj.scales == (16, 2048)`
- `model.layers.0.self_attn.k_proj.qweight == (2048, 64)`
- `model.layers.0.mlp.down_proj.qweight == (8192, 256)`
- `lm_head.qweight == (2048, 16032)`

3. Implement candidate unpack paths using explicit unsigned masks:

```python
u = qweight.astype(np.uint32)
nibble = (u >> shift) & 0xF
```

Never rely on signed right-shift behavior of int32 qweight/qzeros.

4. Test the documented/expected AWQ reorder shuffle:

```python
shuffle = [0, 4, 1, 5, 2, 6, 3, 7]
```

But do not trust it until it matches an independent oracle.

5. Explicitly test qzero variants:

- `zero = unpacked_qzero`
- `zero = unpacked_qzero + 1`
- any Quark/AutoAWQ documented variant discovered from installed loaders

6. Compare one or more selected dequantized linear outputs against an independent AWQ oracle.

Preferred oracle:

- Load the same AWQ model using Transformers/AutoAWQ/Quark in Python.
- Feed a deterministic float input vector to a selected quantized module.
- Compare output to local NumPy dequant + matmul.

Minimum selected modules:

- `model.layers.0.self_attn.q_proj`
- `model.layers.0.mlp.down_proj`
- `lm_head`

**Verification:**

```bash
source /home/jfifield/npu-dev-air/env.sh
cd /home/jfifield/npu-dev-air/mlir-air/programming_examples/llama32_1b
python3 tools/inspect_awq_format.py --model /path/to/awq_model
python3 tools/awq_format_reference.py --model /path/to/awq_model --module model.layers.0.self_attn.q_proj
python3 tools/awq_format_reference.py --model /path/to/awq_model --module model.layers.0.mlp.down_proj
python3 tools/awq_format_reference.py --model /path/to/awq_model --module lm_head
```

Expected:

- Exact tensor shapes match.
- One chosen unpack/qzero semantics matches the independent AWQ oracle within BF16/F32 tolerance.
- Record the chosen qzero rule in comments in `awq_format_reference.py`.

Do not proceed to Stage 2 until this passes.

---

## Stage 2 — Offline AWQ repack tool

**Objective:** Convert Quark/AutoAWQ GEMM-reorder tensors into row-major K-packed uint4 tensors optimized for the AIR GEMV access pattern.

**Files to create:**

- `programming_examples/llama32_1b/tools/repack_awq.py`
- `programming_examples/llama32_1b/tools/test_repack_awq.py`

**Output format:**

For each linear tensor, write:

- `<prefix>.qweight_repacked`: uint8, shape `(N, K/2)`
- `<prefix>.params_interleaved`: bf16, shape `(N, 2 * K/group_size)`

where `prefix` is the original module path, for example:

- `model.layers.0.self_attn.q_proj.qweight_repacked`
- `model.layers.0.self_attn.q_proj.params_interleaved`

Also copy unchanged tensors needed for runtime:

- `model.embed_tokens.weight`
- `model.norm.weight`
- per-layer RMSNorm weights
- tokenizer/config files are kept in the model directory, not necessarily duplicated in safetensors

**Repack algorithm:**

For each linear:

1. Load source tensors:

```python
qweight: (K, N/8) int32
qzeros:  (K/g, N/8) int32
scales:  (K/g, N) bf16
```

2. Unpack qweight using the Stage 1-validated AWQ GEMM/reorder semantics to recover logical `(K, N)` uint4 values.

3. Transpose to `(N, K)` so output rows are contiguous.

4. Re-pack along K naturally:

```python
byte[j] = low_nibble_for_k_even | (high_nibble_for_k_odd << 4)
```

Result: `(N, K/2)` uint8.

5. Unpack qzeros using the Stage 1-validated qzero semantics to recover `(K/g, N)` zero values.

6. Transpose qzeros to `(N, K/g)`.

7. Transpose scales to `(N, K/g)`.

8. Interleave params as bf16:

```python
params[row, 2*g]     = scale[row, g]
params[row, 2*g + 1] = zero[row, g]
```

**Important:** If Stage 1 determines qzeros need an offset such as `+1`, apply it in repack before storing bf16 zeros.

**Verification:**

Unit tests must avoid circularity.

Required tests:

1. Tiny hand-authored tensor test:

- Known K,N,g
- Known int4 values
- Known qzeros/scales
- Known output bytes/params

2. Round-trip repack test:

- Original AWQ tensors -> dequant using Stage 1 reference
- Repacked tensors -> dequant using repacked reference
- Compare elementwise or near-elementwise.

3. Linear-output oracle test:

- For selected real modules, compute output of `x @ W_dequant` from repacked tensors.
- Compare to independent AWQ oracle from Stage 1.

Commands:

```bash
python3 tools/test_repack_awq.py --model /path/to/awq_model
python3 tools/repack_awq.py --src /path/to/awq_model --dst ./awq_repacked
```

Expected:

- Tests pass.
- Repacked safetensors include qweight/params for all quantized linears including `lm_head`.

---

## Stage 3 — AWQ-aware weight loader and CPU-dequant smoke path

**Objective:** Prove loader, mixed prefill/decode data model, and accuracy before writing kernels.

**Files modified:**

- `programming_examples/llama32_1b/llama32_1b_weights.py`
- `programming_examples/llama32_1b/llama32_1b_inference.py`
- `programming_examples/llama32_1b/llama32_1b_reference.py`
- `programming_examples/llama32_1b/Makefile`

**Files to create if cleaner:**

- `programming_examples/llama32_1b/llama32_1b_awq.py`

**Data model:**

Add explicit AWQ containers instead of loose dynamic attributes where practical.

Suggested structures:

```python
@dataclass
class AwqLinear:
    qweight: np.ndarray          # uint8, shape (M, K/2)
    params: np.ndarray           # bf16, shape (M, 2*K/group_size)
    k: int
    m: int
    group_size: int = 128

@dataclass
class AwqLayerWeights:
    wq: AwqLinear
    wk: AwqLinear
    wv: AwqLinear
    wo: AwqLinear
    w_gate: AwqLinear
    w_up: AwqLinear
    w_down: AwqLinear
```

Extend `LlamaWeights` with optional fields:

```python
is_awq: bool = False
awq_layers: list[AwqLayerWeights] | None = None
awq_lm_head: AwqLinear | None = None
```

Keep BF16 fields populated for prefill using CPU-dequant from AWQ tensors for Strategy B.

**CLI:**

Add:

```bash
--quant bf16|awq
--awq-weights PATH
```

For now:

- `--quant bf16` uses existing path.
- `--quant awq` requires `--awq-weights` and loads repacked AWQ.

**Makefile:**

Add:

```make
QUANT ?= bf16
AWQ_WEIGHTS ?=
```

Wire `QUANT=awq` to pass:

```bash
--quant awq --awq-weights $(AWQ_WEIGHTS)
```

Keep current `WEIGHTS=hf|synthetic` behavior for BF16 path unless intentionally replaced.

**CPU-dequant smoke path:**

Before the fused kernel exists, add a mode that dequants AWQ packed tensors into BF16 and feeds the existing BF16 decode/preload path.

This proves:

- AWQ loader finds every tensor.
- Repacked tensors can reconstruct weights.
- Prefill/decode/reference can run with the AWQ-derived BF16 weights.

**LM head:**

Do not silently fall back to tied embeddings for AWQ if `lm_head.weight` is missing. The AWQ checkpoint has quantized `lm_head.qweight/qzeros/scales`; use `awq_lm_head` or explicitly choose a BF16 fallback with a warning.

**Verification:**

```bash
make run QUANT=awq AWQ_WEIGHTS=./awq_repacked MODEL=instruct PROMPT="What is the capital of France?" N_TOKENS=5
make verify QUANT=awq AWQ_WEIGHTS=./awq_repacked N_TOKENS=5
```

Expected:

- Loader succeeds.
- No missing qweight/qzeros/scales.
- CPU-dequant BF16 smoke path produces coherent text.
- CPU reference and runtime match for the CPU-dequant BF16 path at the same tolerance as existing BF16 verification, or any drift is understood and logged.

---

## Stage 4 — Fused dequant+GEMV external kernel (`mv_awq.cc`)

**Objective:** Build and microbenchmark a fused AWQ int4-dequant + BF16 GEMV external AIE2P kernel.

**Files to create:**

- `programming_examples/llama32_1b/kernel_builder/mv_awq.cc`
- `programming_examples/llama32_1b/kernel_builder/mv_awq_test.py`

**Files modified:**

- `programming_examples/llama32_1b/kernel_builder/external_kernels.py`

**Public C entry:**

Use an arg order matching the AWQ builder convention:

```cpp
extern "C" {

void matvec_awq_bf16(uint32_t m, uint32_t k, uint32_t row_offset,
                     const uint8_t *__restrict qweights,
                     const bfloat16 *__restrict params,
                     const bfloat16 *__restrict b_in,
                     bfloat16 *__restrict c_out);

void linalg_fill_bf16(bfloat16 *c_out);

}
```

Semantics:

- `qweights` points to a row-major tile of shape `(m_input, k/2)` or equivalent L1 tile chunk.
- `params` points to matching rows of shape `(m_input, 2*k/group_size)`.
- `b_in` is BF16 input vector shape `(k,)`.
- `c_out += row_offset` only offsets output, preserving current BF16 behavior.
- qweight/params offsets are controlled by AIR DMA tiling, not by `row_offset` inside the kernel.

**Inner loop sketch:**

For each output row:

```cpp
for (uint32_t g = 0; g < k / GROUP_SIZE; ++g) {
  bfloat16 scale = row_params[2*g];
  bfloat16 zero  = row_params[2*g + 1];
  for (uint32_t kv = 0; kv < GROUP_SIZE; kv += 64) {
    // Load 32 bytes = 64 int4 values.
    // Unpack low/high nibbles to 64 values.
    // Convert/subtract/multiply to bf16/float-compatible vector.
    // MAC with b_in[g*GROUP_SIZE + kv : +64].
  }
}
```

**Important performance notes:**

- Do not assume Peano vectorizes scalar unpack well.
- Inspect generated IR/assembly enough to determine whether the inner loop is scalarizing badly.
- Preserve accfloat accumulation as in BF16 `mv.cc`.
- If dequanting to bf16 before MAC causes accuracy drift, document it and compare to runtime behavior.

**External compile functions:**

Add:

```python
def compile_mv_awq(tile_m=8): ... -> mv_awq.o

def compile_mv_awq_k8192(): ... -> mv_awq_k8192.o
```

The K=8192 variant must use symbol renaming analogous to `compile_mv_k8192()`:

```python
-Dmatvec_awq_bf16=dg_matvec_awq_bf16
-Dlinalg_fill_bf16=dg_linalg_fill_bf16
```

**Standalone correctness test:**

`mv_awq_test.py` generates:

- random logical int4 weights
- random scales/zeros
- packed `qweight_repacked`
- `params_interleaved`
- random BF16 input vector

Reference:

```python
W = dequant_repacked(qweight, params).astype(np.float32)
ref = W @ b.astype(np.float32)
```

Compare NPU output to reference.

**Recommended tolerances:**

Start with:

```python
rtol=3e-2, atol=3e-2
```

Only loosen with a written reason. The inherited dequant-only tolerance `rtol=1e-1, atol=5e-2` is too loose for dot-product validation unless measured drift demands it.

**Microbenchmark:**

Measure both BF16 and AWQ under the same runner conditions:

- `(M=2048, K=2048)`
- `(M=512, K=2048)`
- `(M=8192, K=2048)`
- `(M=2048, K=8192)`
- LM partition shape `(M=16384, K=2048)` if compile/runtime permits

**Gate:**

Pass if:

- correctness test passes for all representative shapes; and
- `(M=2048,K=2048)` AWQ GEMV is at least 2x faster than BF16 GEMV; and
- no obvious static rewrite/copy overhead is included in the microbenchmark timing.

If correctness passes but speedup is <2x:

- inspect generated code;
- attempt one focused tuning pass;
- if still <2x, stop full integration and propose pivot.

---

## Stage 5 — AWQ GEMV AIR module builder

**Objective:** Build a single AWQ GEMV AIR module with the same tiling role as BF16 `matvec.py`, but using qweight+params+input vector.

**Files to create:**

- `programming_examples/llama32_1b/kernel_builder/gemv_awq_builder.py`

**Subkernel function signature in AIR:**

The AWQ GEMV builder should expose a function with 4 memref args:

```mlir
func.func @matvec_awq(
  %qweight: memref<MxK/2xi8>,
  %params:  memref<Mx2K/gxbf16>,
  %input:   memref<Kxbf16>,
  %output:  memref<Mxbf16>
)
```

**Arg order convention:**

- arg0 = qweight
- arg1 = params
- arg2 = input vector
- arg3 = output

Use this consistently in all stitch maps.

**Tiling and memory:**

Update capacity checks; do not reuse BF16 checks unchanged.

For L2/L1 staging, account for:

- qweight tile bytes: `herd_m * tile_m * k / 2`
- params tile bytes: `herd_m * tile_m * (2*k/group_size) * 2`
- output tile bytes: `herd_m * tile_m * 2`
- input vector L1 bytes: `k * 2`
- if `m_input > 1`, params L1 must include `m_input` rows, not one row

Initial conservative choice:

- K=2048 GEMVs: keep `tile_m=8`, `m_input=4`, `herd_m=8` if capacity and correctness allow.
- K=8192 down GEMV: start with existing down settings `down_tile_m=2`, `down_m_input=1`.

If the AWQ kernel is simpler with `m_input=1`, use it first for correctness, then tune.

**Object links:**

- K=2048 AWQ GEMVs link with `mv_awq.o`.
- K=8192 down AWQ GEMV links with `mv_awq_k8192.o` and renamed symbols.

**Verification:**

Create a standalone compile/run in `gemv_awq_builder.py` or companion test:

```bash
python3 kernel_builder/gemv_awq_builder.py --m 2048 --k 2048 --compile-mode compile-and-run
python3 kernel_builder/gemv_awq_builder.py --m 2048 --k 8192 --compile-mode compile-and-run
```

Expected:

- Module parses.
- Links correct object files.
- Output matches NumPy reference.
- Timing still meets Stage 4 expectations within reason.

---

## Stage 6 — Decode multi-launch AWQ builders

**Objective:** Mirror existing decode multi-launch builders using AWQ GEMV submodules.

**Files to create:**

- `programming_examples/llama32_1b/multi_launch_builder/rms_gemv_rope_awq_multi.py`
- `programming_examples/llama32_1b/multi_launch_builder/o_gemv_ffn_awq_multi.py`
- `programming_examples/llama32_1b/multi_launch_builder/lm_head_gemv_awq_multi.py`

**Files modified:**

- `programming_examples/llama32_1b/kernel_builder/backend_presets.py`
- possibly `programming_examples/llama32_1b/kernel_builder/stitching.py` if current helpers assume 3-arg GEMV patterns

### 6A — `rms_gemv_rope_awq_multi.py`

Current BF16 function has 13 args. AWQ version should have 16 args:

```text
arg0:  x_in
arg1:  norm_w
arg2:  normed
arg3:  wq_qweight
arg4:  wq_params
arg5:  q
arg6:  wk_qweight
arg7:  wk_params
arg8:  k
arg9:  wv_qweight
arg10: wv_params
arg11: v
arg12: lut_q
arg13: lut_k
arg14: q_roped
arg15: k_roped
```

Stitch maps using AWQ GEMV arg order `(qweight, params, input, output)`:

```python
RMSNorm: {0: 0, 1: 1, 2: 2}
Q AWQ:   {0: 3, 1: 4, 2: 2, 3: 5}
K AWQ:   {0: 6, 1: 7, 2: 2, 3: 8}
V AWQ:   {0: 9, 1: 10, 2: 2, 3: 11}
RoPE Q:  {0: 5, 1: 12, 2: 14}
RoPE K:  {0: 8, 1: 13, 2: 15}
```

### 6B — `o_gemv_ffn_awq_multi.py`

Current BF16 function has 15 args. AWQ version should have 19 args:

```text
arg0:  wo_qweight
arg1:  wo_params
arg2:  attn_out
arg3:  proj
arg4:  x_residual
arg5:  res1
arg6:  ffn_norm_w
arg7:  normed2
arg8:  wgate_qweight
arg9:  wgate_params
arg10: gate
arg11: wup_qweight
arg12: wup_params
arg13: up
arg14: swiglu
arg15: wdown_qweight
arg16: wdown_params
arg17: down
arg18: output
```

Stitch maps:

```python
O AWQ:    {0: 0, 1: 1, 2: 2, 3: 3}
Add1:     {0: 3, 1: 4, 2: 5}
RMSNorm:  {0: 5, 1: 6, 2: 7}
Gate AWQ: {0: 8, 1: 9, 2: 7, 3: 10}
Up AWQ:   {0: 11, 1: 12, 2: 7, 3: 13}
SiLU:     {0: 10, 1: 13, 2: 14}
Down AWQ: {0: 15, 1: 16, 2: 14, 3: 17}
Add2:     {0: 17, 1: 5, 2: 18}
```

Down AWQ must use renamed externs and link with `mv_awq_k8192.o`, mirroring the existing BF16 `dg_` pattern.

### 6C — `lm_head_gemv_awq_multi.py`

Current BF16 lm_head has 17 args:

```text
arg0: input
for p in 0..7:
  arg(1+2p): weight_p
  arg(2+2p): output_p
```

AWQ lm_head should have 25 args:

```text
arg0: input
for p in 0..7:
  arg(1+3p): qweight_p
  arg(2+3p): params_p
  arg(3+3p): output_p
```

Per-partition AWQ stitch map:

```python
{0: 1 + 3*p, 1: 2 + 3*p, 2: 0, 3: 3 + 3*p}
```

**Backend presets:**

Add:

- `RGR_AWQ_BACKEND`
- `OGF_AWQ_BACKEND`
- `LM_GEMV_AWQ_BACKEND`

Start from BF16 presets, but do not assume pingpong settings are optimal. If AWQ L1/L2 pressure differs, adjust based on compile/run results.

**Verification:**

For each AWQ multi-launch module:

```bash
python3 multi_launch_builder/rms_gemv_rope_awq_multi.py --compile-mode compile-and-run
python3 multi_launch_builder/o_gemv_ffn_awq_multi.py --compile-mode compile-and-run
python3 multi_launch_builder/lm_head_gemv_awq_multi.py --compile-mode compile-and-run
```

Expected:

- Module parses.
- Compiles to ELF.
- Links AWQ object files.
- One-layer output matches CPU AWQ reference.
- Compile time remains in the same rough order as BF16. If it exceeds 90s substantially, investigate before proceeding.

---

## Stage 7 — Cache/object plumbing and compile path wiring

**Objective:** Ensure AWQ external objects, cache artifacts, and compile paths are robust and do not collide with BF16.

**Files modified:**

- `programming_examples/llama32_1b/kernel_builder/cache.py`
- `programming_examples/llama32_1b/kernel_builder/external_kernels.py`
- `programming_examples/llama32_1b/llama32_1b_decode.py`
- `programming_examples/llama32_1b/llama32_1b_inference.py`

**Tasks:**

1. Add AWQ objects to `prepare_air_project()` copy list:

```python
"mv_awq.o",
"mv_awq_k8192.o",
```

2. Add AWQ compile functions to `external_kernels.py`.

3. Make compile entrypoints quant-aware:

```python
def compile_decode_kernels(cache, config, quant="bf16"):
    if quant == "bf16": compile existing kernels
    if quant == "awq": compile AWQ builders and cache under AWQ names
```

4. Use distinct cache artifact names:

- `rms_gemv_rope_awq`
- `o_gemv_ffn_awq`
- `lm_head_gemv_awq`

Do not reuse BF16 artifact names.

5. Use distinct per-layer BO keys:

- `rms_gemv_rope_awq_L{i}`
- `o_gemv_ffn_awq_L{i}`

Do not collide with BF16 BO keys.

**Verification:**

```bash
make compile QUANT=bf16
make compile QUANT=awq AWQ_WEIGHTS=./awq_repacked
```

Expected:

- BF16 compile still works.
- AWQ compile produces distinct cached artifacts.
- `air_project/` contains needed AWQ objects during compile.
- No stale/missing `link_with` errors.

---

## Stage 8 — AWQ decode BO preload and runtime wiring

**Objective:** Run full decode with AWQ GEMVs and verify static weights are preloaded once.

**Files modified:**

- `programming_examples/llama32_1b/llama32_1b_inference.py`
- `programming_examples/llama32_1b/llama32_1b_decode.py`

**Preload logic:**

Branch in `_preload_decode_weights` or a new `_preload_decode_weights_awq`.

For each layer, preload:

RMS/QKV block:

- norm weight remains BF16 static input
- `wq.qweight`, `wq.params`
- `wk.qweight`, `wk.params`
- `wv.qweight`, `wv.params`

O/FFN block:

- `wo.qweight`, `wo.params`
- ffn norm remains BF16 static input
- `w_gate.qweight`, `w_gate.params`
- `w_up.qweight`, `w_up.params`
- `w_down.qweight`, `w_down.params`

LM head:

- 8 partitions, each with `qweight`, `params`, output buffer

**Static input indices:**

For `rms_gemv_rope_awq` arg layout from Stage 6A:

```python
static_input_indices={1, 3, 4, 6, 7, 9, 10}
```

For `o_gemv_ffn_awq` arg layout from Stage 6B:

```python
static_input_indices={0, 1, 6, 8, 9, 11, 12, 15, 16}
```

For `lm_head_gemv_awq` arg layout from Stage 6C:

```python
static_input_indices={1 + 3*p for p in range(8)} | {2 + 3*p for p in range(8)}
```

Double-check these indices during implementation. One wrong index can silently rewrite weights every token or fail to update dynamic inputs.

**Runtime branch:**

In `run_decode_block`, branch on `weights.is_awq`:

- BF16 path remains unchanged.
- AWQ path calls:
  - `rms_gemv_rope_awq`
  - CPU attention unchanged
  - `o_gemv_ffn_awq`

LM head decode path should similarly branch to `lm_head_gemv_awq`.

**Verification:**

1. Full forward for one decode token:

```bash
make verify QUANT=awq AWQ_WEIGHTS=./awq_repacked N_TOKENS=1
```

2. Static write check:

Use profiler output from `KernelCache.load_and_run` to confirm that after preload, per-token writes do not include static qweight/params BOs.

Expected:

- Full decode completes.
- Output logits correlate with CPU mixed-mode AWQ reference.
- Static qweight/params are not rewritten per token.
- BF16 path still runs after AWQ path compile/run.

---

## Stage 9 — CPU mixed-mode AWQ reference and `make verify`

**Objective:** Make verification compare runtime against the exact mixed-mode design.

**Files modified:**

- `programming_examples/llama32_1b/llama32_1b_reference.py`
- `programming_examples/llama32_1b/llama32_1b_inference.py`
- `programming_examples/llama32_1b/Makefile`

**Reference behavior for Strategy B:**

- Prefill uses BF16 dequantized AWQ weights, matching runtime prefill.
- Decode linears use CPU dequant from the same repacked AWQ tensors before F32 matmul, matching AWQ decode numerically as closely as practical.
- LM head uses AWQ dequant if runtime uses AWQ lm_head.
- CPU attention and KV cache behavior match runtime.

**Verification levels:**

1. Per-linear correlation/absolute error for representative layers.
2. One-layer transformer block output correlation.
3. Full decode logits correlation.
4. Greedy token match as a final but not sole criterion.

**Gates:**

For `make verify QUANT=awq N_TOKENS=10`:

- logits correlation >= 0.95 initially;
- if stable, tighten after observing actual drift;
- greedy token IDs should match for deterministic short prompts, but investigate rather than blindly fail if logits are close and token boundary is unstable.

**Command:**

```bash
make verify QUANT=awq AWQ_WEIGHTS=./awq_repacked N_TOKENS=10 PROMPT="What is the capital of France?"
```

Expected:

- Verification reports mixed-mode AWQ explicitly.
- No accidental comparison against original BF16 reference.
- Drift source is attributable if tolerance needs adjustment.

---

## Stage 10 — Benchmark and tune

**Objective:** Measure real end-to-end decode speed and determine whether AWQ integration achieved the target.

**Commands:**

```bash
source /home/jfifield/npu-dev-air/env.sh
cd /home/jfifield/npu-dev-air/mlir-air/programming_examples/llama32_1b

python3 tools/repack_awq.py \
  --src /home/jfifield/npu-dev-air/gpu-awq-onnx-bench/models/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead \
  --dst ./awq_repacked

make compile QUANT=awq AWQ_WEIGHTS=./awq_repacked
make verify QUANT=awq AWQ_WEIGHTS=./awq_repacked N_TOKENS=10
make profile QUANT=awq AWQ_WEIGHTS=./awq_repacked N_TOKENS=100 MODEL=base

make profile QUANT=bf16 N_TOKENS=100 MODEL=base
```

**Benchmark requirements:**

- Run on a quiet system.
- Capture per-token and per-kernel breakdown.
- Report write/read/kernel times separately.
- Confirm static AWQ weights are not rewritten after preload.

**Targets:**

Minimum viable success:

- End-to-end decode at least 2x faster than local BF16 baseline under same conditions.

Original target:

- <= 45 ms/token locally.
- <= 25 ms/token in docs-equivalent environment.

If target is missed:

1. Determine whether bottleneck is:
   - fused AWQ GEMV kernel time;
   - BO write/read overhead;
   - CPU attention;
   - LM head;
   - compile/backend scheduling issue.
2. If fused GEMV is the bottleneck, inspect generated code and try one focused tuning pass.
3. If unpack/dequant dominates and cannot be fixed quickly, pivot to W8A16.
4. If LM head AWQ is unstable or too slow, test BF16/tied lm_head fallback and quantify cost.

---

## Files summary

### Create

- `programming_examples/llama32_1b/tools/inspect_awq_format.py`
- `programming_examples/llama32_1b/tools/awq_format_reference.py`
- `programming_examples/llama32_1b/tools/repack_awq.py`
- `programming_examples/llama32_1b/tools/test_repack_awq.py`
- `programming_examples/llama32_1b/llama32_1b_awq.py` if a separate AWQ module is cleaner
- `programming_examples/llama32_1b/kernel_builder/mv_awq.cc`
- `programming_examples/llama32_1b/kernel_builder/mv_awq_test.py`
- `programming_examples/llama32_1b/kernel_builder/gemv_awq_builder.py`
- `programming_examples/llama32_1b/multi_launch_builder/rms_gemv_rope_awq_multi.py`
- `programming_examples/llama32_1b/multi_launch_builder/o_gemv_ffn_awq_multi.py`
- `programming_examples/llama32_1b/multi_launch_builder/lm_head_gemv_awq_multi.py`

### Modify

- `setup.py` — env.sh PYTHONPATH generation
- `programming_examples/llama32_1b/llama32_1b_weights.py` — AWQ loader/data model
- `programming_examples/llama32_1b/llama32_1b_inference.py` — CLI, quant branching, preload
- `programming_examples/llama32_1b/llama32_1b_decode.py` — AWQ decode branch
- `programming_examples/llama32_1b/llama32_1b_reference.py` — mixed-mode AWQ CPU reference
- `programming_examples/llama32_1b/kernel_builder/external_kernels.py` — compile AWQ objects
- `programming_examples/llama32_1b/kernel_builder/cache.py` — copy AWQ objects into air_project
- `programming_examples/llama32_1b/kernel_builder/backend_presets.py` — AWQ backend presets
- `programming_examples/llama32_1b/Makefile` — `QUANT=bf16|awq`, `AWQ_WEIGHTS`

### Read-only references

- `programming_examples/dequant_awq/dequant.cc` — scalar dequant arithmetic example only
- `programming_examples/dequant_awq/dequant_awq.py` — test harness pattern only
- `programming_examples/matrix_vector_multiplication/bf16/mv.cc` — BF16 GEMV external kernel structure
- `programming_examples/matrix_vector_multiplication/bf16/matvec.py` — AIR GEMV tiling/DMA pattern
- `programming_examples/llama32_1b/multi_launch_builder/rms_gemv_rope_multi.py` — current arg order/stitch maps
- `programming_examples/llama32_1b/multi_launch_builder/o_gemv_ffn_multi.py` — K=8192 rename/link pattern
- `programming_examples/llama32_1b/multi_launch_builder/lm_head_gemv_multi.py` — LM head partition stitching
- `programming_examples/llama32_1b/kernel_builder/cache.py` — object copy list and profiler

---

## Verification table

| Stage | Gate |
|---|---|
| 0 | `source env.sh && python -c 'import air, aie'` succeeds |
| 1 | AWQ unpack/qzero semantics match independent oracle for q_proj, down_proj, lm_head |
| 2 | Repacked tensors dequant to same weights/linear outputs as Stage 1 oracle |
| 3 | AWQ loader populates BF16 prefill weights + packed decode weights; CPU-dequant smoke path runs |
| 4 | Standalone fused AWQ GEMV is correct and >=2x faster than BF16 for `(M=2048,K=2048)` |
| 5 | Single AWQ GEMV AIR module compiles/runs for K=2048 and K=8192 shapes |
| 6 | AWQ multi-launch ELFs compile/run and match one-layer CPU references |
| 7 | AWQ object files are compiled/copied; BF16 and AWQ cache artifacts do not collide |
| 8 | Full AWQ decode forward completes; static qweight/params are not rewritten per token |
| 9 | `make verify QUANT=awq N_TOKENS=10` passes mixed-mode AWQ reference gates |
| 10 | `make profile QUANT=awq N_TOKENS=100` shows >=2x end-to-end decode speedup, stretch <=45 ms/token local |

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---:|---|
| qzeros offset/reorder semantics wrong | High | Stage 1 independent oracle gate before repack/integration |
| Repack verification is circular | Medium | Tiny hand-authored tests + independent AWQ oracle linear-output checks |
| Decode-only AWQ lacks BF16 prefill weights | High | Strategy B: CPU-dequant AWQ weights to BF16 for prefill; document mixed mode |
| LM head silently falls back to tied embeddings | Medium | Explicit AWQ lm_head handling; error/warn if missing |
| Fused int4 dequant+GEMV does not vectorize well | Medium/High | Stage 4 microbenchmark gate; inspect generated code; pivot if <2x |
| Static AWQ weights rewritten every token due to bad indices | Medium | Add profiling gate checking bytes_written/n_written after preload |
| AWQ object files missing from air_project | Medium | Update `cache.py` object copy list and compile path in Stage 7 |
| BF16 and AWQ cache/BO keys collide | Medium | Distinct artifact names and BO keys with `_awq` suffix |
| Arg maps wrong in stitched modules | Medium | Use explicit AWQ arg convention and standalone compile/run per module |
| 25-arg LM head AWQ hits backend fragility | Low/Medium | Test LM head builder separately; fallback to BF16/tied lm_head if necessary |
| Accuracy drift from AWQ is visibly worse | Medium | Mixed-mode CPU reference, logit correlation gates, optional better quant recipe later |

---

## Implementation order summary

Do not start with full integration. The safe order is:

1. Fix env imports.
2. Prove AWQ unpack/qzero semantics against an independent oracle.
3. Repack and verify non-circularly.
4. Load AWQ and run CPU-dequant smoke path.
5. Build fused AWQ GEMV and benchmark it.
6. Only if Stage 4 passes, build AIR AWQ GEMV and multi-launch modules.
7. Wire cache/preload/runtime.
8. Verify full mixed-mode AWQ decode.
9. Benchmark and tune.

The plan succeeds only if the Stage 4 fused kernel actually buys bandwidth. Everything before that stage exists to ensure the kernel is tested against the right packed format; everything after that stage should wait until the speed gate passes.
