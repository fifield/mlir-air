# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# LLAMA-3.2-1B BF16/AWQ Inference on NPU2 (MLIR-AIR)
#
# Quick start:
#   make compile        — Compile all kernels (~3 min, cached to disk)
#   make run            — Run inference: NPU prefill + NPU decode
#   make profile        — Run with per-kernel profiling breakdown

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Build directory
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Configurable parameters
N_TOKENS ?= 1000
PROMPT   ?= What is the capital of France?
MODEL    ?= instruct

# WEIGHTS=hf (default)        — load real Meta weights from HuggingFace
# WEIGHTS=synthetic           — deterministic random weights (no HF, for CI)
WEIGHTS ?= hf
QUANT ?= bf16
AWQ_WEIGHTS ?=

# Quote Make variables for safe shell forwarding. PROMPT is intentionally left
# inside double quotes in recipes so normal make run PROMPT="..." remains tidy.
AWQ_WEIGHTS_Q := $(shell python3 -c 'import shlex,sys; print(shlex.quote(sys.argv[1]))' '$(AWQ_WEIGHTS)')

ifeq ($(WEIGHTS),synthetic)
  WEIGHTS_FLAG := --synthetic-weights
else
  WEIGHTS_FLAG :=
endif
ifeq ($(QUANT),awq)
  QUANT_FLAG := --quant awq --awq-weights $(AWQ_WEIGHTS_Q)
else
  QUANT_FLAG := --quant bf16
endif

.PHONY: help compile run profile verify chat run-awq profile-awq verify-awq clean

# ============================================================
# Help
# ============================================================

help:
	@echo "============================================================"
	@echo " LLAMA-3.2-1B Inference on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compile          Compile all kernels (~4 min, one-time)"
	@echo "  make run              Run inference ($(N_TOKENS) tokens)"
	@echo "  make chat             Interactive chat REPL (streaming output)"
	@echo "  make profile          Run with profiling breakdown"
	@echo ""
	@echo "More targets:"
	@echo "  make verify           With CPU reference verification"
	@echo "  make run-awq          Run NPU pipeline with repacked AWQ weights"
	@echo "  make profile-awq      Profile NPU pipeline with repacked AWQ weights"
	@echo "  make verify-awq       Verify NPU pipeline with repacked AWQ weights"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove all build artifacts"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  N_TOKENS=1000         Max decode tokens (instruct model stops early on EOT)"
	@echo "  PROMPT=\"...\"          Input prompt text"
	@echo "  MODEL=base|instruct   Model variant (default: instruct)"
	@echo "  WEIGHTS=hf|synthetic  Weight source (default: hf)"
	@echo "  QUANT=bf16|awq        Weight format (default: bf16)"
	@echo "  AWQ_WEIGHTS=/path     Repacked AWQ model dir for QUANT=awq"
	@echo ""
	@echo "Examples:"
	@echo "  make run N_TOKENS=50"
	@echo "  make run MODEL=base PROMPT=\"The capital of France is\" N_TOKENS=200"
	@echo "  make profile PROMPT=\"How does photosynthesis work?\""
	@echo "  make run QUANT=awq AWQ_WEIGHTS=./awq_repacked PROMPT=\"Hello\" N_TOKENS=5"
	@echo "  make profile-awq AWQ_WEIGHTS=./awq_repacked PROMPT=\"Hello\" N_TOKENS=5"
	@echo "  make verify N_TOKENS=10"

# ============================================================
# Unified Pipeline (NPU prefill + NPU decode)
# ============================================================

## Compile all kernels (prefill + decode, ~4 min)
compile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py --compile-only $(QUANT_FLAG)

## Run unified inference
run:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" --model $(MODEL) $(WEIGHTS_FLAG) $(QUANT_FLAG)

## Run with detailed profiling breakdown
profile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --profile --prompt "$(PROMPT)" --model $(MODEL) $(WEIGHTS_FLAG) $(QUANT_FLAG)

## Run with CPU reference verification
verify:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --verify --profile --prompt "$(PROMPT)" --model $(MODEL) $(WEIGHTS_FLAG) $(QUANT_FLAG)

## AWQ NPU shortcuts: use repacked AWQ weights through normal NPU pipeline
run-awq:
	$(MAKE) run QUANT=awq WEIGHTS=hf

profile-awq:
	$(MAKE) profile QUANT=awq WEIGHTS=hf

verify-awq:
	$(MAKE) verify QUANT=awq WEIGHTS=hf

## Interactive chat: prepare runtime once, then loop on prompts
chat:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py \
		--run-only --interactive --n-tokens $(N_TOKENS) --model $(MODEL) $(WEIGHTS_FLAG) $(QUANT_FLAG)

## Compile and run in one step
all: compile profile

# ============================================================
# Clean
# ============================================================

## Remove all build artifacts
clean:
	rm -r $(BUILD_DIR) 2>/dev/null || true
	@echo "Build directory removed. Run 'make compile' to rebuild."
