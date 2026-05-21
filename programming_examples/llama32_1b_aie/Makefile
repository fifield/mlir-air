# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# LLAMA-3.2-1B BF16 Inference on NPU2 (MLIR-AIE port)
#
# Low-level duplicate of ../llama32_1b. The kernels are compiled by aiecc
# directly (no aircc in the steady state). Each unique kernel's cached IR
# lives at <build>/decode_kernel_cache/<name>.npu.air.mlir and can be
# hand-edited; rerun `make compile` after editing to rebuild the ELF.
#
# Quick start:
#   make compile        - Compile all kernels (~3 min, cached to disk)
#   make run            - Run inference (NPU prefill + NPU decode)
#   make profile        - Run with per-kernel profiling breakdown

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

# WEIGHTS=hf (default)        - load real Meta weights from HuggingFace
# WEIGHTS=synthetic           - deterministic random weights (no HF, for CI)
WEIGHTS ?= hf
ifeq ($(WEIGHTS),synthetic)
  WEIGHTS_FLAG := --synthetic-weights
else
  WEIGHTS_FLAG :=
endif

# Optional: override the HF repo (or pass a local dir). Useful for non-gated
# mirrors, e.g. HF_MODEL_ID=unsloth/Llama-3.2-1B-Instruct
HF_MODEL_ID ?=
ifneq ($(HF_MODEL_ID),)
  HF_MODEL_FLAG := --hf-model-id $(HF_MODEL_ID)
else
  HF_MODEL_FLAG :=
endif

.PHONY: help compile run profile verify chat clean

help:
	@echo "============================================================"
	@echo " LLAMA-3.2-1B Inference on NPU2 (MLIR-AIE port)"
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
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove all build artifacts"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  N_TOKENS=1000         Max decode tokens (instruct model stops early on EOT)"
	@echo "  PROMPT=\"...\"          Input prompt text"
	@echo "  MODEL=base|instruct   Model variant (default: instruct)"
	@echo ""
	@echo "Examples:"
	@echo "  make run N_TOKENS=50"
	@echo "  make run MODEL=base PROMPT=\"The capital of France is\" N_TOKENS=200"
	@echo "  make profile PROMPT=\"How does photosynthesis work?\""

compile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py --compile-only

run:
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" --model $(MODEL) $(WEIGHTS_FLAG) $(HF_MODEL_FLAG)

profile:
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --profile --prompt "$(PROMPT)" --model $(MODEL) $(WEIGHTS_FLAG) $(HF_MODEL_FLAG)

verify:
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --verify --profile --prompt "$(PROMPT)" --model $(MODEL) $(WEIGHTS_FLAG) $(HF_MODEL_FLAG)

chat:
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py \
		--run-only --interactive --n-tokens $(N_TOKENS) --model $(MODEL) $(WEIGHTS_FLAG) $(HF_MODEL_FLAG)

all: compile profile

clean:
	rm -r $(BUILD_DIR) 2>/dev/null || true
	@echo "Build directory removed. Run 'make compile' to rebuild."
