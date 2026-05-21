#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage 3 smoke tests for the AWQ-aware Llama weight loader."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
from ml_dtypes import bfloat16
from safetensors.numpy import save_file

_SCRIPT_DIR = os.path.dirname(__file__)
_EXAMPLE_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _EXAMPLE_DIR)

from repack_awq import repack_module_from_logical, dequant_repacked_awq  # noqa: E402
from llama32_1b_weights import LlamaConfig, load_awq_weights  # noqa: E402
from llama32_1b_reference import transformer_block  # noqa: E402


def _logical_quant(k: int, n: int, offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = (np.arange(k * n, dtype=np.uint8).reshape(k, n) + offset) % 16
    groups = k // 2
    zeros = ((np.arange(groups * n, dtype=np.uint8).reshape(groups, n) + offset) % 4) + 3
    scales = (
        0.05
        + 0.01 * ((np.arange(groups * n, dtype=np.float32).reshape(groups, n) + offset) % 7)
    )
    return q, zeros, scales.astype(np.float32)


def _add_repacked(tensors: dict[str, np.ndarray], prefix: str, k: int, n: int, offset: int):
    q, zeros, scales = _logical_quant(k, n, offset)
    qweight, params = repack_module_from_logical(q, zeros, scales, group_size=2)
    tensors[f"{prefix}.qweight_repacked"] = qweight
    tensors[f"{prefix}.params_interleaved"] = params
    return dequant_repacked_awq(qweight, params, k=k, group_size=2).astype(bfloat16)


def _write_tiny_repacked_awq_model(root: str, config: LlamaConfig):
    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": (np.arange(config.vocab_size * config.emb_dim, dtype=np.float32).reshape(config.vocab_size, config.emb_dim) / 100).astype(bfloat16),
        "model.norm.weight": np.ones(config.emb_dim, dtype=bfloat16),
    }
    expected = {"layers": [], "lm_head": None}
    for layer_idx in range(config.n_layers):
        tensors[f"model.layers.{layer_idx}.input_layernorm.weight"] = np.ones(config.emb_dim, dtype=bfloat16)
        tensors[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = np.ones(config.emb_dim, dtype=bfloat16)
        layer_expected = {}
        specs = {
            "self_attn.q_proj": (config.emb_dim, config.n_heads * config.head_dim, "wq"),
            "self_attn.k_proj": (config.emb_dim, config.n_kv_heads * config.head_dim, "wk"),
            "self_attn.v_proj": (config.emb_dim, config.n_kv_heads * config.head_dim, "wv"),
            "self_attn.o_proj": (config.emb_dim, config.emb_dim, "wo"),
            "mlp.gate_proj": (config.emb_dim, config.hidden_dim, "w_gate"),
            "mlp.up_proj": (config.emb_dim, config.hidden_dim, "w_up"),
            "mlp.down_proj": (config.hidden_dim, config.emb_dim, "w_down"),
        }
        for offset, (suffix, (k, n, field)) in enumerate(specs.items(), start=1):
            rows_nk = _add_repacked(
                tensors,
                f"model.layers.{layer_idx}.{suffix}",
                k,
                n,
                offset + layer_idx * 10,
            )
            layer_expected[field] = np.ascontiguousarray(rows_nk.T)
        expected["layers"].append(layer_expected)
    expected["lm_head"] = _add_repacked(
        tensors,
        "lm_head",
        config.emb_dim,
        config.vocab_size,
        99,
    )
    save_file(tensors, os.path.join(root, "model.safetensors"))
    with open(os.path.join(root, "awq_repack_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"group_size": 2, "modules": sorted(k[:-len(".qweight_repacked")] for k in tensors if k.endswith(".qweight_repacked"))}, handle)
    with open(os.path.join(root, "tokenizer_config.json"), "w", encoding="utf-8") as handle:
        json.dump({"model_max_length": 2048}, handle)
    return expected


def test_load_awq_weights_populates_awq_and_bf16_prefill_fields():
    config = LlamaConfig(
        n_layers=1,
        emb_dim=4,
        n_heads=2,
        head_dim=2,
        n_kv_heads=1,
        hidden_dim=8,
        vocab_size=6,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        expected = _write_tiny_repacked_awq_model(tmpdir, config)
        weights = load_awq_weights(tmpdir, config=config)

    assert weights.is_awq is True
    assert weights.awq_layers is not None and len(weights.awq_layers) == 1
    assert weights.awq_lm_head is not None
    assert weights.awq_layers[0].wq.group_size == 2
    assert weights.awq_layers[0].wq.k == config.emb_dim
    assert weights.awq_layers[0].wq.m == config.emb_dim
    assert weights.layers[0].wq.shape == (config.emb_dim, config.emb_dim)
    assert weights.layers[0].w_down.shape == (config.hidden_dim, config.emb_dim)
    np.testing.assert_allclose(weights.layers[0].wq.astype(np.float32), expected["layers"][0]["wq"].astype(np.float32), rtol=0, atol=0)
    np.testing.assert_allclose(weights.layers[0].w_down.astype(np.float32), expected["layers"][0]["w_down"].astype(np.float32), rtol=0, atol=0)
    np.testing.assert_allclose(weights.lm_head.astype(np.float32), expected["lm_head"].astype(np.float32), rtol=0, atol=0)


def test_awq_dequantized_weights_run_cpu_reference_block():
    config = LlamaConfig(
        n_layers=1,
        emb_dim=4,
        n_heads=2,
        head_dim=2,
        n_kv_heads=1,
        hidden_dim=8,
        vocab_size=6,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_tiny_repacked_awq_model(tmpdir, config)
        weights = load_awq_weights(tmpdir, config=config)

    x = (np.arange(12, dtype=np.float32).reshape(3, 4) / 10).astype(bfloat16)
    rope_lut = np.ones((3, config.head_dim), dtype=bfloat16)
    y, intermediates = transformer_block(x, weights.layers[0], rope_lut, config)

    assert y.shape == (3, config.emb_dim)
    assert "q" in intermediates
    assert np.isfinite(y.astype(np.float32)).all()


def main() -> int:
    test_load_awq_weights_populates_awq_and_bf16_prefill_fields()
    print("PASS test_load_awq_weights_populates_awq_and_bf16_prefill_fields")
    test_awq_dequantized_weights_run_cpu_reference_block()
    print("PASS test_awq_dequantized_weights_run_cpu_reference_block")
    print("PASS test_awq_weight_loader")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
