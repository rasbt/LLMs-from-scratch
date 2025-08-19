# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import importlib
from pathlib import Path

import pytest
import torch

from llms_from_scratch.utils import import_definitions_from_notebook


transformers_installed = importlib.util.find_spec("transformers") is not None


@pytest.fixture
def nb_imports():
    nb_dir = Path(__file__).resolve().parents[1]
    mod = import_definitions_from_notebook(nb_dir, "standalone-gemma3-plus-kvcache.ipynb")
    return mod


@pytest.fixture
def dummy_input():
    torch.manual_seed(123)
    return torch.randint(0, 100, (1, 8))  # batch size 1, seq length 8


@pytest.fixture
def dummy_cfg_base():
    return {
        "vocab_size": 100,
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_layers": 2,
        "n_heads": 4,
        "head_dim": 8,
        "n_kv_groups": 1,
        "qk_norm": True,                # Gemma3 uses q/k RMSNorm
        "dtype": torch.float32,
        "rope_base": 1_000_000.0,       # global RoPE base
        "rope_local_base": 10_000.0,    # local RoPE base (unused in these tests)
        "context_length": 64,
        "sliding_window": 16,
        "layer_types": ["full_attention", "full_attention"],
        "query_pre_attn_scalar": 256,
    }


@torch.inference_mode()
def test_dummy_gemma3_forward(dummy_cfg_base, dummy_input, nb_imports):
    torch.manual_seed(123)
    model = nb_imports.Gemma3Model(dummy_cfg_base)
    out = model(dummy_input)
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_base["vocab_size"])


@torch.inference_mode()
@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_gemma3_base_equivalence_with_transformers(nb_imports):
    from transformers import Gemma3TextConfig, Gemma3ForCausalLM

    # Tiny config so the test is fast
    cfg = {
        "vocab_size": 257,
        "context_length": 8,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "hidden_dim": 64,
        "head_dim": 8,
        "qk_norm": True,
        "n_kv_groups": 2,
        "rope_base": 1_000_000.0,
        "rope_local_base": 10_000.0,
        "sliding_window": 4,
        "layer_types": ["full_attention", "full_attention"],
        "dtype": torch.float32,
        "query_pre_attn_scalar": 256,
    }
    model = nb_imports.Gemma3Model(cfg)

    hf_cfg = Gemma3TextConfig(
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["context_length"],
        hidden_size=cfg["emb_dim"],
        num_attention_heads=cfg["n_heads"],
        num_hidden_layers=cfg["n_layers"],
        intermediate_size=cfg["hidden_dim"],
        head_dim=cfg["head_dim"],
        num_key_value_heads=cfg["n_kv_groups"],
        rope_theta=cfg["rope_base"],
        rope_local_base_freq=cfg["rope_local_base"],
        layer_types=cfg["layer_types"],
        sliding_window=cfg["sliding_window"],
        tie_word_embeddings=False,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
        rope_scaling={"rope_type": "default"},
    )
    hf_model = Gemma3ForCausalLM(hf_cfg)

    hf_state = hf_model.state_dict()
    param_config = {"n_layers": cfg["n_layers"], "hidden_dim": cfg["hidden_dim"]}
    nb_imports.load_weights_into_gemma(model, param_config, hf_state)

    x = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]), dtype=torch.long)
    ours_logits = model(x)
    theirs_logits = hf_model(x).logits
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)
