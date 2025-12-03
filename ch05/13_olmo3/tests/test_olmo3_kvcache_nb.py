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
    mod = import_definitions_from_notebook(nb_dir, "standalone-olmo3-plus-kv-cache.ipynb")
    return mod


@pytest.fixture
def dummy_input():
    torch.manual_seed(123)
    return torch.randint(0, 100, (1, 8))  # batch size 1, seq length 8


@pytest.fixture
def dummy_cfg_base():
    return {
        "vocab_size": 100,
        "context_length": 64,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "hidden_dim": 64,
        "head_dim": 8,
        "n_kv_heads": 1,  # 4 query heads, 1 KV groups -> group_size = 4
        "attention_bias": False,
        "attention_dropout": 0.0,
        "sliding_window": 4,
        "layer_types": ["full_attention"] * 2,

        # RoPE config
        "rope_base": 10_000.0,
        "rope_attention_factor": 1.0,
        "rope_type": "default",
        "rope_factor": 1.0,
        "rope_orig_max": 64,
        "rms_norm_eps": 1e-6,
        "dtype": torch.float32,
    }

@torch.inference_mode()
def test_dummy_olmo3_forward(dummy_cfg_base, dummy_input, nb_imports):
    torch.manual_seed(123)
    model = nb_imports.Olmo3Model(dummy_cfg_base)
    out = model(dummy_input)
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_base["vocab_size"]), \
        f"Expected shape (1, seq_len, vocab_size), got {out.shape}"


@torch.inference_mode()
@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_olmo3_base_equivalence_with_transformers(nb_imports):
    from transformers import Olmo3Config, Olmo3ForCausalLM

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
        "n_kv_heads": 2,
        "sliding_window": 4,
        "layer_types": ["full_attention", "full_attention"],
        "dtype": torch.float32,
        "query_pre_attn_scalar": 256,

        # required by TransformerBlock
        "attention_bias": False,

        # required by RMSNorm and RoPE setup in Olmo3Model
        "rms_norm_eps": 1e-6,
        "rope_base": 1_000_000.0,
        "rope_attention_factor": 1.0,
        "rope_type": "default",
        "rope_factor": 1.0,
        "rope_orig_max": 8,

        # extra HF-only stuff
        "rope_local_base": 10_000.0,
    }

    model = nb_imports.Olmo3Model(cfg)

    hf_cfg = Olmo3Config(
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["context_length"],
        hidden_size=cfg["emb_dim"],
        num_attention_heads=cfg["n_heads"],
        num_hidden_layers=cfg["n_layers"],
        intermediate_size=cfg["hidden_dim"],
        head_dim=cfg["head_dim"],
        num_key_value_heads=cfg["n_kv_heads"],
        rope_theta=cfg["rope_base"],
        rope_local_base_freq=cfg["rope_local_base"],
        layer_types=cfg["layer_types"],
        sliding_window=cfg["sliding_window"],
        tie_word_embeddings=False,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
        rope_scaling={"rope_type": "default"},
        qk_norm=cfg["qk_norm"],
        rms_norm_eps=cfg["rms_norm_eps"],
    )
    hf_model = Olmo3ForCausalLM(hf_cfg)

    hf_state = hf_model.state_dict()
    param_config = {
        "n_layers": cfg["n_layers"],
        "hidden_dim": cfg["hidden_dim"],
    }
    nb_imports.load_weights_into_olmo(model, param_config, hf_state)

    x = torch.randint(
        0,
        cfg["vocab_size"],
        (2, cfg["context_length"]),
        dtype=torch.long,
    )
    ours_logits = model(x)
    theirs_logits = hf_model(x).logits
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)
