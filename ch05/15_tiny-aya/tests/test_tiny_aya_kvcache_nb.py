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
def import_notebook_defs():
    nb_dir = Path(__file__).resolve().parents[1]
    mod = import_definitions_from_notebook(nb_dir, "standalone-tiny-aya-plus-kv-cache.ipynb")
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
        "n_kv_heads": 1,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_base": 10_000.0,
        "layer_norm_eps": 1e-5,
        "logit_scale": 1.0,
        "tie_word_embeddings": False,
        "dtype": torch.float32,
    }


@torch.inference_mode()
def test_dummy_tiny_aya_forward(dummy_cfg_base, dummy_input, import_notebook_defs):
    torch.manual_seed(123)
    model = import_notebook_defs.TinyAyaModel(dummy_cfg_base)
    out = model(dummy_input)
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_base["vocab_size"]), \
        f"Expected shape (1, seq_len, vocab_size), got {out.shape}"


@torch.inference_mode()
@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_tiny_aya_base_equivalence_with_transformers(import_notebook_defs):
    from transformers import Cohere2Config, Cohere2ForCausalLM

    # Tiny config so the test is fast
    cfg = {
        "vocab_size": 257,
        "context_length": 8,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "hidden_dim": 64,
        "head_dim": 8,
        "n_kv_heads": 2,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "dtype": torch.float32,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "layer_norm_eps": 1e-5,
        "rope_base": 10_000.0,
        "logit_scale": 1.0,
        "tie_word_embeddings": False,
    }

    model = import_notebook_defs.TinyAyaModel(cfg)

    hf_cfg = Cohere2Config(
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["context_length"],
        hidden_size=cfg["emb_dim"],
        num_attention_heads=cfg["n_heads"],
        num_hidden_layers=cfg["n_layers"],
        intermediate_size=cfg["hidden_dim"],
        num_key_value_heads=cfg["n_kv_heads"],
        attention_bias=cfg["attention_bias"],
        attention_dropout=cfg["attention_dropout"],
        layer_norm_eps=cfg["layer_norm_eps"],
        layer_types=cfg["layer_types"],
        sliding_window=cfg["sliding_window"],
        logit_scale=cfg["logit_scale"],
        tie_word_embeddings=cfg["tie_word_embeddings"],
        rope_parameters={"rope_type": "default", "rope_theta": cfg["rope_base"]},
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    hf_model = Cohere2ForCausalLM(hf_cfg)

    hf_state = hf_model.state_dict()
    import_notebook_defs.load_weights_into_tiny_aya(model, cfg, hf_state)

    x = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]), dtype=torch.long)
    ours_logits = model(x)
    theirs_logits = hf_model(x).logits
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)
