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
    mod = import_definitions_from_notebook(nb_dir, "standalone-llama32.ipynb")
    return mod


@pytest.fixture
def dummy_input():
    torch.manual_seed(123)
    return torch.randint(0, 100, (1, 8))  # batch size 1, seq length 8


@pytest.fixture
def dummy_cfg_base():
    return {
        "vocab_size": 100,
        "emb_dim": 32,            # hidden_size
        "hidden_dim": 64,         # intermediate_size (FFN)
        "n_layers": 2,
        "n_heads": 4,
        "head_dim": 8,
        "n_kv_groups": 1,
        "dtype": torch.float32,
        "rope_base": 500_000.0,
        "rope_freq": {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,
        },
        "context_length": 64,
    }


@torch.inference_mode()
def test_dummy_llama3_forward(dummy_cfg_base, dummy_input, nb_imports):
    torch.manual_seed(123)
    model = nb_imports.Llama3Model(dummy_cfg_base)
    out = model(dummy_input)
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_base["vocab_size"])


@torch.inference_mode()
@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_llama3_base_equivalence_with_transformers(nb_imports):
    from transformers.models.llama import LlamaConfig, LlamaForCausalLM
    cfg = {
        "vocab_size": 257,
        "context_length": 8192,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "hidden_dim": 64,
        "n_kv_groups": 2,
        "rope_base": 500_000.0,
        "rope_freq": {
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,
        },
        "dtype": torch.float32,
    }

    ours = nb_imports.Llama3Model(cfg)

    hf_cfg = LlamaConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["emb_dim"],
        num_attention_heads=cfg["n_heads"],
        num_key_value_heads=cfg["n_kv_groups"],
        num_hidden_layers=cfg["n_layers"],
        intermediate_size=cfg["hidden_dim"],
        max_position_embeddings=cfg["context_length"],
        rms_norm_eps=1e-5,
        attention_bias=False,
        rope_theta=cfg["rope_base"],
        tie_word_embeddings=False,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        rope_scaling={
            "type": "llama3",
            "factor": cfg["rope_freq"]["factor"],
            "low_freq_factor": cfg["rope_freq"]["low_freq_factor"],
            "high_freq_factor": cfg["rope_freq"]["high_freq_factor"],
            "original_max_position_embeddings": cfg["rope_freq"]["original_context_length"],
        },
    )
    theirs = LlamaForCausalLM(hf_cfg)

    hf_state = theirs.state_dict()
    nb_imports.load_weights_into_llama(ours, {"n_layers": cfg["n_layers"], "hidden_dim": cfg["hidden_dim"]}, hf_state)

    x = torch.randint(0, cfg["vocab_size"], (2, 8), dtype=torch.long)
    ours_logits = ours(x)
    theirs_logits = theirs(x).logits.to(ours_logits.dtype)

    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)
