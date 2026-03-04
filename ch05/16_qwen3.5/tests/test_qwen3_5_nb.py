# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import importlib
import sys
from pathlib import Path

import pytest
import torch

from llms_from_scratch.utils import import_definitions_from_notebook


def _import_qwen3_5_classes():
    try:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        return Qwen3_5TextConfig, Qwen3_5ForCausalLM
    except Exception:
        repo_root = Path(__file__).resolve().parents[3]
        local_src = repo_root / "transformers-main" / "src"
        if not local_src.exists():
            raise

        for name in list(sys.modules):
            if name == "transformers" or name.startswith("transformers."):
                del sys.modules[name]
        sys.path.insert(0, str(local_src))

        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        return Qwen3_5TextConfig, Qwen3_5ForCausalLM


transformers_installed = importlib.util.find_spec("transformers") is not None
if transformers_installed:
    try:
        Qwen3_5TextConfig, Qwen3_5ForCausalLM = _import_qwen3_5_classes()
    except Exception:
        transformers_installed = False
        Qwen3_5TextConfig, Qwen3_5ForCausalLM = None, None
else:
    Qwen3_5TextConfig, Qwen3_5ForCausalLM = None, None


@pytest.fixture
def import_notebook_defs():
    nb_dir = Path(__file__).resolve().parents[1]
    if str(nb_dir) not in sys.path:
        sys.path.insert(0, str(nb_dir))

    mod = import_definitions_from_notebook(nb_dir, "qwen3.5.ipynb")
    return mod


@pytest.fixture
def dummy_input():
    torch.manual_seed(123)
    return torch.randint(0, 100, (1, 8))


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
        "qk_norm": False,
        "dtype": torch.float32,
        "rope_base": 10_000.0,
        "context_length": 64,
        "partial_rotary_factor": 1.0,
        "rms_norm_eps": 1e-6,
        "linear_conv_kernel_dim": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "layer_types": ["linear_attention", "full_attention"],
    }


@torch.inference_mode()
def test_dummy_qwen3_5_forward(dummy_cfg_base, dummy_input, import_notebook_defs):
    torch.manual_seed(123)
    model = import_notebook_defs.Qwen3_5Model(dummy_cfg_base)
    out = model(dummy_input)
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_base["vocab_size"]), (
        f"Expected shape (1, seq_len, vocab_size), got {out.shape}"
    )


@torch.inference_mode()
@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_qwen3_5_base_equivalence_with_transformers(import_notebook_defs):
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
        "partial_rotary_factor": 1.0,
        "rms_norm_eps": 1e-6,
        "linear_conv_kernel_dim": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "dtype": torch.float32,
    }
    model = import_notebook_defs.Qwen3_5Model(cfg)

    hf_cfg = Qwen3_5TextConfig(
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["context_length"],
        hidden_size=cfg["emb_dim"],
        num_attention_heads=cfg["n_heads"],
        num_hidden_layers=cfg["n_layers"],
        intermediate_size=cfg["hidden_dim"],
        head_dim=cfg["head_dim"],
        num_key_value_heads=cfg["n_kv_groups"],
        layer_types=cfg["layer_types"],
        linear_conv_kernel_dim=cfg["linear_conv_kernel_dim"],
        linear_key_head_dim=cfg["linear_key_head_dim"],
        linear_value_head_dim=cfg["linear_value_head_dim"],
        linear_num_key_heads=cfg["linear_num_key_heads"],
        linear_num_value_heads=cfg["linear_num_value_heads"],
        tie_word_embeddings=False,
        use_cache=False,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": cfg["rope_base"],
            "partial_rotary_factor": cfg["partial_rotary_factor"],
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
        },
        torch_dtype=torch.float32,
    )
    hf_cfg._attn_implementation = "eager"
    hf_model = Qwen3_5ForCausalLM(hf_cfg)

    hf_state = hf_model.state_dict()
    param_config = {"n_layers": cfg["n_layers"], "layer_types": cfg["layer_types"]}
    import_notebook_defs.load_weights_into_qwen3_5(model, param_config, hf_state)

    x = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]), dtype=torch.long)
    ours_logits = model(x)
    theirs_logits = hf_model(x, use_cache=False).logits
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)
