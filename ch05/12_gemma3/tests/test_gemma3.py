# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import importlib
import types
import re
from pathlib import Path

import nbformat
import pytest
import torch

transformers_installed = importlib.util.find_spec("transformers") is not None


def _extract_defs_and_classes_from_code(src):
    lines = src.splitlines()
    kept = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        # Keep decorators attached to the next def/class
        if stripped.startswith("@"):
            # Look ahead: if the next non-empty line starts with def/class, keep decorator
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].lstrip().startswith(("def ", "class ")):
                kept.append(line)
                i += 1
                continue
        if stripped.startswith("def ") or stripped.startswith("class "):
            kept.append(line)
            # capture until we leave the indentation block
            base_indent = len(line) - len(stripped)
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.strip() == "":
                    kept.append(nxt)
                    i += 1
                    continue
                indent = len(nxt) - len(nxt.lstrip())
                if indent <= base_indent and not nxt.lstrip().startswith(("#", "@")):
                    break
                kept.append(nxt)
                i += 1
            continue
        i += 1
    code = "\n".join(kept)
    code = re.sub(r"def\s+load_weights_into_gemma\s*\(\s*Gemma3Model\s*,",
                  "def load_weights_into_gemma(model,",
                  code)
    return code


def import_definitions_from_notebook(nb_dir_or_path, notebook_name):
    nb_path = Path(nb_dir_or_path)
    if nb_path.is_dir():
        nb_file = nb_path / notebook_name
    else:
        nb_file = nb_path
    if not nb_file.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_file}")

    nb = nbformat.read(nb_file, as_version=4)
    pieces = ["import torch", "import torch.nn as nn"]
    for cell in nb.cells:
        if cell.cell_type == "code":
            pieces.append(_extract_defs_and_classes_from_code(cell.source))
    src = "\n\n".join(pieces)

    mod = types.ModuleType("gemma3_defs")
    exec(src, mod.__dict__)
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
def test_dummy_gemma3_forward(dummy_cfg_base, dummy_input):
    nb_dir = Path(__file__).resolve().parents[1]
    mod = import_definitions_from_notebook(nb_dir, "standalone-gemma3.ipynb")
    Gemma3Model = mod.Gemma3Model

    torch.manual_seed(123)
    model = Gemma3Model(dummy_cfg_base)
    out = model(dummy_input)
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_base["vocab_size"]),         f"Expected shape (1, seq_len, vocab_size), got {out.shape}"


@torch.inference_mode()
@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_gemma3_base_equivalence_with_transformers():
    nb_dir = Path(__file__).resolve().parents[1]
    mod = import_definitions_from_notebook(nb_dir, "standalone-gemma3.ipynb")
    Gemma3Model = mod.Gemma3Model
    load_weights_into_gemma = mod.load_weights_into_gemma

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
    model = Gemma3Model(cfg)

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
    load_weights_into_gemma(model, param_config, hf_state)

    x = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]), dtype=torch.long)
    ours_logits = model(x)
    theirs_logits = hf_model(x).logits
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)
