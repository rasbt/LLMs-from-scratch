# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# File for internal use (unit tests)

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent


def load_additional_experiments_module(monkeypatch):
    # Stub out gpt_download so that importing the script does not require
    # TensorFlow; the tests below do not download any weights.
    gpt_download_stub = types.ModuleType("gpt_download")
    gpt_download_stub.download_and_load_gpt2 = None
    monkeypatch.setitem(sys.modules, "gpt_download", gpt_download_stub)
    monkeypatch.syspath_prepend(str(THIS_DIR))

    script_path = THIS_DIR / "additional_experiments.py"
    spec = importlib.util.spec_from_file_location("additional_experiments", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_cfg():
    return {
        "vocab_size": 50,
        "context_length": 8,
        "emb_dim": 16,
        "n_heads": 2,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }


@pytest.mark.parametrize("alternative", [False, True])
def test_replace_linear_with_lora_wraps_all_linear_layers(monkeypatch, alternative):
    module = load_additional_experiments_module(monkeypatch)
    expected_cls = module.LinearWithLoRAMerged if alternative else module.LinearWithLoRA

    torch.manual_seed(123)
    model = module.GPTModel(small_cfg())
    num_linear_layers = sum(isinstance(m, torch.nn.Linear) for m in model.modules())
    module.replace_linear_with_lora(model, rank=2, alpha=2, alternative=alternative)

    lora_layers = [m for m in model.modules()
                   if isinstance(m, (module.LinearWithLoRA, module.LinearWithLoRAMerged))]
    assert len(lora_layers) == num_linear_layers
    assert all(type(m) is expected_cls for m in lora_layers)
