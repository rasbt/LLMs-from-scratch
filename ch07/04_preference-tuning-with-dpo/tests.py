# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# File for internal use (unit tests)

import importlib.util
from pathlib import Path

import torch


def load_previous_chapters():
    path = Path(__file__).resolve().parent / "previous_chapters.py"
    spec = importlib.util.spec_from_file_location("dpo_previous_chapters", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GPT_CONFIG = {
    "vocab_size": 50,
    "context_length": 16,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


def make_model(module):
    torch.manual_seed(123)
    model = module.GPTModel(GPT_CONFIG)
    model.eval()
    return model


def test_generate_with_temperature_samples_valid_tokens():
    module = load_previous_chapters()
    model = make_model(module)
    prompt = torch.tensor([[1, 2, 3]])

    torch.manual_seed(123)
    out = module.generate(
        model=model,
        idx=prompt,
        max_new_tokens=5,
        context_size=GPT_CONFIG["context_length"],
        temperature=1.0,
        top_k=10,
    )

    assert out.shape == (1, 8)
    assert torch.equal(out[:, :3], prompt)
    assert ((out >= 0) & (out < GPT_CONFIG["vocab_size"])).all()


def test_generate_with_temperature_and_top_k_1_matches_greedy():
    module = load_previous_chapters()
    model = make_model(module)
    prompt = torch.tensor([[1, 2, 3]])

    greedy = module.generate(
        model=model,
        idx=prompt,
        max_new_tokens=5,
        context_size=GPT_CONFIG["context_length"],
    )
    sampled = module.generate(
        model=model,
        idx=prompt,
        max_new_tokens=5,
        context_size=GPT_CONFIG["context_length"],
        temperature=1.0,
        top_k=1,
    )

    assert torch.equal(sampled, greedy)


def test_generate_with_temperature_follows_softmax_distribution():
    module = load_previous_chapters()
    model = make_model(module)
    temperature = 2.0
    num_samples = 20_000
    prompt = torch.tensor([[1, 2, 3]])

    with torch.no_grad():
        logits = model(prompt)[:, -1, :]
    expected = torch.softmax(logits / temperature, dim=-1)[0]

    torch.manual_seed(123)
    out = module.generate(
        model=model,
        idx=prompt.repeat(num_samples, 1),
        max_new_tokens=1,
        context_size=GPT_CONFIG["context_length"],
        temperature=temperature,
    )
    counts = torch.bincount(out[:, -1], minlength=GPT_CONFIG["vocab_size"])
    observed = counts.float() / num_samples

    assert torch.allclose(observed, expected, atol=0.01)
