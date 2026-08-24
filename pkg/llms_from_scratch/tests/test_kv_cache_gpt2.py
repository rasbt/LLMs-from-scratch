# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).

import pytest
import torch

from llms_from_scratch.kv_cache.generate import generate_text_simple, generate_text_simple_stream
from llms_from_scratch.kv_cache.gpt2 import GPTModel
from llms_from_scratch.kv_cache.utils import KVCache


TEST_CONFIG = {
    "vocab_size": 96,
    "context_length": 16,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


def make_model(seed=123):
    torch.manual_seed(seed)
    return GPTModel(TEST_CONFIG).eval()


def test_model_exposes_cached_generator_contract():
    model = make_model()

    assert model.cfg == TEST_CONFIG
    model.current_pos = 7
    model.reset_kv_cache()
    assert model.current_pos == 0


def test_supplied_cache_activates_cache_path_and_supports_a_second_step():
    model = make_model()
    cache = KVCache(n_layers=TEST_CONFIG["n_layers"])
    prompt = torch.tensor([[4, 8, 15, 16]])

    logits = model(prompt, cache=cache)

    assert logits.shape == (1, 4, TEST_CONFIG["vocab_size"])
    assert model.current_pos == prompt.shape[1]
    for keys, values in cache.get_all():
        assert keys.shape[2] == prompt.shape[1]
        assert values.shape[2] == prompt.shape[1]

    next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
    next_logits = model(next_token, cache=cache)

    assert next_logits.shape == (1, 1, TEST_CONFIG["vocab_size"])
    assert model.current_pos == prompt.shape[1] + 1
    for keys, values in cache.get_all():
        assert keys.shape[2] == prompt.shape[1] + 1
        assert values.shape[2] == prompt.shape[1] + 1


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("prompt_length", [1, 4, 8])
def test_cached_generation_matches_uncached(batch_size, prompt_length):
    model = make_model()
    prompt = torch.randint(0, TEST_CONFIG["vocab_size"], (batch_size, prompt_length))

    uncached = generate_text_simple(
        model, prompt.clone(), max_new_tokens=3, context_size=16, use_cache=False
    )
    cached = generate_text_simple(
        model, prompt.clone(), max_new_tokens=3, context_size=16, use_cache=True
    )

    assert torch.equal(cached, uncached)


def test_uncached_generation_does_not_inherit_cache_position():
    model = make_model(seed=321)
    prompt = torch.tensor([[4, 8, 15, 16]])

    generate_text_simple(
        model, prompt.clone(), max_new_tokens=3, context_size=16, use_cache=True
    )
    observed = generate_text_simple(
        model, prompt.clone(), max_new_tokens=3, context_size=16, use_cache=False
    )

    fresh_model = GPTModel(TEST_CONFIG).eval()
    fresh_model.load_state_dict(model.state_dict())
    expected = generate_text_simple(
        fresh_model, prompt.clone(), max_new_tokens=3, context_size=16, use_cache=False
    )

    assert torch.equal(observed, expected)


def test_repeated_cached_generation_resets_position_state():
    model = make_model()
    prompt = torch.tensor([[4, 8, 15, 16]])

    first = generate_text_simple(
        model, prompt.clone(), max_new_tokens=3, context_size=16, use_cache=True
    )
    second = generate_text_simple(
        model, prompt.clone(), max_new_tokens=3, context_size=16, use_cache=True
    )

    assert torch.equal(first, second)


def test_streaming_generation_matches_cached_generation():
    reference_model = make_model()
    streaming_model = make_model()
    streaming_model.load_state_dict(reference_model.state_dict())
    prompt = torch.tensor([[4, 8, 15, 16]])

    expected = generate_text_simple(
        reference_model, prompt.clone(), max_new_tokens=3, context_size=16, use_cache=True
    )
    streamed_tokens = list(
        generate_text_simple_stream(
            streaming_model,
            prompt.clone(),
            max_new_tokens=3,
            context_size=16,
        )
    )
    observed = torch.cat([prompt, *streamed_tokens], dim=1)

    assert torch.equal(observed, expected)
