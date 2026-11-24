import pytest
import torch

import gpt_with_kv_gqa as gqa
import gpt_with_kv_mha as mha


@pytest.fixture(params=[gqa, mha], ids=["gqa", "mha"])
def model_setup(request):
    cfg = {
        "vocab_size": 32,
        "context_length": 8,
        "emb_dim": 16,
        "n_heads": 4,
        "n_kv_groups": 2,
        "n_layers": 1,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    torch.manual_seed(123)
    return request.param.GPTModel(cfg), request.param.generate_text_simple_cached


@pytest.mark.parametrize("prompt_len, max_new_tokens", [(8, 1), (4, 5), (10, 1), (8, 0)])
def test_cached_generation_at_context_boundary(model_setup, prompt_len, max_new_tokens):
    model, generate = model_setup
    prompt = torch.arange(prompt_len).unsqueeze(0)

    cached = generate(model, prompt, max_new_tokens, use_cache=True)
    uncached = generate(model, prompt, max_new_tokens, use_cache=False)

    assert cached.shape[1] == prompt_len + max_new_tokens
    assert torch.equal(cached[:, :prompt_len], prompt)
    assert torch.equal(cached, uncached)


@pytest.mark.parametrize("prompt_len, max_new_tokens", [(8, 2), (4, 6), (10, 2)])
def test_cached_generation_rejects_context_overflow(model_setup, prompt_len, max_new_tokens):
    model, generate = model_setup
    prompt = torch.arange(prompt_len).unsqueeze(0)

    with pytest.raises(AssertionError, match="context length"):
        generate(model, prompt, max_new_tokens, use_cache=True)


def test_uncached_generation_can_slide_context_window(model_setup):
    model, generate = model_setup
    prompt = torch.arange(8).unsqueeze(0)

    output = generate(model, prompt, max_new_tokens=3, use_cache=False)

    assert output.shape[1] == 11
    assert torch.equal(output[:, :8], prompt)


def test_oversized_context_size_uses_model_limit(model_setup):
    model, generate = model_setup
    prompt = torch.arange(10).unsqueeze(0)

    # Both paths should use the last 8 prompt tokens and can predict one more.
    expected = generate(model, prompt, 1, context_size=8, use_cache=False)
    for use_cache in (False, True):
        actual = generate(model, prompt, 1, context_size=16, use_cache=use_cache)
        assert torch.equal(actual, expected)

    # A second cached prediction would require position 8 in a table indexed 0–7.
    with pytest.raises(AssertionError, match="context length"):
        generate(model, prompt[:, :8], 2, context_size=16, use_cache=True)
