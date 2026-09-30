# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).

import pytest
import torch

from llms_from_scratch.kv_cache import gpt2, llama3, qwen3
from llms_from_scratch.kv_cache.generate import generate_text_simple_stream
from llms_from_scratch.kv_cache_batched import qwen3 as qwen3_batched


@pytest.fixture(params=["gpt2", "llama3", "qwen3", "qwen3_moe", "qwen3_batched"])
def model_setup(request):
    cfg = {
        "vocab_size": 32, "context_length": 8, "emb_dim": 16,
        "n_heads": 4, "n_layers": 2, "hidden_dim": 32,
        "drop_rate": 0.0, "qkv_bias": False,
        "head_dim": 4, "n_kv_groups": 2, "qk_norm": True,
        "rope_base": 1_000_000.0, "rope_freq": None,
        "dtype": torch.float32, "num_experts": 0,
    }
    module = {"gpt2": gpt2, "llama3": llama3, "qwen3": qwen3,
              "qwen3_moe": qwen3, "qwen3_batched": qwen3_batched}[request.param]
    model_class = (gpt2.GPTModel if module is gpt2 else
                   llama3.Llama3Model if module is llama3 else module.Qwen3Model)
    if request.param == "qwen3_moe":
        cfg.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32)
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(cfg).to(device).eval()
    if module is qwen3_batched:
        cache = module.KVCache(n_layers=cfg["n_layers"], batch_size=2)
        model.reset_kv_cache(batch_size=2, device=device)
    else:
        cache = module.KVCache(n_layers=cfg["n_layers"])
    return model, cache


def cached_forward(model, cache, tokens):
    if isinstance(cache, qwen3_batched.KVCache):
        logits = model(tokens, cache=cache, start_pos=model.current_pos.clone())
        model.current_pos += tokens.shape[1]
        return logits
    return model(tokens, cache=cache)


@torch.inference_mode()
@pytest.mark.parametrize("use_cache", [False, True])
def test_overlong_prompt_reports_context_limit(model_setup, use_cache):
    model, cache = model_setup
    tokens = torch.ones((2, 9), dtype=torch.long, device=model.tok_emb.weight.device)
    with pytest.raises(ValueError, match=r"Sequence length 9 exceeds .*8 tokens"):
        if use_cache:
            cached_forward(model, cache, tokens)
        else:
            model(tokens)


@torch.inference_mode()
def test_rejected_decode_preserves_cache_and_last_valid_position(model_setup):
    model, cache = model_setup
    tokens = torch.arange(1, 9, device=model.tok_emb.weight.device).repeat(2, 1)
    cached_forward(model, cache, tokens[:, :7])

    with pytest.raises(ValueError, match=r"Sequence length 9 exceeds .*8 tokens"):
        cached_forward(model, cache, tokens[:, -2:])

    # Rejection must leave the previous prefix usable, including the last position.
    actual = cached_forward(model, cache, tokens[:, -1:])
    expected = model(tokens)[:, -1:]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@torch.inference_mode()
@pytest.mark.parametrize("model_setup", ["qwen3_batched"], indirect=True)
def test_batched_limit_checks_every_sample(model_setup):
    # A later batch item can exceed the limit even when the first one fits.
    model, cache = model_setup
    device = model.tok_emb.weight.device
    with pytest.raises(ValueError, match=r"Sequence length 9 exceeds .*8 tokens"):
        model(torch.ones((2, 1), dtype=torch.long, device=device), cache=cache,
              start_pos=torch.tensor([0, 8], device=device))


@torch.inference_mode()
@pytest.mark.parametrize("model_setup", ["gpt2", "llama3", "qwen3", "qwen3_moe"], indirect=True)
def test_large_generation_budget_can_stop_early_at_eos(model_setup):
    model, _ = model_setup
    model.out_head.weight.zero_()  # Make the next token EOS (token 0).
    prompt = torch.tensor([[1, 2]], device=model.tok_emb.weight.device)
    output = list(generate_text_simple_stream(model, prompt, max_new_tokens=16, eos_token_id=0))
    assert output == []
