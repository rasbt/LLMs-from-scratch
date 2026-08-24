# Code to test the GPT model implementation against the KV cache variants

import pytest
import torch
import tiktoken

from gpt_ch04 import GPTModel as GPTModelBase
from gpt_ch04 import generate_text_simple

from gpt_with_kv_cache import GPTModel as GPTModelKV1
from gpt_with_kv_cache_optimized import GPTModel as GPTModelKV2
from gpt_with_kv_cache import generate_text_simple_cached as generate_text_simple_cachedKV1
from gpt_with_kv_cache_optimized import generate_text_simple_cached as generate_text_simple_cachedKV2


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "kv_window_size": 1024  # NEW: KV cache window size
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("ModelClass", [GPTModelBase, GPTModelKV1, GPTModelKV2])
def test_gpt_model_equivalence_not_cached(ModelClass):
    torch.manual_seed(123)

    model = ModelClass(GPT_CONFIG_124M).to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "Hello, I am"
    encoded = tokenizer.encode(prompt)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    model_name = ModelClass.__module__ + "." + ModelClass.__name__

    token_ids = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=30,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    if not hasattr(test_gpt_model_equivalence_not_cached, "results"):
        test_gpt_model_equivalence_not_cached.results = []

    test_gpt_model_equivalence_not_cached.results.append((model_name, token_ids))

    if len(test_gpt_model_equivalence_not_cached.results) == 3:
        base_name, base_output = test_gpt_model_equivalence_not_cached.results[0]
        for other_name, other_output in test_gpt_model_equivalence_not_cached.results[1:]:
            assert torch.equal(base_output, other_output), (
                f"Mismatch between {base_name} and {other_name}"
            )


@pytest.mark.parametrize("ModelClass", [GPTModelBase, GPTModelKV1, GPTModelKV2])
def test_gpt_model_equivalence_cached(ModelClass):
    torch.manual_seed(123)

    model = ModelClass(GPT_CONFIG_124M).to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "Hello, I am"
    encoded_tensor = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

    model_name = ModelClass.__module__ + "." + ModelClass.__name__

    if ModelClass is GPTModelBase:
        token_ids = generate_text_simple(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=30,
            context_size=GPT_CONFIG_124M["context_length"]
        )
    elif ModelClass is GPTModelKV1:
        token_ids = generate_text_simple_cachedKV1(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=30,
            context_size=GPT_CONFIG_124M["context_length"]
        )
    else:
        token_ids = generate_text_simple_cachedKV2(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=30,
            context_size=GPT_CONFIG_124M["context_length"]
        )

    if not hasattr(test_gpt_model_equivalence_cached, "results"):
        test_gpt_model_equivalence_cached.results = []

    test_gpt_model_equivalence_cached.results.append((model_name, token_ids))

    if len(test_gpt_model_equivalence_cached.results) == 3:
        base_name, base_output = test_gpt_model_equivalence_cached.results[0]
        for other_name, other_output in test_gpt_model_equivalence_cached.results[1:]:
            assert torch.equal(base_output, other_output), (
                f"Mismatch between {base_name} and {other_name}"
            )


def test_context_overflow_bug():
    """
    Test that demonstrates the ptr_current_pos overflow bug.

    In old implementation:
    - context_length = 10 (positions 0-9 available)
    - We try to generate 15 tokens total (5 input + 10 generated)
    - At token 11 (position 10), it crashes trying to access pos_emb[10]
    """
    GPT_CONFIG_SMALL = {
        "vocab_size": 50257,
        "context_length": 10,  # Very small context
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "kv_window_size": 20  # Larger than context_length
    }

    torch.manual_seed(123)

    model = GPTModelKV2(GPT_CONFIG_SMALL).to(device)
    model.eval()

    # 5 input tokens
    input_tokens = torch.randint(0, 50257, (1, 5), device=device)

    generate_text_simple_cachedKV2(
        model=model,
        idx=input_tokens,
        max_new_tokens=10,  # 5 + 10 = 15 > 10 context_length
        context_size=GPT_CONFIG_SMALL["context_length"],
        use_cache=True
    )


def test_prefill_chunking_basic():
    """
    Test that prefill correctly chunks input when input_length > kv_window_size.

    Setup:
    - kv_window_size = 4
    - input_length = 10
    - Should process in 3 chunks: [0:4], [4:8], [8:10]
    """
    config = {
        "vocab_size": 50257,
        "context_length": 20,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "kv_window_size": 4  # Small window to force chunking
    }

    torch.manual_seed(123)
    model = GPTModelKV2(config).to(device)
    model.eval()

    # 10 input tokens (> kv_window_size of 4)
    input_tokens = torch.randint(0, 50257, (1, 10), device=device)

    # Should successfully process all input in chunks
    token_ids = generate_text_simple_cachedKV2(
        model=model,
        idx=input_tokens,
        max_new_tokens=2,
        use_cache=True
    )

    # Should have 10 input + 2 generated = 12 total
    assert token_ids.shape[1] == 12, f"Expected 12 tokens, got {token_ids.shape[1]}"

    # First 10 tokens should match input
    assert torch.equal(token_ids[:, :10], input_tokens), "Input tokens should be preserved"