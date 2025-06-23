# Code to test the GPT model implementation against the KV cache variants

import pytest
import torch
import tiktoken

from gpt_ch04 import GPTModel as GPTModelBase
from gpt_ch04 import generate_text_simple

from gpt_with_kv_cache import GPTModel as GPTModelKV1
from gpt_with_kv_cache_optimized import GPTModel as GPTModelKV2
from gpt_with_kv_cache import generate_text_simple_cached


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
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
    else:
        token_ids = generate_text_simple_cached(
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
