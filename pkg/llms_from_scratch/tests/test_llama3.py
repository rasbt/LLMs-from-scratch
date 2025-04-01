# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from llms_from_scratch.ch04 import generate_text_simple
from llms_from_scratch.llama3 import (
    compute_rope_params,
    apply_rope,
    rescale_theta,
    LLAMA32_CONFIG_1B,
    GroupedQueryAttention,
    GroupedQueryAttentionFast,
    Llama3Model,
)

import importlib
import pytest
import tiktoken
import torch


transformers_installed = importlib.util.find_spec("transformers") is not None


@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_rope():

    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

    # Settings
    batch_size = 1
    context_len = 8192
    num_heads = 4
    head_dim = 16
    rope_theta = 500_000

    rope_config = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }

    # Instantiate RoPE parameters
    cos, sin = compute_rope_params(
        head_dim=head_dim,
        theta_base=rope_theta,
        context_length=context_len,
        freq_config=rope_config,
    )

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # Apply rotary position embeddings
    queries_rot = apply_rope(queries, cos, sin)
    keys_rot = apply_rope(keys, cos, sin)

    # Generate reference RoPE via HF
    hf_rope_params = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    }

    class RoPEConfig:
        rope_type = "llama3"
        rope_scaling = hf_rope_params
        factor = 1.0
        dim: int = head_dim
        rope_theta = 500_000
        max_position_embeddings: int = 8192
        hidden_size = head_dim * num_heads
        num_attention_heads = num_heads

    config = RoPEConfig()

    rot_emb = LlamaRotaryEmbedding(config=config)
    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)

    torch.testing.assert_close(sin, ref_sin.squeeze(0))
    torch.testing.assert_close(cos, ref_cos.squeeze(0))
    torch.testing.assert_close(keys_rot, ref_keys_rot)
    torch.testing.assert_close(queries_rot, ref_queries_rot)


GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}


def test_rescale():

    new_theta = rescale_theta(
        theta_old=500_000.,
        context_length_old=131_072,
        context_length_new=8192
    )
    assert new_theta == 31250.

    old_theta = rescale_theta(
        theta_old=new_theta,
        context_length_old=8192,
        context_length_new=131_072
    )
    assert old_theta == 500_000.


def test_grouped_query_attention_equivalence():
    torch.manual_seed(42)
    b, t, d_in, d_out, num_heads, num_kv_groups = 2, 8, 32, 64, 4, 2

    x = torch.randn(b, t, d_in)
    cos, sin = compute_rope_params(
        head_dim=d_out // num_heads,
        theta_base=50_000,
        context_length=t,
        freq_config={
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": t,
        }
    )

    # Causal mask for the slow version
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)

    attn1 = GroupedQueryAttention(d_in, d_out, num_heads, num_kv_groups)
    attn2 = GroupedQueryAttentionFast(d_in, d_out, num_heads, num_kv_groups)

    # Copy weights to make both models identical
    attn2.load_state_dict(attn1.state_dict())

    # Run both
    y1 = attn1(x, mask, cos, sin)
    y2 = attn2(x, cos, sin)

    # Compare outputs
    max_diff = (y1 - y2).abs().max().item()
    print(f"Max difference between slow and fast outputs: {max_diff:.4e}")
    assert torch.allclose(y1, y2, atol=1e-4)


@pytest.fixture(scope="session")
def llama3_weights_path(tmp_path_factory):
    """Creates and saves a deterministic Llama3 model for testing."""
    path = tmp_path_factory.mktemp("models") / "llama3_test_weights.pt"

    if not path.exists():
        torch.manual_seed(123)
        model = Llama3Model(LLAMA32_CONFIG_1B)
        torch.save(model.state_dict(), path)

    return path


@pytest.mark.parametrize("ModelClass", [Llama3Model])
def test_gpt_model_variants(ModelClass, llama3_weights_path):
    torch.manual_seed(123)
    model = ModelClass(LLAMA32_CONFIG_1B)
    model.load_state_dict(torch.load(llama3_weights_path))
    model.eval()

    start_context = "Llamas eat"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=5,
        context_size=LLAMA32_CONFIG_1B["context_length"]
    )
    print("Encoded output text:", out)
    expect = torch.tensor([
        [43,   2543,    292,   4483, 100383,   8113,  21197,  33804,  54419]
    ])
    assert torch.equal(expect, out)
