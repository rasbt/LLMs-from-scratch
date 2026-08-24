import os

import pytest
import torch
import torch.nn as nn
import tiktoken

from gpt_with_kv_dsa import (
    GPTModel,
    LightningIndexer,
    MultiHeadAttentionWithDSA,
    generate_text_simple_cached,
)


def import_transformers_dsa_model():
    """Import the reference DSA model from the installed Transformers package."""
    try:
        from transformers import GlmMoeDsaConfig, GlmMoeDsaModel
    except ImportError as err:
        pytest.skip(f"Transformers GLM-MoE-DSA reference unavailable: {err}")

    return GlmMoeDsaConfig, GlmMoeDsaModel


def test_output_shape():
    """Output shape must be (batch, seq_len, d_out)."""
    torch.manual_seed(0)
    b, T, d = 2, 20, 128
    attn = MultiHeadAttentionWithDSA(
        d_in=d, d_out=d, dropout=0.0, num_heads=4,
        index_n_heads=2, index_head_dim=16, topk=5,
    )
    x = torch.randn(b, T, d)
    out = attn(x)
    assert out.shape == (b, T, d), f"Wrong shape: {out.shape}"


def test_causal_property():
    """Tokens at position p must not be affected by tokens at positions > p."""
    torch.manual_seed(1)
    b, T, d = 1, 20, 128
    attn = MultiHeadAttentionWithDSA(
        d_in=d, d_out=d, dropout=0.0, num_heads=4,
        index_n_heads=2, index_head_dim=16, topk=5,
    )
    x = torch.randn(b, T, d)
    out_full = attn(x)

    # Replace tokens at positions 6+ with random noise
    x_noisy = x.clone()
    x_noisy[:, 6:, :] = torch.randn(b, T - 6, d)
    out_noisy = attn(x_noisy)

    torch.testing.assert_close(out_noisy[:, :6, :], out_full[:, :6, :], rtol=0, atol=1e-5)


def test_sparsity():
    """Each query must attend to at most topk tokens."""
    torch.manual_seed(2)
    b, T, d = 1, 20, 128
    topk = 5
    attn = MultiHeadAttentionWithDSA(
        d_in=d, d_out=d, dropout=0.0, num_heads=4,
        index_n_heads=2, index_head_dim=16, topk=topk,
    )
    x = torch.randn(b, T, d)

    # Reconstruct the combined (causal + sparse) mask
    q_pos = torch.arange(T)
    k_pos = torch.arange(T)
    causal_bool = q_pos.unsqueeze(-1) < k_pos.unsqueeze(0)
    causal_float = torch.zeros(T, T).masked_fill_(causal_bool, float("-inf"))

    topk_idx = attn.indexer(x, x, topk)
    sparse_mask = torch.full((b, T, T), float("-inf"))
    sparse_mask.scatter_(-1, topk_idx, 0.0)

    combined = causal_float.unsqueeze(0) + sparse_mask   # (1, T, T)
    counts = (combined[0] > float("-inf")).sum(dim=-1).float()

    assert int(counts.max()) <= topk


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Transformers reference test is too expensive for GitHub Actions",
)
def test_indexer_matches_transformers_reference():
    """The indexer must match the Transformers DSA scoring path."""
    torch.manual_seed(4)
    b, T, d = 2, 6, 32
    topk = 3
    indexer = LightningIndexer(d_model=d, index_n_heads=4, index_head_dim=8)
    GlmMoeDsaConfig, GlmMoeDsaModel = import_transformers_dsa_model()
    reference_cfg = GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=d,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=8,
        index_n_heads=indexer.index_n_heads,
        index_head_dim=indexer.index_head_dim,
        index_topk=topk,
        q_lora_rank=d,
        qk_rope_head_dim=0,
        qk_nope_head_dim=8,
        v_head_dim=8,
        n_routed_experts=4,
        num_experts_per_tok=1,
        max_position_embeddings=16,
        mlp_layer_types=["dense"],
    )
    reference_model = GlmMoeDsaModel(reference_cfg)
    reference_model.eval()
    reference = reference_model.layers[0].self_attn.indexer
    reference.k_norm = nn.Identity()
    with torch.no_grad():
        reference.wq_b.weight.copy_(indexer.W_q_index.weight)
        reference.wk.weight.copy_(indexer.W_k_index.weight)
        reference.weights_proj.weight.copy_(indexer.W_weights.weight)

    x = torch.randn(b, T, d)
    q_pos = torch.arange(T)
    k_pos = torch.arange(T)
    causal_bool = q_pos.unsqueeze(-1) < k_pos.unsqueeze(0)
    causal_mask = torch.zeros(T, T).masked_fill_(causal_bool, float("-inf"))

    actual = indexer(x, x, topk=topk, causal_mask=causal_mask)
    empty_rope = torch.empty(b, T, 0)
    expected = reference(x, x, (empty_rope, empty_rope), causal_mask)
    assert torch.equal(actual, expected)


def test_cache_consistency():
    """Cached and non-cached generation must produce identical token sequences."""
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode("Hello, I am")
    cfg = {
        "vocab_size": 50257,
        "context_length": 30,
        "emb_dim": 256,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "index_n_heads": 2,
        "index_head_dim": 32,
        "topk": 200,   # large topk == full attention, so both modes match exactly
    }
    torch.manual_seed(42)
    model = GPTModel(cfg)
    model.eval()
    idx = torch.tensor(encoded).unsqueeze(0)
    out_no_cache = generate_text_simple_cached(model, idx.clone(), max_new_tokens=5, use_cache=False)
    out_with_cache = generate_text_simple_cached(model, idx.clone(), max_new_tokens=5, use_cache=True)
    assert torch.equal(out_no_cache, out_with_cache)


def dense_attention_reference(attn, x):
    """Dense causal attention using the same projections as the DSA module."""
    b, T, _ = x.shape

    queries = attn.W_query(x).view(b, T, attn.num_heads, attn.head_dim).transpose(1, 2)
    keys = attn.W_key(x).view(b, T, attn.num_heads, attn.head_dim).transpose(1, 2)
    values = attn.W_value(x).view(b, T, attn.num_heads, attn.head_dim).transpose(1, 2)

    attn_scores = queries @ keys.transpose(2, 3)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
    attn_scores = attn_scores.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(attn_scores / attn.head_dim ** 0.5, dim=-1)
    context_vec = attn_weights @ values
    context_vec = context_vec.transpose(1, 2).contiguous().view(b, T, attn.d_out)
    return attn.out_proj(context_vec)


def test_topk_full_equals_dense():
    """With topk >= seq_len the sparse mask is all-zeros -> identical to dense attention."""
    torch.manual_seed(3)
    b, T, d = 1, 10, 64

    attn_full = MultiHeadAttentionWithDSA(
        d_in=d, d_out=d, dropout=0.0, num_heads=4,
        index_n_heads=2, index_head_dim=16, topk=T,
    )
    x = torch.randn(b, T, d)
    out_dsa = attn_full(x)
    out_dense = dense_attention_reference(attn_full, x)
    torch.testing.assert_close(out_dsa, out_dense, rtol=0, atol=1e-5)
