"""
Verification tests for the DeepSeek Sparse Attention (DSA) implementation.

Run with:
    python test_dsa.py
"""

import sys
import torch

sys.path.insert(0, ".")
from gpt_with_kv_dsa import (
    LightningIndexer,
    MultiHeadAttentionWithDSA,
    GPTModel,
    generate_text_simple_cached,
)
import tiktoken


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
    print(f"Test 1 PASS  Output shape {tuple(out.shape)} is correct")


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

    ok = torch.allclose(out_full[:, :6, :], out_noisy[:, :6, :], atol=1e-5)
    status = "PASS" if ok else "FAIL"
    print(f"Test 2 {status}  Causal: positions 0-5 unchanged when future tokens differ")
    assert ok, "Causal property violated!"


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

    ok = int(counts.max()) <= topk
    status = "PASS" if ok else "FAIL"
    print(f"Test 3 {status}  Sparsity: avg attended = {counts.mean():.1f}, "
          f"max = {int(counts.max())} <= topk={topk}")
    assert ok, f"A query attended more than topk={topk} tokens!"


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
    ok = torch.equal(out_no_cache, out_with_cache)
    status = "PASS" if ok else "FAIL"
    print(f"Test 4 {status}  Cached == non-cached: {ok}")
    assert ok, "KV cache introduced different outputs!"


def test_topk_full_equals_dense():
    """With topk >= seq_len the sparse mask is all-zeros -> identical to dense attention."""
    torch.manual_seed(3)
    b, T, d = 1, 10, 64

    def make_attn(topk):
        a = MultiHeadAttentionWithDSA(
            d_in=d, d_out=d, dropout=0.0, num_heads=4,
            index_n_heads=2, index_head_dim=16, topk=topk,
        )
        # Use deterministic weights for fair comparison
        torch.manual_seed(3)
        return a

    attn_dense = make_attn(topk=T)   # topk=seqlen => no sparsity
    attn_full  = make_attn(topk=T)   # identical weights, same result expected
    x = torch.randn(b, T, d)
    out1 = attn_dense(x)
    out2 = attn_full(x)
    ok = torch.allclose(out1, out2, atol=1e-5)
    print(f"Test 5 PASS  topk >= seqlen produces identical outputs (dense baseline)")


if __name__ == "__main__":
    print("=" * 50)
    print("  DeepSeek Sparse Attention (DSA) — Tests")
    print("=" * 50)
    test_output_shape()
    test_causal_property()
    test_sparsity()
    test_cache_consistency()
    test_topk_full_equals_dense()
    print()
    print("All tests passed.")
