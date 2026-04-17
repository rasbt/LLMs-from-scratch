# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4, adapted to use DeepSeek Sparse Attention (DSA).
# This file can be run as a standalone script.

# DSA is introduced in DeepSeek-V3.2:
#   https://huggingface.co/deepseek-ai/DeepSeek-V3.2
# Technical report:
#   https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf

import argparse
import time
import tiktoken
import torch
import torch.nn as nn


#####################################
# DeepSeek Sparse Attention (DSA)
#####################################
# DSA combines two components:
#   1. A Lightning Indexer that scores all past tokens for each query
#      using a lightweight sum of ReLU(q · k) dot products.
#   2. A Token Selector that picks the top-K highest-scoring past tokens
#      and masks out the rest, reducing effective context from O(L) to O(k).
#
# This reduces attention complexity from O(L²) to O(L·k).
#
# Reference implementation inspired by:
#   https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py


class LightningIndexer(nn.Module):
    """Lightweight module that scores every past token for each incoming query.

    For each query token t and each candidate past token s, the score is:
        I_{t,s} = sum_j [ w_{t,j} * ReLU(q_{t,j} · k_s) ]

    where w_{t,j} is a learned per-head scalar weight derived from the input,
    and j indexes over the index heads.

    Args:
        d_model:       model dimension (same as emb_dim).
        index_n_heads: number of lightweight index heads (H_I in the paper).
        index_head_dim: dimension of each index head.
    """

    def __init__(self, d_model: int, index_n_heads: int, index_head_dim: int):
        super().__init__()
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim

        # Project input to indexer query vectors: (d_model -> index_n_heads * index_head_dim)
        self.W_q_index = nn.Linear(d_model, index_n_heads * index_head_dim, bias=False)
        # Project input to shared key vectors: (d_model -> index_head_dim)
        self.W_k_index = nn.Linear(d_model, index_head_dim, bias=False)
        # Learn a per-head weight scalar: (d_model -> index_n_heads), as in the V3.2 paper
        self.W_weights = nn.Linear(d_model, index_n_heads, bias=False)

        self.scale = index_head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,         # (b, T, d_model)  current token(s)
        x_ctx: torch.Tensor,     # (b, S, d_model)  all past + current tokens
        topk: int,
        causal_mask: torch.Tensor | None = None,  # (T, S) float mask
    ) -> torch.Tensor:
        """Return top-K token indices shape (b, T, topk)."""
        b, T, _ = x.shape
        _, S, _ = x_ctx.shape

        # Indexer queries: (b, T, H_I, head_dim)
        q = self.W_q_index(x).view(b, T, self.index_n_heads, self.index_head_dim)
        # Indexer keys: (b, S, head_dim)
        k = self.W_k_index(x_ctx)  # (b, S, head_dim)

        # ReLU(q · k^T) for each head: (b, T, H_I, S)
        # k: (b, S, head_dim) -> (b, 1, S, head_dim) for broadcast
        raw = torch.einsum("bthd,bsd->bths", q, k) * self.scale  # (b, T, H_I, S)
        raw = torch.relu(raw)

        # Per-head learned weights: (b, T, H_I)
        w = self.W_weights(x)  # (b, T, H_I)
        w = w.softmax(dim=-1)  # normalise across heads

        # Weighted sum over heads -> index scores (b, T, S)
        index_scores = torch.einsum("bth,bths->bts", w, raw)  # (b, T, S)

        if causal_mask is not None:
            index_scores = index_scores + causal_mask  # broadcast over batch

        # Select top-K positions. topk is capped at available context length S.
        k_val = min(topk, S)
        topk_indices = index_scores.topk(k_val, dim=-1).indices  # (b, T, k)
        return topk_indices


class MultiHeadAttentionWithDSA(nn.Module):
    """Multi-head causal self-attention with DeepSeek Sparse Attention (DSA).

    After computing full attention scores, the Lightning Indexer selects the
    top-K most relevant past tokens for each query, and all other positions
    are masked to -inf before softmax.  This makes the effective attention
    cost O(L·k) instead of O(L²).

    Args:
        d_in:           input dimension.
        d_out:          output dimension (must be divisible by num_heads).
        dropout:        dropout probability.
        num_heads:      number of standard attention heads.
        qkv_bias:       whether to use bias in Q/K/V projections.
        index_n_heads:  number of lightweight index heads (H_I).
        index_head_dim: dimension per index head.
        topk:           number of tokens each query attends to (k).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
        index_n_heads: int = 4,
        index_head_dim: int = 64,
        topk: int = 64,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.topk = topk

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.indexer = LightningIndexer(d_in, index_n_heads, index_head_dim)

        ####################################################
        # KV cache-related code
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        # Keep raw input tokens for the indexer key projection
        self.register_buffer("cache_x", None, persistent=False)
        self.ptr_current_pos = 0
        ####################################################

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None
        self.cache_x = None
        self.ptr_current_pos = 0

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys_new = self.W_key(x)
        values_new = self.W_value(x)

        # Reshape to (b, T, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        # KV cache-related
        if use_cache:
            old_len = 0 if self.cache_k is None else self.cache_k.size(1)
            if self.cache_k is None:
                keys = keys_new
                values = values_new
                x_ctx = x
            else:
                keys = torch.cat([self.cache_k, keys_new], dim=1)
                values = torch.cat([self.cache_v, values_new], dim=1)
                x_ctx = torch.cat([self.cache_x, x], dim=1)
            self.cache_k = keys
            self.cache_v = values
            self.cache_x = x_ctx
            q_start = self.ptr_current_pos
            k_start = 0
            self.ptr_current_pos += num_tokens
        else:
            keys = keys_new
            values = values_new
            x_ctx = x
            q_start = 0
            k_start = 0
        ####################################################

        # Transpose: (b, T, num_heads, head_dim) -> (b, num_heads, T, head_dim)
        queries_t = queries.transpose(1, 2)
        keys_t = keys.transpose(1, 2)
        values_t = values.transpose(1, 2)

        # Full scaled dot-product attention scores: (b, num_heads, T_q, T_k)
        attn_scores = queries_t @ keys_t.transpose(2, 3)

        num_tokens_Q = queries_t.shape[-2]
        num_tokens_K = keys_t.shape[-2]
        device = x.device

        # ---- Build causal mask (float, -inf for masked positions) ----
        q_positions = torch.arange(q_start, q_start + num_tokens_Q, device=device, dtype=torch.long)
        k_positions = torch.arange(k_start, k_start + num_tokens_K, device=device, dtype=torch.long)
        causal_bool = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)  # (T_q, T_k)
        causal_float = torch.zeros(num_tokens_Q, num_tokens_K, device=device, dtype=attn_scores.dtype)
        causal_float.masked_fill_(causal_bool, float("-inf"))

        # ---- DSA: Lightning Indexer → sparse mask ----
        # The indexer receives the current queries (x) and all context tokens (x_ctx).
        # causal_float is passed so future tokens are excluded from index selection.
        topk_indices = self.indexer(x, x_ctx, self.topk, causal_mask=causal_float)
        # topk_indices: (b, T_q, k)

        # Build sparse mask: -inf everywhere, 0 at selected positions
        sparse_mask = torch.full(
            (b, num_tokens_Q, num_tokens_K), float("-inf"), device=device, dtype=attn_scores.dtype
        )
        sparse_mask.scatter_(-1, topk_indices, 0.0)  # (b, T_q, T_k)

        # Combine causal mask and sparse mask, then broadcast over heads
        combined_mask = causal_float.unsqueeze(0) + sparse_mask  # (b, T_q, T_k)
        attn_scores = attn_scores + combined_mask.unsqueeze(1)  # (b, num_heads, T_q, T_k)

        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_heads, T_q, head_dim)
        context_vec = attn_weights @ values_t
        # Transpose and reshape: (b, T_q, d_out)
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttentionWithDSA(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            index_n_heads=cfg["index_n_heads"],
            index_head_dim=cfg["index_head_dim"],
            topk=cfg["topk"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)

        ####################################################
        # KV cache-related
        x = self.att(x, use_cache=use_cache)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        ####################################################
        # KV cache-related
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.current_pos = 0
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        ####################################################
        # KV cache-related
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)

        ####################################################
        # KV cache-related
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    ####################################################
    # KV cache-related
    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    ####################################################


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run GPT with DeepSeek Sparse Attention (DSA)."
    )
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")
    parser.add_argument("--index_n_heads", type=int, default=4,
                        help="Number of lightweight indexer heads (H_I in the DSA paper).")
    parser.add_argument("--index_head_dim", type=int, default=64,
                        help="Dimension of each indexer head.")
    parser.add_argument("--topk", type=int, default=64,
                        help="Number of tokens each query attends to (k). "
                             "For short sequences this is capped at sequence length.")

    args = parser.parse_args()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # Vocabulary size
        "context_length": args.max_new_tokens + len(encoded),
        "emb_dim": args.emb_dim,    # Embedding dimension
        "n_heads": args.n_heads,    # Number of attention heads
        "n_layers": args.n_layers,  # Number of layers
        "drop_rate": 0.0,           # Dropout rate
        "qkv_bias": False,          # Query-Key-Value bias
        "index_n_heads": args.index_n_heads,
        "index_head_dim": args.index_head_dim,
        "topk": args.topk,
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)
    model.eval()  # disable dropout

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=args.max_new_tokens,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
