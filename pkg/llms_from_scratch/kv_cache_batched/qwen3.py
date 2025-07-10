# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from .utils import KVCache   # noqa: F401
from ..qwen3 import (   # noqa: F401
    QWEN_CONFIG_06_B, QWEN3_CONFIG_1_7B, QWEN3_CONFIG_4B,
    QWEN3_CONFIG_8B, QWEN3_CONFIG_14B, QWEN3_CONFIG_32B,
    Qwen3Tokenizer, load_weights_into_qwen,
    download_from_huggingface,
    download_from_huggingface_from_snapshots
)

import torch
import torch.nn as nn


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusable utilities
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = None  # Batched version tracks positions per sample

    def forward(self, in_idx, cache=None, start_pos=None):
        B, num_tokens = in_idx.size()
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        device = x.device

        if cache is not None:
            pos_start = start_pos
            pos_end = pos_start + num_tokens
            max_len = pos_end.max().item()
            full_mask = torch.triu(
                torch.ones(max_len, max_len, device=device, dtype=torch.bool), diagonal=1
            )
            mask = torch.zeros(B, 1, num_tokens, max_len, device=device, dtype=torch.bool)
            for i in range(B):
                ps, pe = pos_start[i].item(), pos_end[i].item()
                mask[i, 0] = full_mask[ps:pe, :pe]
        else:
            pos_start = torch.zeros(B, dtype=torch.long, device=device)
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=device, dtype=torch.bool), diagonal=1
            )[None, None, :, :]

        for i, block in enumerate(self.trf_blocks):
            blk_cache = [cache.get(i, b_idx) for b_idx in range(B)] if cache is not None else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin, start_pos=pos_start, cache=blk_cache)
            if cache is not None:
                for b_idx in range(B):
                    cache.update(i, b_idx, new_blk_cache[b_idx])
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        self.current_pos = torch.zeros(batch_size, dtype=torch.long, device=device)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x, next_cache


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys = apply_rope(keys, cos, sin, offset=start_pos)

        # KV caching
        next_cache = []
        for i in range(b):
            prev = cache[i] if cache else None
            if prev is None:
                k_cat = keys[i:i+1]
                v_cat = values[i:i+1]
            else:
                prev_k, prev_v = prev
                k_cat = torch.cat([prev_k, keys[i:i+1]], dim=2)
                v_cat = torch.cat([prev_v, values[i:i+1]], dim=2)
            next_cache.append((k_cat, v_cat))

        keys = torch.cat([k for k, _ in next_cache], dim=0)
        values = torch.cat([v for _, v in next_cache], dim=0)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Attention
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        # attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        # PyTorch fails to do the implicit casting, so we have to be intentional with the types
        scale = torch.tensor(self.head_dim**0.5, dtype=queries.dtype, device=queries.device)
        attn_weights = torch.softmax(attn_scores / scale, dim=-1).to(values.dtype)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context), next_cache


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin, offset):
    # x: (batch_size, num_heads, seq_len, head_dim)
    bsz, n_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    assert offset.shape[0] == bsz, "Offset must have one value per batch item"

    # Prepare cos/sin: (seq_len, head_dim)
    cos = cos[:cos.shape[0], :].unsqueeze(0).unsqueeze(0)  # (1, 1, total_seq_len, head_dim)
    sin = sin[:sin.shape[0], :].unsqueeze(0).unsqueeze(0)

    # Build position indices per batch item
    position_ids = torch.arange(seq_len, device=offset.device).unsqueeze(0) + offset.unsqueeze(1)  # (bsz, seq_len)
    position_ids = position_ids.clamp(max=cos.shape[2] - 1)

    # Gather cos/sin for each position
    cos = cos[0, 0, position_ids, :]  # (bsz, seq_len, head_dim)
    sin = sin[0, 0, position_ids, :]

    # Expand for multi-heads
    cos = cos.unsqueeze(1)  # (bsz, 1, seq_len, head_dim)
    sin = sin.unsqueeze(1)

    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)
