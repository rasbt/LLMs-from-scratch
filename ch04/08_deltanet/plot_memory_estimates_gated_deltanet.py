# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import argparse
import numpy as np
import matplotlib.pyplot as plt

# Bytes per element
DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
}


def kv_bytes_total_mha(batch, context_length, emb_dim, n_layers, bytes_per_elem, n_heads):
    # Full attention (MHA)
    d_head = emb_dim // n_heads
    per_layer = batch * context_length * n_heads * d_head * 2 * bytes_per_elem
    return per_layer * n_layers


def kv_bytes_total_deltanet_no_conv(batch, emb_dim, n_layers, bytes_per_elem, n_heads):
    # Simple Gated DeltaNet (no convolutional mixing)
    d_head = emb_dim // n_heads
    per_layer = batch * n_heads * d_head * d_head * bytes_per_elem
    return per_layer * n_layers


def gb(x):
    return x / 1e9


def main():
    p = argparse.ArgumentParser(description="Memory vs. Context Length: MHA vs. DeltaNet (3:1 mix)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--emb_dim", type=int, default=2048)
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--n_layers", type=int, default=48)
    p.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="bf16")
    p.add_argument("--min_ctx", type=int, default=128)
    p.add_argument("--max_ctx", type=int, default=131_072)
    args = p.parse_args()

    step = 100
    ctx = np.arange(args.min_ctx, args.max_ctx + 1, step, dtype=int)
    bytes_per_elem = DTYPE_BYTES[args.dtype]

    # 1) Full attention only
    mha_bytes = np.array([
        kv_bytes_total_mha(args.batch, int(t), args.emb_dim, args.n_layers,
                           bytes_per_elem, args.n_heads)
        for t in ctx
    ], dtype=float)

    # 2) DeltaNet only
    dnet_bytes_const = kv_bytes_total_deltanet_no_conv(
        args.batch, args.emb_dim, args.n_layers,
        bytes_per_elem, args.n_heads
    )
    dnet_bytes = np.full_like(mha_bytes, fill_value=dnet_bytes_const, dtype=float)

    # 3) 3:1 layer ratio (3 DeltaNet : 1 Full Attention)
    n_mha_layers = args.n_layers / 4
    n_dnet_layers = args.n_layers - n_mha_layers
    mix_bytes = np.array([
        kv_bytes_total_mha(args.batch, int(t), args.emb_dim, n_mha_layers,
                           bytes_per_elem, args.n_heads)
        + kv_bytes_total_deltanet_no_conv(args.batch, args.emb_dim, n_dnet_layers,
                                          bytes_per_elem, args.n_heads)
        for t in ctx
    ], dtype=float)

    # Convert to GB
    mha_gb = gb(mha_bytes)
    dnet_gb = gb(dnet_bytes)
    mix_gb = gb(mix_bytes)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ctx, mha_gb, label="Full Attention (MHA) KV cache")
    ax.plot(ctx, dnet_gb, label="All Gated DeltaNet (no conv)")
    ax.plot(ctx, mix_gb, label="3:1 layer ratio (3 DeltaNet : 1 Full Attention)")

    ax.set_xlabel("Context length (number of tokens)")
    ax.set_ylabel("KV cache size (GB)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    fig.tight_layout()
    plt.savefig("deltanet_memory_plot.pdf", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
