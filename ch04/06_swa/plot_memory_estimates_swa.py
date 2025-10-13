# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# Sliding Window Attention (SWA) memory usage vs context length plot.
#
# This script mirrors the style and structure of plot_memory_estimates_mla.py.

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Bytes per element
DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
}


def bytes_to_gb(n_bytes):
    return n_bytes / (1000.0 ** 3)


def parse_ratio(ratio_str):
    # "--swa_ratio a:b" means a SWA layers for every b full layers within a block
    try:
        a_str, b_str = ratio_str.split(":")
        a, b = int(a_str), int(b_str)
        assert a >= 0 and b >= 0 and (a + b) > 0
        return a, b
    except Exception:
        raise ValueError("--swa_ratio must be in the form 'a:b' with nonnegative integers and a+b>0")


def kv_bytes_total_mha(batch, context_length, emb_dim, n_layers, bytes_per_elem):
    # For MHA, n_kv_heads = n_heads, which cancels out:
    # total = B * L * E * 2 (K,V) * bytes * n_layers
    return batch * context_length * emb_dim * 2 * bytes_per_elem * n_layers


def kv_bytes_total_gqa(
    batch, context_length, emb_dim, n_layers, bytes_per_elem, n_kv_groups
):
    # For GQA, n_kv_heads = n_heads / n_kv_groups
    # => scale the MHA total by 1 / n_kv_groups
    base = kv_bytes_total_mha(batch, context_length, emb_dim, n_layers, bytes_per_elem)
    return base / n_kv_groups


def kv_bytes_total_mha_swa(
    batch, context_length, emb_dim, n_layers, bytes_per_elem, window, swa_ratio
):
    # Split layers into SWA vs Full
    a, b = parse_ratio(swa_ratio)
    total_blocks = a + b
    n_swa_layers = int(round(n_layers * (a / total_blocks)))
    n_full_layers = n_layers - n_swa_layers

    total_full = kv_bytes_total_mha(
        batch, context_length, emb_dim, n_full_layers, bytes_per_elem
    )
    total_swa = kv_bytes_total_mha(
        batch, window, emb_dim, n_swa_layers, bytes_per_elem
    )
    return total_full + total_swa


def kv_bytes_total_gqa_swa(
    batch,
    context_length,
    emb_dim,
    n_layers,
    bytes_per_elem,
    n_kv_groups,
    window,
    swa_ratio,
):
    a, b = parse_ratio(swa_ratio)
    total_blocks = a + b
    n_swa_layers = int(round(n_layers * (a / total_blocks)))
    n_full_layers = n_layers - n_swa_layers

    total_full = kv_bytes_total_gqa(
        batch,
        context_length,
        emb_dim,
        n_full_layers,
        bytes_per_elem,
        n_kv_groups,
    )
    total_swa = kv_bytes_total_gqa(
        batch, window, emb_dim, n_swa_layers, bytes_per_elem, n_kv_groups
    )
    return total_full + total_swa


def main():
    p = argparse.ArgumentParser(
        description="KV-cache vs Context Length — MHA vs GQA with SWA overlays"
    )
    p.add_argument("--emb_dim", type=int, required=True)
    p.add_argument("--n_heads", type=int, required=True)
    p.add_argument("--n_layers", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="bf16")
    p.add_argument(
        "--sliding_window_size", type=int, required=True, help="SWA window size W"
    )
    p.add_argument("--swa_ratio", type=str, default="5:1", help="SWA:Full ratio, e.g., 5:1")
    p.add_argument(
        "--output", type=Path, default=Path("kv_bytes_vs_context_length.pdf")
    )
    args = p.parse_args()

    batch_size = args.batch_size
    emb_dim = args.emb_dim
    n_heads = args.n_heads
    n_layers = args.n_layers
    bytes_per_elem = DTYPE_BYTES[args.dtype]

    kv_groups = 4
    valid_g4 = (n_heads % kv_groups == 0)

    context_lengths = [
        256, 512, 1024, 2048, 4096, 8192,
        16384, 32768, 65536, 131072
    ]

    series = {
        "MHA (KV total)": [],
        f"SWA on MHA (ratio {args.swa_ratio}, W={args.sliding_window_size})": [],
    }
    if valid_g4:
        series["GQA kv_groups=4 (full)"] = []
        series[
            f"SWA on GQA kv_groups=4 (ratio {args.swa_ratio}, W={args.sliding_window_size})"
        ] = []

    for L in context_lengths:
        total_mha = kv_bytes_total_mha(
            batch_size, L, emb_dim, n_layers, bytes_per_elem
        )
        total_mha_swa = kv_bytes_total_mha_swa(
            batch_size,
            L,
            emb_dim,
            n_layers,
            bytes_per_elem,
            window=args.sliding_window_size,
            swa_ratio=args.swa_ratio,
        )
        series["MHA (KV total)"].append(bytes_to_gb(total_mha))
        series[
            f"SWA on MHA (ratio {args.swa_ratio}, W={args.sliding_window_size})"
        ].append(bytes_to_gb(total_mha_swa))

        if valid_g4:
            total_gqa = kv_bytes_total_gqa(
                batch_size, L, emb_dim, n_layers, bytes_per_elem, n_kv_groups=kv_groups
            )
            total_gqa_swa = kv_bytes_total_gqa_swa(
                batch_size,
                L,
                emb_dim,
                n_layers,
                bytes_per_elem,
                n_kv_groups=kv_groups,
                window=args.sliding_window_size,
                swa_ratio=args.swa_ratio,
            )
            series["GQA kv_groups=4 (full)"].append(bytes_to_gb(total_gqa))
            series[
                f"SWA on GQA kv_groups=4 (ratio {args.swa_ratio}, W={args.sliding_window_size})"
            ].append(bytes_to_gb(total_gqa_swa))

    plt.figure(figsize=(10, 5))
    x = np.array(context_lengths, dtype=float)

    colors = {
        "MHA": "#1f77b4",
        "GQA": "#ff7f0e",
    }

    for label, yvals in series.items():
        y = np.array(yvals, dtype=float)
        if np.all(np.isnan(y)):
            continue

        linestyle = "--" if "SWA" in label else "-"
        if "MHA" in label:
            color = colors["MHA"]
        elif "GQA" in label:
            color = colors["GQA"]
        else:
            color = None

        plt.plot(x, y, marker="o", label=label, linestyle=linestyle, color=color)

    plt.xscale("log")
    plt.xlabel("context_length (log scale)")
    plt.ylabel("Total KV cache (GB)")
    plt.title(
        "KV-cache vs Context Length — MHA vs GQA (SWA overlays)\n"
        f"(n_heads={n_heads}, emb_dim={emb_dim}, n_layers={n_layers}, "
        f"batch={batch_size}, dtype={args.dtype}; "
        f"SWA ratio={args.swa_ratio}, W={args.sliding_window_size})",
        fontsize=8,
    )
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()

    if not valid_g4:
        print(
            f"Skipped GQA kv_groups=4 because n_heads={args.n_heads} "
            "is not divisible by 4."
        )
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
