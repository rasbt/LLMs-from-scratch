# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""Plot KV-cache memory for MHA, GQA, and cross-layer KV sharing."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter


DTYPE_BYTES = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
}


PRESETS = {
    "gemma4_e2b": {
        "n_query_heads": 8,
        "n_kv_heads": 1,
        "head_dim": 256,
        "n_layers": 35,
        "n_kv_producing_layers": 15,
    },
    "gemma4_e4b": {
        "n_query_heads": 8,
        "n_kv_heads": 2,
        "head_dim": 320,
        "n_layers": 42,
        "n_kv_producing_layers": 24,
    },
}


def bytes_to_gb(n):
    return n / (1000 ** 3)


def context_tick_formatter(value, _pos):
    labels = {
        256: "256",
        1024: "1k",
        4096: "4k",
        16384: "16k",
        65536: "64k",
        131072: "128k",
    }
    return labels.get(int(round(value)), "")


def calc_kv_bytes_total(
    batch_size,
    context_length,
    head_dim,
    n_kv_heads,
    n_cached_layers,
    bytes_per_elem,
):
    return (
        batch_size
        * context_length
        * 2
        * head_dim
        * n_kv_heads
        * n_cached_layers
        * bytes_per_elem
    )


def compute_kv_curve(
    context_lengths,
    batch_size,
    head_dim,
    n_kv_heads,
    n_cached_layers,
    bytes_per_elem,
):
    curve = []
    for context_length in context_lengths:
        total_bytes = calc_kv_bytes_total(
            batch_size=batch_size,
            context_length=context_length,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
            n_cached_layers=n_cached_layers,
            bytes_per_elem=bytes_per_elem,
        )
        curve.append(bytes_to_gb(total_bytes))
    return curve


def add_end_label(ax, x_value, y_value, text, color, y_offset=0.0):
    ax.text(
        x_value * 1.16,
        y_value + y_offset,
        text,
        color=color,
        fontsize=8,
        va="center",
        ha="left",
        clip_on=False,
    )


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Plot KV-cache memory for MHA, GQA, and cross-layer KV sharing",
    )
    p.add_argument("--preset", choices=PRESETS.keys(), default="gemma4_e4b")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="bf16")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    cfg = PRESETS[args.preset]
    bytes_per_elem = DTYPE_BYTES[args.dtype]
    context_lengths = [
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
    ]

    curves = {
        "MHA": compute_kv_curve(
            context_lengths,
            batch_size=args.batch_size,
            head_dim=cfg["head_dim"],
            n_kv_heads=cfg["n_query_heads"],
            n_cached_layers=cfg["n_layers"],
            bytes_per_elem=bytes_per_elem,
        ),
        "GQA": compute_kv_curve(
            context_lengths,
            batch_size=args.batch_size,
            head_dim=cfg["head_dim"],
            n_kv_heads=cfg["n_kv_heads"],
            n_cached_layers=cfg["n_layers"],
            bytes_per_elem=bytes_per_elem,
        ),
        "GQA + KV sharing": compute_kv_curve(
            context_lengths,
            batch_size=args.batch_size,
            head_dim=cfg["head_dim"],
            n_kv_heads=cfg["n_kv_heads"],
            n_cached_layers=cfg["n_kv_producing_layers"],
            bytes_per_elem=bytes_per_elem,
        ),
    }

    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    styles = {
        "MHA": {"color": "#2f2f2f", "linewidth": 2.0, "zorder": 3},
        "GQA": {"color": "#6f7f8f", "linewidth": 1.8, "zorder": 3},
        "GQA + KV sharing": {"color": "#2f7ae5", "linewidth": 2.2, "zorder": 4},
    }

    for label, values in curves.items():
        ax.plot(context_lengths, values, solid_capstyle="round", **styles[label])

    max_y = max(curves["MHA"]) * 1.08
    ax.set_xscale("log", base=2)
    ax.set_xlim(context_lengths[0], context_lengths[-1] * 2.1)
    ax.set_ylim(0, max_y)
    ax.xaxis.set_major_locator(FixedLocator([256, 1024, 4096, 16384, 65536, 131072]))
    ax.xaxis.set_major_formatter(FuncFormatter(context_tick_formatter))

    ax.set_xlabel("Context length (tokens, log scale)")
    ax.set_ylabel("KV cache across all layers (GB)")
    ax.set_title("GQA and KV sharing compound the KV-cache savings", loc="left", pad=20)
    ax.text(
        0.0,
        1.02,
        (
            f"{args.preset}: {cfg['n_query_heads']} query heads, "
            f"{cfg['n_kv_heads']} KV heads, {cfg['n_layers']} layers, "
            f"{cfg['n_kv_producing_layers']} K/V-producing layers, "
            f"batch {args.batch_size}, {args.dtype}"
        ),
        transform=ax.transAxes,
        fontsize=8,
        color="#666666",
        ha="left",
        va="bottom",
    )

    mha_128k = curves["MHA"][-1]
    gqa_128k = curves["GQA"][-1]
    sharing_128k = curves["GQA + KV sharing"][-1]

    ax.text(
        0.02,
        0.96,
        (
            f"At 128k tokens, MHA would need {mha_128k:.1f} GB;\n"
            f"GQA cuts this to {gqa_128k:.1f} GB, and KV sharing to {sharing_128k:.1f} GB."
        ),
        transform=ax.transAxes,
        fontsize=8,
        color="#333333",
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none"},
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#a8a8a8")
    ax.spines["bottom"].set_color("#a8a8a8")
    ax.tick_params(axis="both", which="major", length=3, color="#888888", labelsize=8)
    ax.tick_params(axis="both", which="minor", length=0)
    ax.grid(False)

    add_end_label(ax, context_lengths[-1], mha_128k, f"MHA {mha_128k:.1f} GB", "#2f2f2f")
    add_end_label(ax, context_lengths[-1], gqa_128k, f"GQA {gqa_128k:.1f} GB", "#6f7f8f", 0.25)
    add_end_label(
        ax,
        context_lengths[-1],
        sharing_128k,
        f"GQA + KV sharing {sharing_128k:.1f} GB",
        "#2f7ae5",
        -0.25,
    )

    output = args.output
    if output is None:
        output = Path(f"kv_memory_mha_gqa_kvsharing_{args.preset}.pdf")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"Saved plot to: {output}")


if __name__ == "__main__":
    main()
