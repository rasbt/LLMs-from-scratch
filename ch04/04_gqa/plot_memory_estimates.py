# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# Plot KV-cache


import matplotlib.pyplot as plt

# Import from ./memory_estimator.py
from memory_estimator import kv_bytes_total, DTYPE_BYTES


def savings_percent(total_mha, total_gqa):
    return (1.0 - (total_gqa / total_mha)) * 100.0


def plot_savings_vs_nkvgroups():
    n_heads = 24
    emb_dim = 2048
    n_layers = 48
    batch_size = 1
    context_length = 8192
    dtype = "bf16"
    bytes_per_elem = DTYPE_BYTES[dtype]

    total_mha = kv_bytes_total(
        batch_size,
        context_length,
        emb_dim,
        n_heads,
        n_heads,
        n_layers,
        bytes_per_elem,
    )

    xs = []
    ys = []
    for n_kv_groups in range(1, n_heads + 1):
        n_kv_heads = n_heads // n_kv_groups
        total_gqa = kv_bytes_total(
            batch_size,
            context_length,
            emb_dim,
            n_heads,
            n_kv_heads,
            n_layers,
            bytes_per_elem,
        )
        xs.append(n_kv_groups)
        ys.append(savings_percent(total_mha, total_gqa))

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("n_kv_groups")
    plt.ylabel("Savings vs MHA (%)")
    plt.title(
        "KV-cache Savings vs n_kv_groups\n"
        "(n_heads=24, emb_dim=2048, n_layers=48, "
        "batch=1, context=8192, dtype=bf16)", fontsize=8
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("savings_vs_n_kv_groups.pdf")


def plot_abs_kv_vs_context():
    n_heads = 24
    emb_dim = 2048
    n_layers = 48
    batch_size = 1
    n_kv_groups = 4
    dtype = "bf16"
    bytes_per_elem = DTYPE_BYTES[dtype]

    n_kv_heads_mha = n_heads
    n_kv_heads_gqa = n_heads // n_kv_groups

    context_lengths = [
        256, 512, 1024, 2048, 4096, 8192,
        16384, 32768, 65536, 131072
    ]

    xs = []
    mha_gib = []
    gqa_gib = []
    savings_pct = None

    for L in context_lengths:
        total_mha = kv_bytes_total(
            batch_size, L, emb_dim, n_heads,
            n_kv_heads_mha, n_layers, bytes_per_elem
        )
        total_gqa = kv_bytes_total(
            batch_size, L, emb_dim, n_heads,
            n_kv_heads_gqa, n_layers, bytes_per_elem
        )
        xs.append(L)
        mha_gib.append(total_mha / (1024**3))
        gqa_gib.append(total_gqa / (1024**3))
        if savings_pct is None:
            savings_pct = savings_percent(total_mha, total_gqa)

    plt.figure()
    plt.plot(xs, mha_gib, marker="o", label="MHA (KV total)")
    plt.plot(xs, gqa_gib, marker="o", label=f"GQA (n_kv_groups={n_kv_groups})")
    plt.xscale("log")
    plt.xlabel("context_length (log scale)")
    plt.ylabel("Total KV cache (GB)")
    plt.title(
        "KV-cache vs Context Length\n"
        "(n_heads=24, emb_dim=2048, n_layers=48, "
        "batch=1, n_kv_groups=4, dtype=bf16)", fontsize=8
    )
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("kv_bytes_vs_context_length.pdf")
    print(f"Savings is constant across lengths: ~{savings_pct:.2f}%.")


if __name__ == "__main__":
    plot_savings_vs_nkvgroups()
    plot_abs_kv_vs_context()
