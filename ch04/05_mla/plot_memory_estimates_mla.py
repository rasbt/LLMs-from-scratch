# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import matplotlib.pyplot as plt

# Bytes per element
DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
}


def bytes_to_gb(n_bytes):
    return n_bytes / (1000. ** 3)


def kv_bytes_total_mha(batch, context_length, emb_dim, n_heads,
                       n_layers, bytes_per_elem):
    head_dim = emb_dim / n_heads
    per_layer = batch * context_length * head_dim * n_heads * 2 * bytes_per_elem
    return per_layer * n_layers


def kv_bytes_total_mla(batch, context_length, n_layers, latent_dim, bytes_per_elem):
    return batch * context_length * n_layers * latent_dim * bytes_per_elem


def plot_abs_kv_vs_context_multiple():
    n_heads = 24
    emb_dim = 2048
    n_layers = 48
    batch_size = 1
    dtype = "bf16"
    bytes_per_elem = DTYPE_BYTES[dtype]

    context_lengths = [
        256, 512, 1024, 2048, 4096, 8192,
        16384, 32768, 65536, 131072
    ]

    mha_gb = []
    for L in context_lengths:
        total_mha = kv_bytes_total_mha(
            batch_size, L, emb_dim, n_heads, n_layers, bytes_per_elem
        )
        mha_gb.append(bytes_to_gb(total_mha))

    latent_dims = [1024, 512, 256, 64]
    plt.figure()
    plt.plot(context_lengths, mha_gb, marker="o", label="MHA (KV total)")

    L_ref = context_lengths[-1]
    total_mha_ref = kv_bytes_total_mha(batch_size, L_ref, emb_dim, n_heads, n_layers, bytes_per_elem)

    for latent_dim in latent_dims:
        mla_gb = []
        for L in context_lengths:
            total_mla = kv_bytes_total_mla(
                batch_size, L, n_layers, latent_dim, bytes_per_elem
            )
            mla_gb.append(bytes_to_gb(total_mla))

        total_mla_ref = kv_bytes_total_mla(batch_size, L_ref, n_layers, latent_dim, bytes_per_elem)
        comp = total_mha_ref / total_mla_ref if total_mla_ref != 0 else float("inf")

        plt.plot(context_lengths, mla_gb, marker="o",
                 label=f"MLA (latent_dim={latent_dim}, {comp:,.1f}× compression)")

    plt.xscale("log")
    plt.xlabel("context_length (log scale)")
    plt.ylabel("Total KV cache (GB)")
    plt.title(
        "KV-cache vs Context Length — MHA vs MLA\n"
        f"(n_heads={n_heads}, emb_dim={emb_dim}, n_layers={n_layers}, "
        f"batch={batch_size}, dtype={dtype})",
        fontsize=8
    )
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("kv_bytes_vs_context_length.pdf")


if __name__ == "__main__":
    plot_abs_kv_vs_context_multiple()
