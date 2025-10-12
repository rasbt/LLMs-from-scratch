# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# KV-cache memory estimator for MHA vs GQA vs MLA

import argparse
import math

DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
}


def bytes_convert(n):
    gb = n / (1000 ** 3)
    return f"{gb:,.2f} GB"


def kv_bytes_total(batch, context_length, emb_dim, n_heads,
                   n_kv_heads, n_layers, bytes_per_elem):
    # Generic KV-cache: per-head dim is embed_dim / n_heads, times 2 for K and V
    head_dim = math.ceil(emb_dim / n_heads)
    per_layer = batch * context_length * head_dim * n_kv_heads * 2 * bytes_per_elem
    return per_layer * n_layers


def mla_bytes_total(batch, context_length, n_layers, latent_dim, bytes_per_elem):
    # Simple MLA (per-token compressed latent)
    # bytes ≈ batch × seqlen × n_layers × latent_dim × bytes_per_elem
    return batch * context_length * n_layers * latent_dim * bytes_per_elem


def main():
    p = argparse.ArgumentParser(description="Estimate KV-cache memory for MHA vs GQA vs MLA")
    p.add_argument("--context_length", default=1024, type=int)
    p.add_argument("--emb_dim", required=True, type=int)
    p.add_argument("--n_heads", required=True, type=int)
    p.add_argument("--n_layers", required=True, type=int)
    p.add_argument("--n_kv_groups", required=True, type=int)
    p.add_argument("--latent_dim", required=True, type=int, help="MLA per-token latent dimension")
    p.add_argument("--batch_size", default=1, type=int)
    p.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="fp16")
    args = p.parse_args()

    cfg = {
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "n_kv_groups": args.n_kv_groups,
        "latent_dim": args.latent_dim,
    }

    if cfg["n_heads"] % cfg["n_kv_groups"] != 0:
        raise ValueError("n_kv_groups must divide n_heads exactly.")

    bytes_per_elem = DTYPE_BYTES[args.dtype]
    head_dim = math.ceil(cfg["emb_dim"] / cfg["n_heads"])

    n_kv_heads_mha = cfg["n_heads"]
    n_kv_heads_gqa = cfg["n_heads"] // cfg["n_kv_groups"]

    total_mha = kv_bytes_total(
        args.batch_size,
        cfg["context_length"],
        cfg["emb_dim"],
        cfg["n_heads"],
        n_kv_heads_mha,
        cfg["n_layers"],
        bytes_per_elem,
    )

    total_gqa = kv_bytes_total(
        args.batch_size,
        cfg["context_length"],
        cfg["emb_dim"],
        cfg["n_heads"],
        n_kv_heads_gqa,
        cfg["n_layers"],
        bytes_per_elem,
    )

    total_mla = mla_bytes_total(
        args.batch_size,
        cfg["context_length"],
        cfg["n_layers"],
        cfg["latent_dim"],
        bytes_per_elem,
    )

    ratio = total_mha / total_gqa if total_gqa != 0 else float("inf")
    savings = 1 - (total_gqa / total_mha) if total_mha != 0 else 0.0

    ratio_mha_mla = total_mha / total_mla if total_mla != 0 else float("inf")
    savings_mla = 1 - (total_mla / total_mha) if total_mha != 0 else 0.0

    print("==== Config ====")
    for k, v in cfg.items():
        print(f"{k:17}: {v}")
    print(f"batch_size       : {args.batch_size}")
    print(f"dtype            : {args.dtype} ({bytes_per_elem} Bytes/elem)")
    print(f"head_dim         : {head_dim}")
    print(f"GQA n_kv_heads   : {n_kv_heads_gqa}")
    print()

    print("==== KV-cache totals across all layers ====")
    print(f"MHA total KV cache  : {bytes_convert(total_mha)}")
    print(f"GQA total KV cache  : {bytes_convert(total_gqa)}")
    print(f"MLA total KV cache  : {bytes_convert(total_mla)}")
    print(f"Ratio (MHA / GQA)   : {ratio:,.2f}x")
    print(f"Savings (GQA vs MHA): {savings*100:,.2f}%")
    print(f"Ratio (MHA / MLA)   : {ratio_mha_mla:,.2f}x")
    print(f"Savings (MLA vs MHA): {savings_mla*100:,.2f}%")


if __name__ == "__main__":
    main()
