# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# KV-cache memory estimator for MHA, GQA, and cross-layer KV sharing.

import argparse
import math

DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
}


def convert_bytes(n):
    gb = n / (1000 ** 3)
    return f"{gb:,.2f} GB"


def calc_kv_bytes_total(batch, context_length, emb_dim, n_heads,
                        n_kv_heads, n_cached_layers, bytes_per_elem):
    head_dim = math.ceil(emb_dim / n_heads)
    per_layer = batch * context_length * head_dim * n_kv_heads * 2 * bytes_per_elem
    return per_layer * n_cached_layers


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Estimate KV-cache memory for MHA, GQA, and cross-layer KV sharing",
    )
    p.add_argument("--context_length", default=1024, type=int)
    p.add_argument("--emb_dim", required=True, type=int)
    p.add_argument("--n_heads", required=True, type=int)
    p.add_argument("--n_layers", required=True, type=int)
    p.add_argument("--n_kv_groups", required=True, type=int)
    p.add_argument("--n_kv_producing_layers", required=True, type=int)
    p.add_argument("--batch_size", default=1, type=int)
    p.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="fp16")
    args = p.parse_args()

    cfg = {
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "n_kv_groups": args.n_kv_groups,
        "n_kv_producing_layers": args.n_kv_producing_layers,
    }

    if cfg["n_heads"] % cfg["n_kv_groups"] != 0:
        raise ValueError("n_kv_groups must divide n_heads exactly.")
    if not 1 <= cfg["n_kv_producing_layers"] <= cfg["n_layers"]:
        raise ValueError("n_kv_producing_layers must be between 1 and n_layers.")

    bytes_per_elem = DTYPE_BYTES[args.dtype]
    head_dim = math.ceil(cfg["emb_dim"] / cfg["n_heads"])

    n_kv_heads_mha = cfg["n_heads"]
    n_kv_heads_gqa = cfg["n_heads"] // cfg["n_kv_groups"]

    total_mha = calc_kv_bytes_total(
        args.batch_size,
        cfg["context_length"],
        cfg["emb_dim"],
        cfg["n_heads"],
        n_kv_heads_mha,
        cfg["n_layers"],
        bytes_per_elem,
    )
    total_gqa = calc_kv_bytes_total(
        args.batch_size,
        cfg["context_length"],
        cfg["emb_dim"],
        cfg["n_heads"],
        n_kv_heads_gqa,
        cfg["n_layers"],
        bytes_per_elem,
    )
    total_mha_sharing = calc_kv_bytes_total(
        args.batch_size,
        cfg["context_length"],
        cfg["emb_dim"],
        cfg["n_heads"],
        n_kv_heads_mha,
        cfg["n_kv_producing_layers"],
        bytes_per_elem,
    )
    total_gqa_sharing = calc_kv_bytes_total(
        args.batch_size,
        cfg["context_length"],
        cfg["emb_dim"],
        cfg["n_heads"],
        n_kv_heads_gqa,
        cfg["n_kv_producing_layers"],
        bytes_per_elem,
    )

    print("==== Config ====")
    for k, v in cfg.items():
        print(f"{k:23}: {v}")
    print(f"batch_size             : {args.batch_size}")
    print(f"dtype                  : {args.dtype} ({bytes_per_elem} Bytes/elem)")
    print(f"head_dim               : {head_dim}")
    print(f"GQA n_kv_heads         : {n_kv_heads_gqa}")
    print()

    print("==== KV-cache totals across all layers ====")
    print(f"MHA total KV cache        : {convert_bytes(total_mha)}")
    print(f"GQA total KV cache        : {convert_bytes(total_gqa)}")
    print(f"MHA + KV sharing          : {convert_bytes(total_mha_sharing)}")
    print(f"GQA + KV sharing          : {convert_bytes(total_gqa_sharing)}")
    print(f"Ratio (MHA / GQA+sharing) : {total_mha / total_gqa_sharing:,.2f}x")
    print(f"Savings vs MHA            : {(1 - total_gqa_sharing / total_mha) * 100:,.2f}%")


if __name__ == "__main__":
    main()
