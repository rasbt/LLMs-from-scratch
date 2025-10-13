# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# KV-cache memory estimator for MHA vs GQA with SWA.

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


def kv_bytes_per_layer(batch, context_length, head_dim, n_kv_heads, bytes_per_elem):
    # KV = batch * tokens * head_dim * n_kv_heads * 2 (K,V) * bytes
    return batch * context_length * head_dim * n_kv_heads * 2 * bytes_per_elem


def parse_ratio(ratio_str):
    # "--swa_ratio a:b" means a SWA layers for every b full layers within a block
    try:
        a_str, b_str = ratio_str.split(":")
        a, b = int(a_str), int(b_str)
        assert a >= 0 and b >= 0 and (a + b) > 0
        return a, b
    except Exception:
        raise ValueError("--swa_ratio must be in the form 'a:b' with nonnegative integers and a+b>0")


def distribute_layers(n_layers, a, b):
    block = a + b
    blocks = n_layers // block
    rem = n_layers % block
    swa = blocks * a + min(a, rem)
    full = blocks * b + max(0, rem - a)
    return swa, full


def estimate_totals(context_length, sliding_window_size, emb_dim, n_heads, n_layers,
                    n_kv_groups, batch_size, dtype, swa_ratio):
    if n_heads % n_kv_groups != 0:
        raise ValueError("n_kv_groups must divide n_heads exactly.")

    bytes_per_elem = DTYPE_BYTES[dtype]
    head_dim = math.ceil(emb_dim / n_heads)
    n_kv_heads_mha = n_heads
    n_kv_heads_gqa = n_heads // n_kv_groups

    a_swa, b_full = parse_ratio(swa_ratio)
    n_swa_layers, n_full_layers = distribute_layers(n_layers, a_swa, b_full)

    eff_W = min(context_length, sliding_window_size)
    L = context_length

    # Per-layer costs
    per_mha_full = kv_bytes_per_layer(batch_size, L, head_dim, n_kv_heads_mha, bytes_per_elem)
    per_gqa_full = kv_bytes_per_layer(batch_size, L, head_dim, n_kv_heads_gqa, bytes_per_elem)
    per_mha_swa = kv_bytes_per_layer(batch_size, eff_W, head_dim, n_kv_heads_mha, bytes_per_elem)
    per_gqa_swa = kv_bytes_per_layer(batch_size, eff_W, head_dim, n_kv_heads_gqa, bytes_per_elem)

    # Totals
    total_mha_allfull = per_mha_full * n_layers
    total_gqa_allfull = per_gqa_full * n_layers
    total_mixed_mha = n_swa_layers * per_mha_swa + n_full_layers * per_mha_full
    total_mixed_gqa = n_swa_layers * per_gqa_swa + n_full_layers * per_gqa_full

    return {
        "bytes_per_elem": bytes_per_elem,
        "head_dim": head_dim,
        "n_kv_heads_gqa": n_kv_heads_gqa,
        "eff_W": eff_W,
        "n_swa_layers": n_swa_layers,
        "n_full_layers": n_full_layers,
        "total_mha_allfull": total_mha_allfull,
        "total_gqa_allfull": total_gqa_allfull,
        "total_mixed_mha": total_mixed_mha,
        "total_mixed_gqa": total_mixed_gqa,
    }


def main():
    p = argparse.ArgumentParser(description="Estimate KV-cache memory for MHA/GQA with SWA layer ratio")
    p.add_argument("--context_length", default=1024, type=int)
    p.add_argument("--sliding_window_size", required=True, type=int,
                   help="SWA window size W per SWA layer.")
    p.add_argument("--emb_dim", required=True, type=int)
    p.add_argument("--n_heads", required=True, type=int)
    p.add_argument("--n_layers", required=True, type=int)
    p.add_argument("--n_kv_groups", required=True, type=int,
                   help="GQA groups; 1 means MHA-equivalent KV heads.")
    p.add_argument("--batch_size", default=1, type=int)
    p.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="fp16")
    p.add_argument("--swa_ratio", default="1:0",
                   help="SWA:Full layer ratio. Example '5:1' -> 5 SWA for each 1 full. "
                        "'1:5' -> 1 SWA for 5 full. Default '1:0' = all SWA.")
    args = p.parse_args()

    cfg = {
        "context_length": args.context_length,
        "sliding_window_size": args.sliding_window_size,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "n_kv_groups": args.n_kv_groups,
    }

    res = estimate_totals(
        context_length=cfg["context_length"],
        sliding_window_size=cfg["sliding_window_size"],
        emb_dim=cfg["emb_dim"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        n_kv_groups=cfg["n_kv_groups"],
        batch_size=args.batch_size,
        dtype=args.dtype,
        swa_ratio=args.swa_ratio,
    )

    print("==== Config ====")
    for k, v in cfg.items():
        print(f"{k:23}: {v}")
    print(f"batch_size             : {args.batch_size}")
    print(f"dtype                  : {args.dtype} ({res['bytes_per_elem']} Bytes/elem)")
    print(f"head_dim               : {res['head_dim']}")
    print(f"GQA n_kv_heads         : {res['n_kv_heads_gqa']}")
    print(f"Effective SWA window W : {res['eff_W']}")
    print(f"Layer ratio (SWA:Full) : {args.swa_ratio} -> "
          f"{res['n_swa_layers']} SWA, {res['n_full_layers']} Full")
    print()

    print("==== KV-cache totals across all layers ====")
    print(f"MHA KV total           : {bytes_convert(res['total_mha_allfull'])}")
    print(f"GQA KV total           : {bytes_convert(res['total_gqa_allfull'])}")
    print(f"MHA + SWA (ratio {args.swa_ratio})  : {bytes_convert(res['total_mixed_mha'])}")
    print(f"GQA + SWA (ratio {args.swa_ratio})  : {bytes_convert(res['total_mixed_gqa'])}")
    print()


if __name__ == "__main__":
    main()
