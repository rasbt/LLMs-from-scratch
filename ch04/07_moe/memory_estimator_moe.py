# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import argparse

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


def get_num_param_matrices(ffn_type):
    if ffn_type == "gelu":
        return 2
    elif ffn_type == "swiglu":
        return 3
    else:
        raise ValueError("--ffn_type must be 'gelu' or 'swiglu'")


def ffn_params(emb_dim, hidden_dim, ffn_type):
    return get_num_param_matrices(ffn_type) * emb_dim * hidden_dim


def router_params(emb_dim, num_experts):
    return emb_dim * num_experts


def estimate_params_and_hidden(
    emb_dim, hidden_dim, ffn_type, num_experts, match_dense=False
):
    P_dense = ffn_params(emb_dim, hidden_dim, ffn_type)
    R = router_params(emb_dim, num_experts)

    if match_dense:
        num_param_matrices = get_num_param_matrices(ffn_type)
        num = P_dense - R
        den = num_experts * num_param_matrices * emb_dim
        if num <= 0:
            raise ValueError("Dense layer too small for requested num_experts.")
        moe_hidden_dim = int(round(num / float(den)))
    else:
        moe_hidden_dim = hidden_dim

    per_expert_params = ffn_params(emb_dim, moe_hidden_dim, ffn_type)
    moe_total = num_experts * per_expert_params + R

    return {
        "dense_params": P_dense,
        "router": R,
        "moe_hidden_dim": moe_hidden_dim,
        "per_expert_params": per_expert_params,
        "moe_total": moe_total,
    }


def main():
    p = argparse.ArgumentParser(
        description="Estimate FFN vs MoE parameter memory"
    )
    p.add_argument("--emb_dim", type=int, required=True,
                   help="Model embedding dimension.")
    p.add_argument("--hidden_dim", type=int, required=True,
                   help="Dense FFN intermediate size (hidden dimension).")
    p.add_argument("--ffn_type", choices=["gelu", "swiglu"], default="swiglu")
    p.add_argument("--num_experts", type=int, default=8)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="bf16")
    p.add_argument(
        "--match_dense",
        action="store_true",
        help=("Auto-set per-expert hidden so MoE total params ~= dense FFN params "
              "(router included)."),
    )
    args = p.parse_args()

    bytes_per_elem = DTYPE_BYTES[args.dtype]

    res = estimate_params_and_hidden(
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        ffn_type=args.ffn_type,
        num_experts=args.num_experts,
        match_dense=args.match_dense,
    )

    moe_active_params_per_token = (
        res["router"] + args.top_k * res["per_expert_params"]
    )

    print("==== Config ====")
    print(f"{'emb_dim':23}: {args.emb_dim}")
    print(f"{'hidden_dim':23}: {args.hidden_dim}")
    print(f"{'ffn_type':23}: {args.ffn_type}")
    print(f"{'num_experts':23}: {args.num_experts}")
    print(f"{'top_k':23}: {args.top_k}")
    print(f"{'dtype':23}: {args.dtype} ({bytes_per_elem} Bytes/elem)")
    print(f"{'match_dense':23}: {args.match_dense}")
    print()

    print("==== Model weights (parameters) ====")
    print(f"{'Dense FFN params':23}: {res['dense_params']:,} "
          f"({bytes_convert(res['dense_params'] * bytes_per_elem)})")
    print(f"{'Per-expert params':23}: {res['per_expert_params']:,} "
          f"({bytes_convert(res['per_expert_params'] * bytes_per_elem)})")
    print(f"{'Router params':23}: {res['router']:,} "
          f"({bytes_convert(res['router'] * bytes_per_elem)})")
    print(f"{'MoE TOTAL params':23}: {res['moe_total']:,} "
          f"({bytes_convert(res['moe_total'] * bytes_per_elem)})")
    print(f"{'MoE ACTIVE/Token':23}: {moe_active_params_per_token:,} "
          f"({bytes_convert(moe_active_params_per_token * bytes_per_elem)})")
    print(f"{'moe_hidden_dim':23}: {res['moe_hidden_dim']}")
    print()


if __name__ == "__main__":
    main()
