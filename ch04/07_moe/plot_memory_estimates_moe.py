# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import argparse
import matplotlib.pyplot as plt
from ffn_moe_memory_estimator import (
    estimate_params_and_hidden,
    ffn_params,
    router_params,
)


def moe_active_and_total(
    emb_dim,
    hidden_dim,
    ffn_type,
    num_experts,
    top_k,
    match_dense=True,
):
    if match_dense:
        dense_params = ffn_params(emb_dim, hidden_dim, ffn_type)
        router = router_params(emb_dim, num_experts)
        if dense_params <= router:
            match_dense = False

    stats = estimate_params_and_hidden(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        ffn_type=ffn_type,
        num_experts=num_experts,
        match_dense=match_dense,
    )

    active = stats["router"] + top_k * stats["per_expert_params"]
    return active, stats["moe_total"]


def plot_active_params_vs_experts(
    emb_dim,
    hidden_dim,
    ffn_type="swiglu",
    top_k=2,
    max_experts=512,
    y_log=True,
    save_path=None,
    match_dense=True,
):
    experts = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512]
    experts = [e for e in experts if e <= max_experts]

    dense_active = ffn_params(emb_dim, hidden_dim, ffn_type)
    moe_active = []
    moe_total = []
    for e in experts:
        active, total = moe_active_and_total(
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            ffn_type=ffn_type,
            num_experts=e,
            top_k=top_k,
            match_dense=match_dense,
        )
        moe_active.append(active)
        moe_total.append(total)

    plt.figure(figsize=(7, 5))
    plt.plot(experts, moe_active, marker="o", label="MoE active per token")
    plt.plot(experts, moe_total, marker="s", linestyle="--", label="MoE total parameters")
    plt.axhline(dense_active, linestyle=":", color="gray",
                label="FFN dense (active = total)")

    plt.xlabel(f"Number of experts (top_k = {top_k})")
    plt.ylabel("Parameters")
    if y_log:
        plt.yscale("log")
    plt.title(
        f"Active vs Total Parameters per Token\n"
        f"(emb_dim={emb_dim}, hidden_dim={hidden_dim}, ffn={ffn_type}, top_k={top_k})"
    )
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Plot Dense vs MoE active parameters.")
    p.add_argument("--emb_dim", type=int, required=True, help="Embedding dimension")
    p.add_argument("--hidden_dim", type=int, required=True, help="Dense FFN hidden size")
    p.add_argument("--ffn_type", choices=["gelu", "swiglu"], default="swiglu")
    p.add_argument("--top_k", type=int, default=2, help="Active experts per token")
    p.add_argument("--max_experts", type=int, default=512, help="Max experts on x-axis")
    p.add_argument("--no_log", action="store_true", help="Disable log-scale y-axis")
    p.add_argument("--save", type=str, default=None, help="Optional path to save PNG")
    p.add_argument(
        "--no_match_dense",
        action="store_true",
        help=("Disable matching MoE parameters to dense FFN total; "
              "uses provided hidden_dim instead."),
    )
    args = p.parse_args()

    plot_active_params_vs_experts(
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        ffn_type=args.ffn_type,
        top_k=args.top_k,
        max_experts=args.max_experts,
        y_log=not args.no_log,
        save_path=args.save,
        match_dense=not args.no_match_dense,
    )


if __name__ == "__main__":
    main()
