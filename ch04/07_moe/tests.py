import subprocess
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent


def load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_cfg():
    return {
        "vocab_size": 64, "context_length": 32, "emb_dim": 24,
        "hidden_dim": 48, "n_heads": 4, "n_layers": 2,
        "drop_rate": 0.0, "qkv_bias": False,
        "num_experts": 4, "num_experts_per_tok": 2,
    }


def test_moe_forward_matches_naive_per_token_reference():
    moe_module = load_module(THIS_DIR / "gpt_with_kv_moe.py", "gpt_with_kv_moe")
    cfg = small_cfg()
    torch.manual_seed(3)
    moe = moe_module.MoEFeedForward(cfg).eval()
    x = torch.randn(2, 5, cfg["emb_dim"])

    with torch.no_grad():
        out = moe(x)

        # Naive per-token reference: each token is routed to its top-k
        # experts and the expert outputs are combined with the gate probs
        scores = moe.gate(x)
        topk_scores, topk_idx = torch.topk(scores, cfg["num_experts_per_tok"], dim=-1)
        probs = torch.softmax(topk_scores, dim=-1)
        ref = torch.zeros_like(x)
        for b in range(x.shape[0]):
            for s in range(x.shape[1]):
                for k in range(cfg["num_experts_per_tok"]):
                    e = topk_idx[b, s, k].item()
                    h = torch.nn.functional.silu(moe.fc1[e](x[b, s])) * moe.fc2[e](x[b, s])
                    ref[b, s] += probs[b, s, k] * moe.fc3[e](h)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


def test_moe_estimator_counts_match_actual_module_params():
    moe_module = load_module(THIS_DIR / "gpt_with_kv_moe.py", "gpt_with_kv_moe")
    est = load_module(THIS_DIR / "memory_estimator_moe.py", "memory_estimator_moe")
    cfg = small_cfg()
    torch.manual_seed(3)
    moe = moe_module.MoEFeedForward(cfg)

    res = est.estimate_params_and_hidden(
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        ffn_type="swiglu",
        num_experts=cfg["num_experts"],
    )

    router_params = sum(p.numel() for p in moe.gate.parameters())
    per_expert_params = sum(p.numel() for p in moe.fc1[0].parameters())
    per_expert_params += sum(p.numel() for p in moe.fc2[0].parameters())
    per_expert_params += sum(p.numel() for p in moe.fc3[0].parameters())
    total_params = sum(p.numel() for p in moe.parameters())

    assert res["router"] == router_params
    assert res["per_expert_params"] == per_expert_params
    assert res["moe_total"] == total_params


def test_cached_prefill_matches_uncached():
    moe_module = load_module(THIS_DIR / "gpt_with_kv_moe.py", "gpt_with_kv_moe")
    torch.manual_seed(7)
    model = moe_module.GPTModel(small_cfg()).eval()
    prompt = torch.tensor([[1, 5, 9, 2, 40, 3]])

    with torch.no_grad():
        cached = model(prompt, use_cache=True)
        model.reset_kv_cache()
        uncached = model(prompt, use_cache=False)

    torch.testing.assert_close(cached, uncached, rtol=1e-5, atol=1e-6)


def test_cached_generation_matches_uncached():
    moe_module = load_module(THIS_DIR / "gpt_with_kv_moe.py", "gpt_with_kv_moe")
    torch.manual_seed(7)
    model = moe_module.GPTModel(small_cfg()).eval()
    prompt = torch.tensor([[1, 5, 9, 2, 40, 3]])

    cached = moe_module.generate_text_simple_cached(
        model, prompt.clone(), max_new_tokens=8, use_cache=True
    )
    uncached = moe_module.generate_text_simple_cached(
        model, prompt.clone(), max_new_tokens=8, use_cache=False
    )

    assert torch.equal(cached, uncached)


def test_estimator_rejects_top_k_above_num_experts():
    result = subprocess.run(
        [sys.executable, str(THIS_DIR / "memory_estimator_moe.py"),
         "--emb_dim", "96", "--hidden_dim", "48",
         "--num_experts", "4", "--top_k", "8"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "top_k" in result.stderr


def test_estimator_active_params_not_above_total():
    out = subprocess.run(
        [sys.executable, str(THIS_DIR / "memory_estimator_moe.py"),
         "--emb_dim", "96", "--hidden_dim", "48",
         "--num_experts", "4", "--top_k", "2"],
        capture_output=True, text=True, check=True,
    ).stdout

    def value_after(label):
        line = next(l for l in out.splitlines() if l.startswith(label))
        return int(line.split(":")[1].split("(")[0].strip().replace(",", ""))

    assert value_after("MoE ACTIVE/Token") <= value_after("MoE TOTAL params")


def test_plot_active_params_capped_by_num_experts():
    sys.path.insert(0, str(THIS_DIR))
    plot_mod = load_module(THIS_DIR / "plot_memory_estimates_moe.py", "plot_memory_estimates_moe")

    # With fewer experts than top_k, a token can only activate all experts,
    # so active params must not exceed total params
    active, total = plot_mod.calc_moe_active_and_total(
        emb_dim=96, hidden_dim=48, ffn_type="swiglu",
        num_experts=1, top_k=8,
    )
    assert active <= total

    active, total = plot_mod.calc_moe_active_and_total(
        emb_dim=96, hidden_dim=48, ffn_type="swiglu",
        num_experts=4, top_k=2,
    )
    assert active <= total
