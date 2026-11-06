import subprocess
import sys
from pathlib import Path

import pytest
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


@pytest.mark.parametrize("top_k", [-1, 0, 5])
def test_estimator_rejects_invalid_top_k(top_k):
    result = subprocess.run(
        [sys.executable, str(THIS_DIR / "memory_estimator_moe.py"),
         "--emb_dim", "96", "--hidden_dim", "48",
         "--num_experts", "4", "--top_k", str(top_k)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "must be between 1 and --num_experts (4)" in result.stderr


# Independent counts for emb_dim=96, hidden_dim=48, and four experts.
# Matching the dense budget gives a rounded per-expert hidden dimension of 12.
PARAM_COUNTS = [
    ("gelu", False, 9216, 37248),
    ("gelu", True, 2304, 9600),
    ("swiglu", False, 13824, 55680),
    ("swiglu", True, 3456, 14208),
]


@pytest.mark.parametrize("ffn_type,match_dense,per_expert_params,total_params", PARAM_COUNTS)
@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_estimator_active_params_match_selected_experts(ffn_type, match_dense, per_expert_params, total_params, top_k):
    args = [sys.executable, str(THIS_DIR / "memory_estimator_moe.py"),
            "--emb_dim", "96", "--hidden_dim", "48", "--ffn_type", ffn_type,
            "--num_experts", "4", "--top_k", str(top_k)]
    if match_dense:
        args.append("--match_dense")
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout

    def value_after(label):
        line = next(l for l in out.splitlines() if l.startswith(label))
        return int(line.split(":")[1].split("(")[0].strip().replace(",", ""))

    # Every token uses the router and exactly top_k experts, even when all
    # experts are selected. An upper-bound check alone would allow zero counts.
    assert value_after("MoE ACTIVE/Token") == 384 + top_k * per_expert_params
    assert value_after("MoE TOTAL params") == total_params
    if top_k == 4:
        assert value_after("MoE ACTIVE/Token") == total_params


@pytest.fixture
def plot_module(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.syspath_prepend(str(THIS_DIR))
    module = load_module(THIS_DIR / "plot_memory_estimates_moe.py", "plot_memory_estimates_moe")
    yield module
    module.plt.close("all")


@pytest.mark.parametrize("ffn_type,match_dense,per_expert_params,total_params", PARAM_COUNTS)
@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_plot_counts_match_selected_experts(plot_module, ffn_type, match_dense, per_expert_params, total_params, top_k):
    active, total = plot_module.calc_moe_active_and_total(
        emb_dim=96, hidden_dim=48, ffn_type=ffn_type,
        num_experts=4, top_k=top_k, match_dense=match_dense,
    )
    assert active == 384 + top_k * per_expert_params
    assert total == total_params
    if top_k == 4:
        assert active == total


@pytest.mark.parametrize("top_k", [-1, 0, 5])
def test_plot_helper_rejects_invalid_top_k(plot_module, top_k):
    with pytest.raises(ValueError, match="must be between 1 and num_experts"):
        plot_module.calc_moe_active_and_total(
            emb_dim=96, hidden_dim=48, ffn_type="swiglu",
            num_experts=4, top_k=top_k,
        )


@pytest.mark.parametrize("top_k,expected_experts", [
    (2, [2, 4, 8, 16]),  # Default routing count.
    (3, [4, 8, 16]),  # The requested count need not occur in the sweep grid.
    (8, [8, 16]),  # README example.
])
def test_plot_keeps_top_k_fixed(plot_module, tmp_path, top_k, expected_experts):
    plot_module.plot_active_params_vs_experts(
        emb_dim=96, hidden_dim=48, top_k=top_k, max_experts=16,
        save_path=tmp_path / "moe.png",
    )
    ax = plot_module.plt.gca()
    # Check the actual plotted points, so the sweep cannot silently reduce k.
    assert list(ax.lines[0].get_xdata()) == expected_experts
    assert list(ax.lines[1].get_xdata()) == expected_experts


@pytest.mark.parametrize("top_k,max_experts,error", [
    (-1, 16, "--top_k must be at least 1"),
    (0, 16, "--top_k must be at least 1"),
    (8, 4, "No expert counts"),
    (3, 3, "No expert counts"),
    (513, 1024, "No expert counts"),
])
def test_plot_rejects_invalid_sweeps(plot_module, tmp_path, top_k, max_experts, error):
    with pytest.raises(ValueError, match=error):
        plot_module.plot_active_params_vs_experts(
            emb_dim=96, hidden_dim=48, top_k=top_k, max_experts=max_experts,
            save_path=tmp_path / "invalid.png",
        )
