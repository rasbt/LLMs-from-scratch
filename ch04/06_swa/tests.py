import copy

import pytest
import torch

from llms_from_scratch.ch04 import GPTModel as GPTModelBase
from llms_from_scratch.ch04 import generate_text_simple
from gpt_with_kv_swa import GPTModel as GPTModelSWA
from gpt_with_kv_swa import MultiHeadAttentionWithSWA
from gpt_with_kv_swa import generate_text_simple_cached


def test_cached_prefill_matches_uncached_swa():
    torch.manual_seed(123)

    att = MultiHeadAttentionWithSWA(
        d_in=8,
        d_out=8,
        dropout=0.0,
        num_heads=2,
        sliding_window_size=4,
    )
    att.eval()
    ref_att = copy.deepcopy(att)

    x = torch.randn(1, 6, 8)

    expected = ref_att(x, use_cache=False)
    att.reset_cache()
    actual = att(x, use_cache=True)

    assert not torch.isnan(actual).any()
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0)
    assert att.cache_k.size(1) == 4
    assert att.cache_v.size(1) == 4


def test_swa_matches_base_model_when_window_equals_context():
    cfg = {
        "vocab_size": 257,
        "context_length": 8,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    swa_cfg = {
        **cfg,
        "sliding_window_size": cfg["context_length"],
        "sliding_window_stride": -1,
    }

    torch.manual_seed(123)
    base_model = GPTModelBase(cfg)
    swa_model = GPTModelSWA(swa_cfg)
    base_state = {
        key: value
        for key, value in base_model.state_dict().items()
        if not key.endswith(".att.mask")
    }
    load_result = swa_model.load_state_dict(base_state, strict=True)
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys

    base_model.eval()
    swa_model.eval()

    full_input = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"]))
    expected_logits = base_model(full_input)
    actual_logits = swa_model(full_input, use_cache=False)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=1e-6)

    prompt = full_input[:, :6]
    expected_ids = generate_text_simple(
        model=base_model,
        idx=prompt.clone(),
        max_new_tokens=2,
        context_size=cfg["context_length"],
    )
    actual_ids = generate_text_simple_cached(
        model=swa_model,
        idx=prompt.clone(),
        max_new_tokens=2,
        context_size=cfg["context_length"],
        use_cache=True,
    )
    assert torch.equal(actual_ids, expected_ids)


def test_estimator_totals_match_measured_model_cache():
    import importlib.util
    from pathlib import Path

    import memory_estimator_swa as est

    gqa_path = Path(__file__).resolve().parent.parent / "04_gqa" / "gpt_with_kv_gqa.py"
    spec = importlib.util.spec_from_file_location("gpt_with_kv_gqa", gqa_path)
    gqa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gqa)

    emb_dim, n_heads, n_kv_groups = 96, 6, 3
    context_length, n_layers, window = 16, 2, 4

    res = est.estimate_totals(
        context_length=context_length,
        sliding_window_size=window,
        emb_dim=emb_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        n_kv_groups=n_kv_groups,
        batch_size=1,
        dtype="bf16",
        swa_ratio="1:1",
    )
    assert res["n_kv_heads_gqa"] == n_kv_groups

    def measured_cache_bytes(num_kv_heads, seq_len):
        att = gqa.GroupedQueryAttention(
            d_in=emb_dim, d_out=emb_dim, dropout=0.0,
            num_heads=n_heads, num_kv_groups=num_kv_heads, dtype=torch.bfloat16,
        )
        att.eval()
        x = torch.randn(1, seq_len, emb_dim, dtype=torch.bfloat16)
        with torch.no_grad():
            att(x, use_cache=True)
        return (att.cache_k.numel() * att.cache_k.element_size()
                + att.cache_v.numel() * att.cache_v.element_size())

    # Full-attention GQA layers: estimator total equals the measured model cache
    assert res["total_gqa_allfull"] == measured_cache_bytes(n_kv_groups, context_length) * n_layers
    # MHA is GQA with one KV head per query head
    assert res["total_mha_allfull"] == measured_cache_bytes(n_heads, context_length) * n_layers

    # SWA layers: estimator uses the real model's capped window cache
    swa_att = MultiHeadAttentionWithSWA(
        d_in=emb_dim, d_out=emb_dim, dropout=0.0, num_heads=n_heads,
        sliding_window_size=window,
    ).to(torch.bfloat16)
    swa_att.eval()
    x = torch.randn(1, context_length, emb_dim, dtype=torch.bfloat16)
    with torch.no_grad():
        swa_att(x, use_cache=True)
    swa_measured = (swa_att.cache_k.numel() * swa_att.cache_k.element_size()
                    + swa_att.cache_v.numel() * swa_att.cache_v.element_size())
    assert swa_att.cache_k.size(1) == window
    # 1 SWA layer + 1 full MHA layer
    assert res["total_mixed_mha"] == swa_measured + measured_cache_bytes(n_heads, context_length)


@pytest.mark.parametrize("context_length,emb_dim,n_heads,n_layers,n_kv_groups,swa_ratio", [
    (256, 4096, 32, 32, 4, "5:1"),  # Prompt shorter than the SWA window.
    (2048, 4096, 32, 32, 4, "5:1"),  # Prompt exactly fills the window.
    (4096, 4096, 48, 36, 4, "5:1"),  # README plot's rounded head dimension.
    (4096, 96, 6, 2, 3, "1:5"),  # Incomplete layer block starts with SWA.
])
def test_plot_matches_estimator(context_length, emb_dim, n_heads, n_layers, n_kv_groups, swa_ratio):
    import memory_estimator_swa as est
    import plot_memory_estimates_swa as plot

    batch, window, bytes_per_elem = 1, 2048, 2
    expected = est.estimate_totals(
        context_length, window, emb_dim, n_heads, n_layers,
        n_kv_groups, batch, "bf16", swa_ratio,
    )
    assert plot.calc_kv_bytes_total_mha(
        batch, context_length, emb_dim, n_heads, n_layers, bytes_per_elem,
    ) == expected["total_mha_allfull"]
    assert plot.calc_kv_bytes_total_gqa(
        batch, context_length, emb_dim, n_heads, n_layers, bytes_per_elem, n_kv_groups,
    ) == pytest.approx(expected["total_gqa_allfull"])
    assert plot.calc_kv_bytes_total_mha_swa(
        batch, context_length, emb_dim, n_heads, n_layers, bytes_per_elem, window, swa_ratio,
    ) == expected["total_mixed_mha"]
    assert plot.calc_kv_bytes_total_gqa_swa(
        batch, context_length, emb_dim, n_heads, n_layers, bytes_per_elem, n_kv_groups, window, swa_ratio,
    ) == pytest.approx(expected["total_mixed_gqa"])
