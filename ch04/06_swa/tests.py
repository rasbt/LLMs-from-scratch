import copy

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
