import torch

from gpt_with_kv_mha import GPTModel as GPTModelMHA
from gpt_with_kv_mha import generate_text_simple_cached as generate_text_simple_cached_mha
from gpt_with_kv_sharing import GPTModel as GPTModelKVSharing
from gpt_with_kv_sharing import generate_text_simple_cached as generate_text_simple_cached_sharing
from memory_estimator_kv_sharing import calc_kv_bytes_total


def test_kv_sharing_matches_mha_when_all_layers_produce_kv():
    cfg = {
        "vocab_size": 257,
        "context_length": 8,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 3,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    sharing_cfg = {**cfg, "n_kv_producing_layers": cfg["n_layers"]}

    torch.manual_seed(123)
    mha_model = GPTModelMHA(cfg)
    sharing_model = GPTModelKVSharing(sharing_cfg)
    load_result = sharing_model.load_state_dict(mha_model.state_dict(), strict=True)
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys

    mha_model.eval()
    sharing_model.eval()

    full_input = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"]))
    expected_logits = mha_model(full_input, use_cache=False)
    actual_logits = sharing_model(full_input, use_cache=False)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=1e-6)

    prompt = full_input[:, :6]
    expected_ids = generate_text_simple_cached_mha(
        model=mha_model,
        idx=prompt.clone(),
        max_new_tokens=2,
        context_size=cfg["context_length"],
        use_cache=True,
    )
    actual_ids = generate_text_simple_cached_sharing(
        model=sharing_model,
        idx=prompt.clone(),
        max_new_tokens=2,
        context_size=cfg["context_length"],
        use_cache=True,
    )
    assert torch.equal(actual_ids, expected_ids)


def test_only_producer_layers_store_kv_cache():
    cfg = {
        "vocab_size": 257,
        "context_length": 8,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 4,
        "n_kv_producing_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }

    torch.manual_seed(123)
    model = GPTModelKVSharing(cfg)
    model.eval()

    idx = torch.randint(0, cfg["vocab_size"], (1, 6))
    model.reset_kv_cache()
    logits = model(idx, use_cache=True)

    assert not torch.isnan(logits).any()
    for block in model.trf_blocks[:2]:
        assert block.att.cache_k.size(1) == idx.size(1)
        assert block.att.cache_v.size(1) == idx.size(1)
    for block in model.trf_blocks[2:]:
        assert block.att.cache_k is None
        assert block.att.cache_v is None


def test_memory_estimator_counts_cached_layers():
    batch_size = 1
    context_length = 128
    emb_dim = 32
    n_heads = 4
    n_kv_heads = 1
    n_cached_layers = 2
    bytes_per_elem = 2

    actual = calc_kv_bytes_total(
        batch_size,
        context_length,
        emb_dim,
        n_heads,
        n_kv_heads,
        n_cached_layers,
        bytes_per_elem,
    )
    expected = batch_size * context_length * 8 * n_kv_heads * 2 * bytes_per_elem * n_cached_layers
    assert actual == expected
