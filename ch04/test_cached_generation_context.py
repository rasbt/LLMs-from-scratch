import importlib.util
from pathlib import Path

import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent

BASE_CFG = {
    "vocab_size": 32,
    "context_length": 8,
    "emb_dim": 16,
    "n_heads": 4,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
}

# (script, extra config keys)
VARIANTS = {
    "kv_cache": ("03_kv-cache/gpt_with_kv_cache.py", {}),
    "mla_mha": ("05_mla/gpt_with_kv_mha.py", {}),
    "mla": ("05_mla/gpt_with_kv_mla.py", {"latent_dim": 4}),
    "swa_mha": ("06_swa/gpt_with_kv_mha.py", {}),
    "swa": ("06_swa/gpt_with_kv_swa.py", {"sliding_window_size": 4, "sliding_window_stride": 2}),
    "moe_ffn": ("07_moe/gpt_with_kv_ffn.py", {"hidden_dim": 32}),
    "moe": ("07_moe/gpt_with_kv_moe.py", {"hidden_dim": 32, "num_experts": 2, "num_experts_per_tok": 1}),
    "dsa": ("09_dsa/gpt_with_kv_dsa.py", {"index_n_heads": 2, "index_head_dim": 4, "topk": 4}),
    "kv_sharing_mha": ("10_kv-sharing/gpt_with_kv_mha.py", {}),
    "kv_sharing": ("10_kv-sharing/gpt_with_kv_sharing.py", {"n_kv_producing_layers": 1}),
}


def load_module(relpath):
    path = THIS_DIR / relpath
    spec = importlib.util.spec_from_file_location(path.stem + "_" + path.parent.name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=list(VARIANTS), ids=list(VARIANTS))
def model_setup(request):
    relpath, extra_cfg = VARIANTS[request.param]
    module = load_module(relpath)
    torch.manual_seed(123)
    model = module.GPTModel({**BASE_CFG, **extra_cfg})
    return model, module.generate_text_simple_cached


@pytest.mark.parametrize("prompt_len, max_new_tokens", [(8, 1), (4, 5), (10, 1), (8, 0)])
def test_cached_generation_at_context_boundary(model_setup, prompt_len, max_new_tokens):
    model, generate = model_setup
    prompt = torch.arange(prompt_len).unsqueeze(0)

    cached = generate(model, prompt, max_new_tokens, use_cache=True)
    uncached = generate(model, prompt, max_new_tokens, use_cache=False)

    assert cached.shape[1] == prompt_len + max_new_tokens
    assert torch.equal(cached[:, :prompt_len], prompt)
    assert torch.equal(cached, uncached)


@pytest.mark.parametrize("prompt_len, max_new_tokens", [(8, 2), (4, 6), (10, 2)])
def test_cached_generation_rejects_context_overflow(model_setup, prompt_len, max_new_tokens):
    model, generate = model_setup
    prompt = torch.arange(prompt_len).unsqueeze(0)

    with pytest.raises(AssertionError, match="context length"):
        generate(model, prompt, max_new_tokens, use_cache=True)
