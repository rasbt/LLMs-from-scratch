import importlib.util
import subprocess
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
GQA_MODULE_PATH = THIS_DIR.parent / "04_gqa" / "gpt_with_kv_gqa.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measured_gqa_cache_bytes(emb_dim, n_heads, n_kv_groups, context_length, batch=1):
    """Actual KV-cache bytes of the repo's GroupedQueryAttention in bf16."""
    gqa = load_module(GQA_MODULE_PATH, "gpt_with_kv_gqa")
    att = gqa.GroupedQueryAttention(
        d_in=emb_dim, d_out=emb_dim, dropout=0.0,
        num_heads=n_heads, num_kv_groups=n_kv_groups, dtype=torch.bfloat16,
    )
    att.eval()
    x = torch.randn(batch, context_length, emb_dim, dtype=torch.bfloat16)
    with torch.no_grad():
        att(x, use_cache=True)
    return (att.cache_k.numel() * att.cache_k.element_size()
            + att.cache_v.numel() * att.cache_v.element_size())


def test_estimator_total_matches_measured_model_cache():
    emb_dim, n_heads, n_kv_groups = 96, 6, 3
    context_length, n_layers = 16, 2

    est = load_module(THIS_DIR / "memory_estimator_mla.py", "memory_estimator_mla")
    estimator_total = est.calc_kv_bytes_total(
        batch=1,
        context_length=context_length,
        emb_dim=emb_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_groups,  # n_kv_groups is the number of key-value heads
        n_layers=n_layers,
        bytes_per_elem=2,  # bf16
    )

    measured_total = measured_gqa_cache_bytes(
        emb_dim, n_heads, n_kv_groups, context_length
    ) * n_layers

    assert estimator_total == measured_total


def test_cli_reports_n_kv_groups_as_kv_heads():
    out = subprocess.run(
        [sys.executable, str(THIS_DIR / "memory_estimator_mla.py"),
         "--context_length", "16", "--emb_dim", "96", "--n_heads", "6",
         "--n_layers", "2", "--n_kv_groups", "3", "--latent_dim", "16",
         "--dtype", "bf16"],
        capture_output=True, text=True, check=True,
    ).stdout

    # n_kv_groups is the number of key-value heads, so MHA (6 heads)
    # uses exactly 6/3 = 2x the GQA cache
    assert "GQA n_kv_heads   : 3" in out
    assert "Ratio (MHA / GQA)   : 2.00x" in out
