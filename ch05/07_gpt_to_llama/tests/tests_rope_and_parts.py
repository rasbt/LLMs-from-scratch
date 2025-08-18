# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# File for internal use (unit tests)

import io
import os
import sys
import types
import nbformat
from packaging import version
from typing import Optional, Tuple
import torch
import pytest
import transformers
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb


transformers_version = transformers.__version__

# LitGPT code function `litgpt_build_rope_cache` from https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py
# LitGPT is licensed under Apache v2: https://github.com/Lightning-AI/litgpt/blob/main/LICENSE


def litgpt_build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
    extra_config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enhanced Transformer with Rotary Position Embedding.

    Args:
        seq_len (int): Sequence length.
        n_elem (int): Number of elements (head dimension).
        device (torch.device, optional): Device for tensor allocations.
        base (int, optional): Base for computing inverse frequencies.
        condense_ratio (int, optional): Ratio to condense the position indices.
        extra_config (dict, optional): Configuration parameters for frequency adjustments (used by Llama 3.1 and 3.2)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cosine and sine caches for RoPE.
    """

    # Compute the inverse frequencies theta
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    if extra_config is not None:
        orig_context_len = extra_config["original_max_seq_len"]
        factor = extra_config["factor"]
        low_freq_factor = extra_config["low_freq_factor"]
        high_freq_factor = extra_config["high_freq_factor"]

        wavelen = 2 * torch.pi / theta
        ratio = orig_context_len / wavelen
        smooth_factor = (ratio - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_factor = torch.clamp(smooth_factor, min=0.0, max=1.0)

        # Compute adjusted_theta without masked indexing
        adjusted_theta = (1 - smooth_factor) * (theta / factor) + smooth_factor * theta
        theta = adjusted_theta

    # Create position indices `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


# LitGPT code from https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py
# LitGPT is licensed under Apache v2: https://github.com/Lightning-AI/litgpt/blob/main/LICENSE
def litgpt_apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2:]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    if cos.dim() > 1:
        # batch dimensions must align
        # sin/cos are (B, T, hs) so we unsqeeze -3 for nh
        # we count from back because all of apply_rope does
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)

    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


@pytest.fixture(scope="module")
def notebook():
    def import_definitions_from_notebook(notebooks):
        imported_modules = {}

        for fullname, names in notebooks.items():
            # Get the directory of the current test file
            current_dir = os.path.dirname(__file__)
            path = os.path.join(current_dir, "..", fullname + ".ipynb")
            path = os.path.normpath(path)

            # Load the notebook
            if not os.path.exists(path):
                raise FileNotFoundError(f"Notebook file not found at: {path}")

            with io.open(path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            # Create a module to store the imported functions and classes
            mod = types.ModuleType(fullname)
            sys.modules[fullname] = mod

            # Go through the notebook cells and only execute function or class definitions
            for cell in nb.cells:
                if cell.cell_type == "code":
                    cell_code = cell.source
                    for name in names:
                        # Check for function or class definitions
                        if f"def {name}" in cell_code or f"class {name}" in cell_code:
                            exec(cell_code, mod.__dict__)

            imported_modules[fullname] = mod

        return imported_modules

    notebooks = {
        "converting-gpt-to-llama2": ["SiLU", "RMSNorm", "precompute_rope_params", "compute_rope"],
        "converting-llama2-to-llama3": ["precompute_rope_params"]
    }

    return import_definitions_from_notebook(notebooks)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(123)


def test_rope_llama2(notebook):

    this_nb = notebook["converting-gpt-to-llama2"]

    # Settings
    batch_size = 1
    context_len = 4096
    num_heads = 4
    head_dim = 16
    theta_base = 10_000

    # Instantiate RoPE parameters
    cos, sin = this_nb.precompute_rope_params(head_dim=head_dim, context_length=context_len)

    # Dummy query and key tensors
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # Apply rotary position embeddings
    queries_rot = this_nb.compute_rope(queries, cos, sin)
    keys_rot = this_nb.compute_rope(keys, cos, sin)

    # Generate reference RoPE via HF

    if version.parse(transformers_version) < version.parse("4.48"):
        rot_emb = LlamaRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=context_len,
            base=theta_base
        )
    else:
        class RoPEConfig:
            dim: int = head_dim
            rope_theta = theta_base
            max_position_embeddings: int = 8192
            hidden_size = head_dim * num_heads
            num_attention_heads = num_heads

        config = RoPEConfig()
        rot_emb = LlamaRotaryEmbedding(config=config)

    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)
    torch.testing.assert_close(sin, ref_sin.squeeze(0))
    torch.testing.assert_close(cos, ref_cos.squeeze(0))
    torch.testing.assert_close(keys_rot, ref_keys_rot)
    torch.testing.assert_close(queries_rot, ref_queries_rot)

    # Generate reference RoPE via LitGPT
    litgpt_cos, litgpt_sin = litgpt_build_rope_cache(context_len, n_elem=head_dim, base=10_000)
    litgpt_queries_rot = litgpt_apply_rope(queries, litgpt_cos, litgpt_sin)
    litgpt_keys_rot = litgpt_apply_rope(keys, litgpt_cos, litgpt_sin)

    torch.testing.assert_close(sin, litgpt_sin)
    torch.testing.assert_close(cos, litgpt_cos)
    torch.testing.assert_close(keys_rot, litgpt_keys_rot)
    torch.testing.assert_close(queries_rot, litgpt_queries_rot)


def test_rope_llama3(notebook):

    nb1 = notebook["converting-gpt-to-llama2"]
    nb2 = notebook["converting-llama2-to-llama3"]

    # Settings
    batch_size = 1
    context_len = 8192
    num_heads = 4
    head_dim = 16
    theta_base = 500_000

    # Instantiate RoPE parameters
    cos, sin = nb2.precompute_rope_params(
        head_dim=head_dim,
        context_length=context_len,
        theta_base=theta_base
    )

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # Apply rotary position embeddings
    queries_rot = nb1.compute_rope(queries, cos, sin)
    keys_rot = nb1.compute_rope(keys, cos, sin)

    # Generate reference RoPE via HF
    if version.parse(transformers_version) < version.parse("4.48"):
        rot_emb = LlamaRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=context_len,
            base=theta_base
        )
    else:
        class RoPEConfig:
            dim: int = head_dim
            rope_theta = theta_base
            max_position_embeddings: int = 8192
            hidden_size = head_dim * num_heads
            num_attention_heads = num_heads

        config = RoPEConfig()
        rot_emb = LlamaRotaryEmbedding(config=config)

    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)

    torch.testing.assert_close(sin, ref_sin.squeeze(0))
    torch.testing.assert_close(cos, ref_cos.squeeze(0))
    torch.testing.assert_close(keys_rot, ref_keys_rot)
    torch.testing.assert_close(queries_rot, ref_queries_rot)

    # Generate reference RoPE via LitGPT
    litgpt_cos, litgpt_sin = litgpt_build_rope_cache(context_len, n_elem=head_dim, base=theta_base)
    litgpt_queries_rot = litgpt_apply_rope(queries, litgpt_cos, litgpt_sin)
    litgpt_keys_rot = litgpt_apply_rope(keys, litgpt_cos, litgpt_sin)

    torch.testing.assert_close(sin, litgpt_sin)
    torch.testing.assert_close(cos, litgpt_cos)
    torch.testing.assert_close(keys_rot, litgpt_keys_rot)
    torch.testing.assert_close(queries_rot, litgpt_queries_rot)


def test_rope_llama3_12(notebook):

    nb1 = notebook["converting-gpt-to-llama2"]
    nb2 = notebook["converting-llama2-to-llama3"]

    # Settings
    batch_size = 1
    context_len = 8192
    num_heads = 4
    head_dim = 16
    rope_theta = 500_000

    rope_config = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }

    # Instantiate RoPE parameters
    cos, sin = nb2.precompute_rope_params(
        head_dim=head_dim,
        theta_base=rope_theta,
        context_length=context_len,
        freq_config=rope_config,
    )

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # Apply rotary position embeddings
    queries_rot = nb1.compute_rope(queries, cos, sin)
    keys_rot = nb1.compute_rope(keys, cos, sin)

    # Generate reference RoPE via HF
    hf_rope_params = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    }

    class RoPEConfig:
        rope_type = "llama3"
        rope_scaling = hf_rope_params
        factor = 1.0
        dim: int = head_dim
        rope_theta = 500_000
        max_position_embeddings: int = 8192
        hidden_size = head_dim * num_heads
        num_attention_heads = num_heads

    config = RoPEConfig()

    rot_emb = LlamaRotaryEmbedding(config=config)
    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)

    torch.testing.assert_close(sin, ref_sin.squeeze(0))
    torch.testing.assert_close(cos, ref_cos.squeeze(0))
    torch.testing.assert_close(keys_rot, ref_keys_rot)
    torch.testing.assert_close(queries_rot, ref_queries_rot)

    # Generate reference RoPE via LitGPT
    litgpt_rope_config = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_seq_len": 8192
    }

    litgpt_cos, litgpt_sin = litgpt_build_rope_cache(
        context_len,
        n_elem=head_dim,
        base=rope_theta,
        extra_config=litgpt_rope_config
    )
    litgpt_queries_rot = litgpt_apply_rope(queries, litgpt_cos, litgpt_sin)
    litgpt_keys_rot = litgpt_apply_rope(keys, litgpt_cos, litgpt_sin)

    torch.testing.assert_close(sin, litgpt_sin)
    torch.testing.assert_close(cos, litgpt_cos)
    torch.testing.assert_close(keys_rot, litgpt_keys_rot)
    torch.testing.assert_close(queries_rot, litgpt_queries_rot)


def test_silu(notebook):
    example_batch = torch.randn(2, 3, 4)
    silu = notebook["converting-gpt-to-llama2"].SiLU()
    assert torch.allclose(silu(example_batch), torch.nn.functional.silu(example_batch))


@pytest.mark.skipif(torch.__version__ < "2.4", reason="Requires PyTorch 2.4 or newer")
def test_rmsnorm(notebook):
    example_batch = torch.randn(2, 3, 4)
    rms_norm = notebook["converting-gpt-to-llama2"].RMSNorm(emb_dim=example_batch.shape[-1], eps=1e-5)
    rmsnorm_pytorch = torch.nn.RMSNorm(example_batch.shape[-1], eps=1e-5)

    assert torch.allclose(rms_norm(example_batch), rmsnorm_pytorch(example_batch))
