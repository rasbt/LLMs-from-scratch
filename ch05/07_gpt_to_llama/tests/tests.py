import io
import os
import sys
import types
import nbformat
import torch
import pytest
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb


# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# File for internal use (unit tests)


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

    # Instantiate RoPE parameters
    cos, sin = this_nb.precompute_rope_params(head_dim=head_dim, context_length=context_len)

    # Dummy query and key tensors
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # Apply rotary position embeddings
    queries_rot = this_nb.compute_rope(queries, cos, sin)
    keys_rot = this_nb.compute_rope(keys, cos, sin)

    rot_emb = LlamaRotaryEmbedding(
        dim=head_dim,
        max_position_embeddings=context_len,
        base=10_000
    )

    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)

    torch.testing.assert_close(sin, ref_sin.squeeze(0))
    torch.testing.assert_close(cos, ref_cos.squeeze(0))
    torch.testing.assert_close(keys_rot, ref_keys_rot)
    torch.testing.assert_close(queries_rot, ref_queries_rot)


def test_rope_llama3(notebook):

    nb1 = notebook["converting-gpt-to-llama2"]
    nb2 = notebook["converting-llama2-to-llama3"]

    # Settings
    batch_size = 1
    context_len = 8192
    num_heads = 4
    head_dim = 16
    theta_base = 50_000

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

    rot_emb = LlamaRotaryEmbedding(
        dim=head_dim,
        max_position_embeddings=context_len,
        base=theta_base
    )

    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)

    torch.testing.assert_close(sin, ref_sin.squeeze(0))
    torch.testing.assert_close(cos, ref_cos.squeeze(0))
    torch.testing.assert_close(keys_rot, ref_keys_rot)
    torch.testing.assert_close(queries_rot, ref_queries_rot)


def test_rope_llama3_12(notebook):

    nb1 = notebook["converting-gpt-to-llama2"]
    nb2 = notebook["converting-llama2-to-llama3"]

    # Settings
    batch_size = 1
    context_len = 8192
    num_heads = 4
    head_dim = 16
    rope_theta = 50_000

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
        rope_theta = 50_000
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
