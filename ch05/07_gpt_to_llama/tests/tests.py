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
    def import_definitions_from_notebook(fullname, names):
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
        return mod

    # Specify the notebook name and functions/classes to import
    fullname = "converting-gpt-to-llama2"
    names = ["precompute_rope_params", "compute_rope", "SiLU", "RMSNorm"]

    # Import the required functions and classes from the notebook
    return import_definitions_from_notebook(fullname, names)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(123)


def test_rope_llama2(notebook):
    # Settings
    batch_size = 1
    context_len = 4096
    num_heads = 4
    head_dim = 16

    # Instantiate RoPE parameters
    cos, sin = notebook.precompute_rope_params(head_dim=head_dim, context_length=context_len)

    # Dummy query and key tensors
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # Apply rotary position embeddings
    queries_rot = notebook.compute_rope(queries, cos, sin)
    keys_rot = notebook.compute_rope(keys, cos, sin)

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
    # Settings
    batch_size = 1
    context_len = 8192
    num_heads = 4
    head_dim = 16
    theta_base = 50_000

    # Instantiate RoPE parameters
    cos, sin = notebook.precompute_rope_params(
        head_dim=head_dim,
        context_length=context_len,
        theta_base=theta_base
    )

    # Dummy query and key tensors
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # Apply rotary position embeddings
    queries_rot = notebook.compute_rope(queries, cos, sin)
    keys_rot = notebook.compute_rope(keys, cos, sin)

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


def test_silu(notebook):
    example_batch = torch.randn(2, 3, 4)
    silu = notebook.SiLU()
    assert torch.allclose(silu(example_batch), torch.nn.functional.silu(example_batch))


@pytest.mark.skipif(torch.__version__ < "2.4", reason="Requires PyTorch 2.4 or newer")
def test_rmsnorm(notebook):
    example_batch = torch.randn(2, 3, 4)
    rms_norm = notebook.RMSNorm(emb_dim=example_batch.shape[-1], eps=1e-5)
    rmsnorm_pytorch = torch.nn.RMSNorm(example_batch.shape[-1], eps=1e-5)

    assert torch.allclose(rms_norm(example_batch), rmsnorm_pytorch(example_batch))
