import io
import sys
import types
import nbformat
import torch
import pytest
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb


@pytest.fixture(scope="module")
def notebook():
    def import_definitions_from_notebook(fullname, names):
        # Load the notebook
        path = fullname + ".ipynb"
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

    fullname = "converting-gpt-to-llama2"
    names = ["precompute_rope_params", "compute_rope", "SiLU", "RMSNorm"]
    return import_definitions_from_notebook(fullname, names)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(123)


def test_rope(notebook):
    # Settings
    batch_size = 1
    context_len = 5
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

    class RoPEConfig:
        rope_type = "default"
        rope_scaling = None
        factor = 1.0
        dim: int = head_dim
        rope_theta = 10000
        max_position_embeddings: int = 4096
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
    silu = notebook.SiLU()
    assert torch.allclose(silu(example_batch), torch.nn.functional.silu(example_batch))


def test_rmsnorm(notebook):
    example_batch = torch.randn(2, 3, 4)
    rms_norm = notebook.RMSNorm(emb_dim=example_batch.shape[-1])
    rmsnorm_pytorch = torch.nn.RMSNorm(example_batch.shape[-1], eps=1e-6)

    assert torch.allclose(rms_norm(example_batch), rmsnorm_pytorch(example_batch))
