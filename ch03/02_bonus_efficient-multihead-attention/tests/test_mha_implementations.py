from pathlib import Path
import torch
import pytest


from llms_from_scratch.utils import import_definitions_from_notebook


@pytest.fixture
def nb_imports():
    nb_dir = Path(__file__).resolve().parents[1]
    mod = import_definitions_from_notebook(nb_dir, "mha-implementations.ipynb")
    return mod


def copy_weights(from_mha, to_mha):
    with torch.no_grad():
        to_mha.W_query.copy_(from_mha.W_query.weight.T)
        to_mha.W_key.copy_(from_mha.W_key.weight.T)
        to_mha.W_value.copy_(from_mha.W_value.weight.T)

        to_mha.out_proj.weight.copy_(from_mha.out_proj.weight)
        to_mha.out_proj.bias.copy_(from_mha.out_proj.bias)


@pytest.mark.parametrize(
    "d_in,d_out,batch,seq_len,num_heads,seed",
    [
        (768, 768, 2, 4, 12, 123),  # d_in == d_out
        (768, 1536, 2, 4, 12, 456),  # d_in != d_out
        (1024, 512, 2, 4, 8, 789),   # d_in > d_out
    ],
)
def test_mha_einsum_matches_ch03(d_in, d_out, batch, seq_len, num_heads, seed, nb_imports):
    torch.manual_seed(seed)

    x = torch.randn(batch, seq_len, d_in)

    mha_linear = nb_imports.Ch03_MHA(
        d_in=d_in,
        d_out=d_out,
        context_length=seq_len,
        dropout=0.0,
        num_heads=num_heads,
        qkv_bias=False,
    ).eval()

    mha_einsum = nb_imports.MHAEinsum(
        d_in=d_in,
        d_out=d_out,
        context_length=seq_len,
        dropout=0.0,
        num_heads=num_heads,
        qkv_bias=False,
    ).eval()

    copy_weights(mha_linear, mha_einsum)

    out_linear = mha_linear(x)
    out_einsum = mha_einsum(x)

    assert out_linear.shape == out_einsum.shape == torch.Size([batch, seq_len, d_out])
    assert torch.allclose(out_linear, out_einsum, atol=1e-5)
