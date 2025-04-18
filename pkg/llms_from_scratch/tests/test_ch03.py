# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


from llms_from_scratch.ch03 import MultiHeadAttention, PyTorchMultiHeadAttention
import torch


def test_mha():

    context_length = 100
    d_in = 256
    d_out = 16

    mha = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0, num_heads=2)

    batch = torch.rand(8, 6, d_in)
    context_vecs = mha(batch)

    context_vecs.shape == torch.Size([8, 6, d_out])

    # Test bonus class
    mha = PyTorchMultiHeadAttention(d_in, d_out, num_heads=2)

    batch = torch.rand(8, 6, d_in)
    context_vecs = mha(batch)

    context_vecs.shape == torch.Size([8, 6, d_out])
