# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from llms_from_scratch.ch04 import generate_text_simple
from llms_from_scratch.qwen3 import (
    compute_rope_params,
    apply_rope,
    QWEN_CONFIG_06_B,
    RMSNorm,
    Qwen3Model,
    Qwen3Tokenizer
)
from llms_from_scratch.kv_cache.qwen3 import Qwen3Model as Qwen3ModelKV
from llms_from_scratch.kv_cache.generate import generate_text_simple as generate_text_simple_cached

from llms_from_scratch.kv_cache_batched.qwen3 import Qwen3Model as Qwen3ModelKVBatched
from llms_from_scratch.kv_cache_batched.generate import generate_text_simple as generate_text_simple_batched

import importlib
import pytest
import torch
import torch.nn as nn


class Qwen3RMSNorm(nn.Module):
    # Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
    # License: Apache License, Version 2.0 (see file above)
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        print(input_dtype)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


transformers_installed = importlib.util.find_spec("transformers") is not None


@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_rope():

    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding, apply_rotary_pos_emb

    # Settings
    batch_size = 1
    context_len = 8192
    num_heads = 4
    head_dim = 16
    rope_theta = 1_000_000

    # Instantiate RoPE parameters
    cos, sin = compute_rope_params(
        head_dim=head_dim,
        theta_base=rope_theta,
        context_length=context_len,
    )

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # Apply rotary position embeddings
    queries_rot = apply_rope(queries, cos, sin)
    keys_rot = apply_rope(keys, cos, sin)

    # Generate reference RoPE via HF
    class RoPEConfig:
        rope_type = "qwen3"
        factor = 1.0
        dim: int = head_dim
        rope_theta = 1_000_000
        max_position_embeddings: int = 8192
        hidden_size = head_dim * num_heads
        num_attention_heads = num_heads

    config = RoPEConfig()

    rot_emb = Qwen3RotaryEmbedding(config=config)
    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)

    torch.testing.assert_close(sin, ref_sin.squeeze(0))
    torch.testing.assert_close(cos, ref_cos.squeeze(0))
    torch.testing.assert_close(keys_rot, ref_keys_rot)
    torch.testing.assert_close(queries_rot, ref_queries_rot)


@pytest.fixture(scope="session")
def qwen3_weights_path(tmp_path_factory):
    """Creates and saves a deterministic model for testing."""
    path = tmp_path_factory.mktemp("models") / "qwen3_test_weights.pt"

    if not path.exists():
        torch.manual_seed(123)
        model = Qwen3Model(QWEN_CONFIG_06_B)
        torch.save(model.state_dict(), path)

    return path


@pytest.mark.parametrize("ModelClass", [Qwen3Model, Qwen3ModelKV])
@pytest.mark.parametrize("generate_fn", [generate_text_simple])
def test_model_variants(ModelClass, qwen3_weights_path, generate_fn):

    torch.manual_seed(123)
    model = ModelClass(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(qwen3_weights_path))
    model.eval()

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path="tokenizer-base.json",
        repo_id="rasbt/qwen3-from-scratch",
        add_generation_prompt=False,
        add_thinking=False
    )

    prompt = "Give me a short introduction to large language models."
    input_token_ids = tokenizer.encode(prompt)
    input_token_ids = torch.tensor([input_token_ids])

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", prompt)
    print("Encoded input text:", input_token_ids)
    print("encoded_tensor.shape:", input_token_ids.shape)

    out = generate_fn(
        model=model,
        idx=input_token_ids,
        max_new_tokens=5,
        context_size=QWEN_CONFIG_06_B["context_length"]
    )
    print("Encoded output text:", out)
    expect = torch.tensor([
        [151644, 872, 198, 35127, 752, 264, 2805, 16800, 311,
         3460, 4128,  4119, 13, 151645, 198, 112120, 83942, 60483,
         102652, 7414]
    ])
    assert torch.equal(expect, out)


def test_model_KV_noKV(qwen3_weights_path):

    torch.manual_seed(123)
    model_KV = Qwen3ModelKV(QWEN_CONFIG_06_B)
    model_KV.load_state_dict(torch.load(qwen3_weights_path))
    model_KV.eval()

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path="tokenizer-base.json",
        repo_id="rasbt/qwen3-from-scratch",
        add_generation_prompt=False,
        add_thinking=False
    )

    prompt = "Give me a short introduction to large language models."
    input_token_ids = tokenizer.encode(prompt)
    input_token_ids = torch.tensor([input_token_ids])

    out_KV = generate_text_simple_cached(
        model=model_KV,
        idx=input_token_ids,
        max_new_tokens=5,
        context_size=QWEN_CONFIG_06_B["context_length"]
    )
    del model_KV

    torch.manual_seed(123)
    model_noKV = Qwen3Model(QWEN_CONFIG_06_B)
    model_noKV.load_state_dict(torch.load(qwen3_weights_path))
    model_noKV.eval()

    out_noKV = generate_text_simple(
        model=model_noKV,
        idx=input_token_ids,
        max_new_tokens=5,
        context_size=QWEN_CONFIG_06_B["context_length"]
    )

    assert torch.equal(out_noKV, out_KV)


def test_model_batched_KV(qwen3_weights_path):

    torch.manual_seed(123)
    model_KV = Qwen3ModelKV(QWEN_CONFIG_06_B)
    model_KV.load_state_dict(torch.load(qwen3_weights_path))
    model_KV.eval()

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path="tokenizer-base.json",
        repo_id="rasbt/qwen3-from-scratch",
        add_generation_prompt=False,
        add_thinking=False
    )

    # Batch size 1

    prompt = "Give me a short introduction to large language models."
    input_token_ids = tokenizer.encode(prompt)
    input_token_ids = torch.tensor([input_token_ids])

    out_KV = generate_text_simple_cached(
        model=model_KV,
        idx=input_token_ids,
        max_new_tokens=5,
        context_size=QWEN_CONFIG_06_B["context_length"]
    )
    del model_KV

    torch.manual_seed(123)
    model_KV_batched = Qwen3ModelKVBatched(QWEN_CONFIG_06_B)
    model_KV_batched.load_state_dict(torch.load(qwen3_weights_path))
    model_KV_batched.eval()

    out_KV_bs_1 = generate_text_simple_batched(
        model=model_KV_batched,
        idx=input_token_ids,
        max_new_tokens=5,
        context_size=QWEN_CONFIG_06_B["context_length"]
    )

    assert torch.equal(out_KV, out_KV_bs_1)

    # Batch size 2

    prompts = [
        "Give me a short introduction to large language models.",
        "Give me a short introduction to large language models."
    ]
    tokenized_prompts = [tokenizer.encode(p) for p in prompts]
    max_len = max(len(t) for t in tokenized_prompts)
    padded_token_ids = [
        t + [tokenizer.pad_token_id] * (max_len - len(t)) for t in tokenized_prompts
    ]
    input_tensor = torch.tensor(padded_token_ids)
    out_KV_bs_2 = generate_text_simple_batched(
        model=model_KV_batched,
        idx=input_tensor,
        max_new_tokens=5,
        context_size=QWEN_CONFIG_06_B["context_length"],
    )
    assert torch.equal(out_KV.squeeze(0), out_KV_bs_2[0]), (out_KV.squeeze(0).shape, out_KV_bs_2[0].shape)


def test_rmsnorm_equivalence():
    torch.manual_seed(42)

    hidden_size = 64
    batch_size = 8
    seq_len = 16

    rms_norm = RMSNorm(hidden_size)
    ref_norm = Qwen3RMSNorm(hidden_size)

    # Sync weights
    with torch.no_grad():
        ref_norm.weight.copy_(ref_norm.weight)

    x = torch.randn(batch_size, seq_len, hidden_size)

    out1 = rms_norm(x)
    out2 = ref_norm(x)

    torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_tokenizer_equivalence():
    from transformers import AutoTokenizer

    prompt = "Give me a short introduction to large language models."
    messages = [
        {"role": "user", "content": prompt},
    ]

    # Reasoning model tokenizer
    repo_id = "Qwen/Qwen3-0.6B"
    tokenizer_ref = AutoTokenizer.from_pretrained(repo_id)

    for states in ((True, True), (False, False)):
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path="Qwen3-0.6B/tokenizer.json",
            repo_id=repo_id,
            add_generation_prompt=states[0],
            add_thinking=states[1]
        )
        input_token_ids = tokenizer.encode(prompt)
        input_token_ids_ref = tokenizer_ref.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=states[0],
            enable_thinking=states[1],
        )
        assert input_token_ids == input_token_ids_ref, states

        output_text = tokenizer.decode(input_token_ids)
        out_text_ref = tokenizer_ref.decode(input_token_ids_ref)
        assert output_text == out_text_ref, states

        assert tokenizer_ref.eos_token_id == tokenizer.eos_token_id
        assert tokenizer_ref.pad_token_id == tokenizer.pad_token_id

    # Base model tokenizer
    repo_id = "Qwen/Qwen3-0.6B-Base"
    tokenizer_ref = AutoTokenizer.from_pretrained(repo_id)

    for states in ((True, True), (False, False)):
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path="Qwen3-0.6B-Base/tokenizer.json",
            repo_id=repo_id,
            add_generation_prompt=states[0],
            add_thinking=states[1]
        )
        input_token_ids = tokenizer.encode(prompt)
        input_token_ids_ref = tokenizer_ref.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=states[0],
            enable_thinking=states[1],
        )
        assert input_token_ids == input_token_ids_ref, states

        output_text = tokenizer.decode(input_token_ids)
        out_text_ref = tokenizer_ref.decode(input_token_ids_ref)
        assert output_text == out_text_ref, states

        assert tokenizer_ref.eos_token_id == tokenizer.eos_token_id
        assert tokenizer_ref.pad_token_id == tokenizer.pad_token_id

        assert tokenizer.encode("<|endoftext|>") == [tokenizer._special_to_id["<|endoftext|>"]]
        assert tokenizer.encode("<|im_end|>") == [tokenizer._special_to_id["<|im_end|>"]]

        expected_eos_token = "<|im_end|>" if "Base" not in repo_id else "<|endoftext|>"
        expected_pad_token = "<|endoftext|>"
        assert tokenizer.decode([tokenizer.eos_token_id]) == expected_eos_token
        assert tokenizer.decode([tokenizer.pad_token_id]) == expected_pad_token
