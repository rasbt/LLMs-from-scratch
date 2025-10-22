# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from llms_from_scratch.ch04 import generate_text_simple
from llms_from_scratch.qwen3 import (
    apply_rope,
    compute_rope_params,
    load_weights_into_qwen,
    QWEN_CONFIG_06_B,
    Qwen3Model,
    Qwen3Tokenizer,
    MoEFeedForward,
    RMSNorm,
)
from llms_from_scratch.kv_cache.qwen3 import Qwen3Model as Qwen3ModelKV
from llms_from_scratch.kv_cache.utils import KVCache
from llms_from_scratch.kv_cache.generate import generate_text_simple as generate_text_simple_cached

from llms_from_scratch.kv_cache_batched.qwen3 import Qwen3Model as Qwen3ModelKVBatched
from llms_from_scratch.kv_cache_batched.generate import generate_text_simple as generate_text_simple_batched

from llms_from_scratch.utils import download_file

import importlib
import os
import shutil
import tempfile
import platform
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


@pytest.fixture
def dummy_input():
    torch.manual_seed(123)
    return torch.randint(0, 100, (1, 8))  # batch size 1, seq length 8


@pytest.fixture
def dummy_cfg_base():
    return {
        "vocab_size": 100,
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_layers": 2,
        "n_heads": 4,
        "head_dim": 8,
        "n_kv_groups": 1,
        "qk_norm": False,
        "dtype": torch.float32,
        "rope_base": 1000000,
        "context_length": 64,
        "num_experts": 0,
    }


@pytest.fixture
def dummy_cfg_moe(dummy_cfg_base):
    cfg = dummy_cfg_base.copy()
    cfg.update({
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
    })
    return cfg


@torch.inference_mode()
def test_dummy_qwen3_forward(dummy_cfg_base, dummy_input):
    torch.manual_seed(123)
    model = Qwen3Model(dummy_cfg_base)
    out = model(dummy_input)
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_base["vocab_size"]), \
        f"Expected shape (1, seq_len, vocab_size), got {out.shape}"


@torch.inference_mode()
def test_dummy_qwen3_moe_forward(dummy_cfg_moe, dummy_input):
    torch.manual_seed(123)
    model = Qwen3Model(dummy_cfg_moe)
    out = model(dummy_input)
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_moe["vocab_size"]), \
        f"Expected shape (1, seq_len, vocab_size), got {out.shape}"
    assert any(hasattr(block.ff, "gate") for block in model.trf_blocks), \
        "Expected MoEFeedForward in at least one transformer block"


@torch.inference_mode()
def test_moe_forward_matches_reference(dummy_cfg_moe):
    torch.manual_seed(0)
    moe = MoEFeedForward(dummy_cfg_moe)
    x = torch.randn(2, 5, dummy_cfg_moe["emb_dim"])

    scores = moe.gate(x)
    topk_scores, topk_indices = torch.topk(scores, moe.num_experts_per_tok, dim=-1)
    topk_probs = torch.softmax(topk_scores, dim=-1)

    expert_outputs = []
    for e in range(moe.num_experts):
        hidden = torch.nn.functional.silu(moe.fc1[e](x)) * moe.fc2[e](x)
        out = moe.fc3[e](hidden)
        expert_outputs.append(out.unsqueeze(-2))
    expert_outputs = torch.cat(expert_outputs, dim=-2)

    gating_probs = torch.zeros_like(scores)
    for i in range(moe.num_experts_per_tok):
        indices = topk_indices[..., i:i+1]
        prob = topk_probs[..., i:i+1]
        gating_probs.scatter_(dim=-1, index=indices, src=prob)
    gating_probs = gating_probs.unsqueeze(-1)

    expected = (gating_probs * expert_outputs).sum(dim=-2)

    actual = moe(x)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@torch.inference_mode()
@pytest.mark.parametrize("cfg_name", ["dummy_cfg_base", "dummy_cfg_moe"])
def test_qwen3_kvcache_equivalence(cfg_name, request):
    cfg = request.getfixturevalue(cfg_name)

    if cfg["num_experts"] > 0 and platform.system() == "Linux":
        pytest.skip("Skipping MoE KV equivalence test on Linux due to nondeterministic expert routing")

    torch.manual_seed(123)
    model_regular = Qwen3Model(cfg)
    model_regular.eval()

    model_kv = Qwen3ModelKV(cfg)
    model_kv.eval()
    model_kv.load_state_dict(model_regular.state_dict())
    model_kv.reset_kv_cache()
    cache = KVCache(n_layers=cfg["n_layers"])

    torch.manual_seed(123)
    input_ids = torch.randint(0, cfg["vocab_size"], (1, 6))

    out_full = model_regular(input_ids)

    logits_stepwise = []
    for t in range(input_ids.size(1)):
        input_token = input_ids[:, t:t + 1]
        logits = model_kv(input_token, cache=cache)
        logits_stepwise.append(logits)
    out_kv = torch.cat(logits_stepwise, dim=1)

    assert out_full.shape == out_kv.shape, f"Shape mismatch: {out_full.shape} vs {out_kv.shape}"
    assert torch.allclose(out_full, out_kv, atol=1e-5, rtol=1e-3)


@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
@pytest.mark.parametrize("context_len", [1024, 8192, 40960])
def test_rope(context_len):

    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3RotaryEmbedding,
        apply_rotary_pos_emb,
    )

    # Settings
    batch_size = 1
    num_heads = 4
    head_dim = 16
    rope_theta = 1_000_000

    # Instantiate RoPE parameters (our implementation)
    cos, sin = compute_rope_params(
        head_dim=head_dim,
        theta_base=rope_theta,
        context_length=context_len,
    )

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # Apply rotary embeddings with our implementation
    queries_rot = apply_rope(queries, cos, sin)
    keys_rot = apply_rope(keys, cos, sin)

    # Generate reference RoPE via HF
    class RoPEConfig:
        rope_type = "qwen3"
        factor = 1.0
        dim: int = head_dim
        rope_theta = 1_000_000
        max_position_embeddings = context_len
        hidden_size = head_dim * num_heads
        num_attention_heads = num_heads

    config = RoPEConfig()

    rot_emb = Qwen3RotaryEmbedding(config=config)
    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)

    # torch.testing.assert_close(sin, ref_sin.squeeze(0), rtol=1e-5, atol=1e-6)
    # torch.testing.assert_close(cos, ref_cos.squeeze(0), rtol=1e-5, atol=1e-6)

    # torch.testing.assert_close(keys_rot, ref_keys_rot, rtol=1e-5, atol=1e-6)A
    # torch.testing.assert_close(queries_rot, ref_queries_rot, rtol=1e-5, atol=1e-6)

    assert torch.equal(sin, ref_sin.squeeze(0))
    assert torch.equal(cos, ref_cos.squeeze(0))

    assert torch.equal(keys_rot, ref_keys_rot)
    assert torch.equal(queries_rot, ref_queries_rot)


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
@pytest.mark.parametrize("repo_id, tok_file", [
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B/tokenizer.json"),  # Chat / Reasoning
    ("Qwen/Qwen3-0.6B-Base", "Qwen3-0.6B-Base/tokenizer.json"),  # Base
])
def test_all_special_tokens_roundtrip(repo_id, tok_file):
    from transformers import AutoTokenizer as HFTokenizer
    hf_tok = HFTokenizer.from_pretrained(repo_id)

    qt = Qwen3Tokenizer(
        tokenizer_file_path=tok_file,
        repo_id=repo_id,
        add_generation_prompt=False,
        add_thinking=False,
    )

    # Use the instance's actually-available specials
    active_specials = list(qt._special_to_id.keys())

    # Every available special has a concrete id and round-trips
    for sp, sp_id in qt._special_to_id.items():
        assert isinstance(sp_id, int) and sp_id >= 0, f"{sp} missing or invalid id"
        assert qt.encode(sp) == [sp_id], f"{sp} must encode to its single id"
        assert qt.decode([sp_id]) == sp, f"{sp} must decode back to itself"

    # Inline use preserves boundaries for available specials
    for sp in active_specials:
        s = f"hello {sp} world"
        ids = qt.encode(s, chat_wrapped=False)
        sp_id = qt._special_to_id[sp]
        assert sp_id in ids, f"{sp} id not found inline"
        assert qt.decode(ids) == s, f"Inline decode mismatch for {sp}"

    # EOS / PAD expectations
    is_base = ("Base" in repo_id)
    expected_eos = "<|endoftext|>" if is_base else "<|im_end|>"
    expected_pad = "<|endoftext|>"

    assert qt.decode([qt.eos_token_id]) == expected_eos
    assert qt.decode([qt.pad_token_id]) == expected_pad
    assert hf_tok.eos_token_id == qt.eos_token_id
    assert hf_tok.pad_token_id == qt.pad_token_id
    assert hf_tok.decode([hf_tok.eos_token_id], skip_special_tokens=False) == expected_eos
    assert hf_tok.decode([hf_tok.pad_token_id], skip_special_tokens=False) == expected_pad

    # Thinking tokens only on chat models
    if not is_base:
        assert qt._tok.token_to_id("<think>") == 151667
        assert qt._tok.token_to_id("</think>") == 151668
        assert qt.encode("<think>") == [151667]
        assert qt.encode("</think>") == [151668]
    else:
        assert "<think>" not in active_specials and "</think>" not in active_specials


@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
@pytest.mark.parametrize("add_gen, add_think", [(True, True), (True, False), (False, False)])
def test_chat_wrap_and_equivalence(add_gen, add_think):
    from transformers import AutoTokenizer

    prompt = "Give me a short introduction to large language models."
    messages = [{"role": "user", "content": prompt}]

    for repo_id, tok_file in [
        ("Qwen/Qwen3-0.6B", "Qwen3-0.6B/tokenizer.json"),
        ("Qwen/Qwen3-0.6B-Base", "Qwen3-0.6B-Base/tokenizer.json"),
    ]:
        hf_tok = AutoTokenizer.from_pretrained(repo_id)
        qt = Qwen3Tokenizer(
            tokenizer_file_path=tok_file,
            repo_id=repo_id,
            add_generation_prompt=add_gen,
            add_thinking=add_think,
        )

        # Our encode vs HF template
        ours = qt.encode(prompt)
        ref = hf_tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_gen,
            enable_thinking=add_think,
        )

        if add_gen and not add_think:
            pass  # skip edge case as this is not something we use in practice
        else:
            assert ours == ref, (repo_id, add_gen, add_think)

        # Round-trip decode equality
        assert qt.decode(ours) == hf_tok.decode(ref)

        # EOS/PAD parity
        assert qt.eos_token_id == hf_tok.eos_token_id
        assert qt.pad_token_id == hf_tok.pad_token_id


@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
@pytest.mark.parametrize("repo_id, tok_file", [
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B/tokenizer.json"),
    ("Qwen/Qwen3-0.6B-Base", "Qwen3-0.6B-Base/tokenizer.json"),
])
@pytest.mark.parametrize("add_gen, add_think", [
    (True, True),
    (False, False),
])
def test_multiturn_equivalence(repo_id, tok_file, add_gen, add_think):
    from transformers import AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(repo_id)
    qt = Qwen3Tokenizer(
        tokenizer_file_path=tok_file,
        repo_id=repo_id,
        add_generation_prompt=add_gen,
        add_thinking=add_think,
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarize transformers in one sentence."},
        {"role": "assistant", "content": "Transformers use attention to model long-range dependencies efficiently."},
        {"role": "user", "content": "Now add one concrete example."},
    ]

    # HF reference (ids and raw template text)
    ref_ids = hf_tok.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=add_gen, enable_thinking=add_think
    )
    ref_text = hf_tok.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=add_gen, enable_thinking=add_think
    )

    # Our encode over HF's raw template text
    ours_ids = qt.encode(ref_text, chat_wrapped=False)

    assert ours_ids == ref_ids, f"mismatch for ({repo_id}, add_gen={add_gen}, add_think={add_think})"

    # Round-trip decode equality
    ours_dec = qt.decode(ours_ids)
    ref_dec = hf_tok.decode(ref_ids, skip_special_tokens=False)
    assert ours_dec == ref_dec


@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_tokenizer_equivalence():
    from transformers import AutoTokenizer

    prompt = "Give me a short introduction to large language models."
    messages = [
        {"role": "user", "content": prompt},
    ]

    for apply_chat_template in (True, False):
        for s in ("-Base", ""):
            repo_id = f"Qwen/Qwen3-0.6B{s}"
            tokenizer_ref = AutoTokenizer.from_pretrained(repo_id)
            tokenizer_url = f"https://huggingface.co/Qwen/Qwen3-0.6B{s}/resolve/main/tokenizer.json"
            download_file(tokenizer_url, out_dir=".")

            old_name = "tokenizer.json"

            if not s:
                new_name = "tokenizer-reasoning.json"
            else:
                new_name = "tokenizer-base.json"

            try:
                shutil.move(old_name, new_name)
            except Exception:
                with tempfile.NamedTemporaryFile(delete=False, dir=".") as tmp_file:
                    shutil.copyfile(old_name, tmp_file.name)
                    os.replace(tmp_file.name, new_name)
                os.remove(old_name)

            for states in ((True, True), (False, False)):
                tokenizer = Qwen3Tokenizer(
                    tokenizer_file_path=new_name,
                    repo_id=repo_id,
                    apply_chat_template=apply_chat_template,
                    add_generation_prompt=states[0],
                    add_thinking=states[1]
                )
                input_token_ids = tokenizer.encode(prompt)

                if apply_chat_template:
                    input_token_ids_ref = tokenizer_ref.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=states[0],
                        enable_thinking=states[1],
                    )
                else:
                    input_token_ids_ref = input_token_ids

                assert input_token_ids == input_token_ids_ref, states

                output_text = tokenizer.decode(input_token_ids)
                out_text_ref = tokenizer_ref.decode(input_token_ids_ref)
                assert output_text == out_text_ref, states

                assert tokenizer.encode("<|endoftext|>") == [tokenizer._special_to_id["<|endoftext|>"]]
                assert tokenizer.encode("<|im_end|>") == [tokenizer._special_to_id["<|im_end|>"]]

                expected_eos_token = "<|im_end|>" if "base" not in new_name else "<|endoftext|>"
                expected_pad_token = "<|endoftext|>"
                assert tokenizer.decode([tokenizer.eos_token_id]) == expected_eos_token
                assert tokenizer.decode([tokenizer.pad_token_id]) == expected_pad_token


@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
@pytest.mark.parametrize("repo_id, tok_file", [
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B/tokenizer.json"),
])
@pytest.mark.parametrize("add_gen, add_think", [
    (True, True),
    (False, False),
])
def test_multiturn_prefix_stability(repo_id, tok_file, add_gen, add_think):
    from transformers import AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(repo_id)
    qt = Qwen3Tokenizer(
        tokenizer_file_path=tok_file,
        repo_id=repo_id,
        add_generation_prompt=add_gen,
        add_thinking=add_think,
    )

    turns = [
        [{"role": "user", "content": "Define perplexity briefly."}],
        [{"role": "assistant", "content": "A measure of how well a language model predicts a sample."}],
        [{"role": "user", "content": "And why lower is better?"}],
    ]

    prev_ids_qt, prev_ids_hf = None, None
    prev_ref_text = None
    running = []  # grows turn-by-turn

    for delta in turns:
        running += delta

        ref_ids = hf_tok.apply_chat_template(
            running, tokenize=True,
            add_generation_prompt=add_gen, enable_thinking=add_think
        )
        ref_text = hf_tok.apply_chat_template(
            running, tokenize=False,
            add_generation_prompt=add_gen, enable_thinking=add_think
        )

        # Normalize line endings to match our encoder's assumptions
        ref_text_norm = ref_text.replace("\r\n", "\n").replace("\r", "\n")

        # Our encode over HF’s raw template text
        ours_ids = qt.encode(ref_text_norm, chat_wrapped=False)

        # 1) Exact equality per stage
        if ours_ids != ref_ids:
            # Lightweight inline diff to aid debugging
            from itertools import zip_longest
            for i, (a, b) in enumerate(zip_longest(ours_ids, ref_ids, fillvalue=None)):
                if a != b:
                    slice_lo, slice_hi = max(0, i-6), i+6
                    ours_slice = ours_ids[slice_lo:slice_hi]
                    ref_slice = ref_ids[slice_lo:slice_hi]
                    ours_toks = [qt._tok.id_to_token(x) if x is not None else None for x in ours_slice]
                    ref_toks = hf_tok.convert_ids_to_tokens(ref_slice, skip_special_tokens=False)
                    raise AssertionError(
                        f"Stage mismatch for ({repo_id}, add_gen={add_gen}, add_think={add_think}) at index {i}\n"
                        f"OURS ids: {ours_slice}\nREF  ids: {ref_slice}\n"
                        f"OURS tok: {ours_toks}\nREF  tok: {ref_toks}\n"
                        f"OURS dec: {qt.decode(ours_slice)}\nREF  dec: {hf_tok.decode(ref_slice, skip_special_tokens=False)}"
                    )
        # If no raise, they match
        assert ours_ids == ref_ids

        # 2) Prefix stability only when HF's own *text* remained a prefix
        if prev_ids_hf is not None and prev_ref_text is not None:
            if ref_text.startswith(prev_ref_text):
                assert ours_ids[:len(prev_ids_qt)] == prev_ids_qt
                assert ref_ids[:len(prev_ids_hf)] == prev_ids_hf
            # else: HF modified earlier boundaries (e.g., inserted <think>), so skip prefix checks

        # 3) Decode parity at each step
        assert qt.decode(ours_ids) == hf_tok.decode(ref_ids, skip_special_tokens=False)

        prev_ids_qt, prev_ids_hf = ours_ids, ref_ids
        prev_ref_text = ref_text


@torch.inference_mode()
@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_qwen3_base_equivalence_with_transformers():

    from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

    # Tiny config so the test is fast
    cfg = {
        "vocab_size": 257,
        "context_length": 8,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "hidden_dim": 64,
        "head_dim": 8,
        "qk_norm": True,
        "n_kv_groups": 2,
        "rope_base": 1_000_000.0,
        "dtype": torch.float32,
    }
    model = Qwen3Model(cfg)

    hf_cfg = Qwen3Config(
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["context_length"],
        hidden_size=cfg["emb_dim"],
        num_attention_heads=cfg["n_heads"],
        num_hidden_layers=cfg["n_layers"],
        intermediate_size=cfg["hidden_dim"],
        head_dim=cfg["head_dim"],
        num_key_value_heads=cfg["n_kv_groups"],
        rope_theta=cfg["rope_base"],
        tie_word_embeddings=False,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    hf_model = Qwen3ForCausalLM(hf_cfg)

    hf_state = hf_model.state_dict()
    param_config = {"n_layers": cfg["n_layers"], "hidden_dim": cfg["hidden_dim"]}
    load_weights_into_qwen(model, param_config, hf_state)

    x = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]), dtype=torch.long)
    ours_logits = model(x)
    theirs_logits = hf_model(x).logits
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)
