# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from llms_from_scratch.utils import import_definitions_from_notebook


def get_tiny_test_config(dtype=torch.float32):
    return {
        "vocab_size": 257,
        "vocab_size_per_layer_input": 257,
        "context_length": 64,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 3,
        "hidden_dim": 64,
        "head_dim": 8,
        "global_head_dim": 12,
        "qk_norm": True,
        "n_kv_heads": 2,
        "n_kv_groups": 2,
        "num_global_kv_heads": None,
        "rope_local_base": 10_000.0,
        "rope_global_base": 1_000_000.0,
        "rope_global_type": "proportional",
        "rope_global_partial_rotary_factor": 0.25,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention", "full_attention"],
        "dtype": dtype,
        "query_pre_attn_scalar": 1.0,
        "hidden_size_per_layer_input": 8,
        "num_kv_shared_layers": 1,
        "use_double_wide_mlp": True,
        "attention_k_eq_v": False,
        "final_logit_softcap": 30.0,
        "tie_word_embeddings": False,
        "layer_norm_eps": 1e-6,
        "pad_token_id": 0,
    }


@pytest.fixture
def import_notebook_defs():
    nb_dir = Path(__file__).resolve().parents[1]
    mod = import_definitions_from_notebook(nb_dir, "standalone-gemma4.ipynb")
    return mod


@pytest.fixture
def dummy_input():
    torch.manual_seed(123)
    return torch.randint(0, 100, (1, 8))


@pytest.fixture
def dummy_cfg_base():
    return get_tiny_test_config(dtype=torch.float32)


@contextmanager
def gemma4_transformers_module():
    try:
        transformers = importlib.import_module("transformers")
        if hasattr(transformers, "Gemma4ForCausalLM") and hasattr(transformers, "Gemma4TextConfig"):
            yield transformers
            return
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[3]
    transformers_src = repo_root / "temp" / "gemma-4" / "transformers-main" / "src"
    if not transformers_src.exists():
        pytest.skip("Local Gemma 4 Transformers source not found")

    saved_path = list(sys.path)
    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "transformers" or name.startswith("transformers.")
    }

    for name in list(saved_modules):
        sys.modules.pop(name, None)

    sys.path.insert(0, str(transformers_src))
    dummy_dep_module = types.ModuleType("transformers.dependency_versions_check")
    dummy_dep_module.dep_version_check = lambda *args, **kwargs: None
    sys.modules["transformers.dependency_versions_check"] = dummy_dep_module

    try:
        transformers = importlib.import_module("transformers")
        if not hasattr(transformers, "Gemma4ForCausalLM") or not hasattr(transformers, "Gemma4TextConfig"):
            pytest.skip("Gemma 4 is unavailable in the current Transformers environment")
        yield transformers
    finally:
        for name in list(sys.modules):
            if name == "transformers" or name.startswith("transformers."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


@torch.inference_mode()
def test_dummy_gemma4_forward(dummy_cfg_base, dummy_input, import_notebook_defs):
    torch.manual_seed(123)
    model = import_notebook_defs.Gemma4DenseModel(dummy_cfg_base)
    out = model(dummy_input)
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_base["vocab_size"])


@torch.inference_mode()
def test_dummy_gemma4_forward_without_explicit_n_kv_groups(dummy_cfg_base, dummy_input, import_notebook_defs):
    torch.manual_seed(123)
    cfg = dict(dummy_cfg_base)
    cfg.pop("n_kv_groups")
    cfg.pop("qk_norm")
    cfg.pop("query_pre_attn_scalar")
    model = import_notebook_defs.Gemma4DenseModel(cfg)
    out = model(dummy_input)
    assert out.shape == (1, dummy_input.size(1), cfg["vocab_size"])


@pytest.mark.parametrize(
    ("model_size", "expected"),
    [
        (
            "E2B",
            {
                "emb_dim": 1536,
                "hidden_dim": 6144,
                "n_layers": 35,
                "n_kv_heads": 1,
                "global_head_dim": 512,
                "num_kv_shared_layers": 20,
                "use_double_wide_mlp": True,
                "full_attention_layers": 7,
            },
        ),
        (
            "E4B",
            {
                "emb_dim": 2560,
                "hidden_dim": 10240,
                "n_layers": 42,
                "n_kv_heads": 2,
                "global_head_dim": 512,
                "num_kv_shared_layers": 18,
                "use_double_wide_mlp": False,
                "full_attention_layers": 7,
            },
        ),
    ],
)
def test_gemma4_named_configs(import_notebook_defs, model_size, expected):
    cfg = import_notebook_defs.get_gemma4_dense_config(model_size, dtype=torch.float32)
    for key, value in expected.items():
        if key == "full_attention_layers":
            assert cfg["layer_types"].count("full_attention") == value
        else:
            assert cfg[key] == value
    assert len(cfg["layer_types"]) == cfg["n_layers"]
    assert cfg["layer_types"][-1] == "full_attention"


@torch.inference_mode()
def test_gemma4_generation_helper_runs(dummy_cfg_base, dummy_input, import_notebook_defs):
    torch.manual_seed(123)
    model = import_notebook_defs.Gemma4DenseModel(dummy_cfg_base)
    token_ids = dummy_input[:, :4]

    generated = list(
        import_notebook_defs.generate_text_basic_stream(
            model=model,
            token_ids=token_ids,
            max_new_tokens=3,
            eos_token_id=None,
        )
    )

    assert len(generated) == 3
    for token in generated:
        assert token.shape == (1, 1)
        assert 0 <= int(token.item()) < dummy_cfg_base["vocab_size"]


@torch.inference_mode()
def test_gemma4_equivalence_with_transformers(import_notebook_defs):
    with gemma4_transformers_module() as transformers:
        cfg = get_tiny_test_config(dtype=torch.float32)
        model = import_notebook_defs.Gemma4DenseModel(cfg)

        hf_cfg = transformers.Gemma4TextConfig(
            vocab_size=cfg["vocab_size"],
            vocab_size_per_layer_input=cfg["vocab_size_per_layer_input"],
            hidden_size=cfg["emb_dim"],
            intermediate_size=cfg["hidden_dim"],
            num_hidden_layers=cfg["n_layers"],
            num_attention_heads=cfg["n_heads"],
            num_key_value_heads=cfg["n_kv_heads"],
            num_global_key_value_heads=cfg["num_global_kv_heads"],
            head_dim=cfg["head_dim"],
            global_head_dim=cfg["global_head_dim"],
            max_position_embeddings=cfg["context_length"],
            sliding_window=cfg["sliding_window"],
            layer_types=cfg["layer_types"],
            hidden_size_per_layer_input=cfg["hidden_size_per_layer_input"],
            num_kv_shared_layers=cfg["num_kv_shared_layers"],
            use_double_wide_mlp=cfg["use_double_wide_mlp"],
            attention_k_eq_v=cfg["attention_k_eq_v"],
            final_logit_softcapping=cfg["final_logit_softcap"],
            hidden_activation="gelu_pytorch_tanh",
            tie_word_embeddings=cfg["tie_word_embeddings"],
            rms_norm_eps=cfg["layer_norm_eps"],
            attention_bias=False,
            attention_dropout=0.0,
            rope_parameters={
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": cfg["rope_local_base"],
                },
                "full_attention": {
                    "rope_type": cfg["rope_global_type"],
                    "rope_theta": cfg["rope_global_base"],
                    "partial_rotary_factor": cfg["rope_global_partial_rotary_factor"],
                },
            },
            attn_implementation="eager",
            torch_dtype=torch.float32,
        )
        hf_model = transformers.Gemma4ForCausalLM(hf_cfg)

        hf_state = hf_model.state_dict()
        import_notebook_defs.load_weights_into_gemma4_dense(model, cfg, hf_state)

        x = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]), dtype=torch.long)
        ours_logits = model(x)
        theirs_logits = hf_model(x, use_cache=False).logits
        torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)


@torch.inference_mode()
def test_gemma4_loader_supports_multimodal_checkpoint_prefix(import_notebook_defs):
    with gemma4_transformers_module() as transformers:
        cfg = get_tiny_test_config(dtype=torch.float32)
        cfg["tie_word_embeddings"] = True
        model = import_notebook_defs.Gemma4DenseModel(cfg)

        hf_cfg = transformers.Gemma4TextConfig(
            vocab_size=cfg["vocab_size"],
            vocab_size_per_layer_input=cfg["vocab_size_per_layer_input"],
            hidden_size=cfg["emb_dim"],
            intermediate_size=cfg["hidden_dim"],
            num_hidden_layers=cfg["n_layers"],
            num_attention_heads=cfg["n_heads"],
            num_key_value_heads=cfg["n_kv_heads"],
            num_global_key_value_heads=cfg["num_global_kv_heads"],
            head_dim=cfg["head_dim"],
            global_head_dim=cfg["global_head_dim"],
            max_position_embeddings=cfg["context_length"],
            sliding_window=cfg["sliding_window"],
            layer_types=cfg["layer_types"],
            hidden_size_per_layer_input=cfg["hidden_size_per_layer_input"],
            num_kv_shared_layers=cfg["num_kv_shared_layers"],
            use_double_wide_mlp=cfg["use_double_wide_mlp"],
            attention_k_eq_v=cfg["attention_k_eq_v"],
            final_logit_softcapping=cfg["final_logit_softcap"],
            hidden_activation="gelu_pytorch_tanh",
            tie_word_embeddings=cfg["tie_word_embeddings"],
            rms_norm_eps=cfg["layer_norm_eps"],
            attention_bias=False,
            attention_dropout=0.0,
            rope_parameters={
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": cfg["rope_local_base"],
                },
                "full_attention": {
                    "rope_type": cfg["rope_global_type"],
                    "rope_theta": cfg["rope_global_base"],
                    "partial_rotary_factor": cfg["rope_global_partial_rotary_factor"],
                },
            },
            attn_implementation="eager",
            torch_dtype=torch.float32,
        )
        hf_model = transformers.Gemma4ForCausalLM(hf_cfg)

        prefixed_state = {}
        for key, value in hf_model.state_dict().items():
            if key.startswith("model."):
                prefixed_state[f"model.language_model.{key[len('model.') :]}"] = value

        num_loaded = import_notebook_defs.load_weights_into_gemma4_dense(model, cfg, prefixed_state)
        assert num_loaded > 0

        x = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]), dtype=torch.long)
        ours_logits = model(x)
        theirs_logits = hf_model(x, use_cache=False).logits
        torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)


@torch.inference_mode()
def test_gemma4_pretrained_e2b_it_checkpoint_generates_coherent_text():
    checkpoint_dir = Path(__file__).resolve().parents[1] / "gemma-4-E2B-it"
    weights_path = checkpoint_dir / "model.safetensors"
    tokenizer_path = checkpoint_dir / "tokenizer.json"

    if not weights_path.exists() or not tokenizer_path.exists():
        pytest.skip("Local Gemma 4 E2B-it checkpoint not found")

    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    notebook_dir = Path(__file__).resolve().parents[1]
    root_notebook_defs = import_definitions_from_notebook(notebook_dir, "standalone-gemma4.ipynb")

    cfg = root_notebook_defs.get_gemma4_dense_config("E2B", dtype=torch.float32)
    model = root_notebook_defs.Gemma4DenseModel(cfg)
    num_loaded = root_notebook_defs.load_weights_into_gemma4_dense(model, cfg, load_file(weights_path))
    assert num_loaded == 601

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    prompt = "<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n"
    token_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)

    generated = []
    for token in root_notebook_defs.generate_text_basic_stream(
        model=model,
        token_ids=token_ids,
        max_new_tokens=8,
        eos_token_id=tokenizer.token_to_id("<turn|>"),
    ):
        generated.append(int(token.item()))

    response = tokenizer.decode(generated, skip_special_tokens=True)
    assert response == "The capital of France is **Paris**."


@torch.inference_mode()
def test_gemma4_plus_kvcache_pretrained_e2b_it_checkpoint_generates_coherent_text():
    checkpoint_dir = Path(__file__).resolve().parents[1] / "gemma-4-E2B-it"
    weights_path = checkpoint_dir / "model.safetensors"
    tokenizer_path = checkpoint_dir / "tokenizer.json"

    if not weights_path.exists() or not tokenizer_path.exists():
        pytest.skip("Local Gemma 4 E2B-it checkpoint not found")

    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    notebook_dir = Path(__file__).resolve().parents[1]
    kv_notebook_defs = import_definitions_from_notebook(notebook_dir, "standalone-gemma4-plus-kvcache.ipynb")

    cfg = kv_notebook_defs.get_gemma4_dense_config("E2B", dtype=torch.float32)
    model = kv_notebook_defs.Gemma4DenseModel(cfg)
    num_loaded = kv_notebook_defs.load_weights_into_gemma4_dense(model, cfg, load_file(weights_path))
    assert num_loaded == 601

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    prompt = "<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n"
    token_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)

    generated = []
    for token in kv_notebook_defs.generate_text_basic_stream(
        model=model,
        token_ids=token_ids,
        max_new_tokens=8,
        eos_token_id=tokenizer.token_to_id("<turn|>"),
    ):
        generated.append(int(token.item()))

    response = tokenizer.decode(generated, skip_special_tokens=True)
    assert response == "The capital of France is **Paris**."
