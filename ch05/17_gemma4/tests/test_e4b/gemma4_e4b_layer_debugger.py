# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import sys
import types
from pathlib import Path

import torch

from llms_from_scratch.utils import import_definitions_from_notebook


def _import_gemma4_classes():
    try:
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig

        return Gemma4TextConfig, Gemma4ForCausalLM
    except Exception:
        repo_root = Path(__file__).resolve().parents[4]
        local_src = repo_root / "temp" / "gemma-4" / "transformers-main" / "src"
        if not local_src.exists():
            raise

        for name in list(sys.modules):
            if name == "transformers" or name.startswith("transformers."):
                del sys.modules[name]

        sys.path.insert(0, str(local_src))
        dummy_dep_module = types.ModuleType("transformers.dependency_versions_check")
        dummy_dep_module.dep_version_check = lambda *args, **kwargs: None
        sys.modules["transformers.dependency_versions_check"] = dummy_dep_module

        from transformers import Gemma4ForCausalLM, Gemma4TextConfig

        return Gemma4TextConfig, Gemma4ForCausalLM


try:
    Gemma4TextConfig, Gemma4ForCausalLM = _import_gemma4_classes()
except Exception:
    Gemma4TextConfig = None
    Gemma4ForCausalLM = None


def tiny_debug_config():
    return {
        "vocab_size": 257,
        "vocab_size_per_layer_input": 257,
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_layers": 3,
        "n_heads": 4,
        "head_dim": 8,
        "n_kv_heads": 2,
        "num_global_kv_heads": None,
        "global_head_dim": 12,
        "context_length": 8,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention", "full_attention"],
        "hidden_size_per_layer_input": 8,
        "num_kv_shared_layers": 1,
        "use_double_wide_mlp": True,
        "attention_k_eq_v": False,
        "rope_local_base": 10_000.0,
        "rope_global_base": 1_000_000.0,
        "rope_global_type": "proportional",
        "rope_global_partial_rotary_factor": 0.25,
        "layer_norm_eps": 1e-6,
        "final_logit_softcap": 30.0,
        "tie_word_embeddings": False,
        "dtype": torch.float32,
    }


def _hf_config_from_dict(cfg):
    if Gemma4TextConfig is None:
        raise ImportError("Gemma 4 classes are required for the layer debugger.")

    return Gemma4TextConfig(
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
        torch_dtype=cfg.get("dtype", torch.float32),
    )


def load_notebook_defs(nb_name="standalone-gemma4.ipynb"):
    nb_dir = Path(__file__).resolve().parents[2]
    return import_definitions_from_notebook(nb_dir, nb_name)


def build_gemma4_pair(import_notebook_defs, cfg, hf_checkpoint=None):
    if Gemma4ForCausalLM is None:
        raise ImportError("Gemma 4 classes are required for the layer debugger.")

    ours = import_notebook_defs.Gemma4DenseModel(cfg)

    if hf_checkpoint:
        hf_model = Gemma4ForCausalLM.from_pretrained(
            hf_checkpoint,
            torch_dtype=cfg.get("dtype", torch.float32),
            attn_implementation="eager",
        )
    else:
        hf_cfg = _hf_config_from_dict(cfg)
        hf_model = Gemma4ForCausalLM(hf_cfg)

    import_notebook_defs.load_weights_into_gemma4_dense(ours, cfg, hf_model.state_dict())
    hf_model.config.use_cache = False

    ours.eval()
    hf_model.eval()
    return ours, hf_model


def _attach_debug_hooks(model, is_hf):
    traces = {}
    handles = []

    def hook(name):
        def _record(_, __, output):
            if isinstance(output, tuple):
                output = output[0]
            traces[name] = output.detach().to(torch.float32).cpu()

        return _record

    if is_hf:
        core = model.model
        handles.append(core.embed_tokens.register_forward_hook(hook("embedding")))
        for idx, layer in enumerate(core.layers):
            handles.append(layer.register_forward_hook(hook(f"block_{idx}")))
        handles.append(core.norm.register_forward_hook(hook("final_norm")))
        handles.append(model.lm_head.register_forward_hook(hook("logits")))
    else:
        handles.append(model.tok_emb.register_forward_hook(hook("embedding")))
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            blocks = getattr(model, "trf_blocks", None)
        if blocks is None:
            raise AttributeError("Could not locate Gemma 4 blocks on the local model.")
        for idx, block in enumerate(blocks):
            handles.append(block.register_forward_hook(hook(f"block_{idx}")))
        handles.append(model.final_norm.register_forward_hook(hook("final_norm")))
        handles.append(model.out_head.register_forward_hook(hook("logits")))

    return traces, handles


def _layer_sort_key(name):
    if name == "embedding":
        return (0, 0)
    if name.startswith("block_"):
        idx = int(name.split("_")[1])
        return (1, idx)
    if name == "final_norm":
        return (2, 0)
    if name == "logits":
        return (3, 0)
    return (4, name)


def layerwise_differences(ours, hf_model, input_ids, rtol=1e-5, atol=1e-5):
    ours_traces, ours_handles = _attach_debug_hooks(ours, is_hf=False)
    hf_traces, hf_handles = _attach_debug_hooks(hf_model, is_hf=True)

    try:
        with torch.inference_mode():
            ours(input_ids)
            hf_model(input_ids, use_cache=False)
    finally:
        for h in ours_handles + hf_handles:
            h.remove()

    layer_names = sorted(set(ours_traces) | set(hf_traces), key=_layer_sort_key)
    results = []
    for name in layer_names:
        ours_tensor = ours_traces.get(name)
        hf_tensor = hf_traces.get(name)

        if ours_tensor is None or hf_tensor is None:
            results.append(
                {
                    "name": name,
                    "status": "missing",
                    "ours_shape": None if ours_tensor is None else tuple(ours_tensor.shape),
                    "hf_shape": None if hf_tensor is None else tuple(hf_tensor.shape),
                    "max_diff": None,
                    "mean_abs_diff": None,
                }
            )
            continue

        if ours_tensor.shape != hf_tensor.shape:
            results.append(
                {
                    "name": name,
                    "status": "shape_mismatch",
                    "ours_shape": tuple(ours_tensor.shape),
                    "hf_shape": tuple(hf_tensor.shape),
                    "max_diff": None,
                    "mean_abs_diff": None,
                }
            )
            continue

        diff = (ours_tensor - hf_tensor).abs()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())
        allclose = torch.allclose(ours_tensor, hf_tensor, rtol=rtol, atol=atol)
        results.append(
            {
                "name": name,
                "status": "ok" if allclose else "mismatch",
                "ours_shape": tuple(ours_tensor.shape),
                "hf_shape": tuple(hf_tensor.shape),
                "max_diff": max_diff,
                "mean_abs_diff": mean_diff,
            }
        )
    return results


def format_report(differences):
    lines = []
    for diff in sorted(differences, key=lambda d: _layer_sort_key(d["name"])):
        if diff["status"] == "ok":
            lines.append(f"[OK] {diff['name']}: max={diff['max_diff']:.2e}, mean={diff['mean_abs_diff']:.2e}")
        elif diff["status"] == "mismatch":
            lines.append(f"[DIFF] {diff['name']}: max={diff['max_diff']:.2e}, mean={diff['mean_abs_diff']:.2e}")
        elif diff["status"] == "shape_mismatch":
            lines.append(f"[SHAPE] {diff['name']}: ours={diff['ours_shape']}, hf={diff['hf_shape']}")
        else:
            lines.append(f"[MISSING] {diff['name']}: ours={diff['ours_shape']}, hf={diff['hf_shape']}")
    return "\n".join(lines)


if __name__ == "__main__":
    if Gemma4ForCausalLM is None or Gemma4TextConfig is None:
        raise SystemExit("Gemma 4 classes are unavailable; install a compatible transformers build or use temp/gemma-4.")

    import_notebook_defs = load_notebook_defs()
    cfg = tiny_debug_config()

    ours_model, hf_model = build_gemma4_pair(import_notebook_defs, cfg)
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"]), dtype=torch.long)
    diffs = layerwise_differences(ours_model, hf_model, input_ids)
    print(format_report(diffs))
