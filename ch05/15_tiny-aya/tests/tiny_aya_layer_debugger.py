# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import importlib
from pathlib import Path

import torch
from llms_from_scratch.utils import import_definitions_from_notebook

try:
    from transformers import Cohere2Config, Cohere2ForCausalLM
except ImportError:
    Cohere2Config = None
    Cohere2ForCausalLM = None


def tiny_debug_config():
    return {
        "vocab_size": 257,
        "context_length": 8,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "hidden_dim": 64,
        "head_dim": 8,
        "n_kv_heads": 2,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "dtype": torch.float32,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "layer_norm_eps": 1e-5,
        "rope_base": 10_000.0,
        "logit_scale": 1.0,
        "tie_word_embeddings": False,
    }


def _hf_config_from_dict(cfg):
    if Cohere2Config is None:
        raise ImportError("transformers is required for the Tiny Aya debugger.")

    return Cohere2Config(
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["context_length"],
        hidden_size=cfg["emb_dim"],
        num_attention_heads=cfg["n_heads"],
        num_hidden_layers=cfg["n_layers"],
        intermediate_size=cfg["hidden_dim"],
        num_key_value_heads=cfg["n_kv_heads"],
        attention_bias=cfg["attention_bias"],
        attention_dropout=cfg["attention_dropout"],
        layer_norm_eps=cfg["layer_norm_eps"],
        sliding_window=cfg["sliding_window"],
        layer_types=cfg["layer_types"],
        logit_scale=cfg["logit_scale"],
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        rope_parameters={"rope_type": "default", "rope_theta": cfg["rope_base"]},
        torch_dtype=cfg.get("dtype", torch.float32),
    )


def load_notebook_defs(nb_name="standalone-tiny-aya.ipynb"):
    nb_dir = Path(__file__).resolve().parents[1]
    return import_definitions_from_notebook(nb_dir, nb_name)


def build_tiny_aya_pair(import_notebook_defs, cfg, hf_checkpoint=None):
    if Cohere2ForCausalLM is None:
        raise ImportError("transformers is required for the Tiny Aya debugger.")

    ours = import_notebook_defs.TinyAyaModel(cfg)
    hf_cfg = _hf_config_from_dict(cfg)

    if hf_checkpoint:
        hf_model = Cohere2ForCausalLM.from_pretrained(
            hf_checkpoint,
            torch_dtype=cfg.get("dtype", torch.float32),
            attn_implementation="eager",
        )
    else:
        hf_model = Cohere2ForCausalLM(hf_cfg)

    import_notebook_defs.load_weights_into_tiny_aya(ours, cfg, hf_model.state_dict())

    ours.eval()
    hf_model.eval()
    return ours, hf_model


def _attach_debug_hooks(model, is_hf):
    traces = {}
    handles = []

    def hook(name):
        def _record(_, __, output):
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
        blocks = getattr(model, "trf_blocks", None)
        if blocks is None:
            blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise AttributeError("Could not locate Tiny Aya blocks on the local model.")
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
            hf_model(input_ids)
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
    transformers_available = importlib.util.find_spec("transformers") is not None
    if not transformers_available:
        raise SystemExit("transformers is not installed; install it to run the debugger.")

    import_notebook_defs = load_notebook_defs()
    cfg = tiny_debug_config()

    ours_model, hf_model = build_tiny_aya_pair(import_notebook_defs, cfg)
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"]), dtype=torch.long)
    diffs = layerwise_differences(ours_model, hf_model, input_ids)
    print(format_report(diffs))
