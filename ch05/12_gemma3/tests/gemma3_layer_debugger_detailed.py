# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import importlib
from pathlib import Path

import torch

from llms_from_scratch.utils import import_definitions_from_notebook

try:
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
except ImportError:
    Gemma3ForCausalLM = None
    Gemma3TextConfig = None


def tiny_debug_config():
    return {
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
        "rope_local_base": 10_000.0,
        "sliding_window": 4,
        "layer_types": ["full_attention", "full_attention"],
        "dtype": torch.float32,
        "query_pre_attn_scalar": 256,
    }


def _hf_config_from_dict(cfg):
    if Gemma3TextConfig is None:
        raise ImportError("transformers is required for the Gemma 3 debugger.")

    return Gemma3TextConfig(
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["context_length"],
        hidden_size=cfg["emb_dim"],
        num_attention_heads=cfg["n_heads"],
        num_hidden_layers=cfg["n_layers"],
        intermediate_size=cfg["hidden_dim"],
        head_dim=cfg["head_dim"],
        num_key_value_heads=cfg["n_kv_groups"],
        rope_theta=cfg["rope_base"],
        rope_local_base_freq=cfg["rope_local_base"],
        layer_types=cfg["layer_types"],
        sliding_window=cfg["sliding_window"],
        tie_word_embeddings=False,
        attn_implementation="eager",
        torch_dtype=cfg.get("dtype", torch.float32),
        query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
        rope_scaling={"rope_type": "default"},
        ignore_keys_at_rope_validation={"full_attention", "sliding_attention"},
    )


def load_notebook_defs(nb_name="standalone-gemma3.ipynb"):
    nb_dir = Path(__file__).resolve().parents[1]
    return import_definitions_from_notebook(nb_dir, nb_name)


def build_gemma3_pair(import_notebook_defs, cfg, hf_checkpoint=None):
    if Gemma3ForCausalLM is None:
        raise ImportError("transformers is required for the Gemma 3 debugger.")

    torch.manual_seed(123)
    ours = import_notebook_defs.Gemma3Model(cfg)

    if hf_checkpoint:
        hf_model = Gemma3ForCausalLM.from_pretrained(
            hf_checkpoint,
            torch_dtype=cfg.get("dtype", torch.float32),
            attn_implementation="eager",
        )
    else:
        hf_cfg = _hf_config_from_dict(cfg)
        hf_model = Gemma3ForCausalLM(hf_cfg)

    param_config = {"n_layers": cfg["n_layers"], "hidden_dim": cfg["hidden_dim"]}
    import_notebook_defs.load_weights_into_gemma(ours, param_config, hf_model.state_dict())

    ours.eval()
    hf_model.eval()
    return ours, hf_model


def _register_trace_hook(handles, traces, name, module, scale=None):
    if module is None:
        return

    def _record(_, __, output):
        if isinstance(output, tuple):
            output = output[0]
        if scale is not None:
            output = output * scale
        traces[name] = output.detach().to(torch.float32).cpu()

    handles.append(module.register_forward_hook(_record))


def _attach_debug_hooks(model, is_hf, include_block_details=True):
    traces = {}
    handles = []

    if is_hf:
        core = model.model
        _register_trace_hook(handles, traces, "embedding", core.embed_tokens)
        for idx, layer in enumerate(core.layers):
            block_name = f"block_{idx}"
            _register_trace_hook(handles, traces, block_name, layer)

            if include_block_details:
                _register_trace_hook(handles, traces, f"{block_name}.input_layernorm", layer.input_layernorm)
                _register_trace_hook(handles, traces, f"{block_name}.att.q_proj", layer.self_attn.q_proj)
                _register_trace_hook(handles, traces, f"{block_name}.att.k_proj", layer.self_attn.k_proj)
                _register_trace_hook(handles, traces, f"{block_name}.att.v_proj", layer.self_attn.v_proj)
                _register_trace_hook(handles, traces, f"{block_name}.att.q_norm", layer.self_attn.q_norm)
                _register_trace_hook(handles, traces, f"{block_name}.att.k_norm", layer.self_attn.k_norm)
                _register_trace_hook(handles, traces, f"{block_name}.att.o_proj", layer.self_attn.o_proj)
                _register_trace_hook(handles, traces, f"{block_name}.att", layer.self_attn)
                _register_trace_hook(handles, traces, f"{block_name}.post_attention_layernorm", layer.post_attention_layernorm)
                _register_trace_hook(handles, traces, f"{block_name}.pre_feedforward_layernorm", layer.pre_feedforward_layernorm)
                _register_trace_hook(handles, traces, f"{block_name}.ff.gate_proj", layer.mlp.gate_proj)
                _register_trace_hook(handles, traces, f"{block_name}.ff.up_proj", layer.mlp.up_proj)
                _register_trace_hook(handles, traces, f"{block_name}.ff.down_proj", layer.mlp.down_proj)
                _register_trace_hook(handles, traces, f"{block_name}.ff", layer.mlp)
                _register_trace_hook(handles, traces, f"{block_name}.post_feedforward_layernorm", layer.post_feedforward_layernorm)

        _register_trace_hook(handles, traces, "final_norm", core.norm)
        _register_trace_hook(handles, traces, "logits", model.lm_head)
    else:
        emb_scale = float(getattr(model, "cfg", {}).get("emb_dim", model.tok_emb.embedding_dim) ** 0.5)
        _register_trace_hook(handles, traces, "embedding", model.tok_emb, scale=emb_scale)
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            blocks = getattr(model, "trf_blocks", None)
        if blocks is None:
            raise AttributeError("Could not locate Gemma 3 blocks on the local model.")
        for idx, block in enumerate(blocks):
            block_name = f"block_{idx}"
            _register_trace_hook(handles, traces, block_name, block)

            if include_block_details:
                _register_trace_hook(handles, traces, f"{block_name}.input_layernorm", block.input_layernorm)
                _register_trace_hook(handles, traces, f"{block_name}.att.q_proj", block.att.W_query)
                _register_trace_hook(handles, traces, f"{block_name}.att.k_proj", block.att.W_key)
                _register_trace_hook(handles, traces, f"{block_name}.att.v_proj", block.att.W_value)
                _register_trace_hook(handles, traces, f"{block_name}.att.q_norm", block.att.q_norm)
                _register_trace_hook(handles, traces, f"{block_name}.att.k_norm", block.att.k_norm)
                _register_trace_hook(handles, traces, f"{block_name}.att.o_proj", block.att.out_proj)
                _register_trace_hook(handles, traces, f"{block_name}.att", block.att)
                _register_trace_hook(handles, traces, f"{block_name}.post_attention_layernorm", block.post_attention_layernorm)
                _register_trace_hook(handles, traces, f"{block_name}.pre_feedforward_layernorm", block.pre_feedforward_layernorm)
                _register_trace_hook(handles, traces, f"{block_name}.ff.gate_proj", block.ff.fc1)
                _register_trace_hook(handles, traces, f"{block_name}.ff.up_proj", block.ff.fc2)
                _register_trace_hook(handles, traces, f"{block_name}.ff.down_proj", block.ff.fc3)
                _register_trace_hook(handles, traces, f"{block_name}.ff", block.ff)
                _register_trace_hook(handles, traces, f"{block_name}.post_feedforward_layernorm", block.post_feedforward_layernorm)

        _register_trace_hook(handles, traces, "final_norm", model.final_norm)
        _register_trace_hook(handles, traces, "logits", model.out_head)

    return traces, handles


def _layer_sort_key(name):
    block_detail_order = {
        "input_layernorm": 0,
        "att.q_proj": 1,
        "att.k_proj": 2,
        "att.v_proj": 3,
        "att.q_norm": 4,
        "att.k_norm": 5,
        "att.o_proj": 6,
        "att": 7,
        "post_attention_layernorm": 8,
        "pre_feedforward_layernorm": 9,
        "ff.gate_proj": 10,
        "ff.up_proj": 11,
        "ff.down_proj": 12,
        "ff": 13,
        "post_feedforward_layernorm": 14,
    }

    if name == "embedding":
        return (0, 0)
    if name.startswith("block_"):
        block_name, _, detail = name.partition(".")
        idx = int(block_name.split("_")[1])
        if not detail:
            return (1, idx, -1)
        return (2, idx, block_detail_order.get(detail, 100), detail)
    if name == "final_norm":
        return (3, 0)
    if name == "logits":
        return (4, 0)
    return (5, name)


def layerwise_differences(ours, hf_model, input_ids, rtol=1e-5, atol=1e-5, include_block_details=True):
    ours_traces, ours_handles = _attach_debug_hooks(ours, is_hf=False, include_block_details=include_block_details)
    hf_traces, hf_handles = _attach_debug_hooks(hf_model, is_hf=True, include_block_details=include_block_details)

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


def _format_diff_line(diff, indent=""):
    if diff["status"] == "ok":
        return f"{indent}[OK] {diff['name']}: max={diff['max_diff']:.2e}, mean={diff['mean_abs_diff']:.2e}"
    if diff["status"] == "mismatch":
        return f"{indent}[DIFF] {diff['name']}: max={diff['max_diff']:.2e}, mean={diff['mean_abs_diff']:.2e}"
    if diff["status"] == "shape_mismatch":
        return f"{indent}[SHAPE] {diff['name']}: ours={diff['ours_shape']}, hf={diff['hf_shape']}"
    return f"{indent}[MISSING] {diff['name']}: ours={diff['ours_shape']}, hf={diff['hf_shape']}"


def format_report(differences, show_block_details=True, details_for_all_blocks=False):
    lines = []
    top_level_diffs = [diff for diff in differences if "." not in diff["name"]]

    for diff in sorted(top_level_diffs, key=lambda d: _layer_sort_key(d["name"])):
        lines.append(_format_diff_line(diff))

        if not show_block_details or not diff["name"].startswith("block_"):
            continue

        detail_prefix = f"{diff['name']}."
        detail_diffs = [
            other for other in differences
            if other["name"].startswith(detail_prefix)
        ]
        if not detail_diffs:
            continue

        has_detail_mismatch = any(other["status"] != "ok" for other in detail_diffs)
        if not details_for_all_blocks and diff["status"] == "ok" and not has_detail_mismatch:
            continue
        if not details_for_all_blocks and diff["status"] == "ok":
            continue

        for other in sorted(detail_diffs, key=lambda d: _layer_sort_key(d["name"])):
            lines.append(_format_diff_line(other, indent="  "))

    return "\n".join(lines)


if __name__ == "__main__":
    transformers_available = importlib.util.find_spec("transformers") is not None
    if not transformers_available:
        raise SystemExit("transformers is not installed; install it to run the debugger.")

    import_notebook_defs = load_notebook_defs()
    cfg = tiny_debug_config()

    ours_model, hf_model = build_gemma3_pair(import_notebook_defs, cfg)
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"]), dtype=torch.long)
    diffs = layerwise_differences(ours_model, hf_model, input_ids)
    print(format_report(diffs))
