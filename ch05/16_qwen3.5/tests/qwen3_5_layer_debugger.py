# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import sys
from pathlib import Path

import torch

from llms_from_scratch.utils import import_definitions_from_notebook


def _import_qwen3_5_classes():
    try:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        return Qwen3_5TextConfig, Qwen3_5ForCausalLM
    except Exception:
        repo_root = Path(__file__).resolve().parents[3]
        local_src = repo_root / "transformers-main" / "src"
        if not local_src.exists():
            raise

        for name in list(sys.modules):
            if name == "transformers" or name.startswith("transformers."):
                del sys.modules[name]
        sys.path.insert(0, str(local_src))

        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        return Qwen3_5TextConfig, Qwen3_5ForCausalLM


try:
    Qwen3_5TextConfig, Qwen3_5ForCausalLM = _import_qwen3_5_classes()
except Exception:
    Qwen3_5TextConfig = None
    Qwen3_5ForCausalLM = None


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
        "partial_rotary_factor": 1.0,
        "rms_norm_eps": 1e-6,
        "linear_conv_kernel_dim": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "dtype": torch.float32,
    }


def _hf_config_from_dict(cfg):
    if Qwen3_5TextConfig is None:
        raise ImportError("Qwen3.5 classes are required for the layer debugger.")

    hf_cfg = Qwen3_5TextConfig(
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["context_length"],
        hidden_size=cfg["emb_dim"],
        num_attention_heads=cfg["n_heads"],
        num_hidden_layers=cfg["n_layers"],
        intermediate_size=cfg["hidden_dim"],
        head_dim=cfg["head_dim"],
        num_key_value_heads=cfg["n_kv_groups"],
        layer_types=cfg["layer_types"],
        linear_conv_kernel_dim=cfg["linear_conv_kernel_dim"],
        linear_key_head_dim=cfg["linear_key_head_dim"],
        linear_value_head_dim=cfg["linear_value_head_dim"],
        linear_num_key_heads=cfg["linear_num_key_heads"],
        linear_num_value_heads=cfg["linear_num_value_heads"],
        tie_word_embeddings=False,
        use_cache=False,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        rope_parameters={
            "rope_type": "default",
            "rope_theta": cfg["rope_base"],
            "partial_rotary_factor": cfg.get("partial_rotary_factor", 1.0),
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
        },
        torch_dtype=cfg.get("dtype", torch.float32),
    )
    hf_cfg._attn_implementation = "eager"
    return hf_cfg


def load_notebook_defs(nb_name="standalone-qwen3.5.ipynb"):
    nb_dir = Path(__file__).resolve().parents[1]
    if str(nb_dir) not in sys.path:
        sys.path.insert(0, str(nb_dir))
    return import_definitions_from_notebook(nb_dir, nb_name)


def build_qwen3_5_pair(import_notebook_defs, cfg, hf_checkpoint=None):
    if Qwen3_5ForCausalLM is None:
        raise ImportError("Qwen3.5 classes are required for the layer debugger.")

    ours = import_notebook_defs.Qwen3_5Model(cfg)

    if hf_checkpoint:
        hf_model = Qwen3_5ForCausalLM.from_pretrained(
            hf_checkpoint,
            torch_dtype=cfg.get("dtype", torch.float32),
            attn_implementation="eager",
        )
    else:
        hf_cfg = _hf_config_from_dict(cfg)
        hf_model = Qwen3_5ForCausalLM(hf_cfg)

    import_notebook_defs.load_weights_into_qwen3_5(
        ours,
        {"n_layers": cfg["n_layers"], "layer_types": cfg["layer_types"]},
        hf_model.state_dict(),
    )
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
        blocks = getattr(model, "trf_blocks", None)
        if blocks is None:
            blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise AttributeError("Could not locate Qwen3.5 blocks on the local model.")
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
    if Qwen3_5ForCausalLM is None:
        raise SystemExit(
            "Qwen3.5 classes are unavailable. Install a recent transformers version or use local transformers-main."
        )

    import_notebook_defs = load_notebook_defs()
    cfg = tiny_debug_config()

    ours_model, hf_model = build_qwen3_5_pair(import_notebook_defs, cfg)
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"]), dtype=torch.long)
    diffs = layerwise_differences(ours_model, hf_model, input_ids)
    print(format_report(diffs))
