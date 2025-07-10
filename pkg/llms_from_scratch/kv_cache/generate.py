# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from .utils import KVCache
import torch


def generate_text_simple(model, idx, max_new_tokens, context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.cfg["context_length"]

    with torch.no_grad():
        if use_cache:
            cache = KVCache(n_layers=model.cfg["n_layers"])
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], cache=cache)

            for _ in range(max_new_tokens):
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)
                logits = model(next_idx, cache=cache)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], cache=None)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
