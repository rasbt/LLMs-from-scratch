# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from .utils import KVCache
import torch


def generate_text_simple(model, idx, max_new_tokens, context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.cfg["context_length"]
    batch_size = idx.size(0)

    with torch.no_grad():
        if use_cache:
            # initialize cache and positions
            cache = KVCache(n_layers=model.cfg["n_layers"], batch_size=batch_size)
            model.reset_kv_cache(batch_size=batch_size, device=idx.device)

            # initial full-context pass
            input_ids = idx[:, -ctx_len:]
            seq_len = input_ids.size(1)
            start_pos = model.current_pos.clone()
            logits = model(
                input_ids,
                cache=cache,
                start_pos=start_pos
            )
            model.current_pos += seq_len

            # iterative generation
            for _ in range(max_new_tokens):
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)  # (B, 1)
                logits = model(
                    next_token,
                    cache=cache,
                    start_pos=model.current_pos.clone()
                )
                model.current_pos += 1
                idx = torch.cat([idx, next_token], dim=1)
        else:
            # no cache
            for _ in range(max_new_tokens):
                input_ids = idx[:, -ctx_len:]
                logits = model(input_ids, cache=None, start_pos=None)
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_token], dim=1)

    return idx
