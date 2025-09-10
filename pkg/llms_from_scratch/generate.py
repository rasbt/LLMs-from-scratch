# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# Additional utility and helper functions for text generation not covered
# in the main chapters

def trim_input_tensor(input_ids_tensor, context_len, max_new_tokens):
    assert max_new_tokens < context_len
    keep_len = max(1, context_len - max_new_tokens)

    # If the prompt is too long, left-truncate to keep_len
    if input_ids_tensor.shape[1] > keep_len:
        input_ids_tensor = input_ids_tensor[:, -keep_len:]

    return input_ids_tensor
