# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import tiktoken
import torch
import chainlit

from previous_chapters import (
    download_and_load_gpt2,
    generate,
    GPTModel,
    load_weights_into_gpt,
    text_to_token_ids,
    token_ids_to_text,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    Code to load a GPT-2 model with pretrained weights from OpenAI.
    The code is similar to chapter 5.
    The model will be downloaded automatically if it doesn't exist in the current folder, yet.
    """

    CHOOSE_MODEL = "gpt2-small (124M)"  # Optionally replace with another model from the model_configs dir below

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    gpt = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    gpt.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    return tokenizer, gpt, BASE_CONFIG


# Obtain the necessary tokenizer and model files for the chainlit function below
tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    token_ids = generate(  # function uses `with torch.no_grad()` internally already
        model=model,
        idx=text_to_token_ids(message.content, tokenizer).to(device),  # The user text is provided via as `message.content`
        max_new_tokens=50,
        context_size=model_config["context_length"],
        top_k=1,
        temperature=0.0
    )

    text = token_ids_to_text(token_ids, tokenizer)

    await chainlit.Message(
        content=f"{text}",  # This returns the model response to the interface
    ).send()
