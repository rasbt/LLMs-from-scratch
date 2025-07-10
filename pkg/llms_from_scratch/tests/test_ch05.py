# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from llms_from_scratch.ch02 import create_dataloader_v1
from llms_from_scratch.ch04 import GPTModel, GPTModelFast
from llms_from_scratch.ch05 import train_model_simple

import os
import urllib

import pytest
import tiktoken
import torch
from torch.utils.data import Subset, DataLoader


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  # Shortened for test speed
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

OTHER_SETTINGS = {
    "learning_rate": 5e-4,
    "num_epochs": 2,
    "batch_size": 1,
    "weight_decay": 0.1
}


@pytest.mark.parametrize("ModelClass", [GPTModel, GPTModelFast])
def test_train_simple(tmp_path, ModelClass):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################
    # Download data if necessary
    ##############################
    file_path = tmp_path / "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text_data = f.read()

    ##############################
    # Set up dataloaders
    ##############################
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=OTHER_SETTINGS["batch_size"],
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=OTHER_SETTINGS["batch_size"],
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    # Limit to 1 batch for speed
    train_subset = Subset(train_loader.dataset, range(1))
    one_batch_train_loader = DataLoader(train_subset, batch_size=1)
    val_subset = Subset(val_loader.dataset, range(1))
    one_batch_val_loader = DataLoader(val_subset, batch_size=1)

    ##############################
    # Train model
    ##############################
    model = ModelClass(GPT_CONFIG_124M)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=OTHER_SETTINGS["learning_rate"],
        weight_decay=OTHER_SETTINGS["weight_decay"]
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, one_batch_train_loader, one_batch_val_loader, optimizer, device,
        num_epochs=OTHER_SETTINGS["num_epochs"], eval_freq=1, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    assert round(train_losses[0], 1) == 7.6
    assert round(val_losses[0], 1) == 10.1
    assert train_losses[-1] < train_losses[0]
