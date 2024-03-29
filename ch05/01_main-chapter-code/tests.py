# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# File for internal use (unit tests)

import pytest
from gpt_train import main


@pytest.fixture
def gpt_config():
    return {
        "vocab_size": 50257,
        "ctx_len": 12,      # small for testing efficiency
        "emb_dim": 32,      # small for testing efficiency
        "n_heads": 4,       # small for testing efficiency
        "n_layers": 2,      # small for testing efficiency
        "drop_rate": 0.1,
        "qkv_bias": False
    }


@pytest.fixture
def other_hparams():
    return {
        "learning_rate": 5e-4,
        "num_epochs": 1,    # small for testing efficiency
        "batch_size": 2,
        "weight_decay": 0.1
    }


def test_main(gpt_config, other_hparams):
    train_losses, val_losses, tokens_seen, model = main(gpt_config, other_hparams)

    assert len(train_losses) == 39, "Unexpected number of training losses"
    assert len(val_losses) == 39, "Unexpected number of validation losses"
    assert len(tokens_seen) == 39, "Unexpected number of tokens seen"
