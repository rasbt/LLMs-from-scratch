# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.ch06 import (
    download_and_unzip_spam_data, create_balanced_dataset,
    random_split, SpamDataset, train_classifier_simple
)

from pathlib import Path
import urllib

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader, Subset


def test_train_classifier(tmp_path):

    ########################################
    # Download and prepare dataset
    ########################################

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = tmp_path / "sms_spam_collection.zip"
    extracted_path = tmp_path / "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    try:
        download_and_unzip_spam_data(
            url, zip_path, extracted_path, data_file_path
        )
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"Primary URL failed: {e}. Trying backup URL...")
        backup_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        download_and_unzip_spam_data(
            backup_url, zip_path, extracted_path, data_file_path
        )

    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv(tmp_path / "train.csv", index=None)
    validation_df.to_csv(tmp_path / "validation.csv", index=None)
    test_df.to_csv(tmp_path / "test.csv", index=None)

    ########################################
    # Create data loaders
    ########################################
    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = SpamDataset(
        csv_file=tmp_path / "train.csv",
        max_length=None,
        tokenizer=tokenizer
    )

    val_dataset = SpamDataset(
        csv_file=tmp_path / "validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    ########################################
    # Load pretrained model
    ########################################

    # Small GPT model for testing purposes
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 120,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "emb_dim": 12,
        "n_layers": 1,
        "n_heads": 2
    }
    model = GPTModel(BASE_CONFIG)
    model.eval()
    device = "cpu"

    ########################################
    # Modify and pretrained model
    ########################################

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)

    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
    model.to(device)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    ########################################
    # Finetune modified model
    ########################################

    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0)

    train_subset = Subset(train_loader.dataset, range(5))
    batch_train_loader = DataLoader(train_subset, batch_size=5)
    val_subset = Subset(val_loader.dataset, range(5))
    batch_val_loader = DataLoader(val_subset, batch_size=5)

    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, batch_train_loader, batch_val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=1, eval_iter=1,
    )

    assert round(train_losses[0], 1) == 0.8
    assert round(val_losses[0], 1) == 0.8
    assert train_losses[-1] < train_losses[0]
