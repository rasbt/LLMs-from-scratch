# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import itertools
import math
import os
import tiktoken
import torch
from previous_chapters import GPTModel, create_dataloader_v1


# Define a grid of hyperparameters to search over
HPARAM_GRID = {
    "batch_size": [2, 4, 8, 16],
    "drop_rate": [0.0, 0.1, 0.2],
    "warmup_iters": [10, 20, 30],
    "weight_decay": [0.1, 0.01, 0.0],
    "peak_lr": [0.0001, 0.0005, 0.001, 0.005],
    "initial_lr": [0.00005, 0.0001],
    "min_lr": [0.00005, 0.00001, 0.0001],
    "n_epochs": [5, 10, 15, 20, 25],
}


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    logits = model(input_batch)
    logits = logits.view(-1, logits.size(-1))
    loss = torch.nn.functional.cross_entropy(logits, target_batch.view(-1))
    return loss


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter,
                encoded_start_context, tokenizer, warmup_iters=10,
                initial_lr=3e-05, min_lr=1e-6):
    global_step = 0

    max_lr = optimizer.param_groups[0]["lr"]

    # Calculate total number of iterations
    total_training_iters = len(train_loader) * n_epochs

    # Calculate the learning rate increment at each step during warmup
    lr_increment = (optimizer.param_groups[0]["lr"] - initial_lr) / warmup_iters

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            # Increment the global step at the beginning of the iteration
            global_step += 1

            # Warmup: adjust learning rate linearly
            if global_step < warmup_iters:
                lr = initial_lr + global_step * lr_increment
            # Cosine annealing phase
            else:
                progress = (global_step - warmup_iters) / (total_training_iters - warmup_iters)
                lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            # Apply the calculated learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # Apply gradient clipping
            if global_step >= warmup_iters:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)

    return train_loss, val_loss


if __name__ == "__main__":

    # Generate all combinations of hyperparameters
    hyperparameter_combinations = list(itertools.product(*HPARAM_GRID.values()))
    total_combinations = len(hyperparameter_combinations)
    print(f"Total hyperparameter configurations: {total_combinations}")

    # Placeholder for the best loss and best hyperparameters
    best_val_loss = float('inf')
    best_hparams = {}

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    with open(os.path.join(script_dir, "the-verdict.txt"), "r", encoding="utf-8") as file:
        text_data = file.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ratio = 0.95
    split_idx = int(train_ratio * len(text_data))

    torch.manual_seed(123)

    interrupted = False
    current_config = 0
    for combination in hyperparameter_combinations:

        try:
            current_config += 1
            print(f"Evaluating configuration {current_config} of {total_combinations}")

            # Unpack the current combination of hyperparameters
            HPARAM_CONFIG = dict(zip(HPARAM_GRID.keys(), combination))

            GPT_CONFIG_124M = {
                "vocab_size": 50257,    # Vocabulary size
                "context_length": 256,  # Context length -- shortened from original 1024 tokens
                "emb_dim": 768,         # Embedding dimension
                "n_heads": 12,          # Number of attention heads
                "n_layers": 12,         # Number of layers
                "drop_rate": HPARAM_CONFIG["drop_rate"],
                "qkv_bias": False,     # Query-Key-Value bias
            }

            torch.manual_seed(123)
            train_loader = create_dataloader_v1(
                text_data[:split_idx],
                batch_size=HPARAM_CONFIG["batch_size"],
                max_length=GPT_CONFIG_124M["context_length"],
                stride=GPT_CONFIG_124M["context_length"],
                drop_last=True,
                shuffle=True,
                num_workers=0
            )

            val_loader = create_dataloader_v1(
                text_data[split_idx:],
                batch_size=HPARAM_CONFIG["batch_size"],
                max_length=GPT_CONFIG_124M["context_length"],
                stride=GPT_CONFIG_124M["context_length"],
                drop_last=False,
                shuffle=False,
                num_workers=0
            )

            model = GPTModel(GPT_CONFIG_124M)
            model.to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=HPARAM_CONFIG["peak_lr"],
                weight_decay=HPARAM_CONFIG["weight_decay"]
            )

            encoded_start_context = tokenizer.encode("Nevertheless")
            encoded_tensor = torch.tensor(encoded_start_context).unsqueeze(0)

            train_loss, val_loss = train_model(
                model, train_loader, val_loader, optimizer, device,
                n_epochs=HPARAM_CONFIG["n_epochs"],
                eval_freq=5, eval_iter=1,
                encoded_start_context=encoded_tensor,
                tokenizer=tokenizer,
                warmup_iters=HPARAM_CONFIG["warmup_iters"],
                initial_lr=HPARAM_CONFIG["initial_lr"],
                min_lr=HPARAM_CONFIG["min_lr"]
            )

            # Log the best hyperparameters based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_hparams = HPARAM_CONFIG

        except KeyboardInterrupt:
            print("Hyperparameter search completed.")
            print(f"Best hyperparameters: {best_hparams}")
            print(f"Best Val loss: {best_val_loss} | Training loss {train_loss}")
            interrupted = True
            break

    if not interrupted:
        print("Hyperparameter search completed.")
        print(f"Best hyperparameters: {best_hparams}")
        print(f"Best Val loss: {best_val_loss} | Training loss {train_loss}")
