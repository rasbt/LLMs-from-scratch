# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import argparse
from pathlib import Path
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IMDBDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text)[:self.max_length]
            for text in self.data["text"]
        ]
        # Pad sequences to the longest sequence

        # Debug
        pad_token_id = 0

        self.encoded_texts = [
            et + [pad_token_id] * (self.max_length - len(et))
            for et in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        max_length = 0
        for text in self.data["text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # logits = model(input_batch)[:, -1, :]  # Logits of last output token
    logits = model(input_batch).logits
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


# Same as in chapter 5
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


@torch.no_grad()  # Disable gradient tracking for efficiency
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            # logits = model(input_batch)[:, -1, :]  # Logits of last output token
            logits = model(input_batch).logits
            predicted_labels = torch.argmax(logits, dim=1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if max_steps is not None and global_step > max_steps:
                break

        # New: Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        if max_steps is not None and global_step > max_steps:
            break

    return train_losses, val_losses, train_accs, val_accs, examples_seen


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainable_layers",
        type=str,
        default="last_block",
        help=(
            "Which layers to train. Options: 'all', 'last_block', 'last_layer'."
        )
    )
    parser.add_argument(
        "--bert_model",
        type=str,
        default="distilbert",
        help=(
            "Which layers to train. Options: 'all', 'last_block', 'last_layer'."
        )
    )
    args = parser.parse_args()

    ###############################
    # Load model
    ###############################

    torch.manual_seed(123)
    if args.bert_model == "distilbert":

        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        model.out_head = torch.nn.Linear(in_features=768, out_features=2)

        if args.trainable_layers == "last_layer":
            pass
        elif args.trainable_layers == "last_block":
            for param in model.pre_classifier.parameters():
                param.requires_grad = True
            for param in model.distilbert.transformer.layer[-1].parameters():
                param.requires_grad = True
        elif args.trainable_layers == "all":
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    elif args.bert_model == "roberta":

        model = AutoModelForSequenceClassification.from_pretrained(
            "FacebookAI/roberta-large", num_labels=2
        )
        model.classifier.out_proj = torch.nn.Linear(in_features=1024, out_features=2)

        if args.trainable_layers == "last_layer":
            pass
        elif args.trainable_layers == "last_block":
            for param in model.classifier.parameters():
                param.requires_grad = True
            for param in model.roberta.encoder.layer[-1].parameters():
                param.requires_grad = True
        elif args.trainable_layers == "all":
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")

        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")

    else:
        raise ValueError("Selected --bert_model not supported.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ###############################
    # Instantiate dataloaders
    ###############################

    pad_token_id = tokenizer.encode(tokenizer.pad_token)

    base_path = Path(".")

    train_dataset = IMDBDataset(base_path / "train.csv", max_length=256, tokenizer=tokenizer, pad_token_id=pad_token_id)
    val_dataset = IMDBDataset(base_path / "validation.csv", max_length=256, tokenizer=tokenizer, pad_token_id=pad_token_id)
    test_dataset = IMDBDataset(base_path / "test.csv", max_length=256, tokenizer=tokenizer, pad_token_id=pad_token_id)

    num_workers = 0
    batch_size = 8

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

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    ###############################
    # Train model
    ###############################

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    num_epochs = 3
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=20,
        max_steps=None
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    ###############################
    # Evaluate model
    ###############################

    print("\nEvaluating on the full datasets ...\n")

    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
