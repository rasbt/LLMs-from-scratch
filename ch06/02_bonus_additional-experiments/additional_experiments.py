# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import argparse
import math
import os
from pathlib import Path
import time
import zipfile

import pandas as pd
import requests
import tiktoken
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt


# If the `previous_chapters.py` file is not available locally,
# you can import it from the `llms-from-scratch` PyPI package.
# For details, see: https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
# E.g.,
# from llms_from_scratch.ch04 import GPTModel
# from llms_from_scratch.ch05 import download_and_load_gpt2, load_weights_into_gpt


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


# This LoRA code is equivalent to LinearWithLoRA
class LinearWithLoRAMerged(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        lora = self.lora.A @ self.lora.B
        combined_weight = self.linear.weight + self.lora.alpha*lora.T
        return torch.nn.functional.linear(x, combined_weight, self.linear.bias)


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256, no_padding=False):
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text)[:self.max_length]
            for text in self.data["Text"]
        ]

        if not no_padding:
            # Pad sequences to the longest sequence
            self.encoded_texts = [
                et + [pad_token_id] * (self.max_length - len(et))
                for et in self.encoded_texts
            ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        max_length = 0
        for text in self.data["Text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
        # Note: A more pythonic version to implement this method
        # is the following, which is also used in the next chapter:
        # return max(len(encoded_text) for encoded_text in self.encoded_texts)


def download_and_unzip(url, zip_path, extract_to, new_file_path):
    if new_file_path.exists():
        print(f"{new_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    # Renaming the file to indicate its format
    original_file = Path(extract_to) / "SMSSpamCollection"
    os.rename(original_file, new_file_path)
    print(f"File downloaded and saved as {new_file_path}")


def random_split(df, train_frac, validation_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


def create_dataset_csvs(new_file_path):
    df = pd.read_csv(new_file_path, sep="\t", header=None, names=["Label", "Text"])

    # Create balanced dataset
    n_spam = df[df["Label"] == "spam"].shape[0]
    ham_sampled = df[df["Label"] == "ham"].sample(n_spam, random_state=123)
    balanced_df = pd.concat([ham_sampled, df[df["Label"] == "spam"]])
    balanced_df = balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    # Sample and save csv files
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)


def instantiate_model(choose_model, load_weights):

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

    BASE_CONFIG.update(model_configs[choose_model])

    if not load_weights:
        torch.manual_seed(123)
    model = GPTModel(BASE_CONFIG, disable_causal_mask=args.disable_causal_mask)

    if load_weights:
        model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
        load_weights_into_gpt(model, params)

    model.eval()
    return model


def calc_loss_batch(input_batch, target_batch, model, device,
                    trainable_token_pos=-1, ignore_index=-100, average_embeddings=False):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    if trainable_token_pos == "flexible":  # Selects the last tokens before the padding tokens
        # From https://github.com/rasbt/LLMs-from-scratch/discussions/434
        # Find the last non-padding token for each sequence in the batch
        pad_token_id = 50256  # <|endoftext|> token used for padding
        mask = input_batch != pad_token_id
        last_token_pos = mask.sum(dim=1) - 1  # Get position of last real token

        # Get model outputs
        logits = model(input_batch)  # shape: [batch_size, seq_len, num_classes]

        # Select the logits corresponding to the last real token of each sequence
        batch_size = logits.size(0)
        selected_logits = logits[torch.arange(batch_size), last_token_pos]

        loss = torch.nn.functional.cross_entropy(selected_logits, target_batch)
        return loss

    else:
        model_output = model(input_batch)
        if average_embeddings:
            # Average over the sequence dimension (dim=1)
            logits = model_output.mean(dim=1)
        else:
            # Select embeddings at the specified token position
            logits = model_output[:, trainable_token_pos, :]

        loss = torch.nn.functional.cross_entropy(logits, target_batch, ignore_index=ignore_index)
        return loss


def calc_loss_loader(data_loader, model, device,
                     num_batches=None, trainable_token_pos=-1,
                     ignore_index=-100, average_embeddings=False):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
                average_embeddings=average_embeddings
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


@torch.no_grad()  # Disable gradient tracking for efficiency
def calc_accuracy_loader(data_loader, model, device, num_batches=None,
                         trainable_token_pos=-1, average_embeddings=False):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    if trainable_token_pos == "flexible":
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                # Find the last non-padding token for each sequence in the batch
                pad_token_id = 50256  # <|endoftext|> token used for padding
                mask = input_batch != pad_token_id
                last_token_pos = mask.sum(dim=1) - 1  # Get position of last real token

                logits = model(input_batch)  # Logits of last output token
                # Select the logits corresponding to the last real token of each sequence
                batch_size = logits.size(0)
                selected_logits = logits[torch.arange(batch_size), last_token_pos]
                predicted_labels = torch.argmax(selected_logits, dim=-1)

                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break

    else:
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                model_output = model(input_batch)
                if average_embeddings:
                    # Average over the sequence dimension (dim=1)
                    logits = model_output.mean(dim=1)
                else:
                    # Select embeddings at the specified token position
                    logits = model_output[:, trainable_token_pos, :]

                predicted_labels = torch.argmax(logits, dim=-1)

                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break
    return correct_predictions / num_examples


def evaluate_model(model, train_loader, val_loader, device,
                   eval_iter, trainable_token_pos=-1,
                   ignore_index=-100, average_embeddings=False):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
            average_embeddings=average_embeddings
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
            average_embeddings=average_embeddings
        )
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None, trainable_token_pos=-1,
                            accumulation_steps=1, ignore_index=-100, average_embeddings=False):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
                average_embeddings=average_embeddings
            )

            # Use gradient accumulation if accumulation_steps > 1
            # See https://sebastianraschka.com/blog/2023/llm-grad-accumulation.html
            # for an explanation
            loss /= accumulation_steps

            loss.backward()  # Calculate loss gradients

            # Use gradient accumulation if accumulation_steps > 1
            is_update_step = ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader))
            if is_update_step:
                optimizer.step()  # Update model weights using loss gradients
                optimizer.zero_grad()  # Reset loss gradients from previous batch iteration

            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter,
                    trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
                    average_embeddings=average_embeddings
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if max_steps is not None and global_step > max_steps:
                break

        # New: Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        if max_steps is not None and global_step > max_steps:
            break

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def replace_linear_with_lora(model, rank, alpha, alternative=False):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            if alternative:
                setattr(model, name, LinearWithLoRAMerged(module, rank, alpha))
            else:
                setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply the same function to child modules
            replace_linear_with_lora(module, rank, alpha)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_size",
        type=str,
        default="gpt2-small (124M)",
        help=(
            "Which GPT model to use. Options: 'gpt2-small (124M)', 'gpt2-medium (355M)',"
            " 'gpt2-large (774M)', 'gpt2-xl (1558M)'."
        )
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="pretrained",
        help=(
            "Whether to use 'pretrained' or 'random' weights."
        )
    )
    parser.add_argument(
        "--trainable_layers",
        type=str,
        default="last_block",
        help=(
            "Which layers to train. Options: 'all', 'last_block', 'last_two_blocks', 'last_layer', 'lora', 'lora_alternative'."
        )
    )
    parser.add_argument(
        "--trainable_token_pos",
        type=str,
        default="last",
        help=(
            "Which token position to train. Options: 'first', 'last', 'flexible'."
        )
    )
    parser.add_argument(
        "--average_embeddings",
        action="store_true",
        default=False,
        help=(
            "Average the output embeddings from all tokens instead of using"
            " only the embedding at the token position specified by `--trainable_token_pos`."
        )
    )
    parser.add_argument(
        "--context_length",
        type=str,
        default="longest_training_example",
        help=(
            "The context length of the data inputs."
            " Options: 'longest_training_example', 'model_context_length' or integer value."
        )
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help=(
            "The LoRA rank when choosing `--trainable_layers lora`"
        )
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=8,
        help=(
            "The LoRA alpha value when choosing `--trainable_layers lora`"
        )
    )
    parser.add_argument(
        "--no_padding",
        action="store_true",
        default=False,
        help=(
            "Disable padding, which means each example may have a different length."
            " This requires setting `--batch_size 1`."
        )
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help=(
            "Number of training epochs."
        )
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=(
            "The batch size used for training."
        )
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help=(
            "Accumulation steps to allow for gradient accumulation."
            " See https://sebastianraschka.com/blog/2023/llm-grad-accumulation.html for explanation."
            " For example, setting `batch_size=8` and `accumulation_steps=1` compute the exact same"
            " loss and weight updates as setting `batch_size=1` and `accumulation_steps=8`, however,"
            " the latter setting uses more iterations."
        )
    )
    parser.add_argument(
        "--disable_causal_mask",
        action="store_true",
        default=False,
        help=(
            "Disables the causal attention mask."
        )
    )
    parser.add_argument(
        "--ignore_index",
        type=int,
        default=-100,
        help=(
            "Sets the `ignore_index` in the cross-entropy loss."
        )
    )

    args = parser.parse_args()

    if args.trainable_token_pos == "first":
        args.trainable_token_pos = 0
    elif args.trainable_token_pos == "last":
        args.trainable_token_pos = -1
    # The "flexible" setting selects the last tokens before the padding tokens
    # See https://github.com/rasbt/LLMs-from-scratch/discussions/434
    elif args.trainable_token_pos == "flexible":
        args.trainable_token_pos = "flexible"
    else:
        raise ValueError("Invalid --trainable_token_pos argument")

    ###############################
    # Load model
    ###############################

    if args.weights == "pretrained":
        load_weights = True
    elif args.weights == "random":
        load_weights = False
    else:
        raise ValueError("Invalid --weights argument.")

    model = instantiate_model(args.model_size, load_weights)
    for param in model.parameters():
        param.requires_grad = False

    if args.model_size == "gpt2-small (124M)":
        in_features = 768
    elif args.model_size == "gpt2-medium (355M)":
        in_features = 1024
    elif args.model_size == "gpt2-large (774M)":
        in_features = 1280
    elif args.model_size == "gpt2-xl (1558M)":
        in_features = 1600
    else:
        raise ValueError("Invalid --model_size argument")

    torch.manual_seed(123)
    model.out_head = torch.nn.Linear(in_features=in_features, out_features=2)

    if args.trainable_layers == "last_layer":
        pass
    elif args.trainable_layers == "last_block" or args.trainable_layers == "last_two_blocks":
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True
        if args.trainable_layers == "last_two_blocks":
            for param in model.trf_blocks[-2].parameters():
                param.requires_grad = True
    elif args.trainable_layers == "all":
        for param in model.parameters():
            param.requires_grad = True
    elif args.trainable_layers in ("lora", "lora_alternative"):
        if args.trainable_layers == "lora_alternative":
            alternative = True
        else:
            alternative = False
        replace_linear_with_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, alternative=alternative)
    else:
        raise ValueError("Invalid --trainable_layers argument.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ###############################
    # Instantiate dataloaders
    ###############################

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extract_to = "sms_spam_collection"
    new_file_path = Path(extract_to) / "SMSSpamCollection.tsv"

    base_path = Path(".")
    file_names = ["train.csv", "validation.csv", "test.csv"]
    all_exist = all((base_path / file_name).exists() for file_name in file_names)
    
    if not all_exist:
        try:
            download_and_unzip(url, zip_path, extract_to, new_file_path)
        except (requests.exceptions.RequestException, TimeoutError) as e:
            print(f"Primary URL failed: {e}. Trying backup URL...")
            backup_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
            download_and_unzip(backup_url, zip_path, extract_to, new_file_path)
        create_dataset_csvs(new_file_path)

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = None

    if args.no_padding:
        max_length = None

    else:
        if args.context_length == "model_context_length":
            max_length = model.pos_emb.weight.shape[0]
        elif args.context_length == "longest_training_example":
            train_dataset = SpamDataset(base_path / "train.csv", max_length=None, tokenizer=tokenizer, no_padding=args.no_padding)
            max_length = train_dataset.max_length
        else:
            try:
                max_length = int(args.context_length)
            except ValueError:
                raise ValueError("Invalid --context_length argument")

    if train_dataset is None:
        train_dataset = SpamDataset(base_path / "train.csv", max_length=max_length, tokenizer=tokenizer, no_padding=args.no_padding)
    val_dataset = SpamDataset(base_path / "validation.csv", max_length=max_length, tokenizer=tokenizer, no_padding=args.no_padding)
    test_dataset = SpamDataset(base_path / "test.csv", max_length=max_length, tokenizer=tokenizer, no_padding=args.no_padding)

    num_workers = 0

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    assert train_dataset.max_length <= model.pos_emb.weight.shape[0], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {model.pos_emb.weight.shape[0]}. Reinitialize data sets with "
        f"`max_length={model.pos_emb.weight.shape[0]}`"
    )

    ###############################
    # Train model
    ###############################

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs, eval_freq=50, eval_iter=5,
        max_steps=None, trainable_token_pos=args.trainable_token_pos,
        accumulation_steps=args.accumulation_steps, average_embeddings=args.average_embeddings
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    ###############################
    # Evaluate model
    ###############################

    train_accuracy = calc_accuracy_loader(
        train_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )
    val_accuracy = calc_accuracy_loader(
        val_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )
    test_accuracy = calc_accuracy_loader(
        test_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
