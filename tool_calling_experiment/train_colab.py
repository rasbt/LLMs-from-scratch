import torch
import torch.nn as nn
import os
import urllib.request
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from architecture import GPTModel, load_weights_into_gpt
from config import GPT_CONFIG_124M, GPT_CONFIG_355M, GPT_CONFIG_774M, GPT_CONFIG_1558M
from tokenizer_utils import TokenizerWrapper
from dataset_prep import create_dataloader


def download_file(url, destination):
    # Simplified download utility
    import requests
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("content-length", 0))

    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    block_size = 1024
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=url.split("/")[-1]) as progress_bar:
        with open(destination, "wb") as file:
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))
                file.write(chunk)

def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def train(model_size="124M", max_steps=1000, batch_size=2, accumulation_steps=8, max_length=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Download & Load Params
    print(f"Downloading {model_size} weights...")
    settings, params = download_and_load_gpt2(model_size, "models")
    
    # 2. Init Model & Load Weights
    print("Initializing architecture...")
    # Map settings to our config format if needed, but we used GPT_CONFIG_124M as base.
    # We should ensure config matches loaded settings.
    if model_size == "124M":
        cfg = GPT_CONFIG_124M
    elif model_size == "355M":
        cfg = GPT_CONFIG_355M
    elif model_size == "774M":
        cfg = GPT_CONFIG_774M
    elif model_size == "1558M":
        cfg = GPT_CONFIG_1558M
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    # Update max_length from args in case we want shorter context to save memory
    cfg["context_length"] = max_length
    
    model = GPTModel(cfg)
    print("Loading weights into model...")
    load_weights_into_gpt(model, params)
    model.to(device)
    print("Weights loaded successfully.")
    
    # 3. Resize Embeddings for Special Tokens
    tokenizer = TokenizerWrapper()
    new_vocab_size = tokenizer.base_tokenizer.n_vocab + len(tokenizer.special_tokens)
    print(f"Resizing model vocab to {new_vocab_size}...")
    
    old_emb = model.tok_emb
    new_emb = nn.Embedding(new_vocab_size, cfg["emb_dim"])
    # Copy existing
    new_emb.weight.data[:old_emb.num_embeddings] = old_emb.weight.data
    # Init new (mean)
    new_emb.weight.data[old_emb.num_embeddings:] = old_emb.weight.data.mean(dim=0, keepdim=True)
    model.tok_emb = new_emb.to(device)
    
    # Resize Output Head
    old_head = model.out_head
    new_head = nn.Linear(cfg["emb_dim"], new_vocab_size, bias=False)
    new_head.weight.data[:old_head.out_features] = old_head.weight.data
    new_head.weight.data[old_head.out_features:] = new_emb.weight.data[old_head.out_features:]
    model.out_head = new_head.to(device)
    
    # 4. Data Loader
    train_loader = create_dataloader(tokenizer, batch_size=batch_size, max_length=cfg["context_length"])
    
    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    optimizer.zero_grad() # Initialize gradients
    
    # 6. Loop
    model.train()
    step = 0
    print("Starting training...")
    
    for input_chunk, target_chunk in train_loader:
        input_chunk, target_chunk = input_chunk.to(device), target_chunk.to(device)
        
        # optimizer.zero_grad() # Moved to accumulation step
        logits = model(input_chunk)
        
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), 
            target_chunk.flatten(0, 1)
        )
        
        # Gradient Accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            if (step + 1) % (10 * accumulation_steps) == 0:
                print(f"Step {step + 1}: Loss {loss.item() * accumulation_steps:.4f}")
            
        step += 1
        if step >= max_steps * accumulation_steps:
            break
            
    print("Training complete.")
    torch.save(model.state_dict(), "tool_llm.pth")
    print("Model saved to 'tool_llm.pth'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="124M", help="GPT-2 model size (124M, 355M, 774M, 1558M)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per step. Decrease if OOM.")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Total training steps (batches processed).")
    parser.add_argument("--max_length", type=int, default=1024, help="Context length.")
    
    args = parser.parse_args()
    
    print(f"Training with: {args}")
    train(model_size=args.model_size, max_steps=args.max_steps, batch_size=args.batch_size, accumulation_steps=args.accumulation_steps, max_length=args.max_length)
