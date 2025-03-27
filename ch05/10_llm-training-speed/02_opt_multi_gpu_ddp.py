# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import os
import time
import urllib.request

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

# NEW imports (see Appendix A):
import platform
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# NEW: function to initialize a distributed process group (1 process / GPU)
# this allows communication among processes
# (see Appendix A):
def ddp_setup(rank, world_size):
    """
    Arguments:
        rank: a unique process ID
        world_size: total number of processes in the group
    """
    # Only set MASTER_ADDR and MASTER_PORT if not already defined by torchrun
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"

    # initialize process group
    if platform.system() == "Windows":
        # Disable libuv because PyTorch for Windows isn't built with support
        os.environ["USE_LIBUV"] = "0"
        # Windows users may have to use "gloo" instead of "nccl" as backend
        # gloo: Facebook Collective Communication Library
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA Collective Communication Library
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)


#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# NEW: Modify to set shuffle=False and use a sampler
# (See Appendix A):
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # NEW: False because of DistributedSampler below
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        # NEW: chunk batches across GPUs without overlapping samples:
        sampler=DistributedSampler(dataset)  # NEW
    )
    return dataloader


#####################################
# Chapter 3
#####################################
class PyTorchMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec


#####################################
# Chapter 4
#####################################


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = PyTorchMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

#####################################
# Chapter 5
#####################################


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


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


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, device, start_context):
    model.eval()

    # NEW: Modify for DDP
    context_size = model.module.pos_emb.weight.shape[0] if isinstance(model, DDP) else model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tiktoken.get_encoding("gpt2")).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tiktoken.get_encoding("gpt2"))
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model_simple_with_timing(model, train_loader, val_loader, optimizer, device,
                                   num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens = [], [], []
    total_tokens, global_step, last_tokens = 0, -1, 0

    # NEW: Determine the current rank (default to 0 if not distributed)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    # world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    # Variables for cumulative average tokens/sec
    cumulative_tokens, cumulative_time = 0.0, 0.0

    # CUDA-specific timing setup
    use_cuda = device.type == "cuda"
    if use_cuda:
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()  # Ensure all prior CUDA operations are done
        t_start.record()          # Start the timer for the first interval
    else:
        t0 = time.time()          # Start the timer for the first interval

    # Main training loop
    for epoch in range(num_epochs):
        # NEW: set epoch for DistributedSampler so each process gets a unique shuffle order
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        for inp_batch, tgt_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # Forward and backward pass
            loss = calc_loss_batch(inp_batch, tgt_batch, model, device)
            loss.backward()
            optimizer.step()

            total_tokens += inp_batch.numel()

            # At evaluation intervals, measure elapsed time and tokens per second
            if global_step % eval_freq == 0:
                # End timing for the current interval
                if use_cuda:
                    t_end.record()
                    torch.cuda.synchronize()  # Wait for all CUDA ops to complete.
                    elapsed = t_start.elapsed_time(t_end) / 1000  # Convert ms to seconds
                    t_start.record()  # Reset timer for the next interval
                else:
                    elapsed = time.time() - t0
                    t0 = time.time()  # Reset timer for the next interval

                # Calculate local tokens processed during this interval
                local_interval = total_tokens - last_tokens
                last_tokens = total_tokens

                # Aggregate the tokens processed over all devices
                local_tensor = torch.tensor([local_interval], device=device, dtype=torch.float)
                global_tensor = local_tensor.clone()
                torch.distributed.all_reduce(global_tensor, op=torch.distributed.ReduceOp.SUM)
                global_interval = global_tensor.item()

                # Global tokens per second for this interval
                global_tps = global_interval / elapsed if elapsed > 0 else 0

                # Update cumulative tokens (local) and aggregate globally
                cumulative_tokens += local_interval
                local_cum_tensor = torch.tensor([cumulative_tokens], device=device, dtype=torch.float)
                global_cum_tensor = local_cum_tensor.clone()
                torch.distributed.all_reduce(global_cum_tensor, op=torch.distributed.ReduceOp.SUM)
                global_cumulative_tokens = global_cum_tensor.item()
                cumulative_time += elapsed
                global_avg_tps = global_cumulative_tokens / cumulative_time if cumulative_time > 0 else 0

                # Evaluate model performance (this may add overhead)
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens.append(total_tokens)

                # NEW: Only print logs once per GPU (choosing the rank 0 GPU)
                if rank == 0:
                    print(f"Ep {epoch+1}, Step {global_step:06d}, "
                          f"Train: {train_loss:.3f}, Val: {val_loss:.3f}, "
                          f"Step tok/sec: {round(global_tps)}, Global avg tok/sec: {round(global_avg_tps)}")

        # NEW Only rank 0 prints the generated sample and memory usage stats
        if rank == 0 and epoch % 5 == 0:
            generate_and_print_sample(model, device, start_context)

            # Memory stats
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(current_device) / 1024**3  # Convert to GB
                reserved = torch.cuda.memory_reserved(current_device) / 1024**3    # Convert to GB

                print(f"\nAllocated memory: {allocated:.4f} GB")
                print(f"Reserved memory: {reserved:.4f} GB\n")

    return train_losses, val_losses, track_tokens


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


#####################################
# Main function calls
#####################################

# NEW: Add rank and world_size
def main(gpt_config, settings, rank, world_size):

    ddp_setup(rank, world_size)  # NEW: initialize process groups
    device = torch.device("cuda", rank)

    torch.manual_seed(123)

    # NEW: Print info only on 1 GPU
    if rank == 0:
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")

            capability = torch.cuda.get_device_capability()
            if capability[0] >= 7:  # Volta (7.0+), Turing (7.5+), Ampere (8.0+), Hopper (9.0+)
                torch.set_float32_matmul_precision("high")
                print("Uses tensor cores")
            else:
                print("Tensor cores not supported on this GPU. Using default precision.")
        print()

    ##############################
    # Download data if necessary
    ##############################

    file_path = "middlemarch.txt"
    url = "https://www.gutenberg.org/cache/epub/145/pg145.txt"

    # NEW: Only download 1 time
    if rank == 0:
        if not os.path.exists(file_path):
            with urllib.request.urlopen(url) as response:
                text_data = response.read().decode('utf-8')
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_data)

    # NEW: All processes wait until rank 0 is done, using the GPU index.
    torch.distributed.barrier(device_ids=[device.index])

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)
    model = torch.compile(model)
    model = model.to(device)
    model = model.to(torch.bfloat16)
    # NEW: Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
        fused=True
    )

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        num_workers=4
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        num_workers=4
    )

    ##############################
    # Train model
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple_with_timing(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=settings["num_epochs"],
        eval_freq=5,
        eval_iter=1,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    # NEW: Clean up distributed processes
    destroy_process_group()

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    # NEW: Extract rank and world size from environment variables
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = 0

    GPT_CONFIG_124M = {
        "vocab_size": 50304,     # Vocabulary size
        "context_length": 1024,  # Input tokens per training example
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-key-value bias
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,  # * world_size,  # NEW: Increase learning rate to account for multiple GPUs
        "num_epochs": 50,
        "batch_size": 32,
        "weight_decay": 0.1
    }

    ###########################
    # Initiate training
    ###########################

    train_losses, val_losses, tokens_seen, model = main(
        GPT_CONFIG_124M, OTHER_SETTINGS,
        rank, world_size  # NEW
    )

    ###########################
    # After training
    ###########################

    # NEW: Only create 1 plot
    if rank == 0:
        # Plot results
        epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        plt.savefig("loss.pdf")

    # Save and load model
    #
    # compiled = hasattr(model, "_orig_mod")
    # if compiled:
    #     torch.save(model._orig_mod.state_dict(), "model.pth")
    # else:
    #     torch.save(model.state_dict(), "model.pth")
    #
    # model = GPTModel(GPT_CONFIG_124M)
    # model.load_state_dict(torch.load("model.pth", weights_only=True))
