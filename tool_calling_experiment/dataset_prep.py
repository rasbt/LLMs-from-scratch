import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import re

class ToolCallingDataset(IterableDataset):
    def __init__(self, tokenizer, max_length=1024, stride=512, split="train", limit=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.dataset = load_dataset("jtatman/python-code-dataset-500k", split=split, streaming=True)
        self.limit = limit
        
        # Regex to extract code
        self.code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)
        
    def __iter__(self):
        count = 0
        for example in self.dataset:
            if self.limit and count >= self.limit:
                break
                
            instruction = example.get('instruction', '')
            output = example.get('output', '')
            
            # Extract code
            match = self.code_pattern.search(output)
            if not match:
                continue
                
            code = match.group(1).strip()
            
            if not instruction or not code:
                continue
                
            # Format: Instruction: {inst}\n<CODE_START>\n{code}\n<CODE_END>
            text = f"Instruction: {instruction}\n<CODE_START>\n{code}\n<CODE_END>"
            
            # Encode
            token_ids = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            token_ids.append(self.tokenizer.eot_token)
            
            # Sliding window or just yield if it fits?
            # For simplicity in this project (instruction following), we usually want 
            # the full context. If it's too long, we might truncate or skip.
            # GPT-2 small context is 1024.
            
            if len(token_ids) > self.max_length:
                # Truncate or skip? 
                # Let's skip very long ones during training to avoid fragmentation logic here
                # or just take the first chunk.
                token_ids = token_ids[:self.max_length]
                
            # Yield (input, target) where target is shifted by 1
            # But wait, pytorch dataloader expects tensors.
            # Since this is an IterableDataset, we can yield individual samples.
            # BUT, we need to batch them. DataLoader handles batching from iterable? Yes.
            
            # For autoregressive training:
            # Input:  [A, B, C]
            # Target: [B, C, D]
            
            if len(token_ids) <= 1:
                continue
                
            input_chunk = torch.tensor(token_ids[:-1])
            target_chunk = torch.tensor(token_ids[1:])
            
            yield input_chunk, target_chunk
            count += 1

def collate_batch(batch):
    # Batch is a list of (input_chunk, target_chunk) tuples
    # We need to pad them to the longest sequence in the batch
    
    # 1. Separate inputs and targets
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # 2. Pad
    # Use eot_token (50256) for padding? Or 0? EOT is safer for GPT-2 as it ignores it usually?
    # Actually, we should use a pad token. GPT-2 doesn't have one by default, often EOT is used.
    pad_token_id = 50256 # <|endoftext|>
    
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_token_id)
    
    # Create mask? 
    # GPT-2 implementation in the book might not handle padding masks explicitly in the attention 
    # if we supply it. 
    # However, for training formatting, standard causal masking is used. 
    # If we pad at the end, the model will just predict padding from padding.
    # Ideally we should mask the loss for padding tokens.
    # But for simplicity, we'll just pad.
    
    return inputs_padded, targets_padded

def create_dataloader(tokenizer, batch_size=4, max_length=1024, limit=None):
    dataset = ToolCallingDataset(tokenizer, max_length=max_length, limit=limit)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch)
