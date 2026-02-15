import tiktoken
import torch
import re
from config import SPECIAL_TOKENS

class TokenizerWrapper:
    def __init__(self, base_model_name="gpt2"):
        self.base_tokenizer = tiktoken.get_encoding(base_model_name)
        self.special_tokens = SPECIAL_TOKENS
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        # Base vocab size is 50257. New tokens are 50257, 50258.
        # Total size 50259.

    @property
    def eot_token(self):
        return self.base_tokenizer.eot_token

    def encode(self, text, allowed_special={"<|endoftext|>"}):
        """
        Encodes text, handling special tokens manually since tiktoken is frozen.
        We split the text by special token strings, encode the parts, and interleave the IDs.
        """
        # Create a regex pattern to split by special tokens
        # e.g. (<CODE_START>|<CODE_END>)
        pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens.keys()) + ")"
        
        parts = re.split(pattern, text)
        ids = []
        
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                # If the part is empty (e.g. adjacent special tokens), encode returns empty list
                if part:
                    ids.extend(self.base_tokenizer.encode(part, allowed_special=allowed_special))
        return ids

    def decode(self, token_ids):
        """
        Decodes token IDs, handling our custom special tokens.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        decoded_parts = []
        current_chunk = []
        
        for tid in token_ids:
            if tid in self.id_to_token:
                # Decode accumulated regular tokens
                if current_chunk:
                    decoded_parts.append(self.base_tokenizer.decode(current_chunk))
                    current_chunk = []
                # Append special token string
                decoded_parts.append(self.id_to_token[tid])
            else:
                current_chunk.append(tid)
        
        if current_chunk:
            decoded_parts.append(self.base_tokenizer.decode(current_chunk))
            
        return "".join(decoded_parts)
