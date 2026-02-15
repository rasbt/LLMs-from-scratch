# GPT-2 Configurations
# These match the standard GPT-2 sizes.
# The `vocab_size` will be updated at runtime/initialization to 50259 to include special tokens.

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 50257 base + 2 special = 50259
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

GPT_CONFIG_355M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": True
}

GPT_CONFIG_774M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1280,
    "n_heads": 20,
    "n_layers": 36,
    "drop_rate": 0.1,
    "qkv_bias": True
}

GPT_CONFIG_1558M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1600,
    "n_heads": 25,
    "n_layers": 48,
    "drop_rate": 0.1,
    "qkv_bias": True
}

# Special Tokens for Tool Calling
# We append these to the end of the standard vocabulary
SPECIAL_TOKENS = {
    "<CODE_START>": 50257,
    "<CODE_END>": 50258
}

FINAL_VOCAB_SIZE = 50259
