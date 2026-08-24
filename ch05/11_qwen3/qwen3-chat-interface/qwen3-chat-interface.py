# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import torch
import chainlit

# For llms_from_scratch installation instructions, see:
# https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
from llms_from_scratch.kv_cache.qwen3 import (
    Qwen3Model,
    Qwen3Tokenizer,
    download_from_huggingface_from_snapshots,
    load_weights_into_qwen
)
from llms_from_scratch.kv_cache.generate import (
    generate_text_simple_stream
)

# ============================================================
# EDIT ME: Simple configuration
# ============================================================
MODEL = "0.6B"            # options: "0.6B","1.7B","4B","8B","14B","32B","30B-A3B"
REASONING = True          # True = "thinking" chat model, False = Base
DEVICE = "auto"           # "auto" | "cuda" | "mps" | "cpu"
MAX_NEW_TOKENS = 38912
LOCAL_DIR = None          # e.g., "Qwen3-0.6B-Base"; None auto-selects
# ============================================================


def get_qwen_config(name):
    if name == "0.6B":
        from llms_from_scratch.qwen3 import QWEN_CONFIG_06_B as QWEN3_CONFIG
    elif name == "1.7B":
        from llms_from_scratch.qwen3 import QWEN3_CONFIG_1_7B as QWEN3_CONFIG
    elif name == "4B":
        from llms_from_scratch.qwen3 import QWEN3_CONFIG_4B as QWEN3_CONFIG
    elif name == "8B":
        from llms_from_scratch.qwen3 import QWEN3_CONFIG_8B as QWEN3_CONFIG
    elif name == "14B":
        from llms_from_scratch.qwen3 import QWEN3_CONFIG_14B as QWEN3_CONFIG
    elif name == "32B":
        from llms_from_scratch.qwen3 import QWEN3_CONFIG_32B as QWEN3_CONFIG
    elif name == "30B-A3B":
        from llms_from_scratch.qwen3 import QWEN3_CONFIG_30B_A3B as QWEN3_CONFIG
    else:
        raise ValueError(f"Invalid model name: {name}")
    return QWEN3_CONFIG


def build_repo_and_local(model_name, reasoning, local_dir_arg):
    base = f"Qwen3-{model_name}"
    repo_id = f"Qwen/{base}-Base" if not reasoning else f"Qwen/{base}"
    local_dir = local_dir_arg if local_dir_arg else (f"{base}-Base" if not reasoning else base)
    return repo_id, local_dir


def get_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif name == "cuda":
        return torch.device("cuda")
    elif name == "mps":
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_model_and_tokenizer(qwen3_config, repo_id, local_dir, device, use_reasoning):
    model = Qwen3Model(qwen3_config)
    weights_dict = download_from_huggingface_from_snapshots(
        repo_id=repo_id,
        local_dir=local_dir
    )
    load_weights_into_qwen(model, qwen3_config, weights_dict)
    del weights_dict

    model.to(device)  # safe for all but required by the MoE model
    model.eval()

    tok_filename = "tokenizer.json"
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tok_filename,
        repo_id=repo_id,
        apply_chat_template=use_reasoning,
        add_generation_prompt=use_reasoning,
        add_thinking=use_reasoning
    )
    return model, tokenizer


QWEN3_CONFIG = get_qwen_config(MODEL)
REPO_ID, LOCAL_DIR = build_repo_and_local(MODEL, REASONING, LOCAL_DIR)
DEVICE = get_device(DEVICE)
MODEL, TOKENIZER = get_model_and_tokenizer(QWEN3_CONFIG, REPO_ID, LOCAL_DIR, DEVICE, REASONING)


@chainlit.on_chat_start
async def on_start():
    chainlit.user_session.set("history", [])
    chainlit.user_session.get("history").append(
        {"role": "system", "content": "You are a helpful assistant."}
    )


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    # 1) Encode input
    input_ids = TOKENIZER.encode(message.content)
    input_ids_tensor = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)

    # 2) Start an outgoing message we can stream into
    out_msg = chainlit.Message(content="")
    await out_msg.send()

    # 3) Stream generation
    for tok in generate_text_simple_stream(
        model=MODEL,
        token_ids=input_ids_tensor,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=TOKENIZER.eos_token_id
    ):
        token_id = tok.squeeze(0)
        piece = TOKENIZER.decode(token_id.tolist())
        await out_msg.stream_token(piece)

    # 4) Finalize the streamed message
    await out_msg.update()
