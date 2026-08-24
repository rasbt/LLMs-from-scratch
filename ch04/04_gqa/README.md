# Grouped-Query Attention (GQA)

This bonus material illustrates the memory savings when using Grouped-Query Attention (GQA) over regular Multi-Head Attention (MHA).

&nbsp;
## Introduction

Grouped-Query Attention (GQA) has become the new standard replacement for a more compute- and parameter-efficient alternative to Multi-Head Attention (MHA) in recent years. Note that it's not new and goes back to the 2023 [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). And even the larger variants in the good old Llama 2 series used it.

Here's a brief GQA summary. Unlike MHA, where each head also has its own set of keys and values, to reduce memory usage, GQA groups multiple heads to share the same key and value projections.

For example, as further illustrated in the figure below, if there are 3 key-value groups and 6 attention heads, then heads 1 and 2 share one set of keys and values, while heads 3 and 4, as well as heads 5 and 6, share another, respectively.

&nbsp;

![GQA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/1.webp?1)

&nbsp;

This sharing of keys and values reduces the total number of key and value computations, which leads to lower memory usage and improved efficiency.

So, to summarize, the core idea behind GQA is to reduce the number of key and value heads by sharing them across multiple query heads. This (1) lowers the model's parameter count and (2) reduces the memory bandwidth usage for key and value tensors during inference since fewer keys and values need to be stored and retrieved from the KV cache.

While GQA is mainly a computational-efficiency workaround for MHA, ablation studies (such as those in the [original GQA paper](https://arxiv.org/abs/2305.13245) and the [Llama 2 paper](https://arxiv.org/abs/2307.09288)) show it performs comparably to standard MHA in terms of LLM modeling performance.

However, this assumes that the number of key-value groups is chosen carefully. In the extreme case where all attention heads share a single key-value group, known as multi-query attention, the memory usage decreases even more drastically but modeling performance can suffer. (And, on the other extreme, if we set the number of key-value groups equal to the number of query heads, we are back at standard multi-head attention.)

&nbsp;
## GQA Memory Savings

The memory savings are mostly reflected in the KV storage. We can compute the KV storage size with the following formula:

bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

You can use the [memory_estimator_gqa.py](memory_estimator_gqa.py) script in this folder to apply this for different model configs to see how much memory you can save by using GQA over MHA:

```bash
➜ uv run memory_estimator_gqa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16
==== Config ====
context_length   : 32768
emb_dim          : 4096
n_heads          : 32
n_layers         : 32
n_kv_groups      : 4
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 128
GQA n_kv_heads   : 8

==== KV-cache totals across all layers ====
MHA total KV cache  : 17.18 GB
GQA total KV cache  : 4.29 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
```

The savings when using GQA over MHA are further shown in the plot below for different key-value group sizes as a function of the context length:

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/3.webp?4" alt="GQA" width="500px" />

&nbsp;

You can reproduce the plot via `uv run plot_memory_estimates_gqa.py`.

&nbsp;
## GQA Code Examples

The [gpt_with_kv_mha.py](gpt_with_kv_mha.py) and [gpt_with_kv_gqa.py](gpt_with_kv_gqa.py) scripts in this folder provide hands-on examples for comparing the MHA and GQA memory usage in the context of a GPT model implementation.

Note that GQA is also used in the [Llama 3](../../ch05/07_gpt_to_llama), [Gemma 3](../../ch05/12_gemma3), and [Qwen3](../../ch05/11_qwen3) bonus materials. However, for simplicity, the code scripts in this folder modify the GPT architecture, which traditionally didn't use GQA.

Note that the model is not trained and thus generates nonsensical text. However, you can use it as a drop-in replacement for the standard GPT model in chapters 5-7 and train it.

Also, this implementation uses the KV cache explained in [another bonus section](../03_kv-cache) so the memory savings are more pronounced.

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_gqa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--n_kv_groups 4

...

Time: 516.33 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

The reason why we are not seeing such a big saving as in the plots above is 2-fold:

1. I use a smaller configuration to have the model finish the generation in a reasonable time.
2. More importantly, we are looking at the whole model here, not just the attention mechanism; the fully-connected layers in the model take up most of the memory (but this is a topic for a separate analysis).
