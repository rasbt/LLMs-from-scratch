# Sliding Window Attention (SWA)

This bonus material illustrates the memory savings when using Sliding Window Attention (SWA) over regular Multi-Head Attention (MHA).



&nbsp;
## Introduction

What is sliding window attention (SWA)? If we think of regular self-attention as a *global* attention mechanism, since each sequence element can access every other sequence element, then we can think of SWA as *local* attention, because here we restrict the context size around the current query position. This is illustrated in the figure below.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/1.webp?2" alt="Sliding Window Attention" width="500px" />

As shown in the figure above, instead of attending to all previous tokens, each token only attends to a fixed-size local window around its position. This localized attention lowers the size of the KV cache substantially.

In the remainder of this introduction, we will discuss SWA in the context of [Gemma 3](https://arxiv.org/abs/2503.19786), which is implemented from scratch in [../../ch05/12_gemma3](../../ch05/12_gemma3).

Sliding window attention was originally introduced in the [LongFormer paper in 2020](https://arxiv.org/abs/2004.05150), but the reason we focus on Google's Gemma models is that they are very good open-weight models showing that sliding window attention is indeed a feasible approach in recent, capable models.

[Gemma 2](https://arxiv.org/abs/2408.00118) used a hybrid approach that combined local (sliding window) and global attention layers in a 1:1 ratio. Each token could attend to a context window of 4 k tokens. The reason for this 1:1 hybrid is that it strikes a balance between efficiency and global context modeling, since an LLM using only local attention can be too restrictive.

[Gemma 3](https://arxiv.org/abs/2503.19786) then took the design further toward efficiency. It used a 5:1 ratio between sliding window and full attention layers, which means that for every five local attention layers, there is one global layer. In addition, the sliding window size was reduced from 4096 tokens in Gemma 2 to 1024 tokens in Gemma 3. 

Interestingly, the ablation studies in the Gemma 3 technical report indicate that these changes have only a minor effect on overall model quality. In other words, the substantial memory and compute savings achieved through sliding window attention come with minimal loss in modeling performance.



&nbsp;
## Sliding Window Attention (SWA) Memory Savings

The memory savings are mostly reflected in the KV storage. We can compute the KV storage size with the following formula:

bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

When using SWA, we replace the sequence length (seqlen) above by the window size W. So, when using sliding window attention, we reduce the KV cache size by a factor of "W / seqlen". (Note that for simplicity, this assumes that sliding window attention is used in every layer.)


You can use the [memory_estimator_swa.py](memory_estimator_swa.py) script in this folder to apply this for different model configs to see how much memory you can save by using SWA over MHA:

```bash
➜ uv run memory_estimator_swa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 1024 --swa_ratio "5:1"
==== Config ====
context_length         : 32768
sliding_window_size    : 1024
emb_dim                : 4096
n_heads                : 32
n_layers               : 32
n_kv_groups            : 4
batch_size             : 1
dtype                  : bf16 (2 Bytes/elem)
head_dim               : 128
GQA n_kv_heads         : 8
Effective SWA window W : 1024
Layer ratio (SWA:Full) : 5:1
Distributed layers     : 27 SWA, 5 FULL

==== KV-cache totals across all layers ====
MHA KV total           : 17.18 GB
GQA KV total           : 4.29 GB
MHA + SWA (Ratio: 5:1) : 3.14 GB
MHA + GQA (Ratio: 5:1) : 0.78 GB
```

Note that Gemma 3 uses SWA in combination with GQA.

The savings when using SWA over MHA are further shown in the plot below for different context lengths:

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/4.webp?2" alt="SWA" width="800px" />

&nbsp;

You can reproduce thi plots via:

```bash
uv run plot_memory_estimates_swa.py \
  --emb_dim 4096 --n_heads 48 --n_layers 36 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 2048 --swa_ratio "5:1"
```


&nbsp;
## SWA Code Examples

The [gpt_with_kv_mha.py](gpt_with_kv_mha.py) and [gpt_with_kv_swa.py](gpt_with_kv_swa.py) scripts in this folder provide hands-on examples for comparing the MHA and SWA memory usage in the context of a GPT model implementation.

Note that SWA can also be used in combination with MLA and GQA (as mentioned earlier), but for simplicity, this is not done here.

Note that the model is not trained and thus generates nonsensical text. However, you can use it as a drop-in replacement for the standard GPT model in chapters 5-7 and train it.

Also, this implementation uses the KV cache explained in [another bonus section](../03_kv-cache), so the memory savings are more pronounced.

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_swa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--sliding_window_size 1024 \
--sliding_window_stride 5   # like Gemma 3

...

Time: 514.38 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

The reason why we are not seeing such a big saving as in the plots above is 2-fold:

1. I use a smaller configuration to have the model finish the generation in a reasonable time.
2. More importantly, we are looking at the whole model here, not just the attention mechanism; the fully-connected layers in the model take up most of the memory (but this is a topic for a separate analysis).
