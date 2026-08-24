# Multi-Head Latent Attention (MLA)

This bonus material illustrates the memory savings when using Multi-Head Latent Attention (MLA) over regular Multi-Head Attention (MHA).

&nbsp;
## Introduction

In [../04_gqa](../04_gqa), we discussed Grouped-Query Attention (GQA) as a computational-efficiency workaround for MHA. And ablation studies (such as those in the[ original GQA paper](https://arxiv.org/abs/2305.13245) and the [Llama 2 paper](https://arxiv.org/abs/2307.09288)) show it performs comparably to standard MHA in terms of LLM modeling performance.

Now, Multi-Head Latent Attention (MLA), which is used in [DeepSeek V2, V3, and R1](https://arxiv.org/abs/2412.19437), offers a different memory-saving strategy that also pairs particularly well with KV caching. Instead of sharing key and value heads like GQA, MLA compresses the key and value tensors into a lower-dimensional space before storing them in the KV cache. 

At inference time, these compressed tensors are projected back to their original size before being used, as shown in the figure below. This adds an extra matrix multiplication but reduces memory usage.

&nbsp;

![MLA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/1.webp)

&nbsp;

(As a side note, the queries are also compressed, but only during training, not inference.)

By the way, as mentioned earlier, MLA is not new in DeepSeek V3, as its [DeepSeek V2 predecessor](https://arxiv.org/abs/2405.04434) also used (and even introduced) it. Also, the V2 paper contains a few interesting ablation studies that may explain why the DeepSeek team chose MLA over GQA (see the figure below).

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/2.webp" alt="GQA" width="500px" />

&nbsp;

As shown in the figure above, GQA appears to perform worse than MHA, whereas MLA offers better modeling performance than MHA, which is likely why the DeepSeek team chose MLA over GQA. (It would have been interesting to see the "KV Cache per Token" savings comparison between MLA and GQA as well!)

To summarize this section, before we move on to the next architecture component, MLA is a clever trick to reduce KV cache memory use while even slightly outperforming MHA in terms of modeling performance.

&nbsp;
## MLA Memory Savings

The memory savings are mostly reflected in the KV storage. We can compute the KV storage size with the following formula:

bytes ≈ batch_size × seqlen × n_layers × latent_dim × bytes_per_elem

In contrast, MHA KV cache memory is computed as follows:

bytes ≈ batch_size × seqlen × n_layers × embed_dim × 2 (K,V) × bytes_per_elem

This means, in MLA, we reduce "embed_dim × 2 (K,V)" to "latent_dim", since we only stored the compressed latent representation instead of the full key and value vectors as shown in the earlier figure above.



You can use the [memory_estimator_mla.py](memory_estimator_mla.py) script in this folder to apply this for different model configs to see how much memory you can save by using MLA over MHA:

```bash
➜ uv run memory_estimator_mla.py \
  --context_length 8192 \
  --emb_dim 2048 \
  --n_heads 24 \
  --n_layers 48 \
  --n_kv_groups 4 \
  --batch_size 1 \
  --dtype bf16 \
  --latent_dim 1024
==== Config ====
context_length   : 8192
emb_dim          : 2048
n_heads          : 24
n_layers         : 48
n_kv_groups      : 4
latent_dim       : 1024
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 86
GQA n_kv_heads   : 6

==== KV-cache totals across all layers ====
MHA total KV cache  : 3.25 GB
GQA total KV cache  : 0.81 GB
MLA total KV cache  : 0.81 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
Ratio (MHA / MLA)   : 4.03x
Savings (MLA vs MHA): 75.19%
```

Note that the compression above (`--emb_dim 2048 -> latent_dim 1024`) to achieve a similar saving as for GQA. In practice, the compression is a hyperparameter that needs to be carefully investigated, as choosing `latent_dim` to be too small can have negative impact on the modeling performance (similar to choosing too many `n_kv_groups` in GQA).

The savings when using MLA over MHA are further shown in the plot below for different `latent_dim` values as a function of the context length:

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/3.webp?2" alt="GQA" width="500px" />

&nbsp;

You can reproduce the plot via `uv run plot_memory_estimates_mla.py`.



&nbsp;
## MLA Code Examples

The [gpt_with_kv_mha.py](gpt_with_kv_mha.py) and [gpt_with_kv_mla.py](gpt_with_kv_mla.py) scripts in this folder provide hands-on examples for comparing the MHA and MLA memory usage in the context of a GPT model implementation. 

Here, the MLA code is inspired by the [https://huggingface.co/bird-of-paradise/deepseek-mla](https://huggingface.co/bird-of-paradise/deepseek-mla) implementation.

Note that MLA can also be used in combination with [GQA](../04_gqa), but for simplicity, I this is not done here. (Currently, I am also not aware of a prominent LLM doing this.)

Also note that the model is not trained and thus generates nonsensical text. However, you can use it as a drop-in replacement for the standard GPT model in chapters 5-7 and train it.

Lastly, this implementation uses the KV cache explained in [another bonus section](../03_kv-cache) so the memory savings are more pronounced.

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
uv run gpt_with_kv_mla.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--latent_dim 192 # (768×2)/192 = 8× compression

...

Time: 487.21 sec
67 tokens/sec
Max memory allocated: 0.68 GB
```

The reason why we are not seeing such a big saving as in the plots above is 2-fold:

1. I use a smaller configuration to have the model finish the generation in a reasonable time.
2. More importantly, we are looking at the whole model here, not just the attention mechanism; the fully-connected layers in the model take up most of the memory (but this is a topic for a separate analysis).
