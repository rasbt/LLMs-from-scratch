# Mixture of Experts (MoE)

This bonus material illustrates the memory savings (per token) when using Mixture-of-Experts (MoE) layers instead of regular feed-forward (FFN) layers.



&nbsp;
## Introduction

The core idea in MoE is to replace each feed-forward module in a transformer block with multiple expert layers, where each of these expert layers is also a feed-forward module. This means we replace a single feed-forward block with multiple feed-forward blocks, as illustrated in the figure below.



&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/1.webp" alt="SWA" width="800px" />

The feed-forward block inside a transformer block (shown as the dark gray block in the figure above) typically contains a large number of the model's total parameters. (Note that the transformer block, and thereby the feed-forward block, is repeated many times in an LLM; in the case of DeepSeek-V3, 61 times.)

So, replacing *a single* feed-forward block with *multiple* feed-forward blocks (as done in a MoE setup) substantially increases the model's total parameter count. However, the key trick is that we don't use ("activate") all experts for every token. Instead, a router selects only a small subset of experts per token.

Because only a few experts are active at a time, MoE modules are often referred to as *sparse*, in contrast to *dense* modules that always use the full parameter set. However, the large total number of parameters via an MoE increases the capacity of the LLM, which means it can take up more knowledge during training. The sparsity keeps inference efficient, though, as we don't use all the parameters at the same time.

For example, DeepSeek-V3 has 256 experts per MoE module and a total of 671 billion parameters. Yet during inference, only 9 experts are active at a time (1 shared expert plus 8 selected by the router). This means just 37 billion parameters are used for each token inference step as opposed to all 671 billion.

One notable feature of DeepSeek-V3's MoE design is the use of a shared expert. This is an expert that is always active for every token. This idea is not new and was already introduced in the [2022 DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) and the [2024 DeepSeek MoE](https://arxiv.org/abs/2401.06066) papers.

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/3.webp?1" alt="MoE shared expert" width="500px" />

(An annotated figure from the [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) paper.)

&nbsp;

The benefit of having a shared expert was first noted in the [DeepSpeed-MoE paper](https://arxiv.org/abs/2201.05596), where they found that it boosts overall modeling performance compared to no shared experts. This is likely because common or repeated patterns don't have to be learned by multiple individual experts, which leaves them with more room for learning more specialized patterns.

&nbsp;
## Mixture of Experts (MoE) Memory Savings

The memory savings in MoE models primarily come from reduced activation storage and compute. In a regular (dense) feed-forward layer (FFN), every token activates the full intermediate dimension. 

In contrast, an MoE layer routes each token through only a small subset of experts (for example, `top_k` out of `num_experts`) per token.

When using an MoE layer, only `top_k` experts are active per token, so the effective memory (and compute) scales by roughly a factor of `top_k / num_experts` relative to a dense FFN of the same total capacity.


You can use the [memory_estimator_moe.py](memory_estimator_moe.py) script in this folder to apply this for different model configs to see how much memory you can save by using MoE over FFN (note that this is for a single transformer block, to get the total savings, multiply by the number of transformer blocks in your model):

```bash
uv run memory_estimator_moe.py --emb_dim 7168 --hidden_dim 14336 --ffn_type swiglu \
  --num_experts 8 --top_k 2 --match_dense 
==== Config ====
emb_dim                : 7168
hidden_size            : 14336
ffn_type               : swiglu
num_experts            : 8
top_k                  : 2
dtype                  : bf16 (2 Bytes/elem)
match_dense            : True

==== Model weights (parameters) ====
Dense FFN params       : 308,281,344 (0.62 GB)
Per-expert params      : 38,535,168 (0.08 GB)
Router params          : 57,344 (0.00 GB)
MoE TOTAL params       : 308,338,688 (0.62 GB)
MoE ACTIVE/Token       : 77,127,680 (0.15 GB)
moe_hidden_size        : 1792
```

So, based on the results above, we can see that if we have a FFN with an input/output dimension (`emb_dim`) of 7,168 and an intermediate size (`hidden_dim`) of 14,336, we have ~308M parameters in this layer, and all these parameters are active in the forward pass.

Now, if we use an MoE layer with roughly the same number of total parameters (~308M), with 8 experts where 2 experts are active, only ~77M parameters are active in each forward pass. 

Moreover, at a constant number of experts, the more experts we have, the lower the number of active parameters becomes, and the greater the "savings":

&nbsp;

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/2.webp" alt="SWA" width="500px" />



&nbsp;

You can reproduce this plot via:

```bash
uv run plot_memory_estimates_moe.py \
    --emb_dim 7168 \
    --hidden_dim 28672 \
    --ffn_type swiglu \
    --top_k 8
```


&nbsp;
## MoE Code Examples

The [gpt_with_kv_ffn.py](gpt_with_kv_ffn.py) and [gpt_with_kv_moe.py](gpt_with_kv_moe.py) scripts in this folder provide hands-on examples for comparing the regular FFN and MoE memory usage in the context of a GPT model implementation. Note that both scripts use [SwiGLU](https://arxiv.org/abs/2002.05202) feed-forward modules as shown in the first figure of this page (GPT-2 traditionally uses GELU).

**Note: The model is not trained and thus generates nonsensical text. You can find a trained MoE in the bonus materials at [../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb](../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb).**



First, let's run the model with a regular FFN:


```bash
uv run gpt_with_kv_ffn.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 32768

...
Avg FFN time/call: 0.759 ms
Avg FFN mem delta/call: 0.19 MB (max 0.75 MB)
...
Time: 25.13 sec
40 tokens/sec
Max memory allocated: 11.47 GB
```

For a fair comparison with an MoE, we have to shrink the expert size. E.g., of we use 32 experts, we have to set `--hidden_dim 32768/32`:


```bash
uv run gpt_with_kv_moe.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 1024 \
--num_experts 32 \
--num_experts_per_tok 2

...
Avg MoE FF time/call: 1.555 ms
Avg MoE FF mem delta/call: 0.04 MB (max 0.11 MB)
...
Time: 35.11 sec
29 tokens/sec
Max memory allocated: 11.48 GB
```

We can see that the dense feed-forward layer processes a token in about 0.76 ms and uses roughly 0.19 MB of activations (peaking near 0.75 MB),

The sparse MoE layer keeps only about 0.04 MB of memory (peaking at 0.11). However, this comes at the cost of roughly twice the compute time. (There is an added routing overhead, and my implementation may also not be the most efficient one.)

Overall generation still peaks around 11.5 GB of GPU memory in both cases, since both versions load the same number of weight parameters and have the same KV cache size, which dominate here.

Either way, we can see the trade-off here where MoE reduces the FFN memory by about 4-5× while roughly doubling the feed-forward compute time.

Note that if we processed more tokens at one, e.g., with a batch size larger than 1 (here we don't have batches due to code simplicity), the savings would be more pronounced.



