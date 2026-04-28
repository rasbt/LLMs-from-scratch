# DeepSeek Sparse Attention (DSA)

This bonus material implements the **DeepSeek Sparse Attention (DSA)** mechanism introduced in [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) and first published in the experimental [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) release.

&nbsp;
## Introduction

Standard causal self-attention attends to **all** previous tokens for each query, yielding O(L²) compute and O(L) KV-cache growth with sequence length L.

[Sliding Window Attention (SWA)](../06_swa) already showed that restricting attention to a *fixed local window* substantially reduces this cost. DSA takes a different approach: instead of a fixed window, it **learns which past tokens are most relevant** for each query and attends only to those.

### Architecture overview

DSA adds two components on top of standard attention:

**1. Lightning Indexer**

For each query token t and every candidate past token s, the indexer computes a scalar relevance score

$$I_{t,s} = \sum_{j=1}^{H_I} w_{t,j} \cdot \text{ReLU}(q_{t,j} \cdot k_s)$$

where:
- $H_I$ is the number of lightweight index heads,
- $q_{t,j}$ is the indexer query vector for token $t$ and head $j$,
- $k_s$ is a shared indexer key vector for past token $s$,
- $w_{t,j}$ is a learned per-head weight (normalised over heads at query time).

The ReLU zeroes out negative dot-product contributions, and the weighted sum aggregates across index heads into a single relevance score per past token.

**2. Token Selector**

After computing all indexer scores, only the **top-K** highest-scoring positions are kept. All other positions are masked to −∞ *before* the standard softmax, so the model effectively attends to only $k \ll L$ tokens.

This lowers the effective attention complexity from O(L²) to O(L·k).

The figure below illustrates the process (taken from Sebastian Raschka's blog post [From DeepSeek V3 to V3.2](https://magazine.sebastianraschka.com/p/technical-deepseek)):

> In DSA, the current token can attend a select number of tokens in the past (instead of all tokens like in regular causal attention).

&nbsp;
## Implementation

`gpt_with_kv_dsa.py` provides:

| Class | Description |
|---|---|
| `LightningIndexer` | Lightweight multi-head scorer for past-token relevance. |
| `MultiHeadAttentionWithDSA` | Standard MHA with DSA sparse masking + optional KV cache. |
| `GPTModel` | GPT-style model swapping in `MultiHeadAttentionWithDSA`. |

The implementation follows the style of the other bonus material in this repository and can be run as a standalone script.

&nbsp;
## Usage

```bash
uv run gpt_with_kv_dsa.py \
  --emb_dim 768 \
  --n_heads 12 \
  --n_layers 12 \
  --max_new_tokens 200 \
  --index_n_heads 4 \
  --index_head_dim 64 \
  --topk 64
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--index_n_heads` | 4 | Number of lightweight indexer heads (H_I). |
| `--index_head_dim` | 64 | Dimension of each indexer head. |
| `--topk` | 64 | Number of tokens each query attends to (k). Capped at sequence length for short sequences. |

&nbsp;
## Relation to DeepSeek V3.2

The full-scale DeepSeek-V3.2 model also uses Multi-Head Latent Attention (MLA, see [../05_mla](../05_mla)) alongside DSA, and the indexer queries are derived from the shared compressed latent representation rather than the raw input. This implementation uses standard multi-head projections for clarity and compatibility with the rest of the repository.

The key insight—using a cheap, learned dot-product scorer to limit the attention span to the most relevant tokens—is faithfully reproduced here.

&nbsp;
## References

- DeepSeek V3.2 technical report: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf
- DeepSeek V3.2-Exp model card & reference code: https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
- Sebastian Raschka's overview: https://magazine.sebastianraschka.com/p/technical-deepseek
