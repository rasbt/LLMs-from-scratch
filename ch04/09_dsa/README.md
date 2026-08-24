# DeepSeek Sparse Attention (DSA)

This bonus material implements the DeepSeek Sparse Attention (DSA) mechanism introduced in [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) and first published in the experimental [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) release.

The overview below follows the DSA discussion in [From DeepSeek V3 to V3.2: Architecture, Sparse Attention, and RL Updates](https://magazine.sebastianraschka.com/p/technical-deepseek).

&nbsp;
## Introduction

Standard causal self-attention attends to all previous tokens for each query, yielding O(L²) compute and O(L) KV-cache growth with sequence length L.

[Sliding Window Attention (SWA)](../06_swa) already showed that restricting attention to a fixed local window substantially reduces this cost. In SWA, each query token attends only to a local span of nearby previous tokens.

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/09.png" alt="Sliding window attention" width="800px" />

*Figure 1. Sliding-window attention restricts each query token to a fixed local context window.*

&nbsp;

DSA uses the same broad idea of attending to only a subset of previous tokens. However, it replaces the fixed window with a learned selection mechanism. For each query token, the model scores candidate past tokens and keeps only the most relevant ones.

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/10.png" alt="DeepSeek Sparse Attention selected-token pattern" width="800px" />

*Figure 2. DeepSeek Sparse Attention selects a learned subset of past tokens for each query token.*

&nbsp;

### Architecture overview

DSA adds two components on top of standard attention.

**1. Lightning Indexer**

For each query token $t$ and every candidate past token $s$, the indexer computes a scalar relevance score. This implementation makes the scale factors from the reference code explicit:

$$I_{t,s} = \sum_{j=1}^{H_I} \frac{w_{t,j}}{\sqrt{H_I}} \cdot \text{ReLU}\left(\frac{q_{t,j} \cdot k_s}{\sqrt{d_I}}\right)$$

where:
- $H_I$ is the number of lightweight index heads,
- $q_{t,j}$ is the indexer query vector for token $t$ and head $j$,
- $k_s$ is a shared indexer key vector for past token $s$,
- $w_{t,j}$ is a learned per-head gate scaled by $1 / \sqrt{H_I}$.

The ReLU zeroes out negative dot-product contributions, and the gated sum aggregates across index heads into a single relevance score per past token.

In the full DeepSeek model, the indexer works with the compressed token representations from Multi-Head Latent Attention (MLA). This folder keeps the GPT implementation simpler and computes the indexer queries and keys from the regular hidden states.

**2. Token Selector**

After computing all indexer scores, only the top-K highest-scoring positions are kept. All other positions are masked to −∞ *before* the standard softmax, so the model effectively attends to only $k \ll L$ tokens.

The ReLU in the indexer is not where the final sparsity comes from. Since the scores are summed over multiple index heads, most final scores can still be nonzero. The token selector creates the sparse pattern by keeping only the top-K positions.

In a fused production implementation, this can lower attention compute from O(L²) to O(L·k). The implementation here keeps the standard dense attention score matrix and applies the DSA-selected top-K mask before softmax. This makes the selection logic easy to inspect, but it does not provide the fused-kernel compute savings.

The figure below summarizes the flow. The lightning indexer scores candidate tokens, the selector keeps top-K positions, and the resulting mask restricts the usual attention softmax.

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/11.png" alt="DeepSeek Sparse Attention flowchart" width="700px" />

*Figure 3. DSA first scores candidate tokens, then keeps the top-K tokens for the final attention mask.*

&nbsp;
## Implementation

`gpt_with_kv_dsa.py` provides:

| Class | Description |
|---|---|
| `LightningIndexer` | Lightweight multi-head scorer for past-token relevance. |
| `MultiHeadAttentionWithDSA` | Standard MHA with DSA sparse masking + optional KV cache. |
| `GPTModel` | GPT-style model swapping in `MultiHeadAttentionWithDSA`. |

The implementation follows the style of the other bonus material in this repository and can be run as a standalone script. It is meant to make the DSA mechanism inspectable in a small GPT-style model. It does not implement DeepSeek's full MLA stack, fused sparse kernels, or deployment-specific optimizations.

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

The full-scale DeepSeek-V3.2 model uses Multi-Head Latent Attention (MLA, see [../05_mla](../05_mla)) alongside DSA, and the indexer queries are derived from the shared compressed latent representation rather than the raw input. DeepSeek-V3.2 uses the same architecture as DeepSeek-V3.2-Exp, where DSA was first introduced and tested.

The key selection idea is reproduced here. A cheap learned dot-product scorer limits each query to the most relevant tokens before the attention softmax.

The reported inference-cost comparison below is useful context for why DSA matters in long-context deployments. The savings depend on production kernels and serving infrastructure, so this figure should not be read as a benchmark for the teaching implementation in this folder.

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/19.png" alt="Inference cost comparison for DeepSeek Sparse Attention" width="800px" />

*Figure 4. DeepSeek's reported inference-cost savings from DSA in long-context serving, from the [DeepSeek V3.2 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf).*

&nbsp;
## References

- DeepSeek V3.2 technical report: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf
- DeepSeek V3.2-Exp model card & reference code: https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
- Sebastian Raschka's "From DeepSeek V3 to V3.2: Architecture, Sparse Attention, and RL Updates": https://magazine.sebastianraschka.com/p/technical-deepseek
