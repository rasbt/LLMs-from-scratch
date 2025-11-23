# Olmo 3 7B and 32B From Scratch

This [standalone-olmo3.ipynb](standalone-olmo3.ipynb) Jupyter notebook in this folder contains a from-scratch implementation of Olmo 3 7B and 32B and requires about 13 GB of RAM to run. 

The alternative [standalone-olmo3-plus-kvcache.ipynb](standalone-olmo3-plus-kv-cache.ipynb) notebook adds a KV cache for better runtime performance (but adds more code complexity). To learn more about KV caching, see my [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) article.

Below is a side-by-side comparison with Qwen3 as a reference model; if you are interested in the Qwen3 0.6B standalone notebook, you can find it [here](../11_qwen3).

<br>

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-7B.webp?1">

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-32B.webp?1">

Olmo 3 also comes in different flavors, as shown below (the architecture is the same, only the training pipeline differs):

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-pipeline.webp?1">


&nbsp;
## How does Olmo 3 compare to Qwen3

Focusing on the architecture, not the training details, this section provides a brief comparison to Qwen3.


The 7B model:

1. As we can see in the figures above, the Olmo 3 architecture is relatively similar to Qwen3. However, it's worth noting that this is essentially likely inspired by the Olmo 2 predecessor, not Qwen3. 

2) Similar to Olmo 2, Olmo 3 still uses a post-norm flavor instead of pre-norm, as they found in the Olmo 2 paper that it stabilizes the training.

3) Interestingly, the 7B model still uses multi-head attention similar to Olmo 2. 
However, to make things more efficient and reduce the KV cache size, they now use sliding-window attention (e.g., similar to Gemma 3).

Next, the 32B model:

4) Overall, it's the same architecture but just scaled up. Also, the proportions (e.g., going from the input to the intermediate size in the feed-forward layer, and so on) roughly match the ones in Qwen3. 

5) My guess is the architecture was initially somewhat smaller than Qwen3 due to the smaller vocabulary, and they then scaled up the intermediate size expansion from 5x in Qwen3 to 5.4 in Olmo 3 to have a 32B model for a direct comparison. 

6) Also, note that the 32B model (finally!) uses grouped query attention.





<br>

To learn more about the architecture differences and read about comparisons with other architectures, see my [The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2: A Look At Modern LLM Architecture Design](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) article.





