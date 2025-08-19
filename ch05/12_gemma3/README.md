# Gemma 3 270M From Scratch

This [standalone-gemma3.ipynb](standalone-gemma3.ipynb) Jupyter notebook in this folder contains a from-scratch implementation of Gemma 3 270M. It requires about 2 GB of RAM to run. 

The alternative [standalone-gemma3-plus-kvcache.ipynb](standalone-gemma3-plus-kvcache.ipynb) notebook adds a KV cache for better runtime performance (but adds more code complexity). To learn more about KV caching, see my [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) article.

| Model             | Mode              | Hardware        | Tokens/sec | GPU Memory (VRAM) |
| ----------------- | ----------------- | --------------- | ---------- | ----------------- |
| Gemma3Model 270M  | Regular           | Mac Mini M4 CPU | 8          | -                 |
| Gemma3Model 270M  | Regular compiled  | Mac Mini M4 CPU | 9          | -                 |
| Gemma3Model 270M  | KV cache          | Mac Mini M4 CPU | 130        | -                 |
| Gemma3Model 270M  | KV cache compiled | Mac Mini M4 CPU | 224        | -                 |
|                   |                   |                 |            |                   |
| Gemma3Model 270M  | Regular           | Mac Mini M4 GPU | 16         | -                 |
| Gemma3Model 270M  | Regular compiled  | Mac Mini M4 GPU | Error      | -                 |
| Gemma3Model 270M  | KV cache          | Mac Mini M4 GPU | 23         | -                 |
| Gemma3Model 270M  | KV cache compiled | Mac Mini M4 GPU | Error      | -                 |
|                   |                   |                 |            |                   |
| Gemma3Model 270M  | Regular           | Nvidia A100 GPU | 28         | 1.84 GB           |
| Gemma3Model 270M  | Regular compiled  | Nvidia A100 GPU | 128        | 2.12 GB           |
| Gemma3Model 270M  | KV cache          | Nvidia A100 GPU | 26         | 1.77 GB           |
| Gemma3Model 270M  | KV cache compiled | Nvidia A100 GPU | 99         | 2.12 GB           |


Below is a side-by-side comparison with Qwen3 0.6B as a reference model; if you are interested in the Qwen3 0.6B standalone notebook, you can find it [here](../11_qwen3).

<br>

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gemma3/gemma3-vs-qwen3.webp">

<br>

To learn more about the architecture differences and read about comparisons with other architectures, see my [The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2: A Look At Modern LLM Architecture Design](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) article.





