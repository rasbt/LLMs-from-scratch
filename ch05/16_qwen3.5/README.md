# Qwen3.5 0.8B From Scratch

This folder contains a from-scratch style implementation of [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B).

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen3.5/01.webp" width="500px">

Qwen3.5 is based on the Qwen3-Next architecture, which I described in more detail in section [2. (Linear) Attention Hybrids](https://magazine.sebastianraschka.com/i/177848019/2-linear-attention-hybrids) of my [Beyond Standard LLMs](https://magazine.sebastianraschka.com/p/beyond-standard-llms) article


Note that Qwen3.5 alternates `linear_attention` and `full_attention` layers.  
The notebooks keep the full model flow readable while reusing the linear-attention building blocks from the [qwen3_5_transformers.py](qwen3_5_transformers.py), which contains the linear attention code from Hugging Face under an Apache version 2.0 open source license.

&nbsp;
## Files

- [qwen3.5.ipynb](qwen3.5.ipynb): Main Qwen3.5 0.8B notebook implementation.
- [qwen3.5-plus-kv-cache.ipynb](qwen3.5-plus-kv-cache.ipynb): Same model with KV-cache decoding for efficiency.
- [qwen3_5_transformers.py](qwen3_5_transformers.py): Some helper components from Hugging Face Transformers used for Qwen3.5 linear attention.

