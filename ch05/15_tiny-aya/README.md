# Tiny Aya 3.35B From Scratch

Tiny Aya is a new, "small" LLM by Cohere that is said to be the "most capable multi-lingual open-weight model" at the 3B parameter size class. (Tiny Aya outperforms Qwen3-4B, Gemma 3 4B, and Ministral 3 3B according to the [announcement post](https://cohere.com/blog/cohere-labs-tiny-aya)).

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/tiny-aya/01.webp">



This is a great model to run and experiment with locally. The only caveat is that while it's an open-weight model, its licensing terms are relatively restricted and only allow non-commercial use.

That aside, Arya is a 3.35B parameter model that comes in several flavors that are useful for
personal and (non-commercial) research use:

  - [tiny-aya-base](https://huggingface.co/CohereLabs/tiny-aya-base) (base model)
  - [tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) (best balance across languages and regions; notebook default)
  - [tiny-aya-fire](https://huggingface.co/CohereLabs/tiny-aya-fire) (optimized for South Asian languages)
  - [tiny-aya-water](https://huggingface.co/CohereLabs/tiny-aya-water) (optimized for European and Asia Pacific languages)
  - [tiny-aya-earth](https://huggingface.co/CohereLabs/tiny-aya-earth) (optimized for West Asian and African languages)



More specifically, here's a list of languages the models are optimized for:

| Region           | Languages                                                    | Optimized Model |
| ---------------- | ------------------------------------------------------------ | --------------- |
| **Asia Pacific** | Traditional Chinese, Cantonese, Vietnamese, Tagalog, Javanese, Khmer, Thai, Burmese, Malay, Korean, Lao, Indonesian, Simplified Chinese, Japanese | tiny-aya-water  |
| **Africa**       | Zulu, Amharic, Hausa, Igbo, Swahili, Xhosa, Wolof, Shona, Yoruba, Nigerian Pidgin, Malagasy | tiny-aya-earth  |
| **South Asia**   | Telugu, Marathi, Bengali, Tamil, Hindi, Punjabi, Gujarati, Urdu, Nepali | tiny-aya-fire   |
| **Europe**       | Catalan, Galician, Dutch, Danish, Finnish, Czech, Portuguese, French, Lithuanian, Slovak, Basque, English, Swedish, Polish, Spanish, Slovenian, Ukrainian, Greek, Bokmål, Romanian, Serbian, German, Italian, Russian, Irish, Hungarian, Bulgarian, Croatian, Estonian, Latvian, Welsh | tiny-aya-water  |
| **West Asia**    | Arabic, Maltese, Turkish, Hebrew, Persian                    | tiny-aya-earth  |


Architecture-wise, Tiny Aya is a classic decoder-style transformer with a few noteworthy modifications (besides the obvious ones like SwiGLU and Grouped Query Attention):

1. **Parallel transformer blocks.** A parallel transformer block computes attention and MLP from the same normalized input, then adds both to the residual in one step. I assume this is to reduce serial dependencies inside a layer to improve computational throughput.

2. **Sliding window attention.** Specifically, it uses a 3:1 local:global ratio similar to Arcee Trinity and Olmo 3. The window size is also 4096. Also, similar to Arcee, the sliding window layers use RoPE whereas the full attention layers use NoPE.

3. **LayerNorm.** Most architectures moved to RMSNorm as it's computationally a bit cheaper and performs well. Tiny Aya is keeping it more classic with a modified version of LayerNorm (the implementation here is like standard LayerNorm but without shift, i.e., bias, parameter).



&nbsp;
## Files

The [standalone-tiny-aya.ipynb](standalone-tiny-aya.ipynb) is a standalone Jupyter notebook that implements the Tiny Aya architecture and loads the pre-trained weights.


The alternative [standalone-tiny-aya-plus-kvcache.ipynb](standalone-tiny-aya-plus-kv-cache.ipynb) notebook adds a KV cache for better runtime performance (but adds more code complexity). To learn more about KV caching, see my [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) article.


<br>

To learn more about the architecture differences and read about comparisons with other architectures, see my [The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2: A Look At Modern LLM Architecture Design](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) article.





