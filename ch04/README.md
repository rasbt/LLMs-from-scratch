# Chapter 4: Implementing a GPT Model from Scratch to Generate Text

&nbsp;
## Main Chapter Code

- [01_main-chapter-code](01_main-chapter-code) contains the main chapter code.

&nbsp;
## Bonus Materials

- [02_performance-analysis](02_performance-analysis) contains optional code analyzing the performance of the GPT model(s) implemented in the main chapter
- [03_kv-cache](03_kv-cache) implements a KV cache to speed up the text generation during inference
- [07_moe](07_moe) explanation and implementation of Mixture-of-Experts (MoE)
- [ch05/07_gpt_to_llama](../ch05/07_gpt_to_llama) contains a step-by-step guide for converting a GPT architecture implementation to Llama 3.2 and loads pretrained weights from Meta AI (it might be interesting to look at alternative architectures after completing chapter 4, but you can also save that for after reading chapter 5)


&nbsp;
## Attention Alternatives

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/attention-alternatives/attention-alternatives.webp">

&nbsp;

- [04_gqa](04_gqa) contains an introduction to Grouped-Query Attention (GQA), which is used by most modern LLMs (Llama 4, gpt-oss, Qwen3, Gemma 3, and many more) as alternative to regular Multi-Head Attention (MHA)
- [05_mla](05_mla) contains an introduction to Multi-Head Latent Attention (MLA), which is used by DeepSeek V3, as alternative to regular Multi-Head Attention (MHA)
- [06_swa](06_swa) contains an introduction to Sliding Window Attention (SWA), which is used by Gemma 3 and others
- [08_deltanet](08_deltanet) explanation of Gated DeltaNet as a popular linear attention variant (used in Qwen3-Next and Kimi Linear)


&nbsp;
## More

In the video below, I provide a code-along session that covers some of the chapter contents as supplementary material.

<br>
<br>

[![Link to the video](https://img.youtube.com/vi/YSAkgEarBGE/0.jpg)](https://www.youtube.com/watch?v=YSAkgEarBGE)
