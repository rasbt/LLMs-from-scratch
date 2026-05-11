# Troubleshooting Guide

This page collects common issues and setup tips encountered while working through the book.

&nbsp;
## Notebook Image Loading Issues

The chapter notebooks use Markdown image links hosted at `https://sebastianraschka.com/images/LLMs-from-scratch-images/...`. This keeps the repository download size manageable, but it also means the images depend on the image host and your network connection.

If images in the `.ipynb` notebooks do not render:

- Open one of the image URLs directly in your browser, for example [https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/02.webp](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/02.webp).
- If the URL does not load in the browser either, the issue is likely a temporary website, DNS, VPN, proxy, firewall, or local network problem rather than a notebook problem.
- I recommend double-checking the URL on a different device or network (e.g., try opening the image on your phone); if the image loads fine on your phone, it likely points to a VPN or firewall issue on your computer.
- If the images also don't load on your phone, please feel  free to open GitHub [Issue](https://github.com/rasbt/LLMs-from-scratch/issues) to help me debug this further.

&nbsp;
## Apple Silicon and MPS Support

Some notebooks and scripts use `cuda` when available and otherwise fall back to `cpu`, without selecting Apple's `mps` backend. This omission of `mps` support is intentional in many places because earlier PyTorch/MPS versions produced unstable or different results in several examples, especially during training and finetuning.

If you are using an Apple Silicon Mac and see diverging losses, sharp loss spikes, poor generated text, or results that do not match the book, rerun the example on `cpu` first. For faster training with book-matching behavior, I recommend using `cuda` on a local NVIDIA GPU or a cloud GPU.

Newer PyTorch versions may improve MPS behavior, and you can experiment with `mps` locally if you validate the results carefully. However, if you add `mps` support to a script yourself, keep in mind that CUDA-specific options such as `pin_memory=True`, `torch.compile`, and DDP/multi-GPU code may need separate guards.

For more context, see [#977](https://github.com/rasbt/LLMs-from-scratch/issues/977), [#625](https://github.com/rasbt/LLMs-from-scratch/discussions/625), [#644](https://github.com/rasbt/LLMs-from-scratch/discussions/644), [#442](https://github.com/rasbt/LLMs-from-scratch/discussions/442), and [#846](https://github.com/rasbt/LLMs-from-scratch/issues/846).

&nbsp;
## Other Issues

For other issues, please feel free to open a new GitHub [Issue](https://github.com/rasbt/LLMs-from-scratch/issues).
