# Alternative Approaches to Loading Pretrained Weights

This folder contains alternative weight loading strategies in case the weights become unavailable from OpenAI.

- [weight-loading-pytorch.ipynb](weight-loading-pytorch.ipynb): (Recommended) contains code to load the weights from PyTorch state dicts that I created by converting the original TensorFlow weights

- [weight-loading-hf-transformers.ipynb](weight-loading-hf-transformers.ipynb): contains code to load the weights from the Hugging Face Model Hub via the `transformers` library

- [weight-loading-hf-safetensors.ipynb](weight-loading-hf-safetensors.ipynb): contains code to load the weights from the Hugging Face Model Hub via the `safetensors` library directly (skipping the instantiation of a Hugging Face transformer model)