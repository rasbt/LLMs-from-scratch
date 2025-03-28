# `llms-from-scratch` PyPI Package

This optional PyPI package lets you conveniently import code from various chapters of the *Build a Large Language Model From Scratch* book.

&nbsp;
## Installation

&nbsp;
### From PyPI

Install the `llms-from-scratch` package from the official [Python Package Index](https://pypi.org/project/llms-from-scratch/) (PyPI):

```bash
pip install llms-from-scratch
```

> **Note:** If you're using [`uv`](https://github.com/astral-sh/uv), replace `pip` with `uv pip` or use `uv add`:

```bash
uv add llms-from-scratch
```



&nbsp;
### Editable Install from GitHub

If you'd like to modify the code and have those changes reflected during development:

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
pip install -e .
```

> **Note:** With `uv`, use:

```bash
uv add --editable . --dev
```



&nbsp;
## Using the Package

Once installed, you can import code from any chapter using:

```python
from llms_from_scratch.ch02 import GPTDatasetV1, create_dataloader_v1

from llms_from_scratch.ch03 import (
    SelfAttention_v1,
    SelfAttention_v2,
    CausalAttention,
    MultiHeadAttentionWrapper,
    MultiHeadAttention,
    PyTorchMultiHeadAttention # Bonus: Faster variant using PyTorch's scaled_dot_product_attention
)

from llms_from_scratch.ch04 import (
    LayerNorm,
    GELU,
    FeedForward,
    TransformerBlock,
    GPTModel,
    GPTModelFast # Bonus: Faster variant using PyTorch's scaled_dot_product_attention
    generate_text_simple
)

from llms_from_scratch.ch05 import (
    generate,
    train_model_simple,
    evaluate_model,
    generate_and_print_sample,
    assign,
    load_weights_into_gpt,
    text_to_token_ids,
    token_ids_to_text,
    calc_loss_batch,
    calc_loss_loader,
    plot_losses,
    download_and_load_gpt2
)

from llms_from_scratch.ch06 import (
    download_and_unzip_spam_data,
    create_balanced_dataset,
    random_split,
    SpamDataset,
    calc_accuracy_loader,
    evaluate_model,
    train_classifier_simple,
    plot_values,
    classify_review
)

from llms_from_scratch.ch07 import (
    download_and_load_file,
    format_input,
    InstructionDataset,
    custom_collate_fn,
    check_if_running,
    query_model,
    generate_model_scores
)

	
from llms_from_scratch.appendix_a import NeuralNetwork, ToyDataset

from llms_from_scratch.appendix_d import find_highest_gradient, train_model
```

