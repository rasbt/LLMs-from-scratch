# Converting GPT to Llama



This folder contains code for converting the GPT implementation from chapter 4 and 5 to Meta AI's Llama architecture in the following recommended reading order:

- [converting-gpt-to-llama2.ipynb](converting-gpt-to-llama2.ipynb): contains code to convert GPT to Llama 2 7B step by step and loads pretrained weights from Meta AI
- [converting-llama2-to-llama3.ipynb](converting-llama2-to-llama3.ipynb): contains code to convert the Llama 2 model to Llama 3, Llama 3.1, and Llama 3.2
- [standalone-llama32.ipynb](standalone-llama32.ipynb): a standalone notebook implementing Llama 3.2

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/gpt-and-all-llamas.webp">


&nbsp;
### Using Llama 3.2 via the `llms-from-scratch` package

For an easy way to use the Llama 3.2 1B and 3B models, you can also use the `llms-from-scratch` PyPI package based on the source code in this repository at [pkg/llms_from_scratch](../../pkg/llms_from_scratch).

&nbsp;
#### 1) Installation

```bash
pip install llms_from_scratch blobfile
```

(Note that `blobfile` is needed to load the tokenizer.)

&nbsp;
#### 2) Model and text generation settings

Specify which model to use:

```python
MODEL_FILE = "llama3.2-1B-instruct.pth"
# MODEL_FILE = "llama3.2-1B-base.pth"
# MODEL_FILE = "llama3.2-3B-instruct.pth"
# MODEL_FILE = "llama3.2-3B-base.pth"
```

Basic text generation settings that can be defined by the user. Note that the recommended 8192-token context size requires approximately 3 GB of VRAM for the text generation example.

```python
# Text generation settings
if "instruct" in MODEL_FILE:
    PROMPT = "What do llamas eat?"
else:
    PROMPT = "Llamas eat"

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
```

&nbsp;
#### 3) Weight download and loading

This automatically downloads the weight file based on the model choice above:

```python
import os
import urllib.request

url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{MODEL_FILE}"

if not os.path.exists(MODEL_FILE):
    urllib.request.urlretrieve(url, MODEL_FILE)
    print(f"Downloaded to {MODEL_FILE}")
```

The model weights are then loaded as follows:

```python
import torch
from llms_from_scratch.llama3 import Llama3Model

if "1B" in MODEL_FILE:
    from llms_from_scratch.llama3 import LLAMA32_CONFIG_1B as LLAMA32_CONFIG
elif "3B" in MODEL_FILE:
    from llms_from_scratch.llama3 import LLAMA32_CONFIG_3B as LLAMA32_CONFIG
else:
    raise ValueError("Incorrect model file name")

model = Llama3Model(LLAMA32_CONFIG)
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True, map_location="cpu"))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device)
```

&nbsp;
#### 4) Initialize tokenizer

The following code downloads and initializes the tokenizer:

```python
from llms_from_scratch.llama3 import Llama3Tokenizer, ChatFormat, clean_text

TOKENIZER_FILE = "tokenizer.model"

url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{TOKENIZER_FILE}"

if not os.path.exists(TOKENIZER_FILE):
    urllib.request.urlretrieve(url, TOKENIZER_FILE)
    print(f"Downloaded to {TOKENIZER_FILE}")
    
tokenizer = Llama3Tokenizer("tokenizer.model")

if "instruct" in MODEL_FILE:
    tokenizer = ChatFormat(tokenizer)
```

&nbsp;
#### 5) Generating text

Lastly, we can generate text via the following code:

```python
import time

from llms_from_scratch.ch05 import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

torch.manual_seed(123)

start = time.time()

token_ids = generate(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=LLAMA32_CONFIG["context_length"],
    top_k=TOP_K,
    temperature=TEMPERATURE
)

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(token_ids[0])/total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = token_ids_to_text(token_ids, tokenizer)

if "instruct" in MODEL_FILE:
    output_text = clean_text(output_text)

print("\n\nOutput text:\n\n", output_text)
```

When using the Llama 3.2 1B Instruct model, the output should look similar to the one shown below:

```
Time: 3.17 sec
50 tokens/sec
Max memory allocated: 2.91 GB


Output text:

 Llamas are herbivores, which means they primarily eat plants. Their diet consists mainly of:

1. Grasses: Llamas love to graze on various types of grasses, including tall grasses and grassy meadows.
2. Hay: Llamas also eat hay, which is a dry, compressed form of grass or other plants.
3. Alfalfa: Alfalfa is a legume that is commonly used as a hay substitute in llama feed.
4. Other plants: Llamas will also eat other plants, such as clover, dandelions, and wild grasses.

It's worth noting that the specific diet of llamas can vary depending on factors such as the breed,
```

&nbsp;
#### Pro tip 1: speed up inference with FlashAttention

Instead of using `Llama3Model`, you can use `Llama3ModelFast` as a drop-in replacement. For more information, I encourage you to inspect the [pkg/llms_from_scratch/llama3.py](../../pkg/llms_from_scratch/llama3.py) code.

The `Llama3ModelFast` replaces my from-scratch scaled dot-product code in the `GroupedQueryAttention` module with PyTorch's `scaled_dot_product` function, which uses `FlashAttention` on Ampere GPUs or newer.

The following table shows a performance comparison on an A100:

|                 | Tokens/sec | Memory  |
| --------------- | ---------- | ------- |
| Llama3Model     | 42         | 2.91 GB |
| Llama3ModelFast | 54         | 2.91 GB |

&nbsp;
#### Pro tip 2: speed up inference with compilation


For up to a 4× speed-up, replace

```python
model.to(device)
```

with

```python
model = torch.compile(model)
model.to(device)
```

Note: There is a significant multi-minute upfront cost when compiling, and the speed-up takes effect after the first `generate` call. 

The following table shows a performance comparison on an A100 for consequent `generate` calls:

|                 | Tokens/sec | Memory  |
| --------------- | ---------- | ------- |
| Llama3Model     | 170        | 3.12 GB |
| Llama3ModelFast | 177        | 3.61 GB |

&nbsp;
#### Pro tip 3: speed up inference with compilation

You can significantly boost inference performance using the KV cache `Llama3Model` drop-in replacement when running the model on a CPU. (See my [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) article to learn more about KV caches.)

```python
from llms_from_scratch.kv_cache.llama3 import Llama3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple

model = Llama3Model(LLAMA32_CONFIG)
# ...
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=LLAMA32_CONFIG["context_length"],
)
```

Note that the peak memory usage is only listed for Nvidia CUDA devices, as it is easier to calculate. However, the memory usage on other devices is likely similar as it uses a similar precision format, and the KV cache storage results in even lower memory usage here for the generated 150-token text (however, different devices may implement matrix multiplication differently and may result in different peak memory requirements; and KV-cache memory may increase prohibitively for longer contexts lengths).

| Model       | Mode              | Hardware        | Tokens/sec | GPU Memory (VRAM) |
| ----------- | ----------------- | --------------- | ---------- | ----------------- |
| Llama3Model | Regular           | Mac Mini M4 CPU | 1          | -                 |
| Llama3Model | Regular compiled  | Mac Mini M4 CPU | 1          | -                 |
| Llama3Model | KV cache          | Mac Mini M4 CPU | 68         | -                 |
| Llama3Model | KV cache compiled | Mac Mini M4 CPU | 86         | -                 |
|             |                   |                 |            |                   |
| Llama3Model | Regular           | Mac Mini M4 GPU | 15         | -                 |
| Llama3Model | Regular compiled  | Mac Mini M4 GPU | Error      | -                 |
| Llama3Model | KV cache          | Mac Mini M4 GPU | 62         | -                 |
| Llama3Model | KV cache compiled | Mac Mini M4 GPU | Error      | -                 |
|             |                   |                 |            |                   |
| Llama3Model | Regular           | Nvidia A100 GPU | 42         | 2.91 GB           |
| Llama3Model | Regular compiled  | Nvidia A100 GPU | 170        | 3.12 GB           |
| Llama3Model | KV cache          | Nvidia A100 GPU | 58         | 2.87 GB           |
| Llama3Model | KV cache compiled | Nvidia A100 GPU | 161        | 3.61 GB           |

Note that all settings above have been tested to produce the same text outputs.
