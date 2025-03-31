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
##### 1) Installation

```bash
pip install llms_from_scratch blobfile
```
&nbsp;
##### 2) Model and text generation settings

Specify which model to use:

```python
MODEL_FILE = "llama3.2-1B-instruct.pth"
# MODEL_FILE = "llama3.2-1B-base.pth"
# MODEL_FILE = "llama3.2-3B-instruct.pth"
# MODEL_FILE = "llama3.2-3B-base.pth"
```

Basic text generation settings that can be defined by the user. Note that the recommended 8192-token context size requires approximately 3 GB of VRAM for the text generation example.

```python
MODEL_CONTEXT_LENGTH = 8192  # Supports up to 131_072

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
##### 3) Weight download and loading

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

LLAMA32_CONFIG["context_length"] = MODEL_CONTEXT_LENGTH

model = Llama3Model(LLAMA32_CONFIG)
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device)
```

&nbsp;
##### 4) Initialize tokenizer

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
##### 5) Generating text

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

print(f"Time: {time.time() - start:.2f} sec")

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
Time: 4.12 sec
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
**Pro tip**

For up to a 4× speed-up, replace

```python
model.to(device)
```

with

```python
model = torch.compile(model)
model.to(device)
```

Note: the speed-up takes effect after the first `generate` call.

