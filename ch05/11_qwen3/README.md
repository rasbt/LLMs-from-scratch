# Qwen3 From Scratch

This [standalone-qwen3.ipynb](standalone-qwen3.ipynb) Jupyter notebook in this folder contains a from-scratch implementation of Qwen3 0.6B, 1.7B, 4B, 8B, and 32 B.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen-overview.webp">


&nbsp;
### Using Qwen3 via the `llms-from-scratch` package

For an easy way to use the Qwen3 from-scratch implementation, you can also use the `llms-from-scratch` PyPI package based on the source code in this repository at [pkg/llms_from_scratch](../../pkg/llms_from_scratch).

&nbsp;
#### 1) Installation

```bash
pip install llms_from_scratch tokenizers
```

&nbsp;
#### 2) Model and text generation settings

Specify which model to use:

```python
USE_REASONING_MODEL = True   # The "thinking" model
USE_REASONING_MODEL = False  # The base model
```

Basic text generation settings that can be defined by the user. With 150 tokens, the model requires approximately 1.5 GB memory.

```python
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
```

&nbsp;
#### 3a) Weight download and loading of the 0.6B model

The following automatically downloads the weight file based on the model choice (reasoning or base) above. Note that this section focuses on the 0.6B model. Skip this section and continue with section 3b) if you want to work with any of the larger models (1.7B, 4B, 8B, or 32B).

```python
from llms_from_scratch.qwen3 import download_from_huggingface

repo_id = "rasbt/qwen3-from-scratch"

if USE_REASONING_MODEL:
    filename = "qwen3-0.6B.pth"
    local_dir = "Qwen3-0.6B"    
else:
    filename = "qwen3-0.6B-base.pth"   
    local_dir = "Qwen3-0.6B-Base"

download_from_huggingface(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir
)
```

The model weights are then loaded as follows:

```python
from pathlib import Path
import torch

from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

model_file = Path(local_dir) / filename

model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_file, weights_only=True, map_location="cpu"))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device);
```

&nbsp;
#### 3b) Weight download and loading of the larger Qwen models

If you are interested in working with any of the larger Qwen models, for instance, 1.7B, 4B, 8B, or 32B, please use the following code below instead of the code under 3a), which requires additional code dependencies:

```bash
pip install safetensors huggingface_hub
```

Then use the following code (make appropriate changes to `USE_MODEL` to select the desired model size)

```python
USE_MODEL = "1.7B"

if USE_MODEL == "1.7B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_1_7B as QWEN3_CONFIG
elif USE_MODEL == "4B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_4B as QWEN3_CONFIG
elif USE_MODEL == "8B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_8B as QWEN3_CONFIG
elif USE_MODEL == "14B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_14B as QWEN3_CONFIG
elif USE_MODEL == "32B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_32B as QWEN3_CONFIG
else:
    raise ValueError("Invalid USE_MODEL name.")
    
repo_id = f"Qwen/Qwen3-{USE_MODEL}"
local_dir = f"Qwen3-{USE_MODEL}"

if not USE_REASONING_MODEL:
  repo_id = f"{repo_id}-Base"
  local_dir = f"{local_dir}-Base"
```

Now, download and load the weights into the `model`:

```python
from llms_from_scratch.qwen3 import (
    Qwen3Model,
    download_from_huggingface_from_snapshots,
    load_weights_into_qwen
)

model = Qwen3Model(QWEN3_CONFIG)

weights_dict = download_from_huggingface_from_snapshots(
    repo_id=repo_id,
    local_dir=local_dir
)
load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
del weights_dict  # delete weight dictionary to free up disk space

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)

model.to(device);
```


&nbsp;

#### 4) Initialize tokenizer

The following code downloads and initializes the tokenizer:

```python
from llms_from_scratch.qwen3 import Qwen3Tokenizer

if USE_REASONING_MODEL:
    tok_filename = "tokenizer.json"    
else:
    tok_filename = "tokenizer-base.json"   

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tok_filename,
    repo_id=repo_id,
    add_generation_prompt=USE_REASONING_MODEL,
    add_thinking=USE_REASONING_MODEL
)
```



&nbsp;

#### 5) Generating text

Lastly, we can generate text via the following code:

```python
prompt = "Give me a short introduction to large language models."
input_token_ids = tokenizer.encode(prompt)
```





```python
from llms_from_scratch.ch05 import generate
import time

torch.manual_seed(123)

start = time.time()

output_token_ids = generate(
    model=model,
    idx=torch.tensor(input_token_ids, device=device).unsqueeze(0),
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
    top_k=1,
    temperature=0.
)

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(output_token_ids[0])/total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())

print("\n\nOutput text:\n\n", output_text + "...")
```

When using the Qwen3 0.6B reasoning model, the output should look similar to the one shown below (this was run on an A100):

```
Time: 6.35 sec
25 tokens/sec
Max memory allocated: 1.49 GB


Output text:

 <|im_start|>user
Give me a short introduction to large language models.<|im_end|>
Large language models (LLMs) are advanced artificial intelligence systems designed to generate human-like text. They are trained on vast amounts of text data, allowing them to understand and generate coherent, contextually relevant responses. LLMs are used in a variety of applications, including chatbots, virtual assistants, content generation, and more. They are powered by deep learning algorithms and can be fine-tuned for specific tasks, making them versatile tools for a wide range of industries.<|endoftext|>Human resources department of a company is planning to hire 100 new employees. The company has a budget of $100,000 for the recruitment process. The company has a minimum wage of $10 per hour. The company has a total of...
```

&nbsp;
#### Pro tip 1: speed up inference with compilation


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

|                     | Tokens/sec | Memory  |
| ------------------- | ---------- | ------- |
| Qwen3Model          | 25         | 1.49 GB |
| Qwen3Model compiled | 107        | 1.99 GB |

&nbsp;
#### Pro tip 2: speed up inference with compilation

You can significantly boost inference performance using the KV cache `Qwen3Model` drop-in replacement when running the model on a CPU. (See my [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) article to learn more about KV caches.)

```python
from llms_from_scratch.kv_cache.qwen3 import Qwen3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple

model = Qwen3Model(QWEN_CONFIG_06_B)
# ...
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```

Note that the peak memory usage is only listed for Nvidia CUDA devices, as it is easier to calculate. However, the memory usage on other devices is likely similar as it uses a similar precision format, and the KV cache storage results in even lower memory usage here for the generated 150-token text (however, different devices may implement matrix multiplication differently and may result in different peak memory requirements; and KV-cache memory may increase prohibitively for longer contexts lengths).

| Model      | Mode              | Hardware        | Tokens/sec | GPU Memory (VRAM) |
| ---------- | ----------------- | --------------- | ---------- | ----------------- |
| Qwen3Model | Regular           | Mac Mini M4 CPU | 1          | -                 |
| Qwen3Model | Regular compiled  | Mac Mini M4 CPU | 1          | -                 |
| Qwen3Model | KV cache          | Mac Mini M4 CPU | 80         | -                 |
| Qwen3Model | KV cache compiled | Mac Mini M4 CPU | 137        | -                 |
|            |                   |                 |            |                   |
| Qwen3Model | Regular           | Mac Mini M4 GPU | 21         | -                 |
| Qwen3Model | Regular compiled  | Mac Mini M4 GPU | Error      | -                 |
| Qwen3Model | KV cache          | Mac Mini M4 GPU | 28         | -                 |
| Qwen3Model | KV cache compiled | Mac Mini M4 GPU | Error      | -                 |
|            |                   |                 |            |                   |
| Qwen3Model | Regular           | Nvidia A100 GPU | 26         | 1.49 GB           |
| Qwen3Model | Regular compiled  | Nvidia A100 GPU | 107        | 1.99 GB           |
| Qwen3Model | KV cache          | Nvidia A100 GPU | 25         | 1.47 GB           |
| Qwen3Model | KV cache compiled | Nvidia A100 GPU | 90         | 1.48 GB           |

Note that all settings above have been tested to produce the same text outputs.

&nbsp;

#### Pro tip 3: batched inference

We can further increase the throughput via batched inference. While it's not an apples-to-apples comparison, as we are now running inference with a higher number of input sequences, this increases the tokens per second throughput while trading it off against increased memory usage.

This only requires a small code modification with respect to preparing the prompt. For example, consider this batched prompt below:

```python
from llms_from_scratch.ch04 import generate_text_simple
from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B
# ...

prompts = [
    "Give me a short introduction to neural networks.",
    "Give me a short introduction to machine learning.",
    "Give me a short introduction to deep learning models.",
    "Give me a short introduction to natural language processing.",
    "Give me a short introduction to generative AI systems.",
    "Give me a short introduction to transformer architectures.",
    "Give me a short introduction to supervised learning methods.",
    "Give me a short introduction to unsupervised learning.",
]

tokenized_prompts = [tokenizer.encode(p) for p in prompts]
max_len = max(len(t) for t in tokenized_prompts)
padded_token_ids = [
    t + [tokenizer.pad_token_id] * (max_len - len(t)) for t in tokenized_prompts
]
input_tensor = torch.tensor(padded_token_ids).to(device)

output_token_ids = generate_text_simple(
    model=model,
    idx=input_tensor,
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```

The code for the KV cache version is similar, except that it requires using these drop-in replacements:

```python
from llms_from_scratch.kv_cache_batched.generate import generate_text_simple
from llms_from_scratch.kv_cache_batched.qwen3 import Qwen3Model
```


The experiments below are run with a batch size of 8.

| Model      | Mode              | Hardware        | Batch size | Tokens/sec | GPU Memory (VRAM) |
| ---------- | ----------------- | --------------- | ---------- | ---------- | ----------------- |
| Qwen3Model | Regular           | Mac Mini M4 CPU | 8          | 2          | -                 |
| Qwen3Model | Regular compiled  | Mac Mini M4 CPU | 8          | -          | -                 |
| Qwen3Model | KV cache          | Mac Mini M4 CPU | 8          | 92         | -                 |
| Qwen3Model | KV cache compiled | Mac Mini M4 CPU | 8          | 128        | -                 |
|            |                   |                 |            |            |                   |
| Qwen3Model | Regular           | Mac Mini M4 GPU | 8          | 36         | -                 |
| Qwen3Model | Regular compiled  | Mac Mini M4 GPU | 8          | -          | -                 |
| Qwen3Model | KV cache          | Mac Mini M4 GPU | 8          | 61         | -                 |
| Qwen3Model | KV cache compiled | Mac Mini M4 GPU | 8          | -          | -                 |
|            |                   |                 |            |            |                   |
| Qwen3Model | Regular           | Nvidia A100 GPU | 8          | 184        | 2.19 GB           |
| Qwen3Model | Regular compiled  | Nvidia A100 GPU | 8          | 351        | 2.19 GB           |
| Qwen3Model | KV cache          | Nvidia A100 GPU | 8          | 140        | 3.13 GB           |
| Qwen3Model | KV cache compiled | Nvidia A100 GPU | 8          | 280        | 1.75 GB           |


