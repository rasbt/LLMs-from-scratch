# Qwen3 From Scratch with Chat Interface



This bonus folder contains code for running a ChatGPT-like user interface to interact with the pretrained Qwen3 model.



![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-chainlit.gif)



To implement this user interface, we use the open-source [Chainlit Python package](https://github.com/Chainlit/chainlit).

&nbsp;
## Step 1: Install dependencies

First, we install the `chainlit` package and dependencies from the [requirements-extra.txt](requirements-extra.txt) list via

```bash
pip install -r requirements-extra.txt
```

Or, if you are using `uv`:

```bash
uv pip install -r requirements-extra.txt
```



&nbsp;

## Step 2: Run `app` code

This folder contains 2 files:

1. [`qwen3-chat-interface.py`](qwen3-chat-interface.py): This file loads and uses the Qwen3 0.6B model in thinking mode. 
2. [`qwen3-chat-interface-multiturn.py`](qwen3-chat-interface-multiturn.py): The same as above, but configured to remember the message history.

(Open and inspect these files to learn more.)

Run one of the following commands from the terminal to start the UI server:

```bash
chainlit run qwen3-chat-interface.py
```

or, if you are using `uv`:

```bash
uv run chainlit run qwen3-chat-interface.py
```

Running one of the commands above should open a new browser tab where you can interact with the model. If the browser tab does not open automatically, inspect the terminal command and copy the local address into your browser address bar (usually, the address is `http://localhost:8000`).
