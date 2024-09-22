# Building a User Interface to Interact With the Pretrained LLM



This bonus folder contains code for running a ChatGPT-like user interface to interact with the pretrained LLMs from chapter 5, as shown below.



![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-orig.webp)



To implement this user interface, we use the open-source [Chainlit Python package](https://github.com/Chainlit/chainlit).

&nbsp;
## Step 1: Install dependencies

First, we install the `chainlit` package via

```bash
pip install chainlit
```

(Alternatively, execute `pip install -r requirements-extra.txt`.)

&nbsp;
## Step 2: Run `app` code

This folder contains 2 files:

1. [`app_orig.py`](app_orig.py): This file loads and uses the original GPT-2 weights from OpenAI. 
2. [`app_own.py`](app_own.py): This file loads and uses the GPT-2 weights we generated in chapter 5. This requires that you execute the [`../01_main-chapter-code/ch05.ipynb`](../01_main-chapter-code/ch05.ipynb) file first.

(Open and inspect these files to learn more.)

Run one of the following commands from the terminal to start the UI server:

```bash
chainlit run app_orig.py
```

or

```bash
chainlit run app_own.py
```

Running one of the commands above should open a new browser tab where you can interact with the model. If the browser tab does not open automatically, inspect the terminal command and copy the local address into your browser address bar (usually, the address is `http://localhost:8000`).