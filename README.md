# Build a Large Language Model (From Scratch)

This repository contains the code for developing, pretraining, and finetuning a GPT-like LLM and is the official code repository for the book [Build a Large Language Model (From Scratch)](https://amzn.to/4fqvn0D).

<br>
<br>

<a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="250px"></a>

<br>

In [*Build a Large Language Model (From Scratch)*](http://mng.bz/orYv), you'll learn and understand how large language models (LLMs) work from the inside out by coding them from the ground up, step by step. In this book, I'll guide you through creating your own LLM, explaining each stage with clear text, diagrams, and examples.

The method described in this book for training and developing your own small-but-functional model for educational purposes mirrors the approach used in creating large-scale foundational models such as those behind ChatGPT. In addition, this book includes code for loading the weights of larger pretrained models for finetuning.

- Link to the official [source code repository](https://github.com/rasbt/LLMs-from-scratch)
- [Link to the book at Manning (the publisher's website)](http://mng.bz/orYv)
- [Link to the book page on Amazon.com](https://www.amazon.com/gp/product/1633437167)
- ISBN 9781633437166

<a href="http://mng.bz/orYv#reviews"><img src="https://sebastianraschka.com//images/LLMs-from-scratch-images/other/reviews.png" width="220px"></a>


<br>
<br>

To download a copy of this repository, click on the [Download ZIP](https://github.com/rasbt/LLMs-from-scratch/archive/refs/heads/main.zip) button or execute the following command in your terminal:

```bash
git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git
```

<br>

(If you downloaded the code bundle from the Manning website, please consider visiting the official code repository on GitHub at [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) for the latest updates.)

<br>
<br>


# Table of Contents

Please note that this `README.md` file is a Markdown (`.md`) file. If you have downloaded this code bundle from the Manning website and are viewing it on your local computer, I recommend using a Markdown editor or previewer for proper viewing. If you haven't installed a Markdown editor yet, [MarkText](https://www.marktext.cc) is a good free option.

You can alternatively view this and other files on GitHub at [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) in your browser, which renders Markdown automatically.

<br>
<br>
<!--  -->

> [!TIP]
> If you're seeking guidance on installing Python and Python packages and setting up your code environment, I suggest reading the [README.md](setup/README.md) file located in the [setup](setup) directory.

<br>
<br>

[![Code tests (Linux)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux.yml)
[![Code tests (Windows)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows.yml)
[![Code tests (macOS)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos.yml)



<br>

| Chapter Title                                              | Main Code (for Quick Access)                                                                                                    | All Code + Supplementary      |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| [Setup recommendations](setup)                             | -                                                                                                                               | -                             |
| Ch 1: Understanding Large Language Models                  | No code                                                                                                                         | -                             |
| Ch 2: Working with Text Data                               | - [ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb)<br/>- [dataloader.ipynb](ch02/01_main-chapter-code/dataloader.ipynb) (summary)<br/>- [exercise-solutions.ipynb](ch02/01_main-chapter-code/exercise-solutions.ipynb)               | [./ch02](./ch02)            |
| Ch 3: Coding Attention Mechanisms                          | - [ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb)<br/>- [multihead-attention.ipynb](ch03/01_main-chapter-code/multihead-attention.ipynb) (summary) <br/>- [exercise-solutions.ipynb](ch03/01_main-chapter-code/exercise-solutions.ipynb)| [./ch03](./ch03)             |
| Ch 4: Implementing a GPT Model from Scratch                | - [ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb)<br/>- [gpt.py](ch04/01_main-chapter-code/gpt.py) (summary)<br/>- [exercise-solutions.ipynb](ch04/01_main-chapter-code/exercise-solutions.ipynb) | [./ch04](./ch04)           |
| Ch 5: Pretraining on Unlabeled Data                        | - [ch05.ipynb](ch05/01_main-chapter-code/ch05.ipynb)<br/>- [gpt_train.py](ch05/01_main-chapter-code/gpt_train.py) (summary) <br/>- [gpt_generate.py](ch05/01_main-chapter-code/gpt_generate.py) (summary) <br/>- [exercise-solutions.ipynb](ch05/01_main-chapter-code/exercise-solutions.ipynb) | [./ch05](./ch05)              |
| Ch 6: Finetuning for Text Classification                   | - [ch06.ipynb](ch06/01_main-chapter-code/ch06.ipynb)  <br/>- [gpt_class_finetune.py](ch06/01_main-chapter-code/gpt_class_finetune.py)  <br/>- [exercise-solutions.ipynb](ch06/01_main-chapter-code/exercise-solutions.ipynb) | [./ch06](./ch06)              |
| Ch 7: Finetuning to Follow Instructions                    | - [ch07.ipynb](ch07/01_main-chapter-code/ch07.ipynb)<br/>- [gpt_instruction_finetuning.py](ch07/01_main-chapter-code/gpt_instruction_finetuning.py) (summary)<br/>- [ollama_evaluate.py](ch07/01_main-chapter-code/ollama_evaluate.py) (summary)<br/>- [exercise-solutions.ipynb](ch07/01_main-chapter-code/exercise-solutions.ipynb) | [./ch07](./ch07)  |
| Appendix A: Introduction to PyTorch                        | - [code-part1.ipynb](appendix-A/01_main-chapter-code/code-part1.ipynb)<br/>- [code-part2.ipynb](appendix-A/01_main-chapter-code/code-part2.ipynb)<br/>- [DDP-script.py](appendix-A/01_main-chapter-code/DDP-script.py)<br/>- [exercise-solutions.ipynb](appendix-A/01_main-chapter-code/exercise-solutions.ipynb) | [./appendix-A](./appendix-A) |
| Appendix B: References and Further Reading                 | No code                                                                                                                         | -                             |
| Appendix C: Exercise Solutions                             | No code                                                                                                                         | -                             |
| Appendix D: Adding Bells and Whistles to the Training Loop | - [appendix-D.ipynb](appendix-D/01_main-chapter-code/appendix-D.ipynb)                                                          | [./appendix-D](./appendix-D)  |
| Appendix E: Parameter-efficient Finetuning with LoRA       | - [appendix-E.ipynb](appendix-E/01_main-chapter-code/appendix-E.ipynb)                                                          | [./appendix-E](./appendix-E) |

<br>
&nbsp;

The mental model below summarizes the contents covered in this book.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg" width="650px">

<br>
&nbsp;

## Hardware Requirements

The code in the main chapters of this book is designed to run on conventional laptops within a reasonable timeframe and does not require specialized hardware. This approach ensures that a wide audience can engage with the material. Additionally, the code automatically utilizes GPUs if they are available. (Please see the [setup](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/README.md) doc for additional recommendations.)

&nbsp;
## Bonus Material

Several folders contain optional materials as a bonus for interested readers:

- **Setup**
  - [Python Setup Tips](setup/01_optional-python-setup-preferences)
  - [Installing Python Packages and Libraries Used In This Book](setup/02_installing-python-libraries)
  - [Docker Environment Setup Guide](setup/03_optional-docker-environment)
- **Chapter 2: Working with text data**
  - [Byte Pair Encoding (BPE) Tokenizer From Scratch](ch02/05_bpe-from-scratch/bpe-from-scratch.ipynb)
  - [Comparing Various Byte Pair Encoding (BPE) Implementations](ch02/02_bonus_bytepair-encoder)
  - [Understanding the Difference Between Embedding Layers and Linear Layers](ch02/03_bonus_embedding-vs-matmul)
  - [Dataloader Intuition with Simple Numbers](ch02/04_bonus_dataloader-intuition)
- **Chapter 3: Coding attention mechanisms**
  - [Comparing Efficient Multi-Head Attention Implementations](ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)
  - [Understanding PyTorch Buffers](ch03/03_understanding-buffers/understanding-buffers.ipynb)
- **Chapter 4: Implementing a GPT model from scratch**
  - [FLOPS Analysis](ch04/02_performance-analysis/flops-analysis.ipynb)
- **Chapter 5: Pretraining on unlabeled data:**
  - [Alternative Weight Loading from Hugging Face Model Hub using Transformers](ch05/02_alternative_weight_loading/weight-loading-hf-transformers.ipynb)
  - [Pretraining GPT on the Project Gutenberg Dataset](ch05/03_bonus_pretraining_on_gutenberg)
  - [Adding Bells and Whistles to the Training Loop](ch05/04_learning_rate_schedulers)
  - [Optimizing Hyperparameters for Pretraining](ch05/05_bonus_hparam_tuning)
  - [Building a User Interface to Interact With the Pretrained LLM](ch05/06_user_interface)
  - [Converting GPT to Llama](ch05/07_gpt_to_llama)
  - [Llama 3.2 From Scratch](ch05/07_gpt_to_llama/standalone-llama32.ipynb)
  - [Memory-efficient Model Weight Loading](ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict.ipynb)
  - [Extending the Tiktoken BPE Tokenizer with New Tokens](ch05/09_extending-tokenizers/extend-tiktoken.ipynb)
- **Chapter 6: Finetuning for classification**
  - [Additional experiments finetuning different layers and using larger models](ch06/02_bonus_additional-experiments)
  - [Finetuning different models on 50k IMDB movie review dataset](ch06/03_bonus_imdb-classification)
  - [Building a User Interface to Interact With the GPT-based Spam Classifier](ch06/04_user_interface)
- **Chapter 7: Finetuning to follow instructions**
  - [Dataset Utilities for Finding Near Duplicates and Creating Passive Voice Entries](ch07/02_dataset-utilities)
  - [Evaluating Instruction Responses Using the OpenAI API and Ollama](ch07/03_model-evaluation)
  - [Generating a Dataset for Instruction Finetuning](ch07/05_dataset-generation/llama3-ollama.ipynb)
  - [Improving a Dataset for Instruction Finetuning](ch07/05_dataset-generation/reflection-gpt4.ipynb)
  - [Generating a Preference Dataset with Llama 3.1 70B and Ollama](ch07/04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb)
  - [Direct Preference Optimization (DPO) for LLM Alignment](ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb)
  - [Building a User Interface to Interact With the Instruction Finetuned GPT Model](ch07/06_user_interface)

<br>
&nbsp;

## Questions, Feedback, and Contributing to This Repository


I welcome all sorts of feedback, best shared via the [Manning Forum](https://livebook.manning.com/forum?product=raschka&page=1) or [GitHub Discussions](https://github.com/rasbt/LLMs-from-scratch/discussions). Likewise, if you have any questions or just want to bounce ideas off others, please don't hesitate to post these in the forum as well.

Please note that since this repository contains the code corresponding to a print book, I currently cannot accept contributions that would extend the contents of the main chapter code, as it would introduce deviations from the physical book. Keeping it consistent helps ensure a smooth experience for everyone.


&nbsp;
## Citation

If you find this book or code useful for your research, please consider citing it.

Chicago-style citation:

> Raschka, Sebastian. *Build A Large Language Model (From Scratch)*. Manning, 2024. ISBN: 978-1633437166.

BibTeX entry:

```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Large Language Model (From Scratch)},
  publisher    = {Manning},
  year         = {2024},
  isbn         = {978-1633437166},
  url          = {https://www.manning.com/books/build-a-large-language-model-from-scratch},
  github       = {https://github.com/rasbt/LLMs-from-scratch}
}
```
