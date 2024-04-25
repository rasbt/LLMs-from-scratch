# Build a Large Language Model (From Scratch)

This repository contains the code for coding, pretraining, and finetuning a GPT-like LLM and is the official code repository for the book [Build a Large Language Model (From Scratch)](http://mng.bz/orYv).

(If you downloaded the code bundle from the Manning website, please consider visiting the official code repository on GitHub at [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch).)

<br>
<br>

<a href="http://mng.bz/orYv"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg" width="250px"></a>

In [*Build a Large Language Model (From Scratch)*](http://mng.bz/orYv), you'll discover how LLMs work from the inside out. In this book, I'll guide you step by step through creating your own LLM, explaining each stage with clear text, diagrams, and examples.

The method described in this book for training and developing your own small-but-functional model for educational purposes mirrors the approach used in creating large-scale foundational models such as those behind ChatGPT.

- Link to the official [source code repository](https://github.com/rasbt/LLMs-from-scratch)
- [Link to the early access version](http://mng.bz/orYv) at Manning
- ISBN 9781633437166
- Publication in Early 2025 (estimated)

<br>
<br>


# Table of Contents

Please note that this `README.md` file is a Markdown (`.md`) file. If you have downloaded this code bundle from the Manning website and are viewing it on your local computer, I recommend using a Markdown editor or previewer for proper viewing. If you haven't installed a Markdown editor yet, [MarkText](https://www.marktext.cc) is a good free option.

Alternatively, you can view this and other files on GitHub at [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch).

<br>
<br>
<!--  -->

> [!TIP]
> If you're seeking guidance on installing Python and Python packages and setting up your code environment, I suggest reading the [README.md](setup/README.md) file located in the [setup](setup) directory.

<br>

[![Python PEP8 linting](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/pep8-linter.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/pep8-linter.yml)
[![Python tests](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests.yml)
[![Check hyperlinks](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/check-links.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/check-links.yml)

<br>

| Chapter Title                                              | Main Code (for quick access)                                                                                                    | All Code + Supplementary      |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| Ch 1: Understanding Large Language Models                  | No code                                                                                                                         | -                             |
| Ch 2: Working with Text Data                               | - [ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb)<br/>- [dataloader.ipynb](ch02/01_main-chapter-code/dataloader.ipynb) (summary)<br/>- [exercise-solutions.ipynb](ch02/01_main-chapter-code/exercise-solutions.ipynb)               | [./ch02](./ch02)              |
| Ch 3: Coding Attention Mechanisms                          | - [ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb)<br/>- [multihead-attention.ipynb](ch03/01_main-chapter-code/multihead-attention.ipynb) (summary) <br/>- [exercise-solutions.ipynb](ch03/01_main-chapter-code/exercise-solutions.ipynb)| [./ch03](./ch03)              |
| Ch 4: Implementing a GPT Model from Scratch                | - [ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb)<br/>- [gpt.py](ch04/01_main-chapter-code/gpt.py) (summary)<br/>- [exercise-solutions.ipynb](ch04/01_main-chapter-code/exercise-solutions.ipynb) | [./ch04](./ch04)           |
| Ch 5: Pretraining on Unlabeled Data                        | - [ch05.ipynb](ch05/01_main-chapter-code/ch05.ipynb)<br/>- [gpt_train.py](ch05/01_main-chapter-code/gpt_train.py) (summary) <br/>- [gpt_generate.py](ch05/01_main-chapter-code/gpt_generate.py) (summary) <br/>- [exercise-solutions.ipynb](ch05/01_main-chapter-code/exercise-solutions.ipynb) | [./ch05](./ch05)              |
| Ch 6: Finetuning for Text Classification                   | Q2 2024                                                                                                                         | ...                           |
| Ch 7: Finetuning with Human Feedback                       | Q2 2024                                                                                                                         | ...                           |
| Ch 8: Using Large Language Models in Practice              | Q2/3 2024                                                                                                                       | ...                           |
| Appendix A: Introduction to PyTorch                        | - [code-part1.ipynb](appendix-A/01_main-chapter-code/code-part1.ipynb)<br/>- [code-part2.ipynb](appendix-A/01_main-chapter-code/code-part2.ipynb)<br/>- [DDP-script.py](appendix-A/01_main-chapter-code/DDP-script.py)<br/>- [exercise-solutions.ipynb](appendix-A/01_main-chapter-code/exercise-solutions.ipynb) | [./appendix-A](./appendix-A) |
| Appendix B: References and Further Reading                 | No code                                                                                                                         | -                             |
| Appendix C: Exercise Solutions                             | No code                                                                                                                         | -                             |
| Appendix D: Adding Bells and Whistles to the Training Loop | - [appendix-D.ipynb](appendix-D/01_main-chapter-code/appendix-D.ipynb)                                                          | [./appendix-D](./appendix-D)  |



<br>
<br>

Shown below is a mental model summarizing the contents covered in this book.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg" width="650px">

<br>
<br>
&nbsp

## Bonus Material

Several folders contain optional materials as a bonus for interested readers:

- **Setup**
  - [Python Setup Tips](setup/01_optional-python-setup-preferences)
  - [Installing Libraries Used In This Book](setup/02_installing-python-libraries)
  - [Docker Environment Setup Guide](setup/03_optional-docker-environment)
- **Chapter 2:**
  - [Comparing Various Byte Pair Encoding (BPE) Implementations](ch02/02_bonus_bytepair-encoder)
  - [Understanding the Difference Between Embedding Layers and Linear Layers](ch02/03_bonus_embedding-vs-matmul)
- **Chapter 3:**
  - [Comparing Efficient Multi-Head Attention Implementations](ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)
- **Chapter 5:**
  - [Alternative Weight Loading from Hugging Face Model Hub using Transformers](ch05/02_alternative_weight_loading/weight-loading-hf-transformers.ipynb)
  - [Pretraining GPT on the Project Gutenberg Dataset](ch05/03_bonus_pretraining_on_gutenberg)
  - [Adding Bells and Whistles to the Training Loop](ch05/04_learning_rate_schedulers)
  - [Optimizing Hyperparameters for Pretraining](ch05/05_bonus_hparam_tuning)
- **Chapter 6:**
  - [Additional experiments finetuning different layers and using larger models](ch06/02_bonus_additional-experiments) 	
  - [Finetuning different models on 50k IMDB movie review dataset](ch06/03_bonus_imdb-classification) 	

<br>
<br>
&nbsp



### Citation

If you find this book  or code useful for your research, please consider citing it:

```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Large Language Model (From Scratch)},
  publisher    = {Manning},
  year         = {2023},
  isbn         = {978-1633437166},
  url          = {https://www.manning.com/books/build-a-large-language-model-from-scratch},
  note         = {Work in progress},
  github       = {https://github.com/rasbt/LLMs-from-scratch}
}
```
