# Additional Experiments Classifying the Sentiment of 50k IMDB Movie Reviews

## Overview

This folder contains additional experiments to compare the (decoder-style) GPT-2 (2018) model from chapter 6 to encoder-style LLMs like [BERT (2018)](https://arxiv.org/abs/1810.04805), [RoBERTa (2019)](https://arxiv.org/abs/1907.11692), and [ModernBERT (2024)](https://arxiv.org/abs/2412.13663). Instead of using the small SPAM dataset from Chapter 6, we are using the 50k movie review dataset from IMDb ([dataset source](https://ai.stanford.edu/~amaas/data/sentiment/)) with a binary classification objective, predicting whether a reviewer liked the movie or not. This is a balanced dataset, so a random prediction should yield 50% accuracy.





|       | Model                        | Test accuracy |
| ----- | ---------------------------- | ------------- |
| **1** | 124M GPT-2 Baseline          | 91.88%        |
| **2** | 340M BERT                    | 90.89%        |
| **3** | 66M DistilBERT               | 91.40%        |
| **4** | 355M RoBERTa                 | 92.95%        |
| **5** | 304M DeBERTa-v3              | 94.69%        |
| **6** | 149M ModernBERT Base         | 93.79%        |
| **7** | 395M ModernBERT Large        | 95.07%        |
| **8** | Logistic Regression Baseline | 88.85%        |






&nbsp;
## Step 1: Install Dependencies

Install the extra dependencies via

```bash
pip install -r requirements-extra.txt
```

&nbsp;
## Step 2: Download Dataset

The codes are using the 50k movie reviews from IMDb ([dataset source](https://ai.stanford.edu/~amaas/data/sentiment/)) to predict whether a movie review is positive or negative.

Run the following code to create the `train.csv`, `validation.csv`, and `test.csv` datasets:

```bash
python download_prepare_dataset.py
```


&nbsp;
## Step 3: Run Models

&nbsp;
### 1) 124M GPT-2 Baseline

The 124M GPT-2 model used in chapter 6, starting with pretrained weights, and finetuning all weights:

```bash
python train_gpt.py --trainable_layers "all" --num_epochs 1
```

```
Ep 1 (Step 000000): Train loss 3.706, Val loss 3.853
Ep 1 (Step 000050): Train loss 0.682, Val loss 0.706
...
Ep 1 (Step 004300): Train loss 0.199, Val loss 0.285
Ep 1 (Step 004350): Train loss 0.188, Val loss 0.208
Training accuracy: 95.62% | Validation accuracy: 95.00%
Training completed in 9.48 minutes.

Evaluating on the full datasets ...

Training accuracy: 95.64%
Validation accuracy: 92.32%
Test accuracy: 91.88%
```


<br>

---

<br>

&nbsp;
### 2) 340M BERT


A 340M parameter encoder-style [BERT](https://arxiv.org/abs/1810.04805) model:

```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "bert"
```

```
Ep 1 (Step 000000): Train loss 0.848, Val loss 0.775
Ep 1 (Step 000050): Train loss 0.655, Val loss 0.682
...
Ep 1 (Step 004300): Train loss 0.146, Val loss 0.318
Ep 1 (Step 004350): Train loss 0.204, Val loss 0.217
Training accuracy: 92.50% | Validation accuracy: 88.75%
Training completed in 7.65 minutes.

Evaluating on the full datasets ...

Training accuracy: 94.35%
Validation accuracy: 90.74%
Test accuracy: 90.89%
```

<br>

---

<br>

&nbsp;
### 3) 66M DistilBERT

A 66M parameter encoder-style [DistilBERT](https://arxiv.org/abs/1910.01108) model (distilled down from a 340M parameter BERT model), starting for the pretrained weights and only training the last transformer block plus output layers:



```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "distilbert"
```

```
Ep 1 (Step 000000): Train loss 0.693, Val loss 0.688
Ep 1 (Step 000050): Train loss 0.452, Val loss 0.460
...
Ep 1 (Step 004300): Train loss 0.179, Val loss 0.272
Ep 1 (Step 004350): Train loss 0.199, Val loss 0.182
Training accuracy: 95.62% | Validation accuracy: 91.25%
Training completed in 4.26 minutes.

Evaluating on the full datasets ...

Training accuracy: 95.30%
Validation accuracy: 91.12%
Test accuracy: 91.40%
```
<br>

---

<br>

&nbsp;
### 4) 355M RoBERTa

A 355M parameter encoder-style [RoBERTa](https://arxiv.org/abs/1907.11692) model, starting for the pretrained weights and only training the last transformer block plus output layers:


```bash
python train_bert_hf.py --trainable_layers "last_block" --num_epochs 1 --model "roberta" 
```

```
Ep 1 (Step 000000): Train loss 0.695, Val loss 0.698
Ep 1 (Step 000050): Train loss 0.670, Val loss 0.690
...
Ep 1 (Step 004300): Train loss 0.083, Val loss 0.098
Ep 1 (Step 004350): Train loss 0.170, Val loss 0.086
Training accuracy: 98.12% | Validation accuracy: 96.88%
Training completed in 11.22 minutes.

Evaluating on the full datasets ...

Training accuracy: 96.23%
Validation accuracy: 94.52%
Test accuracy: 94.69%
```

<br>

---

<br>

&nbsp;
### 5) 304M DeBERTa-v3

A 304M parameter encoder-style [DeBERTa-v3](https://arxiv.org/abs/2111.09543) model. DeBERTa-v3 improves upon earlier versions with disentangled attention and improved position encoding.


```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "deberta-v3-base"
```

```
Ep 1 (Step 000000): Train loss 0.689, Val loss 0.694
Ep 1 (Step 000050): Train loss 0.673, Val loss 0.683
...
Ep 1 (Step 004300): Train loss 0.126, Val loss 0.149
Ep 1 (Step 004350): Train loss 0.211, Val loss 0.138
Training accuracy: 92.50% | Validation accuracy: 94.38%
Training completed in 7.20 minutes.

Evaluating on the full datasets ...

Training accuracy: 93.44%
Validation accuracy: 93.02%
Test accuracy: 92.95%
```

<br>

---

<br>



&nbsp;
### 6) 149M ModernBERT Base

[ModernBERT (2024)](https://arxiv.org/abs/2412.13663) is an optimized reimplementation of BERT that incorporates architectural improvements like parallel residual connections and gated linear units (GLUs) to boost efficiency and performance. It maintains BERT’s original pretraining objectives while achieving faster inference and better scalability on modern hardware.

```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "modernbert-base"
```



```
Ep 1 (Step 000000): Train loss 0.699, Val loss 0.698
Ep 1 (Step 000050): Train loss 0.564, Val loss 0.606
...
Ep 1 (Step 004300): Train loss 0.086, Val loss 0.168
Ep 1 (Step 004350): Train loss 0.160, Val loss 0.131
Training accuracy: 95.62% | Validation accuracy: 93.75%
Training completed in 10.27 minutes.

Evaluating on the full datasets ...

Training accuracy: 95.72%
Validation accuracy: 94.00%
Test accuracy: 93.79%
```

<br>

---

<br>


&nbsp;
### 7) 395M ModernBERT Large

Same as above but using the larger ModernBERT variant.

```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "modernbert-large"
```



```
Ep 1 (Step 000000): Train loss 0.666, Val loss 0.662
Ep 1 (Step 000050): Train loss 0.548, Val loss 0.556
...
Ep 1 (Step 004300): Train loss 0.083, Val loss 0.115
Ep 1 (Step 004350): Train loss 0.154, Val loss 0.116
Training accuracy: 96.88% | Validation accuracy: 95.62%
Training completed in 27.69 minutes.

Evaluating on the full datasets ...

Training accuracy: 97.04%
Validation accuracy: 95.30%
Test accuracy: 95.07%
```





<br>

---

<br>

&nbsp;
### 8) Logistic Regression Baseline

A scikit-learn [logistic regression](https://sebastianraschka.com/blog/2022/losses-learned-part1.html) classifier as a baseline:


```bash
python train_sklearn_logreg.py
```

```
Dummy classifier:
Training Accuracy: 50.01%
Validation Accuracy: 50.14%
Test Accuracy: 49.91%


Logistic regression classifier:
Training Accuracy: 99.80%
Validation Accuracy: 88.62%
Test Accuracy: 88.85%
```
