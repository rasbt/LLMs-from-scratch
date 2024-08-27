# Additional Experiments Classifying the Sentiment of 50k IMDB Movie Reviews

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

The 124M GPT-2 model used in the main chapter, starting with pretrained weights, and finetuning all weights:

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

A 355M parameter encoder-style [RoBERTa](https://arxiv.org/abs/1907.11692) model, starting for the pretrained weights and only training the last transformer block plus output layers:


```bash
python train_bert_hf.py --trainable_layers "last_block" --num_epochs 1 --model "roberta" 
```

```
Ep 1 (Step 000000): Train loss 0.695, Val loss 0.698
Ep 1 (Step 000050): Train loss 0.670, Val loss 0.690
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

A scikit-learn logistic regression classifier as a baseline:


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
