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
python download-prepare-dataset.py
```


&nbsp;
## Step 3: Run Models

The 124M GPT-2 model used in the main chapter, starting for the pretrained weights and only training the last transformer block plus output layers:

```bash
python train-gpt.py
```

```
Ep 1 (Step 000000): Train loss 2.829, Val loss 3.433
Ep 1 (Step 000050): Train loss 1.440, Val loss 1.669
Ep 1 (Step 000100): Train loss 0.879, Val loss 1.037
Ep 1 (Step 000150): Train loss 0.838, Val loss 0.866
...
Ep 1 (Step 004300): Train loss 0.174, Val loss 0.202
Ep 1 (Step 004350): Train loss 0.309, Val loss 0.190
Training accuracy: 88.75% | Validation accuracy: 91.25%
Ep 2 (Step 004400): Train loss 0.263, Val loss 0.205
Ep 2 (Step 004450): Train loss 0.226, Val loss 0.188
...
Ep 2 (Step 008650): Train loss 0.189, Val loss 0.171
Ep 2 (Step 008700): Train loss 0.225, Val loss 0.179
Training accuracy: 85.00% | Validation accuracy: 90.62%
Ep 3 (Step 008750): Train loss 0.206, Val loss 0.187
Ep 3 (Step 008800): Train loss 0.198, Val loss 0.172
...
Training accuracy: 96.88% | Validation accuracy: 90.62%
Training completed in 18.62 minutes.

Evaluating on the full datasets ...

Training accuracy: 93.66%
Validation accuracy: 90.02%
Test accuracy: 89.96%
```

---

A 66M parameter encoder-style [DistilBERT](https://arxiv.org/abs/1910.01108) model (distilled down from a 340M parameter BERT model), starting for the pretrained weights and only training the last transformer block plus output layers:


```bash
python train-bert-hf.py
```

```
Ep 1 (Step 000000): Train loss 0.693, Val loss 0.697
Ep 1 (Step 000050): Train loss 0.532, Val loss 0.596
Ep 1 (Step 000100): Train loss 0.431, Val loss 0.446
...
Ep 1 (Step 004300): Train loss 0.234, Val loss 0.351
Ep 1 (Step 004350): Train loss 0.190, Val loss 0.222
Training accuracy: 88.75% | Validation accuracy: 88.12%
Ep 2 (Step 004400): Train loss 0.258, Val loss 0.270
Ep 2 (Step 004450): Train loss 0.204, Val loss 0.295
...
Ep 2 (Step 008650): Train loss 0.088, Val loss 0.246
Ep 2 (Step 008700): Train loss 0.084, Val loss 0.247
Training accuracy: 98.75% | Validation accuracy: 90.62%
Ep 3 (Step 008750): Train loss 0.067, Val loss 0.209
Ep 3 (Step 008800): Train loss 0.059, Val loss 0.256
...
Ep 3 (Step 013050): Train loss 0.068, Val loss 0.280
Ep 3 (Step 013100): Train loss 0.064, Val loss 0.306
Training accuracy: 99.38% | Validation accuracy: 87.50%
Training completed in 16.70 minutes.

Evaluating on the full datasets ...

Training accuracy: 98.87%
Validation accuracy: 90.98%
Test accuracy: 90.81%
```

---

A 355M parameter encoder-style [RoBERTa](https://arxiv.org/abs/1907.11692) model, starting for the pretrained weights and only training the last transformer block plus output layers:


```bash
python train-bert-hf.py --bert_model roberta
```

---

A scikit-learn Logistic Regression model as a baseline.

```bash
python train-sklearn-logreg.py
```

```
Dummy classifier:
Training Accuracy: 50.01%
Validation Accuracy: 50.14%
Test Accuracy: 49.91%


Logistic regression classifier:
Training Accuracy: 99.80%
Validation Accuracy: 88.60%
Test Accuracy: 88.84%
```
