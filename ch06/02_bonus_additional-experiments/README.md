# Additional Experiments

The table below adds experiments to answer additional questions about various design choices. The first row uses the same settings as the main chapter and is used as a reference.
For example, 

- comparing rows 1 and 2 answers the question: "What is the performance difference when we train the last or first token?";
- comparing rows 1 and 3 answers the question: "What is the performance difference when we train only the last layer instead of the last block?";
- and so forth.

&nbsp;

|      | Model              | Weights    | Trainable token | Trainable layers | Context length          | CPU/GPU | Training time | Training acc | Validation acc | Test acc |
| ---- | ------------------ | ---------- | --------------- | ---------------- | ----------------------- | ------- | ------------- | ------------ | -------------- | -------- |
| 1    | gpt2-small (124M)  | pretrained | last            | last_block       | longest train ex. (120) | V100    | 0.39 min      | 96.63%       | 99.33%         | 95.00%   |
| 2    | gpt2-small (124M)  | pretrained | first           | last_block       | longest train ex. (120) | V100    | 0.37 min      | 78.46%       | 80.54%         | 75.00%   |
| 3    | gpt2-small (124M)  | pretrained | last            | last_layer       | longest train ex. (120) | V100    | 0.33 min      | 78.65%       | 79.87%         | 72.00%   |
| 4    | gpt2-small (124M)  | pretrained | last            | all              | longest train ex. (120) | V100    | 0.94 min      | 99.62%       | 96.64%         | 96.67%   |
| 5    | gpt2-medium (355M) | pretrained | last            | last_block       | longest train ex. (120) | V100    | 0.91 min      | 87.50%       | 91.28%         | 84.67%   |
| 6    | gpt2-large (774M)  | pretrained | last            | last_block       | longest train ex. (120) | V100    | 1.91 min      | 99.52%       | 98.66%         | 96.67%   |
| 7    | gpt2-small (124M)  | random     | last            | all              | longest train ex. (120) | V100    | 0.93 min      | 100%         | 96.64%         | 93.67%   |
| 8    | gpt2-small (124M)  | pretrained | last            | last_block       | context length (1024)   | V100    | 3.24 min      | 83.08%       | 87.92%         | 78.33%   |

&nbsp;

### Usage

You can use the following code to reproduce the experiments:

- Row 1: `python additional-experiments.py`
- Row 2: `python additional-experiments.py --trainable_token first` 
- Row 3: `python additional-experiments.py --trainable_layers last_layer`
- Row 4: `python additional-experiments.py --trainable_layers all`
- Row 5: `python additional-experiments.py --model_size "gpt2-medium (355M)"`
- Row 6: `python additional-experiments.py --model_size "gpt2-large (774M)"`
- Row 7: `python additional-experiments.py --weights random --trainable_layers all`
- Row 8: `python additional-experiments.py --context_length "model_context_length"`

I've kept the LLM and dataset small on purpose, so you can run the training on a regular laptop like a MacBook Air M3 in about 15 minutes in case you don't have access to a GPU.

&nbsp;

### Interpretation

1. **Training the Last vs. First Output Token (Row 1 vs. 2)**: Training the last output token results in substantially better performance compared to the first. This improvement is expected due to the causal self-attention mask.

2. **Training the Last Transformer Block vs. Last Layer (Row 1 vs. 3)**: Training the entire last transformer block is also results in substantially better results than training only the last layer.

3. **Training All Layers vs. Last Transformer Block (Row 1 vs. 4)**: Training all layers shows a modest improvement of ~2% over just training the last transformer block, but it requires almost three times longer in terms of training duration.

4. **Using Larger Pretrained Models (Row 1 vs 5, and Row 1 vs. 6)**: Employing a 3x larger pretrained model leads to worse results. However, using a 5x larger model improves performance compared to the initial model, as was anticipated. (The medium model was perhaps not well pretrained or the particular finetuning configuration works not as well for this model.)

5. **Using a Model with Random Weights vs. Pretrained Weights (Row 1 vs. 7)**: Utilizing a model with random weights yields results that are only slightly worse by 1.3% compared to using pretrained weights.

6. **Padding Input to Full Context Length vs. Longest Training Example (Row 1 vs. 8)**: Padding the input to the full supported context length results is significantly worse.
