# Additional Experiments

The table below adds experiments to answer additional questions about various design choices. The first row uses the same settings as the main chapter and is used as a reference.
For example, 

- comparing rows 1 and 2 answers the question: "What is the performance difference when we train the last or first token?";
- comparing rows 1 and 3 answers the question: "What is the performance difference when we train only the last layer instead of the last block?";
- and so forth.

&nbsp;

|   | Model              | Weights    | Trainable token | Trainable layers | Context length          | CPU/GPU | Training time | Training acc | Validation acc | Test acc |
|---|--------------------|------------|-----------------|------------------|-------------------------|---------|---------------|--------------|----------------|----------|
| 1 | gpt2-small (124M)  | pretrained | last            | last_block       | longest train ex. (120) | V100    | 0.39 min      | 96.63%       | 97.99%         | 94.33%   |
| 2 | gpt2-small (124M)  | pretrained | first           | last_block       | longest train ex. (120) | V100    | 0.37 min      | 78.46%       | 80.54%         | 75.00%   |
| 3 | gpt2-small (124M)  | pretrained | last            | last_layer       | longest train ex. (120) | V100    | 0.33 min      | 78.65%       | 87.25%         | 78.33%   |
| 4 | gpt2-small (124M)  | pretrained | last            | all              | longest train ex. (120) | V100    | 0.94 min      | 99.62%       | 96.64%         | 96.33%   |
| 5 | gpt2-medium (355M) | pretrained | last            | last_block       | longest train ex. (120) | V100    | 0.91 min      | 87.50%       | 51.01%         | 56.67%   |
| 6 | gpt2-large (774M)  | pretrained | last            | last_block       | longest train ex. (120) | V100    | 1.91 min      | 99.52%       | 98.66%         | 96.67%   |
| 7 | gpt2-small (124M)  | random     | last            | all              | longest train ex. (120) | V100    | 0.93 min      | 100%         | 97.32%         | 93.00%   |
| 8 | gpt2-small (124M)  | pretrained | last            | last_block       | context length (1024)   | V100    | 3.24 min      | 83.08%       | 87.92%         | 78.33%   |

&nbsp;

### Usage:

- Row 1: `python additional-experiments.py`
- Row 2: `python additional-experiments.py --trainable_token first` 
- Row 3: `python additional-experiments.py --trainable_layers last_layer`
- Row 4: `python additional-experiments.py --trainable_layers all`
- Row 5: `python additional-experiments.py --model_size gpt2-medium (355M)`
- Row 6: `python additional-experiments.py --model_size gpt2-large (774M)`
- Row 7: `python additional-experiments.py --weights random --trainable_layers all`
- Row 8: `python additional-experiments.py --context_length "model_context_length"`