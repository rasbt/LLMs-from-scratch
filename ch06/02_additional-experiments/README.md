# Additional Experiments

| Model              | Trainable token | Trainable layers | CPU/GPU | Training time | Training acc | Validation acc | Test acc |
|--------------------|-----------------|------------------|---------|---------------|--------------|----------------|----------|
| gpt2-small (124M)  | last            | last_block       | V100    | 0.39 min      | 96.63%       | 97.99%         | 94.33%   |
| gpt2-small (124M)  | first           | last_block       | V100    | 0.37 min      | 78.46%       | 80.54%         | 75.00%   |
| gpt2-small (124M)  | last            | last_layer       | V100    | 0.33 min      | 78.65%       | 87.25%         | 78.33%   |
| gpt2-small (124M)  | last            | all              | V100    | 0.94 min      | 99.62%       | 96.64%         | 96.33%   |
| gpt2-medium (355M) | last            | last_block       | V100    | 0.91 min      | 87.50%       | 51.01%         | 56.67%   |
| gpt2-large (774M)  | last            | last_block       | V100    | 1.91 min      | 99.52%       | 98.66%         | 96.67%   |