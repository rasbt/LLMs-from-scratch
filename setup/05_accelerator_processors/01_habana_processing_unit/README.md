# Running Code on Habana Gaudi (HPU)

This directory contains instructions for running inference part from [Chapter 6](../../../ch06/01_main-chapter-code/ch06.ipynb) on Habana Gaudi processors. The code demonstrates how to leverage HPU acceleration.

## Prerequisites

1. **Habana Driver and Libraries**  
   Make sure you have the correct driver and libraries installed for Gaudi processors. You can follow the official installation guide from Habana Labs:  
   [Habana Labs Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html)

2. **SynapseAI SDK**  
   The SynapseAI SDK includes the compiler, runtime, and various libraries needed to compile and run models on Gaudi hardware.

## Getting Started
1. **Model Configuration**  
   The code supports various GPT-2 model sizes:
   - GPT-2 Small (124M parameters)
   - GPT-2 Medium (355M parameters)
   - GPT-2 Large (774M parameters)
   - GPT-2 XL (1558M parameters)

2. **Running the Code**  
   - Open the `inference_on_gaudi.ipynb` notebook
   - Follow the cells to:
     - Initialize the HPU device
     - Load and configure the model
     - Run inference on the Gaudi processor

3. **Performance Monitoring**  
   The notebook includes performance comparison tools to measure inference time on CPU vs HPU

## Code Structure

- `inference_on_gaudi.ipynb`: Main notebook for running inference on Gaudi
- `gpt_download.py`: Utilities for downloading GPT-2 models
- `previous_chapters.py`: Supporting code from Chapter 6

## Troubleshooting

- **Driver Issues**: Make sure the driver version matches the SDK version.
- **Performance**: For optimal performance, monitor logs and use Habana's profiling tools to identify bottlenecks.

## Additional Resources

- [Habana Developer Site](https://developer.habana.ai/)  
- [SynapseAI Reference](https://docs.habana.ai/en/latest/)  
