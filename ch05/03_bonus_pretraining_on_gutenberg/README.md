# Pretraining GPT on the Project Gutenberg Dataset

The code in this directory contains code for training a small GPT model on the free books provided by Project Gutenberg.

As the Project Gutenberg website states, "the vast majority of Project Gutenberg eBooks are in the public domain in the US." 

Please read the [Project Gutenberg Permissions, Licensing and other Common Requests](https://www.gutenberg.org/policy/permission.html) page for more information about using the resources provided by Project Gutenberg. 

&nbsp;
## How to Use This Code

&nbsp;

### 1) Download the dataset

In this section, we download books from Project Gutenberg using code from the [`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) GitHub repository.

As of this writing, this will require approximately 50 GB of disk space, but it may be more depending on how much Project Gutenberg grew since then.

&nbsp;
#### Download instructions for Linux and macOS users


Linux and macOS users can follow these steps to download the dataset (if you are a Windows user, please see the note below):

1. Set the `03_bonus_pretraining_on_gutenberg` folder as working directory to clone the `gutenberg` repository locally in this folder (this is necessary to run the provided scripts `prepare_dataset.py` and `pretraining_simple.py`). For instance, when being in the `LLMs-from-scratch` repository's folder, navigate into the *03_bonus_pretraining_on_gutenberg* folder via:
```bash
cd ch05/03_bonus_pretraining_on_gutenberg
```

2. Clone the `gutenberg` repository in there:
```bash
git clone https://github.com/pgcorpus/gutenberg.git
```

3. Navigate into the locally cloned `gutenberg` repository's folder:
```bash
cd gutenberg
```

4. Install the required packages defined in *requirements.txt* from the `gutenberg` repository's folder:
```bash
pip install -r requirements.txt
```

5. Download the data:
```bash
python get_data.py
```

6. Go back into the `03_bonus_pretraining_on_gutenberg` folder
```bash
cd ..
```

&nbsp;
#### Special instructions for Windows users

The [`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) code is compatible with both Linux and macOS. However, Windows users would have to make small adjustments, such as adding `shell=True` to the `subprocess` calls and replacing `rsync`. 

Alternatively, an easier way to run this code on Windows is by using the "Windows Subsystem for Linux" (WSL) feature, which allows users to run a Linux environment using Ubuntu in Windows. For more information, please read [Microsoft's official installation instruction](https://learn.microsoft.com/en-us/windows/wsl/install) and [tutorial](https://learn.microsoft.com/en-us/training/modules/wsl-introduction/). 

When using WSL, please make sure you have Python 3 installed (check via `python3 --version`, or install it for instance with `sudo apt-get install -y python3.10` for Python 3.10) and install following packages there:

```bash
sudo apt-get update && \
sudo apt-get upgrade -y && \
sudo apt-get install -y python3-pip && \
sudo apt-get install -y python-is-python3 && \
sudo apt-get install -y rsync && \
```

> [!NOTE]
> Instructions about how to set up Python and installing packages can be found in [Optional Python Setup Preferences](../../setup/01_optional-python-setup-preferences/README.md) and [Installing Python Libraries](../../setup/02_installing-python-libraries/README.md).
>
> Optionally, a Docker image running Ubuntu is provided with this repository. Instructions about how to run a container with the provided Docker image can be found in [Optional Docker Environment](../../setup/03_optional-docker-environment/README.md).

&nbsp;
### 2) Prepare the dataset

Next, run the `prepare_dataset.py` script, which concatenates the (as of this writing, 60,173) text files into fewer larger files so that they can be more efficiently transferred and accessed:

```bash
python prepare_dataset.py \
  --data_dir gutenberg/data \
  --max_size_mb 500 \
  --output_dir gutenberg_preprocessed
```

> [!TIP] 
> Note that the produced files are stored in plaintext format and are not pre-tokenized for simplicity. However, you may want to update the codes to store the dataset in a pre-tokenized form to save computation time if you are planning to use the dataset more often or train for multiple epochs. See the *Design Decisions and Improvements* at the bottom of this page for more information.

> [!TIP]
> You can choose smaller file sizes, for example, 50 MB. This will result in more files but might be useful for quicker pretraining runs on a small number of files for testing purposes.


&nbsp;
### 3) Run the pretraining script

You can run the pretraining script as follows. Note that the additional command line arguments are shown with the default values for illustration purposes:

```bash
python pretraining_simple.py \
  --data_dir "gutenberg_preprocessed" \
  --n_epochs 1 \
  --batch_size 4 \
  --output_dir model_checkpoints
```

The output will be formatted in the following way:

> Total files: 3  
> Tokenizing file 1 of 3: data_small/combined_1.txt  
> Training ...  
> Ep 1 (Step 0): Train loss 9.694, Val loss 9.724  
> Ep 1 (Step 100): Train loss 6.672, Val loss 6.683  
> Ep 1 (Step 200): Train loss 6.543, Val loss 6.434  
> Ep 1 (Step 300): Train loss 5.772, Val loss 6.313  
> Ep 1 (Step 400): Train loss 5.547, Val loss 6.249  
> Ep 1 (Step 500): Train loss 6.182, Val loss 6.155  
> Ep 1 (Step 600): Train loss 5.742, Val loss 6.122  
> Ep 1 (Step 700): Train loss 6.309, Val loss 5.984  
> Ep 1 (Step 800): Train loss 5.435, Val loss 5.975  
> Ep 1 (Step 900): Train loss 5.582, Val loss 5.935  
> ...  
> Ep 1 (Step 31900): Train loss 3.664, Val loss 3.946  
> Ep 1 (Step 32000): Train loss 3.493, Val loss 3.939  
> Ep 1 (Step 32100): Train loss 3.940, Val loss 3.961  
> Saved model_checkpoints/model_pg_32188.pth  
> Book processed 3h 46m 55s   
> Total time elapsed 3h 46m 55s   
> ETA for remaining books: 7h 33m 50s  
> Tokenizing file 2 of 3: data_small/combined_2.txt  
> Training ...  
> Ep 1 (Step 32200): Train loss 2.982, Val loss 4.094  
> Ep 1 (Step 32300): Train loss 3.920, Val loss 4.097  
> ...


&nbsp;
> [!TIP] 
> In practice, if you are using macOS or Linux, I recommend using the `tee` command to save the log outputs to a `log.txt` file in addition to printing them on the terminal:

```bash
python -u pretraining_simple.py | tee log.txt
```

&nbsp;
> [!WARNING]  
> Note that training on 1 of the ~500 Mb text files in the `gutenberg_preprocessed` folder will take approximately 4 hours on a V100 GPU. 
> The folder contains 47 files and will take approximately 200 hours (more than 1 week) to complete. You may want to run it on a smaller number of files.


&nbsp;
## Design Decisions and Improvements

Note that this code focuses on keeping things simple and minimal for educational purposes. The code could be improved in the following ways to improve modeling performance and training efficiency:

1. Modify the `prepare_dataset.py` script to strip the Gutenberg boilerplate text from each book file.
2. Update the data preparation and loading utilities to pre-tokenize the dataset and save it in a tokenized form so that it doesn't have to be re-tokenized each time when calling the pretraining script.
3. Update the `train_model_simple` script by adding the features introduced in [Appendix D: Adding Bells and Whistles to the Training Loop](../../appendix-D/01_main-chapter-code/appendix-D.ipynb), namely, cosine decay, linear warmup, and gradient clipping.
4. Update the pretraining script to save the optimizer state (see section *5.4 Loading and saving weights in PyTorch* in chapter 5; [ch05.ipynb](../../ch05/01_main-chapter-code/ch05.ipynb)) and add the option to load an existing model and optimizer checkpoint and continue training if the training run was interrupted.
5. Add a more advanced logger (for example, Weights and Biases) to view the loss and validation curves live
6. Add distributed data parallelism (DDP) and train the model on multiple GPUs (see section *A.9.3 Training with multiple GPUs* in appendix A; [DDP-script.py](../../appendix-A/01_main-chapter-code/DDP-script.py)).
7. Swap the from scratch `MultiheadAttention` class in the `previous_chapter.py` script with the efficient `MHAPyTorchScaledDotProduct` class implemented in the [Efficient Multi-Head Attention Implementations](../../ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb) bonus section, which uses Flash Attention via PyTorch's `nn.functional.scaled_dot_product_attention` function.
8. Speeding up the training by optimizing the model via [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) (`model = torch.compile`) or [thunder](https://github.com/Lightning-AI/lightning-thunder) (`model = thunder.jit(model)`).
9. Implement Gradient Low-Rank Projection (GaLore) to further speed up the pretraining process. This can be achieved by just replacing the `AdamW` optimizer with the provided `GaLoreAdamW` provided in the [GaLore Python library](https://github.com/jiaweizzhao/GaLore).