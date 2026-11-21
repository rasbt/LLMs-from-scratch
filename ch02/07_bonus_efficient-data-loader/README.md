# Bonus: A sliding-window data loader with `Tensor.unfold`

The `GPTDatasetV1` implementation in Chapter 2 aims to strike a blaance between efficiency, typical dataset setups, and code readability.

However, for advanced users, this can be further optimized as Dmitry Labazkin (@labdmitriy) shared the `GPTDatasetStrided` implementation in [GitHub discussion #745](https://github.com/rasbt/LLMs-from-scratch/discussions/745).



For instance, the original `GPTDatasetV1` uses a loop that creates a tensor for each input and another one for the target:

```python
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

```

The proposed `GPTDatasetStrided` stores the token IDs once and uses `Tensor.unfold` to create windows that share storage:

```python
class GPTDatasetStrided(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        token_ids = torch.tensor(
            tokenizer.encode(txt, allowed_special={"<|endoftext|>"}),
            dtype=torch.long,
        )
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        windows = token_ids.unfold(0, max_length + 1, stride)
        self.input_view = windows[..., :-1]
        self.target_view = windows[..., 1:]

    def __len__(self):
        return self.input_view.size(0)

    def __getitem__(self, idx):
        return self.input_view[idx], self.target_view[idx]
```



To reproduce the comparison below, you can execute the code in the [efficient-dataloader.ipynb](efficient-dataloader.ipynb) notebook.



The benchmark below is based on the sample text used in the chapter (*The Verdict* , 5,145 GPT-2 tokens), using a window length of 256 and a stride of 128, which gives us 39 examples. These are median CPU timings over 30 runs on an Apple M4 Pro with PyTorch 2.10.0. The benchmark includes includes the tokenization step. Each pass reads all 10 batches with a batch size of 4.

| Dataset | Build dataset (ms) | Read one pass (ms) | Token storage (KiB) |
| --- | ---: | ---: | ---: |
| `GPTDatasetV1` (Chapter 2) | 1.92 | 0.15 | 156.0 |
| `GPTDatasetStrided` | 1.06 | 0.17 | 40.2 |

As we can see above, at stride 128, the alternative `unfold` approach in `GPTDatasetStrided` saves less than a millisecond during setup but about 116 KiB (74%) of token storage. Reading the batches takes about the same time. However, keep in mind that dataset loading and preprocessing is incredibly cheap anyway, and the dataset loader works in the background while the model is training, so it's usually not a blocker or bottleneck. And so we prefer the loop in the main chapter because its two slices make the one-token shift easier to see and experiment with.

Also, as we can see above, the memory savings grow as the windows overlap more. At stride 1, the chapter dataset stores 4,889 input/target pairs and uses 19.1 MiB. However, we would not use strides of 1 in practice but set `stride=max_length` . The views still use 40.2 KiB for the same 5,145 token IDs. These figures count each underlying tensor allocation once. Python objects, the tokenizer, and batches also use RAM.

All in all, these is an interesting additional experiment, but using this in practice would be overkill and not result in any training speed up, since the training on the GPU is the bottleneck, not the background data loading and processing on the CPU.
