# Gemma 4

This directory contains a standalone, text-only Gemma 4 notebook built from the Gemma 3 reference notebook and adapted for the dense `google/gemma-4-E2B` and `google/gemma-4-E4B` checkpoints.

- [standalone-gemma4.ipynb](./standalone-gemma4.ipynb) implements the shared Gemma 4 dense architecture in pure PyTorch and switches between the E2B and E4B reference configs via `CHOOSE_MODEL`.
