"""Device selection helpers for the PyTorch examples."""

import torch


def get_device() -> torch.device:
    """Return the fastest locally available PyTorch device.

    CUDA is preferred when available, followed by Apple's MPS backend and
    finally the CPU.  Keeping this choice in one helper lets examples run on
    Apple Silicon without requiring CUDA.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
