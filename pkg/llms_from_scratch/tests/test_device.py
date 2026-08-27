from unittest.mock import patch

import torch

from llms_from_scratch.device import get_device


def test_get_device_prefers_cuda_over_mps():
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.backends.mps.is_available", return_value=True
    ):
        assert get_device() == torch.device("cuda")


def test_get_device_uses_mps_when_cuda_is_unavailable():
    with patch("torch.cuda.is_available", return_value=False), patch(
        "torch.backends.mps.is_available", return_value=True
    ):
        assert get_device() == torch.device("mps")


def test_get_device_falls_back_to_cpu():
    with patch("torch.cuda.is_available", return_value=False), patch(
        "torch.backends.mps.is_available", return_value=False
    ):
        assert get_device() == torch.device("cpu")
