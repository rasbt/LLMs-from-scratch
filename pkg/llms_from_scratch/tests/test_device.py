from unittest.mock import patch

import pytest
import torch

from llms_from_scratch.device import get_device


def _mps_is_available():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


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


@pytest.mark.skipif(not _mps_is_available(), reason="MPS is not available")
def test_mps_forward_matches_cpu():
    torch.manual_seed(123)
    cpu_model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    mps_model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    mps_model.load_state_dict(cpu_model.state_dict())
    inputs = torch.randn(6, 4)

    cpu_output = cpu_model(inputs)
    mps_output = mps_model.to("mps")(inputs.to("mps")).cpu()

    assert torch.isfinite(mps_output).all()
    assert torch.allclose(mps_output, cpu_output, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not _mps_is_available(), reason="MPS is not available")
def test_mps_training_matches_cpu():
    torch.manual_seed(123)
    initial_model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    inputs = torch.randn(12, 4)
    targets = torch.randint(0, 2, (12,))
    cpu_model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    mps_model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    cpu_model.load_state_dict(initial_model.state_dict())
    mps_model.load_state_dict(initial_model.state_dict())
    mps_model.to("mps")
    cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=0.1)
    mps_optimizer = torch.optim.SGD(mps_model.parameters(), lr=0.1)
    cpu_losses = []
    mps_losses = []

    for _ in range(8):
        cpu_optimizer.zero_grad()
        cpu_loss = torch.nn.functional.cross_entropy(cpu_model(inputs), targets)
        cpu_loss.backward()
        cpu_optimizer.step()
        cpu_losses.append(cpu_loss.detach())

        mps_optimizer.zero_grad()
        mps_loss = torch.nn.functional.cross_entropy(mps_model(inputs.to("mps")), targets.to("mps"))
        mps_loss.backward()
        mps_optimizer.step()
        mps_losses.append(mps_loss.detach().cpu())

    assert all(torch.isfinite(loss) for loss in mps_losses)
    assert torch.allclose(torch.stack(mps_losses), torch.stack(cpu_losses), atol=1e-5, rtol=1e-4)
