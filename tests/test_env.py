"""Sanity checks on the Python environment.

These tests do not check any of our code yet: they only make sure the
libraries we depend on are installed and that PyTorch can see the GPU.
"""

import torch


def test_torch_imports():
    # If PyTorch is installed correctly it exposes a version string like "2.15.0".
    assert isinstance(torch.__version__, str)

    # MPS ("Metal Performance Shaders") is PyTorch's backend for the GPU of
    # Apple-silicon Macs. The call must return a plain bool on any machine:
    # True on an M1/M2/M3 Mac, False elsewhere (e.g. a Linux server).
    assert isinstance(torch.backends.mps.is_available(), bool)
