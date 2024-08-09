"""gpu_utils.

This module contains varied utility functions related to GPUs.
"""

from __future__ import annotations

import numpy as np
import torch


def print_gpu_mem(step_name=""):
    print(f"{step_name} ~ {np.round(torch.cuda.memory_allocated()/2e30, 2)} GiB allocated on GPU.")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Parse the PyTorch version to check if it's below version 2.0
        major_version = int(torch.__version__.split(".")[0])
        if major_version >= 2:
            return torch.device("mps")

    return torch.device("cpu")
