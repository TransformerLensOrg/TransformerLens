"""gpu_utils.

This module contains varied utility functions related to GPUs.
"""

from __future__ import annotations

import numpy as np
import torch


def print_gpu_mem(step_name=""):
    print(f"{step_name} ~ {np.round(torch.cuda.memory_allocated()/2e30, 2)} GiB allocated on GPU.")
