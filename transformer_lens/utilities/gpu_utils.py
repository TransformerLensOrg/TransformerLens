"""gpu_utils.

This module contains varied utility functions related to GPUs.
"""

from __future__ import annotations

import numpy as np
import torch


def print_gpu_mem(step_name=""):
    gib_allocated = torch.cuda.memory_allocated() / 2**30
    print(f"{step_name} ~ {np.round(gib_allocated, 2)} GiB allocated on GPU.")
