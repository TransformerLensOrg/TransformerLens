import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc


def get_sample_from_dataset(sequences, nb_sample=2, print_len=10):
    rd_idx = np.random.randint(0, len(sequences), 3)
    return "\n".join([str(sequences[k][:print_len]) + " ... " for k in rd_idx])


def print_gpu_mem(step_name=""):
    print(
        f"{step_name} ~ {np.round(torch.cuda.memory_allocated()/2e30, 2)} GiB allocated on GPU."
    )


def get_corner(tensor, n=2):
    # Prints the top left corner of the tensor
    if len(tensor.shape) == 0:
        return tensor
    elif len(tensor.shape) == 1:
        return tensor[:n]
    elif len(tensor.shape) == 2:
        return tensor[:n, :n]
    elif len(tensor.shape) == 3:
        return tensor[:n, :n, :n]
    elif len(tensor.shape) == 4:
        return tensor[:n, :n, :n, :n]
    elif len(tensor.shape) == 5:
        return tensor[:n, :n, :n, :n, :n]
    elif len(tensor.shape) == 6:
        return tensor[:n, :n, :n, :n, :n, :n]
    else:
        # I never need tensors of rank > 6
        raise ValueError(f"Tensor of shape {tensor.shape} is too big")


def to_numpy(tensor, flat=False):
    if (type(tensor) != torch.Tensor) and (
        type(tensor) != torch.nn.parameter.Parameter
    ):
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()


def gelu_new(input):
    # Implementation of GeLU used by GPT2 - subtly different from PyTorch's
    return (
        0.5
        * input
        * (
            1.0
            + torch.tanh(
                np.sqrt(2.0 / np.pi) * (input + 0.044715 * torch.pow(input, 3.0))
            )
        )
    )


def solu(input):
    """
    SoLU activation function as described by
    https://transformer-circuits.pub/2022/solu/index.html.
    
    LayerNorm implemented by the MLP class.
    """
    return input * F.softmax(input, dim=-1)


def reglu(input, gate):
    """
    ReGLU activation function as described by
    https://arxiv.org/pdf/2002.05202.pdf.
    """
    return F.relu(gate) * input


def geglu(input, gate, use_gelu_new=False):
    """
    GeGLU activation function as described by
    https://arxiv.org/pdf/2002.05202.pdf.
    """
    if use_gelu_new:
        return gelu_new(gate) * input
    else:
        return F.gelu(gate) * input


def swiglu(input, gate):
    """
    SwiGLU activation function as described by
    https://arxiv.org/pdf/2002.05202.pdf.
    """
    return F.silu(gate) * input
