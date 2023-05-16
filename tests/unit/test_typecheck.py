import re
from textwrap import dedent
import pytest
import torch
import typeguard
from typeguard import typechecked
from jaxtyping import Float

from transformer_lens.typecheck import typecheck_fail_callback

typeguard.config.typecheck_fail_callback = typecheck_fail_callback

@typechecked
def foo(x: int) -> int:
    return x + 1

@typechecked
def tensor_foo(
    x: Float[torch.Tensor, "a b"],
    y: Float[torch.Tensor, "b c"],
    z: Float[torch.Tensor, "c d"],
    ) -> Float[torch.Tensor, "a d"]:
    return x @ y @ z

def test_error_message_is_correct_with_tensor_args():

    ref_err_msg = "argument \"x\" (str) is not an instance of int"

    with pytest.raises(typeguard.TypeCheckError, match=re.escape(ref_err_msg)):
        foo("a")

def test_error_message_is_correct_without_tensor_args():

    ref_err_msg = dedent("""argument "y" (torch.Tensor) is not an instance of jaxtyping.Float[Tensor, 'b c']
tensor argument info:
\tx  shape=(1, 2)     dtype=torch.float32
\ty  shape=(2, 3, 5)  dtype=torch.float32
\tz  shape=(3, 4)     dtype=torch.float32"""
    )

    with pytest.raises(typeguard.TypeCheckError, match=re.escape(ref_err_msg)):
        tensor_foo(torch.randn((1,2)), torch.randn((2,3,5)), torch.randn((3,4)))
