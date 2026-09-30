import pytest
import torch

from transformer_lens.model_bridge.composition_scores import CompositionScores


@pytest.mark.parametrize("operation", [torch.stack, torch.cat], ids=["stack", "cat"])
@pytest.mark.parametrize("container", [list, tuple], ids=["list", "tuple"])
@pytest.mark.parametrize("use_kwargs", [False, True], ids=["positional", "keyword"])
@pytest.mark.parametrize("mixed", [False, True], ids=["wrapped", "mixed"])
def test_sequence_operations_match_tensors(
    operation,
    container,
    use_kwargs: bool,
    mixed: bool,
) -> None:
    first = torch.arange(4, dtype=torch.float32).reshape(1, 2, 1, 2)
    second = first + 10
    wrapped_first = CompositionScores(first, [0], ["L0H0", "L0H1"])
    wrapped_second = CompositionScores(second, [0], ["L0H0", "L0H1"])

    values = container([wrapped_first, second if mixed else wrapped_second])
    expected = operation(container([first, second]), dim=1)

    if use_kwargs:
        actual = operation(tensors=values, dim=1)
    else:
        actual = operation(values, dim=1)

    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, expected)
