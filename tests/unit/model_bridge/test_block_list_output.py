"""BlockBridge hooks the residual in a list output, as OpenAI GPT's HF block returns."""

import torch

from transformer_lens.model_bridge.generalized_components import BlockBridge


def test_list_output_fires_hook_out_and_stays_a_list() -> None:
    block = BlockBridge(name="blocks.0")
    edited = torch.zeros(1, 3, 4)
    extra = torch.ones(1)
    calls = []

    def edit(tensor: torch.Tensor, *, hook) -> torch.Tensor:
        calls.append(tensor)
        return edited

    block.hook_out.add_hook(edit)
    residual = torch.ones(1, 3, 4)
    output = block._apply_output_hook([residual, extra])

    assert len(calls) == 1 and calls[0] is residual
    assert isinstance(output, list)
    assert output[0] is edited and output[1] is extra


def test_single_element_list_stays_a_list() -> None:
    output = BlockBridge(name="blocks.0")._apply_output_hook([torch.ones(1, 3, 4)])
    assert isinstance(output, list) and len(output) == 1


def test_list_without_a_leading_tensor_is_returned_untouched() -> None:
    block = BlockBridge(name="blocks.0")
    calls = []
    block.hook_out.add_hook(lambda tensor, *, hook: calls.append(tensor))
    for output in ([], [None, torch.ones(1)]):
        assert block._apply_output_hook(output) is output
    assert calls == []
