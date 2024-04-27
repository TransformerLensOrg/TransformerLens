import time

import pytest
import torch

from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.utilities.devices import get_device_for_block_index


@pytest.fixture
def gpt2_medium_on_1_device():
    model = HookedTransformer.from_pretrained("gpt2-medium", fold_ln=False, n_devices=1)
    return model


@pytest.fixture
def gpt2_medium_on_4_devices():
    model = HookedTransformer.from_pretrained("gpt2-medium", fold_ln=False, n_devices=4)
    return model


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 CUDA devices")
def test_get_device_for_block_index(gpt2_medium_on_4_devices):
    config = gpt2_medium_on_4_devices.cfg
    n_layers = config.n_layers
    n_devices = config.n_devices
    layers_per_device = n_layers // n_devices
    config_device = torch.device(config.device)

    # Test with default device (config.device)
    for i in range(n_layers):
        expected_device = torch.device(config_device.type, i // layers_per_device)
        assert get_device_for_block_index(i, config) == expected_device

    # Test with explicit device
    device_override = "cuda"
    for i in range(n_layers):
        expected_device = torch.device(device_override, i // layers_per_device)
        assert get_device_for_block_index(i, config, device_override) == expected_device

    # Test with explicit torch.device object
    device_override_obj = torch.device("cuda")
    for i in range(n_layers):
        expected_device = torch.device(device_override_obj.type, i // layers_per_device)
        assert get_device_for_block_index(i, config, device_override_obj) == expected_device

    # Test when index is out of bounds
    # with pytest.raises(IndexError):
    # get_device_for_block_index(n_layers, config)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 CUDA devices")
@pytest.mark.parametrize("n_devices", [1, 2, 3, 4])
def test_device_separation_and_cache(gpt2_medium_on_1_device, n_devices):
    model_1_device = gpt2_medium_on_1_device
    model_n_devices = HookedTransformer.from_pretrained(
        "gpt2-medium", fold_ln=False, n_devices=n_devices
    )

    model_description_text = """## Loading Models
    HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. 
    See my explainer for documentation of all supported models, and this table for hyper-parameters and the name used to load them. 
    Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 
    For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"""

    # run model on single device
    start_time_1_device = time.time()
    loss_1_device = model_1_device(model_description_text, return_type="loss")
    elapsed_time_1_device = time.time() - start_time_1_device

    # get model on n_devices
    start_time_n_devices = time.time()
    loss_n_devices = model_n_devices(model_description_text, return_type="loss")
    elapsed_time_n_devices = time.time() - start_time_n_devices

    gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
    gpt2_tokens = model_1_device.to_tokens(gpt2_text)

    gpt2_logits_1_device, gpt2_cache_1_device = model_1_device.run_with_cache(
        gpt2_tokens, remove_batch_dim=True
    )
    gpt2_logits_n_devices, gpt2_cache_n_devices = model_n_devices.run_with_cache(
        gpt2_tokens, remove_batch_dim=True
    )

    # Make sure the tensors in cache remain on their respective devices
    for i in range(model_n_devices.cfg.n_layers):
        expected_device = get_device_for_block_index(i, cfg=model_n_devices.cfg)
        cache_device = gpt2_cache_n_devices[f"blocks.{i}.mlp.hook_post"].device
        assert cache_device == expected_device

    assert torch.allclose(gpt2_logits_1_device.to("cpu"), gpt2_logits_n_devices.to("cpu"))
    for key in gpt2_cache_1_device.keys():
        assert torch.allclose(
            gpt2_cache_1_device[key].to("cpu"), gpt2_cache_n_devices[key].to("cpu")
        )

    cuda_devices = set()
    n_params_on_device = {}
    for name, param in model_n_devices.named_parameters():
        if param.device.type == "cuda":
            cuda_devices.add(param.device.index)
        if param.device.index not in n_params_on_device:
            n_params_on_device[param.device.index] = 0
        n_params_on_device[param.device.index] += 1

    for device in cuda_devices:
        prop_device = n_params_on_device[device] / len(model_n_devices.state_dict())
        expected_prop_device = 1 / n_devices
        assert prop_device == pytest.approx(expected_prop_device, rel=0.20)

    print(
        f"Number of devices: {n_devices}, Model loss (1 device): {loss_1_device}, Model loss ({n_devices} devices): {loss_n_devices}, Time taken (1 device): {elapsed_time_1_device:.4f} seconds, Time taken ({n_devices} devices): {elapsed_time_n_devices:.4f} seconds"
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices")
def test_cache_device():
    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda:1")

    logits, cache = model.run_with_cache("Hello there")
    assert norm_device(cache["blocks.0.mlp.hook_post"].device) == norm_device(
        torch.device("cuda:1")
    )

    logits, cache = model.run_with_cache("Hello there", device="cpu")
    assert norm_device(cache["blocks.0.mlp.hook_post"].device) == norm_device(torch.device("cpu"))

    model.to("cuda")
    logits, cache = model.run_with_cache("Hello there")
    assert norm_device(cache["blocks.0.mlp.hook_post"].device) == norm_device(logits.device)


def norm_device(device):
    """
    Convenience function to normalize device strings for comparison.
    """
    device_str = str(device)
    if device_str.startswith("cuda") and ":" not in device_str:
        device_str += ":0"
    return device_str
