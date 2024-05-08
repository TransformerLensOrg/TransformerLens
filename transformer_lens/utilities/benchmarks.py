import gc

import pandas as pd
import torch
from typing import Dict, List, NamedTuple, Optional, Tuple, Union, cast, overload
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformer import DTYPE_FROM_STRING
from transformer_lens.loading_from_pretrained import (
    get_official_model_name,
)

BenchmarkStatus = Dict[str, torch.Tensor]

def check_norm_folding(
    model_name: str,
    hf_model: torch.Tensor = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    prompt: str = "Hello, world!",
    device: Optional[Union[str, torch.device]] = None,
    dtype=None,
) -> torch.Tensor:
    """
    Checks that loading a model with Layer/RMS Norm folding enabled does not (significantly) change its outputs.

    Returns the maximum difference between the logits produced by the same model with and without norm folding enabled.

    Also asserts that this difference is within some tolerance, although this is deliberately set to a high value
    in order to account for lower precision models.
    """

    # If a device/dtype is not specified, and hf_model is provided, use its device/dtype
    # Otherwise, default to cuda (if available)/float32
    if device is None:
        if hf_model:
            device = hf_model.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        if hf_model:
            dtype = hf_model.dtype
        else:
            dtype = "float32"

    folded_model = HookedTransformer.from_pretrained(
        model_name=model_name,
        hf_model=hf_model,
        device=device,
        tokenizer=tokenizer,
        dtype=dtype,
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
    )
    tokens = folded_model.to_tokens(prompt)
    folded_logits = folded_model(tokens).detach()
    del folded_model
    torch.cuda.empty_cache()

    unfolded_model = HookedTransformer.from_pretrained(
        model_name=model_name,
        hf_model=hf_model,
        device=device,
        tokenizer=tokenizer,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    unfolded_logits = unfolded_model(tokens).detach()
    del unfolded_model
    torch.cuda.empty_cache()

    assert torch.allclose(
        torch.softmax(folded_logits, dim=-1),
        torch.softmax(unfolded_logits, dim=-1),
        atol=1e-2,
    )

    return torch.max(
        torch.abs(torch.softmax(folded_logits, dim=-1) - torch.softmax(unfolded_logits, dim=-1))
    )


def calculate_error(logits1: torch.Tensor, logits2: torch.Tensor) -> BenchmarkStatus:
    t1 = torch.softmax(logits1, dim=-1).to("cpu")
    t2 = torch.softmax(logits2, dim=-1).to("cpu")
    err = torch.abs(t1 - t2)
    return {
        "max": torch.max(err).item(),
        "mean": torch.mean(err).item(),
        "median": torch.median(err).item(),
        "std": torch.std(err).item(),
    }


def benchmark_model_option(
    options: Dict[str, bool],
    option: str,
    model_name: str,
    hf_logits: torch.Tensor,
    tokens: torch.Tensor,
    hf_model: Optional[AutoModelForCausalLM] = None,
    device: Optional[Union[str, torch.device]] = "cuda",
    n_devices: int = 1,
    dtype = torch.float16,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> BenchmarkStatus:
    gc.collect()
    new_options = options.copy()
    new_options[option] = True
    tl_model = HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        n_devices=n_devices,
        dtype=dtype,
        **new_options,
    )
    tl_logits = tl_model(tokens).detach().to("cpu")
    result = calculate_error(hf_logits, tl_logits)

    del tl_model, tl_logits
    torch.cuda.empty_cache()
    gc.collect()
    
    return result

def benchmark_model_options(
    model_name: str,
    hf_model: torch.Tensor = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    device: Optional[Union[str, torch.device]] = "cuda",
    n_devices: int = 1,
    dtype=torch.float16,
    cache_in_cpu=True,
) -> List[BenchmarkStatus]:
    options = {
        "fold_ln": False,
        "center_writing_weights": False,
        "center_unembed": False,
        "fold_value_biases": False,
    }

    prompts = [
        "Hello, world!",
        "This is a test.",
        "What is it about?",
        "I don't know.",
    ]

    model_name = get_official_model_name(model_name)

    if hf_model is None:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokens = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=4).input_ids.to(
        device
    )

    # hf_model = hf_model.to(device)
    hf_logits = hf_model(tokens).logits.detach()
    hf_logits = hf_logits.to("cpu")

    if cache_in_cpu:
        hf_model = hf_model.to("cpu")
    else:
        del hf_model
        hf_model = None

    torch.cuda.empty_cache()
    gc.collect()

    results = {}

    # Check the error when all processing options are disabled
    tl_model = HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        n_devices=n_devices,
        dtype=dtype,
        **options,
    )
    tl_logits = tl_model(tokens).detach().to("cpu")
    results["no_options"] = calculate_error(hf_logits, tl_logits)
    del tl_model, tl_logits
    torch.cuda.empty_cache()

    # Check the error when each processing option is enabled individually
    for option in options:
        results[option] = benchmark_model_option(
            options=options,
            option=option,
            model_name=model_name,
            hf_logits=hf_logits,
            tokens=tokens,
            hf_model=hf_model,
            device=device,
            n_devices=n_devices,
            dtype=dtype,
            tokenizer=tokenizer,
        )

    # Check the error when all processing options are enabled
    all_options = {k: True for k, v in options.items()}
    tl_model = HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        n_devices=n_devices,
        dtype=dtype,
        **all_options,
    )
    tl_logits = tl_model(tokens).detach().to("cpu")
    results["all_options"] = calculate_error(hf_logits, tl_logits)

    del tl_model, tl_logits

    del hf_model
    del tokens
    gc.collect()
    torch.cuda.empty_cache()

    return results


def benchmark_models(models, device="cuda", n_devices=1, cache_in_cpu=True, verbose=False):
    """
    Benchmark the error introduced by different options and data types for a list of models.
    :param models: A dict mapping model names to a list of dtypes to test
    """
    rows = []

    for model in models:
        dtypes = models[model]
        for dtype in dtypes:
            if verbose:
                print(f"Testing {model} with dtype {dtype}")
            results = benchmark_model_options(
                model,
                device=device,
                n_devices=n_devices,
                dtype=DTYPE_FROM_STRING[dtype],
                cache_in_cpu=cache_in_cpu,
            )
            for option, result in results.items():
                rows.append({"model": model, "dtype": dtype, "options": option, **result})

    return pd.DataFrame(rows)


def check_similarity_with_hf_model(tl_model, hf_model, prompt="Hello, world!"):
    """
    Check that the TransformerLens model and the HuggingFace model
    give approximately the same results.

    The logits typically differ by a constant value, but check only the results
    after the softmax because this is what matters most.
    """
    tokens = tl_model.tokenizer.encode(prompt, return_tensors="pt")
    logits = tl_model(tokens, prepend_bos=False)
    hf_logits = hf_model(tokens).logits
    assert torch.allclose(
        torch.softmax(logits, dim=-1), torch.softmax(hf_logits, dim=-1), atol=1e-5
    )


def check_performance(tl_model, hf_model, margin):
    """
    Check that the TransformerLens model and the HuggingFace have
    approximately the same confidence in the expected answer.
    """
    prompt = " Unable"
    tokens = tl_model.tokenizer(prompt, return_tensors="pt")["input_ids"].to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    expected_token = tl_model.tokenizer.encode(" to")[
        0
    ]  # Assume this is the expected token to predict

    tl_logits = tl_model(tokens, prepend_bos=False)[0, -1].float()
    hf_logits = hf_model(tokens).logits[0, -1].float()
    tl_prob = torch.softmax(tl_logits, dim=-1)[expected_token].item()
    hf_prob = torch.softmax(hf_logits, dim=-1)[expected_token].item()
    assert tl_prob + margin > hf_prob


def check_dtype(dtype, margin, no_processing=False):
    """Check the loading and inferences for different dtypes."""
    for model_path in ["gpt2", "roneneldan/TinyStories-33M", "EleutherAI/pythia-70m"]:
        if no_processing:
            # For low precision, the processing is not advised.
            model = HookedTransformer.from_pretrained_no_processing(model_path, torch_dtype=dtype)
        else:
            model = HookedTransformer.from_pretrained(model_path, torch_dtype=dtype)

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        for layer_name, layer in model.state_dict().items():
            assert layer.dtype in [dtype, torch.bool] or "IGNORE" in layer_name

        check_performance(model, hf_model, margin)

        # Check that generate doesn't throw an error
        _ = model.generate("Hello, World!")

        del model
        del hf_model
        gc.collect()