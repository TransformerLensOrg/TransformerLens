"""
Model accuracy tests: verify TransformerLens models match HuggingFace outputs.

NOT run by default. Use when adding or modifying model support.

Usage:
    # Run all supported models that fit in memory (default: ~1GB f32 per model)
    poetry run pytest tests/acceptance/test_model_accuracy.py -m model_accuracy

    # Filter by model name substring
    poetry run pytest tests/acceptance/test_model_accuracy.py -m model_accuracy -k "gemma"
    poetry run pytest tests/acceptance/test_model_accuracy.py -m model_accuracy -k "pythia-70m"

    # Adjust memory limit (GB, float32 model size; test needs ~2x for TL+HF)
    poetry run pytest tests/acceptance/test_model_accuracy.py -m model_accuracy --max-model-gb 16

    # Run a specific model not in OFFICIAL_MODEL_NAMES
    poetry run pytest tests/acceptance/test_model_accuracy.py -m model_accuracy --model "my-org/my-model"

Tests per model:
1. Weights loaded correctly (no all-zero weight matrices)
2. Forward pass logits match HuggingFace (compared after softmax, atol=5e-3)
3. Weight processing (fold_ln, etc.) preserves model behavior (atol=1e-3)
"""

import gc
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pytest
import torch
from huggingface_hub.errors import GatedRepoError
from transformers import AutoModelForCausalLM

from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import (
    OFFICIAL_MODEL_NAMES,
    get_pretrained_model_config,
)

# Prefixes for custom models that don't have a standard HF equivalent
# (loaded from .pth files, not AutoModelForCausalLM)
_CUSTOM_MODEL_PREFIXES = ("NeelNanda/", "ArthurConmy/", "Baidicoot/", "stanford-crfm/")

# Models that aren't hosted on HuggingFace (require passing hf_model manually)
_NON_HF_MODELS = {"llama-7b-hf", "llama-13b-hf", "llama-30b-hf", "llama-65b-hf"}

# Prefixes for models needing special HF loading (encoder-only, encoder-decoder)
_SPECIAL_ARCH_PREFIXES = ("google-bert/", "google-t5/")

# Prompt used for forward pass comparison
TEST_PROMPT = "The quick brown fox jumps over the lazy dog."

# Default memory limit per model in GB (float32 size)
DEFAULT_MAX_GB = 8.0


def _is_testable(model_name: str) -> bool:
    """Check if a model can be tested with AutoModelForCausalLM comparison."""
    if model_name in _NON_HF_MODELS:
        return False
    if any(model_name.startswith(p) for p in _CUSTOM_MODEL_PREFIXES):
        return False
    if any(model_name.startswith(p) for p in _SPECIAL_ARCH_PREFIXES):
        return False
    return True


def _get_model_size_gb(model_name: str) -> float:
    """Get estimated float32 model size in GB from config (without downloading weights)."""
    try:
        cfg = get_pretrained_model_config(model_name)
        if cfg.n_params is not None:
            return cfg.n_params * 4 / 1e9
    except Exception as e:
        warnings.warn(f"Could not get model size for {model_name}: {e!r}")
    return float("inf")


def _get_testable_models() -> List[str]:
    """Get all testable HF models from OFFICIAL_MODEL_NAMES."""
    return [name for name in OFFICIAL_MODEL_NAMES if _is_testable(name)]


@dataclass
class ModelTestResults:
    """Pre-computed results from loading models, so models can be freed before tests run."""

    model_name: str
    param_info: List[Tuple[str, int, bool, bool]] = field(default_factory=list)
    tl_probs: Optional[torch.Tensor] = None
    hf_probs: Optional[torch.Tensor] = None
    processed_probs: Optional[torch.Tensor] = None


def _compute_results(model_name: str) -> ModelTestResults:
    """Load models, compute all needed tensors, then free the models.

    Loads at most one full model at a time to minimize memory usage.
    Raises pytest.skip for gated models without access.
    """
    results = ModelTestResults(model_name=model_name)

    # Phase 1: Load raw TL model, capture weight info and logits
    try:
        raw_model = HookedTransformer.from_pretrained_no_processing(model_name, device="cpu")
    except (GatedRepoError, OSError) as e:
        if "gated repo" in str(e).lower() or "Cannot access gated repo" in str(e):
            pytest.skip(f"Gated model {model_name} not accessible: {e}")
        raise
    tokens = raw_model.to_tokens(TEST_PROMPT, prepend_bos=True)

    for name, param in raw_model.named_parameters():
        is_bias = "b_" in name or name.endswith(".b")
        results.param_info.append((name, param.shape.numel(), is_bias, bool(torch.all(param == 0))))

    with torch.no_grad():
        raw_logits = raw_model(tokens, prepend_bos=False).float()
    results.tl_probs = torch.softmax(raw_logits, dim=-1)

    del raw_model
    gc.collect()

    # Phase 2: Load HF model, compare logits, then free it
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
    with torch.no_grad():
        hf_logits = hf_model(tokens).logits.float()
    results.hf_probs = torch.softmax(hf_logits, dim=-1)

    del hf_model
    gc.collect()

    # Phase 3: Load processed TL model, compare logits, then free it
    processed_model = HookedTransformer.from_pretrained(model_name, device="cpu")
    with torch.no_grad():
        processed_logits = processed_model(tokens, prepend_bos=False).float()
    results.processed_probs = torch.softmax(processed_logits, dim=-1)

    del processed_model
    gc.collect()

    return results


def pytest_generate_tests(metafunc):
    """Dynamically parametrize model_name based on CLI options."""
    if "model_name" not in metafunc.fixturenames:
        return
    if metafunc.cls is not TestModelAccuracy:
        return

    max_gb = metafunc.config.getoption("--max-model-gb", DEFAULT_MAX_GB)
    explicit_models = metafunc.config.getoption("--model", None)

    if explicit_models:
        models = explicit_models
    else:
        models = _get_testable_models()

    # Filter by size
    filtered = []
    for name in models:
        size_gb = _get_model_size_gb(name)
        if size_gb <= max_gb:
            filtered.append(name)

    if not filtered:
        filtered = ["NO_MODELS_MATCH_FILTERS"]

    metafunc.parametrize("model_name", filtered, scope="class")


@pytest.mark.model_accuracy
class TestModelAccuracy:
    """Tests that TL models faithfully reproduce HuggingFace model outputs."""

    @pytest.fixture(scope="class")
    def results(self, model_name):
        if model_name == "NO_MODELS_MATCH_FILTERS":
            pytest.skip("No models match the current size/filter criteria")
        return _compute_results(model_name)

    def test_weights_loaded(self, results):
        """Verify all parameters were loaded (no accidentally all-zero weight matrices)."""
        for name, numel, is_bias, all_zero in results.param_info:
            if numel == 0:
                pytest.fail(f"Empty parameter: {name}")
            if not is_bias and numel > 1 and all_zero:
                pytest.fail(f"All-zero weight matrix (likely not loaded): {name}")

    def test_logits_match_huggingface(self, results):
        """End-to-end logits should match HF model (compared after softmax)."""
        # Generous tolerance is fine here: correct models diff at <5e-4,
        # broken conversions (e.g. missing Gemma +1) diff at ~1.0.
        assert torch.allclose(results.tl_probs, results.hf_probs, atol=5e-3), (
            f"Logit mismatch for {results.model_name}. "
            f"Max diff: {(results.tl_probs - results.hf_probs).abs().max().item():.2e}"
        )

    def test_processing_preserves_output(self, results):
        """Verify that fold_ln and other weight processing preserves model behavior."""
        # Looser tolerance: fold_ln introduces floating point differences
        assert torch.allclose(results.tl_probs, results.processed_probs, atol=1e-3), (
            f"Processing changed outputs too much for {results.model_name}. "
            f"Max diff: {(results.tl_probs - results.processed_probs).abs().max().item():.2e}"
        )
