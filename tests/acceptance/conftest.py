"""Session fixtures for acceptance tests.

transformer_lens imports stay inside fixture bodies — jaxtyping's pytest_configure
hook must install before the package is first imported.
"""

import pytest


@pytest.fixture(scope="session")
def gpt2_model():
    """Session-scoped HookedTransformer gpt2 with default weight processing."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained("gpt2", device="cpu")


@pytest.fixture(scope="session")
def bloom_560m_hooked():
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        "bigscience/bloom-560m", default_prepend_bos=False, device="cpu"
    )


@pytest.fixture(scope="session")
def bloom_560m_hf_model():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")


@pytest.fixture(scope="session")
def bloom_560m_hf_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("bigscience/bloom-560m")
