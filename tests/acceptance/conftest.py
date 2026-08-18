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


# Module-scoped, not session: only one module uses these, and holding a bloom
# pair resident costs ~4.5GB fp32 — enough to push the 7GB coverage runner into
# the swap it already needs for xdist.
@pytest.fixture(scope="module")
def bloom_560m_hooked():
    import gc

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        "bigscience/bloom-560m", default_prepend_bos=False, device="cpu"
    )
    yield model
    del model
    gc.collect()


@pytest.fixture(scope="module")
def bloom_560m_hf_model():
    import gc

    import torch
    from transformers import AutoModelForCausalLM

    # transformers 5.x loads at the checkpoint's dtype (fp16 here) while TL loads
    # fp32; comparing across that gap measures HF's own fp16 error, not TL.
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", dtype=torch.float32)
    yield model
    del model
    gc.collect()


@pytest.fixture(scope="session")
def bloom_560m_hf_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("bigscience/bloom-560m")
