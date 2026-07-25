"""External-oracle parity test for TL JacobianLens.

Compares TransformerBridge-based JacobianLens readouts against the reference
anthropics/jacobian-lens oracle on google/gemma-2-2b-it.

Pass criteria (per the #1539 Tier-1 spec, matching the #1505 spike numbers):
  - Worst-case top-8 token overlap >= 7/8 across all 75 layer x prompt cells
  - Spearman rank-correlation >= 0.95 on the top-64 logit union per cell

The oracle is pinned to commit 581d398613e5602a5af361e1c34d3a92ea82ba8e
(Apache-2.0) so the threshold is reproducible independent of upstream drift.

Marked @pytest.mark.slow; requires ~4 GB VRAM and network access.
"""

import subprocess
import sys
from typing import Dict, List, Set, Tuple

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ORACLE_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
ORACLE_PKG = f"git+https://github.com/anthropics/jacobian-lens.git@{ORACLE_COMMIT}"

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"
GEMMA_MODEL = "google/gemma-2-2b-it"
GEMMA_LENS_FILE = "gemma-2-2b-it/jlens/Salesforce-wikitext/gemma-2-2b-it_jacobian_lens.pt"

# 5 prompts × 15 layers = 75 layer x prompt cells
PARITY_PROMPTS: List[str] = [
    "The Eiffel Tower is located in the city of",
    "Water molecules consist of two hydrogen atoms and one",
    "The mitochondria is often called the powerhouse of the",
    "In supervised learning, gradient descent is used to minimize the",
    "The speed of light in a vacuum is approximately three hundred",
]

# 15 layers sampled from Gemma-2-2b-it's 26 (indices 0-25).
# Layer 25 is the final (identity transport) and is checked separately.
PARITY_LAYERS: List[int] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23, 24, 25]
assert len(PARITY_PROMPTS) * len(PARITY_LAYERS) == 75

TOP_K = 8
TOP_K_MIN_OVERLAP = 7  # >= 7/8 in worst cell
SPEARMAN_TOP_K = 64
SPEARMAN_MIN_R = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spearman_r(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation — avoids a scipy dependency."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    n = len(ra)
    d2 = float(((ra - rb) ** 2).sum())
    denom = n * (n * n - 1)
    return 1.0 - 6.0 * d2 / denom


def _top_k_set(logits: torch.Tensor, k: int) -> Set[int]:
    return set(logits.topk(k).indices.tolist())


def _spearman_top_union(tl_logits: torch.Tensor, oracle_logits: torch.Tensor, k: int) -> float:
    tl_topk = tl_logits.topk(k).indices
    or_topk = oracle_logits.topk(k).indices
    union = torch.unique(torch.cat([tl_topk, or_topk])).tolist()
    return _spearman_r(
        tl_logits[union].float().numpy(),
        oracle_logits[union].float().numpy(),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oracle_package():
    """Install the pinned anthropics/jacobian-lens and return its key symbols."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", ORACLE_PKG],
        check=True,
    )
    # Imports must happen after install so they resolve to the just-installed pkg.
    from jlens.hf import from_hf
    from jlens.lens import JacobianLens as OracleJL

    return OracleJL, from_hf


@pytest.fixture(scope="module")
def gemma_bridge():
    from transformer_lens.model_bridge import TransformerBridge

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TransformerBridge.boot_transformers(GEMMA_MODEL, device=device)


@pytest.fixture(scope="module")
def tl_lens(gemma_bridge):
    from transformer_lens.tools.analysis import JacobianLens

    return JacobianLens.from_pretrained(
        LENS_REPO,
        filename=GEMMA_LENS_FILE,
        revision=LENS_REVISION,
        model=gemma_bridge,
    )


@pytest.fixture(scope="module")
def oracle_lens_and_model(oracle_package, gemma_bridge):
    """Oracle JacobianLens + HFLensModel sharing the bridge's underlying HF model."""
    OracleJL, from_hf = oracle_package
    oracle_lens = OracleJL.from_pretrained(
        LENS_REPO,
        filename=GEMMA_LENS_FILE,
        revision=LENS_REVISION,
    )
    # Reuse the HF model already loaded inside the bridge — no second copy.
    oracle_model = from_hf(
        gemma_bridge.original_model,
        gemma_bridge.tokenizer,
        force_bos=False,  # bridge already controls BOS; don't mutate its tokenizer
    )
    return oracle_lens, oracle_model


# ---------------------------------------------------------------------------
# Parity test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_jacobian_lens_oracle_parity(tl_lens, oracle_lens_and_model, gemma_bridge):
    """TL JacobianLens readouts match the anthropics oracle within #1539 tolerances.

    Checks all 75 (5 prompts × 15 layers) layer x prompt cells:
      - top-8 token overlap >= 7 in every cell
      - Spearman r >= 0.95 on the top-64 logit union in every cell
    """
    oracle_lens, oracle_model = oracle_lens_and_model

    final_layer = gemma_bridge.cfg.n_layers - 1
    source_layers = [l for l in PARITY_LAYERS if l != final_layer]

    failures: List[str] = []

    for prompt in PARITY_PROMPTS:
        # TL: one readout call for all layers at once.
        tl_result = tl_lens.readout(
            gemma_bridge,
            prompt,
            layers=PARITY_LAYERS,
            positions=[-1],
            return_full_logits=True,
            top_k=TOP_K,
        )
        assert tl_result.lens_logits is not None

        # Oracle: source layers only (raises if passed the final/identity layer).
        oracle_logits_dict, oracle_model_logits, _ids = oracle_lens.apply(
            oracle_model,
            prompt,
            layers=source_layers,
            positions=[-1],
        )

        # --- source layers: compare oracle transport vs TL transport ---
        for layer in source_layers:
            tl_vec = tl_result.lens_logits[layer][-1].float()  # [vocab]
            or_vec = oracle_logits_dict[layer][-1].float()  # [vocab]

            overlap = len(_top_k_set(tl_vec, TOP_K) & _top_k_set(or_vec, TOP_K))
            if overlap < TOP_K_MIN_OVERLAP:
                failures.append(
                    f"top-{TOP_K} overlap={overlap}/{TOP_K} "
                    f"(layer={layer}, prompt={prompt[:35]!r})"
                )

            r = _spearman_top_union(tl_vec, or_vec, SPEARMAN_TOP_K)
            if r < SPEARMAN_MIN_R:
                failures.append(
                    f"Spearman r={r:.4f} < {SPEARMAN_MIN_R} "
                    f"(layer={layer}, prompt={prompt[:35]!r})"
                )

        # --- final layer: TL identity == oracle model_logits ---
        tl_final = tl_result.lens_logits[final_layer][-1].float()  # [vocab]
        or_final = oracle_model_logits[-1].float()  # [vocab]

        final_overlap = len(_top_k_set(tl_final, TOP_K) & _top_k_set(or_final, TOP_K))
        if final_overlap < TOP_K_MIN_OVERLAP:
            failures.append(
                f"top-{TOP_K} overlap={final_overlap}/{TOP_K} "
                f"(final layer={final_layer}, prompt={prompt[:35]!r})"
            )

        final_r = _spearman_top_union(tl_final, or_final, SPEARMAN_TOP_K)
        if final_r < SPEARMAN_MIN_R:
            failures.append(
                f"Spearman r={final_r:.4f} < {SPEARMAN_MIN_R} "
                f"(final layer={final_layer}, prompt={prompt[:35]!r})"
            )

    assert not failures, f"{len(failures)} cell(s) failed parity out of 75:\n" + "\n".join(
        f"  {f}" for f in failures
    )
