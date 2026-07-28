"""Image-conditioned seq2seq (Florence-2) must score P4 from real captions.

Florence-2 needs pixel_values to generate: given a text-only prompt its decoder
emits a 1-token EOS, so every continuation is "too short" and P4 scored 0 (a
misleading failure for a working model). The fix drives real image-conditioned
caption generation (<DETAILED_CAPTION> on synthetic test images) and scores that
grammatical output instead — a genuine quality signal, not a skip.
"""

import pytest

pytest.importorskip("transformers")
pytest.importorskip("PIL")


def test_build_caption_test_images_are_distinct_rgb():
    from transformer_lens.benchmarks.text_quality import _build_caption_test_images

    images = _build_caption_test_images(n=3)
    assert len(images) == 3
    assert all(im.mode == "RGB" and im.size == (224, 224) for im in images)
    # Distinct backgrounds -> distinct pixel data (averaging over samples is real).
    assert len({im.tobytes() for im in images}) == 3


def test_florence2_text_quality_scores_image_captions():
    from transformer_lens.benchmarks.text_quality import benchmark_text_quality
    from transformer_lens.model_bridge import TransformerBridge

    try:
        bridge = TransformerBridge.boot_transformers(
            "florence-community/Florence-2-base-ft", device="cpu"
        )
    except (OSError, ConnectionError, TimeoutError) as exc:
        pytest.skip(f"florence-2 unavailable offline: {exc}")

    # Preconditions: this is the image-conditioned seq2seq path.
    assert bridge.original_model.config.is_encoder_decoder
    assert getattr(bridge.cfg, "is_multimodal", False)

    result = benchmark_text_quality(
        bridge, "The theory of relativity explains that", max_new_tokens=50, device="cpu"
    )
    # Pre-fix: "Scoring failed for all prompts" (score absent -> registry P4=0).
    assert result.details is not None, result.message
    assert "score" in result.details, result.message
    assert result.details["score"] > 0
    # Scored the model's actual captions, not the 4 text-only prompts.
    assert result.details["num_prompts"] >= 1
