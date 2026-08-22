"""Caption test images must be real, distinct RGB inputs — averaging caption
scores over identical images would be a fake sample size. (The end-to-end
Florence-2 caption test lives in tests/integration/benchmarks/.)"""

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
