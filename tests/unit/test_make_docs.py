"""Make Docs Tests."""

import pytest

from docs.make_docs import get_config, get_model_info, get_property


def test_get_config():
    """Test get config with attn-only-1l model."""
    config = get_config("attn-only-1l")
    assert config.attn_only is True


def test_get_property():
    """Test get property with attn-only-1l model."""
    act_fn = get_property("act_fn", "attn-only-1l")
    assert act_fn == "attn_only"

    n_params = get_property("n_params", "attn-only-1l")
    assert n_params == "1.0M"

    n_layers = get_property("n_layers", "attn-only-1l")
    assert n_layers == 1

    d_model = get_property("d_model", "attn-only-1l")
    assert d_model == 512

    n_heads = get_property("n_heads", "attn-only-1l")
    assert n_heads == 8

    n_ctx = get_property("n_ctx", "attn-only-1l")
    assert n_ctx == 1024

    d_vocab = get_property("d_vocab", "attn-only-1l")
    assert d_vocab == 48262

    d_head = get_property("d_head", "attn-only-1l")
    assert d_head == 64

    d_mlp = get_property("d_mlp", "attn-only-1l")
    assert d_mlp == 2048

    n_key_value_heads = get_property("n_key_value_heads", "attn-only-1l")
    assert n_key_value_heads is None

    # Test an unknown property
    with pytest.raises(KeyError):
        get_property("unknown_property", "attn-only-1l")


def test_get_model_info():
    get_model_info("attn-only-1l")


def test_render_bridge_models_page_interpolates_registry_constants():
    """Download-free: the bridge-models page template resolves every placeholder
    and sources its status labels from registry_io.STATUS_LABELS."""
    import json

    from docs.make_docs import render_bridge_models_page
    from transformer_lens.tools.model_registry.registry_io import STATUS_LABELS

    page = render_bridge_models_page()
    for placeholder in ("__STATUS_MAP_JSON__", "__STATUS_OPTIONS__", "__OFFICIAL_ORGS_JSON__"):
        assert placeholder not in page
    assert f"const SM = {json.dumps(STATUS_LABELS, sort_keys=True)};" in page
    # Curated select order: Verified, Provisional, Unverified, Failed; no
    # option for status 2 (it shares the "Unverified" label with status 0).
    assert (
        '<option value="1">Verified</option><option value="4">Provisional</option>'
        '<option value="0">Unverified</option><option value="3">Failed</option>' in page
    )
    # Status 2 rows render with the s0 badge, so no .bt-s2 CSS rule exists.
    assert "m.status===2?0" in page
    assert "#bt-root .bt-s2" not in page
