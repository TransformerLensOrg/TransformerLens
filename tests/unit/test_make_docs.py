"""Make Docs Tests."""


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


def test_copy_demos_creates_generated_dir_when_absent(tmp_path, monkeypatch):
    """copy_demos() is the first step of build_docs() on a clean checkout, where the
    gitignored generated/ directory does not exist yet. Copies from the real demos/ so a
    renamed notebook fails here rather than in the docs job."""
    from docs import make_docs

    generated = tmp_path / "generated"
    monkeypatch.setattr(make_docs, "GENERATED_DIR", generated)
    assert not generated.exists()

    make_docs.copy_demos()

    copied = sorted(p.name for p in (generated / "demos").iterdir())
    assert copied == [
        "Exploratory_Analysis_Demo.ipynb",
        "Jacobian_Lens_Coordinate_Patch_Benchmark_Demo.ipynb",
        "Jacobian_Lens_Decomposition_Demo.ipynb",
        "Main_Demo.ipynb",
    ]
