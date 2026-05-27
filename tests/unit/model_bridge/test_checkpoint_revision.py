"""Unit tests for the bridge revision/checkpoint API (issue #453)."""

from unittest.mock import patch

import pytest

from transformer_lens.model_bridge.sources.transformers import (
    _CHECKPOINT_REVISION_FORMATS,
    _resolve_checkpoint_to_revision,
)


class TestResolveCheckpointToRevision:
    def test_pythia_index_resolves_to_step_revision(self):
        labels = [0, 1000, 3000, 10000]
        with patch(
            "transformer_lens.loading_from_pretrained.get_checkpoint_labels",
            return_value=(labels, "step"),
        ):
            revision = _resolve_checkpoint_to_revision(
                "EleutherAI/pythia-70m", checkpoint_index=2, checkpoint_value=None
            )
        assert revision == "step3000"

    def test_pythia_value_resolves_to_step_revision(self):
        labels = [0, 1000, 3000, 10000]
        with patch(
            "transformer_lens.loading_from_pretrained.get_checkpoint_labels",
            return_value=(labels, "step"),
        ):
            revision = _resolve_checkpoint_to_revision(
                "EleutherAI/pythia-70m", checkpoint_index=None, checkpoint_value=10000
            )
        assert revision == "step10000"

    def test_stanford_crfm_uses_checkpoint_prefix(self):
        labels = [100, 200, 400]
        with patch(
            "transformer_lens.loading_from_pretrained.get_checkpoint_labels",
            return_value=(labels, "step"),
        ):
            revision = _resolve_checkpoint_to_revision(
                "stanford-crfm/alias-gpt2-small-x21", checkpoint_index=1, checkpoint_value=None
            )
        assert revision == "checkpoint-200"

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="known checkpoint revision convention"):
            _resolve_checkpoint_to_revision("gpt2", checkpoint_index=0, checkpoint_value=None)

    def test_index_out_of_range_raises(self):
        labels = [0, 1000]
        with patch(
            "transformer_lens.loading_from_pretrained.get_checkpoint_labels",
            return_value=(labels, "step"),
        ):
            with pytest.raises(ValueError, match="out of range"):
                _resolve_checkpoint_to_revision(
                    "EleutherAI/pythia-70m", checkpoint_index=5, checkpoint_value=None
                )

    def test_unknown_value_raises(self):
        labels = [0, 1000]
        with patch(
            "transformer_lens.loading_from_pretrained.get_checkpoint_labels",
            return_value=(labels, "step"),
        ):
            with pytest.raises(ValueError, match="not in available checkpoints"):
                _resolve_checkpoint_to_revision(
                    "EleutherAI/pythia-70m", checkpoint_index=None, checkpoint_value=99999
                )

    def test_neither_provided_raises(self):
        with pytest.raises(ValueError, match="Must specify"):
            _resolve_checkpoint_to_revision(
                "EleutherAI/pythia-70m", checkpoint_index=None, checkpoint_value=None
            )

    def test_known_families_registered(self):
        assert "EleutherAI/pythia" in _CHECKPOINT_REVISION_FORMATS
        assert "stanford-crfm" in _CHECKPOINT_REVISION_FORMATS


class _AbortBoot(Exception):
    """Raised by the model-load patch to short-circuit ``boot()`` before any real load."""


class TestBootRevisionPlumbing:
    """Verify that ``revision`` and ``checkpoint_*`` reach HF's from_pretrained calls.

    Uses pythia-70m's real cached config (avoids MagicMock fragility through the
    adapter/config-mapping path) and aborts at the model-load step.
    """

    def _patched_boot(self, **boot_kwargs):
        from transformer_lens.model_bridge.sources import transformers as bridge_src

        captured: dict = {}
        real_autoconfig = bridge_src.AutoConfig.from_pretrained

        def capture_autoconfig(name, **kwargs):
            captured["autoconfig_kwargs"] = dict(kwargs)
            # Strip the (possibly fake) revision so the real call hits the CI cache.
            kwargs.pop("revision", None)
            return real_autoconfig(name, **kwargs)

        def capture_model_load(*args, **kwargs):
            captured["model_kwargs"] = kwargs
            raise _AbortBoot()

        with patch.object(
            bridge_src.AutoConfig, "from_pretrained", side_effect=capture_autoconfig
        ), patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            side_effect=capture_model_load,
        ):
            with pytest.raises(_AbortBoot):
                bridge_src.boot(model_name="EleutherAI/pythia-70m", device="cpu", **boot_kwargs)

        return captured

    def test_revision_forwarded_to_autoconfig(self):
        captured = self._patched_boot(revision="step3000")
        assert captured["autoconfig_kwargs"].get("revision") == "step3000"

    def test_revision_forwarded_to_model_load(self):
        captured = self._patched_boot(revision="step3000")
        assert captured.get("model_kwargs", {}).get("revision") == "step3000"

    def test_checkpoint_index_resolves_to_revision(self):
        labels = [0, 1000, 3000, 10000]
        with patch(
            "transformer_lens.loading_from_pretrained.get_checkpoint_labels",
            return_value=(labels, "step"),
        ):
            captured = self._patched_boot(checkpoint_index=2)
        assert captured["autoconfig_kwargs"].get("revision") == "step3000"
        assert captured.get("model_kwargs", {}).get("revision") == "step3000"

    def test_conflicting_revision_and_checkpoint_raises(self):
        from transformer_lens.model_bridge.sources import transformers as bridge_src

        with pytest.raises(ValueError, match="not both"):
            bridge_src.boot(
                model_name="EleutherAI/pythia-70m",
                revision="step1000",
                checkpoint_index=2,
            )

    def test_default_revision_is_none(self):
        """With no revision/checkpoint args, revision is not added to model_kwargs."""
        captured = self._patched_boot()
        assert captured["autoconfig_kwargs"].get("revision") is None
        assert "revision" not in captured.get("model_kwargs", {})


class TestHookedTransformerCheckpointLabelAlias:
    def test_checkpoint_label_routes_to_checkpoint_value(self):
        from transformer_lens import HookedTransformer

        with patch("transformer_lens.loading.get_pretrained_model_config") as mock_get_cfg:
            mock_get_cfg.side_effect = RuntimeError("stop after config call")
            with pytest.raises(RuntimeError, match="stop after config call"):
                HookedTransformer.from_pretrained("EleutherAI/pythia-70m", checkpoint_label=3000)

        _, kwargs = mock_get_cfg.call_args
        assert kwargs["checkpoint_value"] == 3000

    def test_checkpoint_label_and_value_together_raises(self):
        from transformer_lens import HookedTransformer

        with pytest.raises(ValueError, match="aliases"):
            HookedTransformer.from_pretrained(
                "EleutherAI/pythia-70m", checkpoint_label=3000, checkpoint_value=1000
            )
