"""Per-provider request/response codecs for the Inspect driver.

The driver speaks one internal shape; each provider speaks its own. A Profile
encapsulates everything provider-specific: which hooks it serves, whether it
returns full-sequence logits, how to phrase the ``generate`` request (prompt +
``extra_args``), how to translate interventions, and how to read logits back.
Torch-free (numpy only) so the driver stays torch-free.

``tl_bridge`` is our own HF provider (``provider.py``). ``vllm-lens`` is the
third-party provider. **The vllm-lens codec is written from its documented API and
is NOT verified against a live provider** (CI has none — it needs their GPU-served
provider). It's isolated here so this is the single place to fix once validated.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from . import hooks, intervention, wire


class TLBridgeProfile:
    """Codec for our own ``tl_bridge`` provider: full hooks, full-sequence logits."""

    provides_sequence_logits = True

    def supported_hooks(self, n_layers: int) -> frozenset[str]:
        return hooks.supported_hook_points(n_layers)

    def translate_interventions(self, intervene, supported):
        return intervention.build_interventions(intervene, supported)  # {wire_key: spec}

    def build_request(self, ids, wire_keys, interventions, return_logits, tokenizer):
        extra: dict[str, Any] = {
            "input_ids": ids,
            "capture": wire_keys,
            "return_logits": return_logits,
        }
        if interventions:
            extra["interventions"] = interventions
        return "", extra  # our provider reads input_ids from extra_args; prompt unused

    def decode_logits(self, output, n_tokens, d_vocab, tokenizer):
        entry = (getattr(output, "metadata", None) or {}).get("tl_logits")
        if entry is not None:
            return wire.decode_array(entry)[np.newaxis, ...]  # full (1, seq, d_vocab)
        return np.full((1, n_tokens, d_vocab), -np.inf, dtype=np.float32)


class VLLMLensProfile:
    """Codec for the third-party vllm-lens provider.

    Residual-stream-only, additive-steering-only, and last-token logits synthesized
    one-hot from the generated token (argmax-only — vllm-lens doesn't hand back full
    logits through this path). Prompt is the detokenized text, so vllm-lens
    re-tokenizes it: activations reflect that re-tokenization, which may differ from
    the exact ids. UNVERIFIED against a live provider.
    """

    provides_sequence_logits = False

    def supported_hooks(self, n_layers: int) -> frozenset[str]:
        return frozenset(f"blocks.{i}.hook_resid_post" for i in range(n_layers))

    def translate_interventions(self, intervene: Mapping[str, Any], supported) -> list:
        """op='add' with a width-shaped vector → vllm-lens SteeringVector; others raise.

        Validates before importing ``vllm_lens`` so non-additive ops (and the
        no-intervention case) don't require the package installed.
        """
        if not intervene:
            return []
        steering_cls: Any = None  # imported lazily once a valid additive spec is seen
        vectors = []
        for name, spec in intervene.items():
            if callable(spec):
                raise NotImplementedError("vllm-lens requires intervention specs, not callables.")
            if not isinstance(spec, Mapping) or spec.get("op") != "add":
                raise NotImplementedError(
                    f"vllm-lens supports only additive steering (op='add' with a width vector); "
                    f"got {spec!r} for {name!r}. Use boot_transformers() for suppress/scale/set."
                )
            if name not in supported:
                raise ValueError(f"Cannot intervene on {name!r}: not in supported_hook_points.")
            resolved = hooks.resolve(name)
            assert resolved is not None  # supported ⇒ resolvable
            if steering_cls is None:
                from vllm_lens import SteeringVector

                steering_cls = SteeringVector
            vectors.append(
                steering_cls(
                    activations=np.asarray(spec["value"], dtype=np.float32),
                    layer_indices=[resolved[0]],
                    scale=float(spec.get("scale", 1.0)),
                    norm_match=bool(spec.get("norm_match", True)),
                )
            )
        return vectors

    def build_request(self, ids, wire_keys, interventions, return_logits, tokenizer):
        layers = sorted({int(key.split(":")[0]) for key in wire_keys})
        extra: dict[str, Any] = {"output_residual_stream": layers}
        if interventions:
            extra["apply_steering_vectors"] = interventions
        prompt = tokenizer.decode(ids) if tokenizer is not None else ""
        return prompt, extra

    def decode_logits(self, output, n_tokens, d_vocab, tokenizer):
        # vllm-lens returns no full logits here; one-hot the generated token (argmax only).
        logits = np.full((1, n_tokens, d_vocab), -np.inf, dtype=np.float32)
        text = getattr(output, "completion", "") or ""
        ids = tokenizer.encode(text) if (tokenizer is not None and text) else []
        if len(ids):
            logits[0, -1, int(ids[0])] = 0.0
        return logits


def for_provider(provider: str) -> Any:
    """Pick the codec for a provider name (the part before ``/`` in get_model)."""
    if provider.startswith("vllm-lens"):
        return VLLMLensProfile()
    return TLBridgeProfile()
