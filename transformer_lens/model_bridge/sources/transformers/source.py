"""``boot`` — load a model via HuggingFace ``transformers`` and wrap it in a TransformerBridge."""
from __future__ import annotations

import contextlib
import copy
import logging
import os
import warnings
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
    detect_tokenizer_bos_eos,
)
from transformer_lens.model_bridge.sources._hf_format import (
    determine_architecture_from_hf_config,
    setup_tokenizer,
)
from transformer_lens.tools.model_registry.registry_io import resolve_model_alias
from transformer_lens.utilities import get_device

from .helpers import (
    _resolve_checkpoint_to_revision,
    get_hf_model_class_for_architecture,
)

# Suppress transformers warnings that go to stderr; otherwise notebook tests fail
# on unexpected stderr output.
warnings.filterwarnings("ignore", message=".*generation flags.*not valid.*")
logging.getLogger("transformers").setLevel(logging.ERROR)


def boot(
    model_name: str,
    hf_config_overrides: dict | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    tokenizer: PreTrainedTokenizerBase | None = None,
    load_weights: bool = True,
    trust_remote_code: bool = False,
    model_class: Any | None = None,
    hf_model: Any | None = None,
    n_ctx: int | None = None,
    revision: str | None = None,
    checkpoint_index: int | None = None,
    checkpoint_value: int | None = None,
    # Multi-device placement (accelerate-dispatched). GPU-validated 2026-07-16:
    # tests/acceptance/model_bridge/test_bridge_multigpu*.py + scripts/bridge_multi_device_parity.py.
    device_map: str | dict[str, str | int] | None = None,
    n_devices: int | None = None,
    max_memory: dict[str | int, str | int] | None = None,
) -> TransformerBridge:
    """Boot a model from HuggingFace (exposed as ``TransformerBridge.boot_transformers``).

    Returns raw HF weights by default — logits/activations match HF, *not*
    legacy ``HookedTransformer`` (which folds LayerNorm + centers weights).
    Call ``enable_compatibility_mode()`` on the result for HookedTransformer-
    equivalent numerics. Generation, argmax, and CE loss are unaffected.

    Attention implementation is forced to ``"eager"`` so hooks can capture scores
    and patterns. For an apples-to-apples HF comparison, load the HF model with
    ``attn_implementation="eager"`` too; comparing against the default ``"sdpa"``
    shows ~1e-3 fp32 drift from kernel-level op reordering, not a bridge bug.

    Args:
        model_name: The name of the model to load.
        hf_config_overrides: Optional overrides applied to the HuggingFace config before model load.
        device: The device to use. If None, will be determined automatically. Mutually exclusive
            with ``device_map``.
        dtype: The dtype to use for the model.
        tokenizer: Optional pre-initialized tokenizer to use; if not provided one will be created.
        load_weights: If False, load model without weights (on meta device) for config inspection only.
        model_class: Optional HuggingFace model class to use instead of the default auto-detected
            class. When the class name matches a key in SUPPORTED_ARCHITECTURES, the corresponding
            adapter is selected automatically (e.g., BertForNextSentencePrediction).
        hf_model: Optional pre-loaded HuggingFace model to use instead of loading one. Useful for
            models loaded with custom configurations (e.g., quantization via BitsAndBytesConfig).
            When provided, load_weights is ignored.
        device_map: HuggingFace-style device map (``"auto"``, ``"balanced"``, dict, etc.) for
            dispatched inference. Explicit maps may include CPU targets; disk / meta offload
            targets are still rejected because Bridge component wrappers need additional
            offload-hook routing work. Mutually exclusive with ``device``.
        n_devices: Convenience: split the model across this many CUDA devices (translated to a
            ``max_memory`` dict internally). Requires CUDA with at least this many visible devices.
        max_memory: Optional per-device memory budget for HF's dispatcher.
        n_ctx: Optional context length override. The bridge normally uses the model's documented
            max context from the HF config. Setting this writes to whichever HF field the model
            uses (n_positions / max_position_embeddings / etc.), so callers don't need to know
            the field name. If larger than the model's default, a warning is emitted — quality
            may degrade past the trained length for rotary models.
        revision: Optional HF revision string (branch, tag, or commit). Forwarded to
            config, model, and tokenizer loading.
            Mutually exclusive with ``checkpoint_index`` and ``checkpoint_value``.
        checkpoint_index: Index into the available training checkpoints for the model family.
            Convenience over ``revision`` for checkpointed models like EleutherAI/pythia* and
            stanford-crfm/*. Resolved to a revision string via the known per-family naming
            conventions (``step{value}`` for Pythia, ``checkpoint-{value}`` for stanford-crfm).
        checkpoint_value: Training step or token count of the desired checkpoint. Alternative to
            ``checkpoint_index``; must be one of the labels returned by ``get_checkpoint_labels``.

    Returns:
        The bridge to the loaded model.
    """
    official_name = resolve_model_alias(model_name)
    if official_name is not None:
        logging.warning(
            f"DEPRECATED: You are using a deprecated, model_name alias '{model_name}'. TransformerLens will now load the official transformers model name, '{official_name}' instead.\n Please update your code to use the official name by changing model_name from '{model_name}' to '{official_name}'.\nSince TransformerLens v3, all model names should be the official transformers model names.\nThe aliases will be removed in the next version of TransformerLens, so please do the update now."
        )
        model_name = official_name
    if checkpoint_index is not None or checkpoint_value is not None:
        if revision is not None:
            raise ValueError(
                "Specify either revision= or checkpoint_index/checkpoint_value, not both."
            )
        revision = _resolve_checkpoint_to_revision(model_name, checkpoint_index, checkpoint_value)
    # Pass HF token for gated model access (e.g. meta-llama/*)
    from transformer_lens.utilities.hf_utils import get_hf_token

    _hf_token = get_hf_token()
    if hf_model is not None:
        # Reuse the pre-loaded model's config to avoid a Hub call when model_name
        # is a Hub repo ID but the model is already loaded locally.
        hf_config = copy.deepcopy(hf_model.config)
    else:
        hf_config = AutoConfig.from_pretrained(
            model_name,
            output_attentions=True,
            trust_remote_code=trust_remote_code,
            token=_hf_token,
            revision=revision,
        )
    _n_ctx_field: str | None = None
    if n_ctx is not None:
        if n_ctx <= 0:
            raise ValueError(f"n_ctx must be a positive integer, got n_ctx={n_ctx}.")
        # Resolve n_ctx to whichever HF config field this model uses. Mirrors the order in
        # map_default_transformer_lens_config so the TL config derivation picks up the override.
        for _field in (
            "n_positions",
            "max_position_embeddings",
            "max_context_length",
            "max_length",
            "seq_length",
            "max_sequence_length",
        ):
            if hasattr(hf_config, _field):
                _n_ctx_field = _field
                break
        if _n_ctx_field is None:
            raise ValueError(
                f"Cannot apply n_ctx={n_ctx}: no recognized context-length field on "
                f"HF config for {model_name}. Use hf_config_overrides instead."
            )
        _default_n_ctx = getattr(hf_config, _n_ctx_field)
        if _default_n_ctx is not None and n_ctx > _default_n_ctx:
            logging.warning(
                "Setting n_ctx=%d which is larger than the model's default "
                "context length of %d. The model was not trained on sequences "
                "this long and may produce unreliable results (especially for "
                "rotary models without RoPE scaling).",
                n_ctx,
                _default_n_ctx,
            )
        # Warn if the caller also set the same field via hf_config_overrides — explicit n_ctx wins.
        if hf_config_overrides and _n_ctx_field in hf_config_overrides:
            _conflicting_value = hf_config_overrides[_n_ctx_field]
            if _conflicting_value != n_ctx:
                logging.warning(
                    "Both n_ctx=%d and hf_config_overrides['%s']=%s were provided. "
                    "The explicit n_ctx takes precedence.",
                    n_ctx,
                    _n_ctx_field,
                    _conflicting_value,
                )
        hf_config_overrides = dict(hf_config_overrides or {})
        hf_config_overrides[_n_ctx_field] = n_ctx
    if hf_config_overrides:
        hf_config.__dict__.update(hf_config_overrides)
    architecture = determine_architecture_from_hf_config(hf_config)
    bridge_config = build_bridge_config_from_hf(hf_config, architecture, model_name, dtype)
    bridge_config.trust_remote_code = trust_remote_code
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_config)
    # Pre-loaded models carry their own weight placement (possibly set by the caller via
    # device_map). Passing device_map / n_devices / max_memory alongside hf_model= is ambiguous
    # and would silently be ignored, so fail loudly.
    if hf_model is not None and (
        device_map is not None or n_devices is not None or max_memory is not None
    ):
        raise ValueError(
            "device_map / n_devices / max_memory are only supported when the bridge loads "
            "the HF model itself. When passing hf_model=..., apply device_map via "
            "AutoModel.from_pretrained before handing the model to the bridge."
        )
    # Stateful/SSM (Mamba) models keep a per-layer recurrent cache that must live on that
    # layer's device. The bridge allocates the stateful cache on a single cfg.device, so
    # cross-device splits would silently misplace the cache. Blocked until v2.
    if (n_devices is not None and n_devices > 1) or device_map is not None:
        if getattr(bridge_config, "is_stateful", False):
            raise ValueError(
                "Multi-device splits are not yet supported for stateful (SSM / Mamba) "
                "architectures: the stateful cache allocation is single-device. "
                "Load on one device, or wait for v2 support."
            )
    # Resolve device_map before defaulting `device` — the two are mutually exclusive and the
    # resolver raises on conflict. If n_devices>1 is passed it's translated into a device_map +
    # max_memory pair here so downstream code only needs to check the resolved values.
    from transformer_lens.utilities.multi_gpu import (
        MIXED_CPU_GPU_ERROR,
        cast_floating_params_to_dtype,
        count_unique_devices,
        find_embedding_device,
        find_misplaced_modules,
        is_mixed_cpu_gpu,
        resolve_device_map,
    )

    resolved_device_map, resolved_max_memory = resolve_device_map(
        n_devices, device_map, device, max_memory
    )
    if resolved_device_map is None:
        if device is None:
            device = get_device()
        adapter.cfg.device = str(device)
    else:
        # cfg.device set from hf_device_map after the model is loaded; provisionally None.
        adapter.cfg.device = None
    if model_class is None:
        model_class = get_hf_model_class_for_architecture(architecture)
    # Ensure pad_token_id exists (v5 raises AttributeError if missing).
    if not hasattr(hf_config, "pad_token_id") or "pad_token_id" not in hf_config.__dict__:
        fallback_pad = getattr(hf_config, "eos_token_id", None)
        # eos_token_id can be a list (Gemma3 uses [1, 106]); take the first.
        if isinstance(fallback_pad, list):
            fallback_pad = fallback_pad[0] if fallback_pad else None
        hf_config.pad_token_id = fallback_pad
    model_kwargs = {"config": hf_config, "torch_dtype": dtype}
    if _hf_token:
        model_kwargs["token"] = _hf_token
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    if revision is not None:
        model_kwargs["revision"] = revision
    if resolved_device_map is not None:
        model_kwargs["device_map"] = resolved_device_map
    if resolved_max_memory is not None:
        model_kwargs["max_memory"] = resolved_max_memory
    if hasattr(adapter.cfg, "attn_implementation") and adapter.cfg.attn_implementation is not None:
        model_kwargs["attn_implementation"] = adapter.cfg.attn_implementation
    else:
        # Eager is required for output_attentions hooks.
        model_kwargs["attn_implementation"] = "eager"
    adapter.prepare_loading(model_name, model_kwargs)
    # Meta device_map targets crash at boot when loading weights
    # (NotImplementedError in HF tie_weights, KeyError in Accelerate offload hooks).
    # Only accepted with load_weights=False (config inspection; map not applied).
    if load_weights and isinstance(resolved_device_map, dict):
        _meta_targets = [
            k
            for k, v in resolved_device_map.items()
            if isinstance(v, str) and v.strip().lower() == "meta"
        ]
        if _meta_targets:
            raise ValueError(
                f"device_map contains meta target(s): {_meta_targets}. "
                "Meta device_map values crash at boot when loading weights. "
                "Set load_weights=False for config inspection only "
                "(the map is not applied; parameters load on CPU via from_config)."
            )
    if hf_model is not None:
        # Use the pre-loaded model as-is (quantized models with custom device_map, etc).
        pass
    elif not load_weights:
        from_config_kwargs = {}
        if trust_remote_code:
            from_config_kwargs["trust_remote_code"] = True
        # adapter.prepare_loading may have replaced model_kwargs["config"] (e.g. Qwen3.5
        # text-only extraction); honor that here so the no-weights path uses the
        # same config the load-weights path would.
        prepared_config = model_kwargs.get("config", hf_config)
        with contextlib.redirect_stdout(None):
            hf_model = model_class.from_config(prepared_config, **from_config_kwargs)
    else:
        try:
            hf_model = model_class.from_pretrained(model_name, **model_kwargs)
        except RuntimeError as e:
            # HF refuses to load when positional-weight shapes don't match. If the user
            # requested an n_ctx that conflicts with the saved weights (common for
            # learned-pos-embed models like GPT-2), re-raise with a clearer message.
            if n_ctx is not None and "ignore_mismatched_sizes" in str(e):
                raise RuntimeError(
                    f"Failed to load {model_name} with n_ctx={n_ctx}: the pretrained "
                    f"weights' positional-embedding shape does not match the requested "
                    f"context length. This affects models with learned positional "
                    f"embeddings (e.g. GPT-2, OPT). Options: (1) use the model's "
                    f"default n_ctx, (2) pass load_weights=False if you only need "
                    f"config inspection, or (3) choose a rotary-embedding model "
                    f"(e.g. Llama, Mistral) which supports n_ctx changes without "
                    f"weight mismatch."
                ) from e
            raise
        # Skip explicit .to(device) when accelerate has placed weights via device_map.
        if resolved_device_map is None and device is not None:
            hf_model = hf_model.to(device)
        # Cast params to dtype; preserve float32 buffers (e.g. RotaryEmbedding.inv_freq).
        # Use module-level alignment so Accelerate can temporarily materialize offloaded
        # parameters before we touch them.
        cast_floating_params_to_dtype(hf_model, dtype)
    # Derive cfg.device / cfg.n_devices from hf_device_map when present. Covers fresh loads
    # with a resolved device_map AND pre-loaded models with caller-dispatched device_map="auto".
    hf_device_map_post = getattr(hf_model, "hf_device_map", None)
    if hf_device_map_post:
        # All-CPU placement is supported (real parameters, no offload). Disk / meta —
        # and CPU entries in a MIXED map, which accelerate implements as CPU offload —
        # are rejected: offload materializes weights via forward hooks that wrapped
        # Bridge components bypass (e.g. NormalizationBridge computes from raw params).
        offload_values = {str(v).lower() for v in hf_device_map_post.values() if isinstance(v, str)}
        unsupported = offload_values & {"disk", "meta"}
        if unsupported:
            raise ValueError(
                f"hf_device_map contains unsupported offload targets: {sorted(unsupported)}. "
                "TransformerBridge currently supports CPU device_map targets, but disk / meta "
                "offload can bypass Accelerate hooks inside wrapped Bridge components."
            )
        if is_mixed_cpu_gpu(hf_device_map_post.values()):
            raise ValueError(f"Realized hf_device_map is unsupported: {MIXED_CPU_GPU_ERROR}")
        if (
            "cpu" in offload_values
            and device_map is None
            and n_devices is not None
            and n_devices > 1
        ):
            raise ValueError(
                "hf_device_map contains CPU targets. n_devices is GPU-only; pass device_map "
                "explicitly for CPU placement."
            )
        misplaced = find_misplaced_modules(hf_model)
        if misplaced:
            details = "; ".join(
                f"{name!r} mapped to {mapped} but loaded on {actual}"
                for name, mapped, actual in misplaced
            )
            raise ValueError(
                f"device_map entries were not honored: {details}. This usually means the "
                "map splits tied parameters (e.g. GPT-2's wte/lm_head share one tensor) "
                "across devices — accelerate places a tied parameter once, leaving a module "
                "executing on a device its weights aren't on, which crashes mid-forward. "
                "Map tied modules to the same device."
            )
    embedding_device = find_embedding_device(hf_model)
    if embedding_device is not None:
        adapter.cfg.device = str(embedding_device)
        adapter.cfg.n_devices = count_unique_devices(hf_model)
    elif adapter.cfg.device is None:
        # Pre-loaded single-device model with no hf_device_map — fall back to first param.
        try:
            adapter.cfg.device = str(next(hf_model.parameters()).device)
        except StopIteration:
            adapter.cfg.device = "cpu"
    # Verify the n_ctx override actually took effect on the loaded model. If HF's config class
    # silently dropped or normalized the value, warn so the user isn't misled.
    if n_ctx is not None and _n_ctx_field is not None and hf_model is not None:
        _actual = getattr(hf_model.config, _n_ctx_field, None)
        if _actual != n_ctx:
            logging.warning(
                "n_ctx=%d was requested but hf_model.config.%s=%s after load. "
                "The override may not have taken effect; the model may not "
                "accept sequences longer than %s.",
                n_ctx,
                _n_ctx_field,
                _actual,
                _actual,
            )
    adapter.prepare_model(hf_model)
    default_padding_side = getattr(adapter.cfg, "default_padding_side", None)
    use_fast = getattr(adapter.cfg, "use_fast", True)
    # Audio models use feature extractors, not text tokenizers.
    _is_audio = getattr(adapter.cfg, "is_audio_model", False)
    if _is_audio and tokenizer is None:
        tokenizer = None
    elif tokenizer is not None:
        tokenizer = setup_tokenizer(tokenizer, default_padding_side=default_padding_side)
    else:
        token_arg = get_hf_token()
        # Some adapters override tokenizer source (e.g. OpenELM has no tokenizer of its own).
        tokenizer_source = model_name
        if hasattr(adapter.cfg, "tokenizer_name") and adapter.cfg.tokenizer_name is not None:
            tokenizer_source = adapter.cfg.tokenizer_name
        # Encoder-decoder models like T5 don't have a BOS token and raise on add_bos_token=True.
        try:
            base_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                add_bos_token=True,
                use_fast=use_fast,
                token=token_arg,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )
        except ValueError:
            base_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                use_fast=use_fast,
                token=token_arg,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )
        tokenizer = setup_tokenizer(
            base_tokenizer,
            default_padding_side=default_padding_side,
        )
    if tokenizer is not None:
        (
            adapter.cfg.tokenizer_prepends_bos,
            adapter.cfg.tokenizer_appends_eos,
        ) = detect_tokenizer_bos_eos(tokenizer)
    from transformer_lens.model_bridge.sources.transformers_driver import (
        TransformersDriver,
    )

    driver = TransformersDriver(hf_model, adapter, tokenizer)
    bridge = TransformerBridge(hf_model, adapter, tokenizer, driver=driver)

    # Multimodal models: load the image preprocessor.
    if getattr(adapter.cfg, "is_multimodal", False):
        try:
            from transformers import AutoProcessor

            huggingface_token = os.environ.get("HF_TOKEN", "")
            token_arg = huggingface_token if len(huggingface_token) > 0 else None
            bridge.processor = AutoProcessor.from_pretrained(
                model_name,
                token=token_arg,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            # Some processors need torchvision (e.g. LlavaOnevision); install if missing.
            _torchvision_available = False
            try:
                import torchvision  # noqa: F401

                _torchvision_available = True
            except Exception:
                import shutil
                import subprocess
                import sys

                try:
                    if shutil.which("uv"):
                        subprocess.check_call(
                            ["uv", "pip", "install", "torchvision", "-q"],
                        )
                    else:
                        subprocess.check_call(
                            [sys.executable, "-m", "pip", "install", "torchvision", "-q"],
                        )
                    import importlib

                    importlib.invalidate_caches()
                    _torchvision_available = True
                except Exception:
                    pass

            if _torchvision_available:
                try:
                    from transformers import AutoProcessor

                    huggingface_token = os.environ.get("HF_TOKEN", "")
                    token_arg = huggingface_token if len(huggingface_token) > 0 else None
                    bridge.processor = AutoProcessor.from_pretrained(
                        model_name,
                        token=token_arg,
                        trust_remote_code=trust_remote_code,
                    )
                except Exception:
                    pass

    # Audio models: load the feature extractor.
    if getattr(adapter.cfg, "is_audio_model", False):
        try:
            from transformers import AutoFeatureExtractor

            huggingface_token = os.environ.get("HF_TOKEN", "")
            token_arg = huggingface_token if len(huggingface_token) > 0 else None
            bridge.processor = AutoFeatureExtractor.from_pretrained(
                model_name,
                token=token_arg,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            pass

    return bridge
