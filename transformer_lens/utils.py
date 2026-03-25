"""utils.

This module is deprecated, but imports from the new utilities to maintain backwards compatibility.

New code in this file: MPS device safety utilities (get_device, warn_if_mps) that
override the simpler versions in utilities.devices.  These are kept here to avoid
circular imports (utilities.devices imports warn_if_mps from here).
"""

import logging
import os
import warnings

import torch

from transformer_lens.utilities import *  # noqa: F401,F403

# ---------------------------------------------------------------------------
# MPS device safety utilities (added from main)
# ---------------------------------------------------------------------------


def get_device() -> torch.device:  # type: ignore[no-redef]  # intentionally overrides utilities.devices.get_device
    """Get the best available device, with MPS safety checks.

    MPS is only auto-selected when the environment variable
    ``TRANSFORMERLENS_ALLOW_MPS=1`` is set **and** the installed PyTorch
    version meets or exceeds ``_MPS_MIN_SAFE_TORCH_VERSION``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        major_version = int(torch.__version__.split(".")[0])
        if major_version >= 2:
            # Only auto-select MPS when explicitly opted-in via env var
            if os.environ.get("TRANSFORMERLENS_ALLOW_MPS", "") == "1":
                return torch.device("mps")
            logging.info(
                "MPS device available but not auto-selected due to known correctness issues "
                "(PyTorch %s). Set TRANSFORMERLENS_ALLOW_MPS=1 to override. See: "
                "https://github.com/TransformerLensOrg/TransformerLens/issues/1178",
                torch.__version__,
            )

    return torch.device("cpu")


_mps_warned = False

# MPS silent correctness issues are known in PyTorch <= 2.7.
# Bump this when a PyTorch release ships verified MPS fixes.
_MPS_MIN_SAFE_TORCH_VERSION: tuple[int, ...] | None = None


def _torch_version_tuple() -> tuple[int, ...]:
    """Parse torch.__version__ into a comparable tuple, ignoring pre-release suffixes."""
    return tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])


def warn_if_mps(device):
    """Emit a one-time warning if device is MPS and TRANSFORMERLENS_ALLOW_MPS is not set.

    Automatically suppressed when the installed PyTorch version meets or exceeds
    _MPS_MIN_SAFE_TORCH_VERSION (currently unset — no version is considered safe yet).
    """
    global _mps_warned
    if _mps_warned:
        return
    if isinstance(device, torch.device):
        device = device.type
    if isinstance(device, str) and device == "mps":
        if (
            _MPS_MIN_SAFE_TORCH_VERSION is not None
            and _torch_version_tuple() >= _MPS_MIN_SAFE_TORCH_VERSION
        ):
            return
        if os.environ.get("TRANSFORMERLENS_ALLOW_MPS", "") != "1":
            _mps_warned = True
            warnings.warn(
                "MPS backend may produce silently incorrect results (PyTorch "
                f"{torch.__version__}). "
                "Set TRANSFORMERLENS_ALLOW_MPS=1 to suppress this warning. "
                "See: https://github.com/TransformerLensOrg/TransformerLens/issues/1178",
                UserWarning,
                stacklevel=2,
            )


__all__ = [
    "download_file_from_hf",
    "clear_huggingface_cache",
    "keep_single_column",
    "get_dataset",
    "print_gpu_mem",
    "get_device",
    "get_corner",
    "to_numpy",
    "remove_batch_dim",
    "transpose",
    "is_square",
    "is_lower_triangular",
    "check_structure",
    "composition_scores",
    "get_offset_position_ids",
    "get_cumsum_along_dim",
    "repeat_along_head_dimension",
    "filter_dict_by_prefix",
    "lm_cross_entropy_loss",
    "lm_accuracy",
    "gelu_new",
    "gelu_fast",
    "solu",
    "calc_fan_in_and_fan_out",
    "init_xavier_uniform_",
    "init_xavier_normal_",
    "init_kaiming_uniform_",
    "init_kaiming_normal_",
    "is_library_available",
    "tokenize_and_concatenate",
    "get_tokenizer_with_bos",
    "get_input_with_manually_prepended_bos",
    "get_tokens_with_bos_removed",
    "get_attention_mask",
    "sample_logits",
    "SliceInput",
    "Slice",
    "get_act_name",
    "get_nested_attr",
    "set_nested_attr",
    "override_or_use_default_value",
    "LocallyOverridenDefaults",
    "USE_DEFAULT_VALUE",
    "test_prompt",
    "warn_if_mps",
    "_mps_warned",
    "_MPS_MIN_SAFE_TORCH_VERSION",
    "_torch_version_tuple",
]
