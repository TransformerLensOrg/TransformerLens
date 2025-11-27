"""utils.

This module is deprecated, but imports from the new utilities to maintain backwards compatibility
"""
import warnings

from transformer_lens.utilities import *

warnings.warn(
    "The 'utils' module has been deprecated. Please use 'transformer_lens.utilities' instead. Importing from utils.py will be removed in TransformerLens 4.0.",
    DeprecationWarning,
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
]
