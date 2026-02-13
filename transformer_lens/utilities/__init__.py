from .activation_functions import (
    SUPPORTED_ACTIVATIONS,
    ActivationFunction,
    gelu_fast,
    gelu_new,
    solu,
)
from .attribute_utils import get_nested_attr, set_nested_attr
from .components_utils import get_act_name
from .defaults_utils import (
    USE_DEFAULT_VALUE,
    LocallyOverridenDefaults,
    override_or_use_default_value,
)
from .devices import get_device, move_to_and_update_config
from .exploratory_utils import test_prompt
from .gpu_utils import print_gpu_mem
from .hf_utils import (
    clear_huggingface_cache,
    download_file_from_hf,
    get_dataset,
    get_rotary_pct_from_config,
    keep_single_column,
    select_compatible_kwargs,
)
from .initialization_utils import (
    NonlinearityType,
    calc_fan_in_and_fan_out,
    init_kaiming_normal_,
    init_kaiming_uniform_,
    init_xavier_normal_,
    init_xavier_uniform_,
)
from .library_utils import is_library_available
from .lm_utils import lm_accuracy, lm_cross_entropy_loss
from .logits_utils import sample_logits
from .matrix import (
    composition_scores,
    get_matrix_corner,
)

# Re-export multi-GPU helpers here (devices.py must not import multi_gpu directly)
from .multi_gpu import (
    calculate_available_device_cuda_memory,
    determine_available_memory_for_available_devices,
    get_best_available_cuda_device,
    get_best_available_device,
    get_device_for_block_index,
    sort_devices_based_on_available_memory,
)
from .slice import Slice, SliceInput
from .tensors import (
    check_structure,
    filter_dict_by_prefix,
    get_corner,
    get_cumsum_along_dim,
    get_offset_position_ids,
    is_lower_triangular,
    is_square,
    remove_batch_dim,
    repeat_along_head_dimension,
    to_numpy,
    transpose,
)
from .tokenize_utils import (
    get_attention_mask,
    get_input_with_manually_prepended_bos,
    get_tokenizer_with_bos,
    get_tokens_with_bos_removed,
    tokenize_and_concatenate,
)
