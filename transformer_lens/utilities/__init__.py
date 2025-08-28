from .devices import (
    get_device,
    get_device_for_block_index,
    move_to_and_update_config,
)
from .hf_utils import (
    download_file_from_hf,
    clear_huggingface_cache,
    keep_single_column,
    get_dataset,
    select_compatible_kwargs,
)
from .gpu_utils import print_gpu_mem
from .tensors import (
    to_numpy,
    remove_batch_dim,
    transpose,
    is_square,
    is_lower_triangular,
    check_structure,
    get_offset_position_ids,
    get_cumsum_along_dim,
    repeat_along_head_dimension,
    get_corner,
)
from .lm_utils import lm_cross_entropy_loss, lm_accuracy
from .activation_functions import (
    gelu_new,
    gelu_fast,
    solu,
    ActivationFunction,
    SUPPORTED_ACTIVATIONS,
)
from .initialization_utils import (
    calc_fan_in_and_fan_out,
    init_xavier_uniform_,
    init_xavier_normal_,
    init_kaiming_uniform_,
    init_kaiming_normal_,
)
from .tokenize_utils import (
    tokenize_and_concatenate,
    get_tokenizer_with_bos,
    get_input_with_manually_prepended_bos,
    get_tokens_with_bos_removed,
    get_attention_mask,
)
from .matrix import (
    composition_scores,
    get_matrix_corner,
)
from .logits_utils import sample_logits
from .slice import SliceInput, Slice
from .components_utils import get_act_name
from .attribute_utils import get_nested_attr, set_nested_attr
from .defaults_utils import (
    override_or_use_default_value,
    LocallyOverridenDefaults,
    USE_DEFAULT_VALUE,
)
from .exploratory_utils import test_prompt
