from transformer_lens.utilities.hf_utils import (
    download_file_from_hf,
    clear_huggingface_cache,
    keep_single_column,
    get_dataset,
    select_compatible_kwargs,
)
from transformer_lens.utilities.gpu_utils import print_gpu_mem, get_device
from transformer_lens.utilities.tensor_utils import (
    get_corner,
    to_numpy,
    remove_batch_dim,
    transpose,
    is_square,
    is_lower_triangular,
    check_structure,
    composition_scores,
    get_offset_position_ids,
    get_cumsum_along_dim,
    repeat_along_head_dimension,
)
from transformer_lens.utilities.lm_utils import lm_cross_entropy_loss, lm_accuracy
from transformer_lens.utilities.activation_utils import gelu_new, gelu_fast, solu
from transformer_lens.utilities.initialization_utils import (
    calc_fan_in_and_fan_out,
    init_xavier_uniform_,
    init_xavier_normal_,
    init_kaiming_uniform_,
    init_kaiming_normal_,
)
from transformer_lens.utilities.tokenize_utils import (
    tokenize_and_concatenate,
    get_tokenizer_with_bos,
    get_input_with_manually_prepended_bos,
    get_tokens_with_bos_removed,
    get_attention_mask,
)
from transformer_lens.utilities.logits_utils import sample_logits
from transformer_lens.utilities.slice import SliceInput, Slice
from transformer_lens.utilities.components_utils import get_act_name
from transformer_lens.utilities.attribute_utils import get_nested_attr, set_nested_attr
from transformer_lens.utilities.defaults_utils import (
    override_or_use_default_value,
    LocallyOverridenDefaults,
    USE_DEFAULT_VALUE,
)
from transformer_lens.utilities.exploratory_utils import test_prompt
