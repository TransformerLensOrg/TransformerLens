"""Architecture adapter conversion steps.

This module contains the conversion steps for converting between different model architectures.
"""

from transformer_lens.conversion_utils.conversion_steps.arithmetic_hook_conversion import (
    ArithmeticHookConversion,
)
from transformer_lens.conversion_utils.conversion_steps.attention_auto_conversion import (
    AttentionAutoConversion,
)
from transformer_lens.conversion_utils.conversion_steps.base_hook_conversion import (
    BaseHookConversion,
)
from transformer_lens.conversion_utils.conversion_steps.callable_hook_conversion import (
    CallableHookConversion,
)
from transformer_lens.conversion_utils.conversion_steps.hook_conversion_set import (
    HookConversionSet,
)
from transformer_lens.conversion_utils.conversion_steps.rearrange_hook_conversion import (
    RearrangeHookConversion,
)
from transformer_lens.conversion_utils.conversion_steps.repeat_hook_conversion import (
    RepeatHookConversion,
)
from transformer_lens.conversion_utils.conversion_steps.split_hook_conversion import (
    SplitHookConversion,
)
from transformer_lens.conversion_utils.conversion_steps.ternary_hook_conversion import (
    TernaryHookConversion,
)
from transformer_lens.conversion_utils.conversion_steps.zeros_like_conversion import (
    ZerosLikeConversion,
)

__all__ = [
    "ArithmeticHookConversion",
    "AttentionAutoConversion",
    "BaseHookConversion",
    "CallableHookConversion",
    "RearrangeHookConversion",
    "RepeatHookConversion",
    "SplitHookConversion",
    "TernaryHookConversion",
    "HookConversionSet",
    "ZerosLikeConversion",
]
