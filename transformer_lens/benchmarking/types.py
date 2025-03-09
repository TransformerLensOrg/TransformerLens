from typing import TypedDict
from transformer_lens.weight_conversion.conversion_utils.conversion_steps.types import WeightConversionInterface

class BenchMarkFieldResult(TypedDict):
    transformer_lens_name: str
    ramote_name: str
    diff: float
    conversion: WeightConversionInterface | None
    
    
LAYER_RESULT = list[BenchMarkFieldResult|list]