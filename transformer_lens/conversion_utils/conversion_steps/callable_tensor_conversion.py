from collections.abc import Callable

from .base_tensor_conversion import BaseTensorConversion


class CallableTensorConversion(BaseTensorConversion):
    def __init__(self, convert_callable: Callable):
        super().__init__()
        self.convert_callable = convert_callable

    def handle_conversion(self, input_value: dict, *full_context):
        return self.convert_callable(input_value, *full_context)

    def __repr__(self):
        return f"Is the following callable operation: {self.convert_callable}"
