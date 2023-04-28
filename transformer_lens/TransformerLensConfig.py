from typing import Protocol


class TransformerLensConfig(Protocol):
    model_type: str

    def from_dict():
        ...

    def to_dict():
        ...
