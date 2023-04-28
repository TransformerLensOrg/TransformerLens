from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class TransformerLensConfig(ABC):
    @classmethod
    @abstractmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> TransformerLensConfig:
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass
