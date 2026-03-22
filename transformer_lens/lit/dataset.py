"""LIT Dataset wrapper for TransformerLens.

This module provides LIT-compatible Dataset wrappers for use with TransformerLens
models. It includes utilities for loading common datasets and creating custom
datasets for model analysis.

Example usage:
    >>> from transformer_lens.lit import SimpleTextDataset  # doctest: +SKIP
    >>>
    >>> # Create a dataset from examples
    >>> examples = [  # doctest: +SKIP
    ...     {"text": "The capital of France is Paris."},
    ...     {"text": "Machine learning is a subset of AI."},
    ... ]
    >>> dataset = SimpleTextDataset(examples)  # doctest: +SKIP
    >>>
    >>> # Use with LIT server
    >>> from lit_nlp import dev_server  # doctest: +SKIP
    >>> server = dev_server.Server(models, {"my_data": dataset})  # doctest: +SKIP

References:
    - LIT Dataset API: https://pair-code.github.io/lit/documentation/api#datasets
    - TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from .constants import INPUT_FIELDS
from .utils import check_lit_installed

if TYPE_CHECKING:
    from lit_nlp.api import dataset as lit_dataset_types  # noqa: F401
    from lit_nlp.api import types as lit_types_module  # noqa: F401

# Check for LIT installation
if check_lit_installed():
    from lit_nlp.api import (  # type: ignore[import-not-found]  # noqa: F401
        dataset as lit_dataset,
    )
    from lit_nlp.api import (  # type: ignore[import-not-found]  # noqa: F401
        types as lit_types,
    )

    _LIT_AVAILABLE = True
    # Dynamic base class for proper LIT Dataset inheritance
    _LITDatasetBase = lit_dataset.Dataset
else:
    _LIT_AVAILABLE = False
    lit_dataset = None  # type: ignore[assignment]
    lit_types = None  # type: ignore[assignment]
    _LITDatasetBase = object  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


def _ensure_lit_available():
    """Raise ImportError if LIT is not available."""
    if not _LIT_AVAILABLE:
        raise ImportError(
            "LIT (lit-nlp) is not installed. " "Please install it with: pip install lit-nlp"
        )


@dataclass
class DatasetConfig:
    """Configuration for LIT datasets."""

    max_examples: Optional[int] = None
    """Maximum number of examples to load."""
    shuffle: bool = False
    """Whether to shuffle the examples."""
    seed: int = 42
    """Random seed for shuffling."""


class SimpleTextDataset(_LITDatasetBase):  # type: ignore[misc, valid-type]
    """Simple text dataset for use with HookedTransformerLIT.

    This is a basic dataset class that holds text examples for analysis
    with LIT. Each example is a dictionary with at least a "text" field.

    Example:
        >>> dataset = SimpleTextDataset([  # doctest: +SKIP
        ...     {"text": "Hello world"},
        ...     {"text": "How are you?"},
        ... ])
        >>> len(dataset.examples)  # doctest: +SKIP
        2
    """

    def __init__(
        self,
        examples: Optional[List[Dict[str, Any]]] = None,
        name: str = "SimpleTextDataset",
    ):
        """Initialize the dataset.

        Args:
            examples: List of example dictionaries with "text" field.
            name: Name for the dataset (shown in LIT UI).
        """
        _ensure_lit_available()

        self._examples = examples or []
        self._name = name

        # Validate examples
        for i, ex in enumerate(self._examples):
            if INPUT_FIELDS.TEXT not in ex:
                raise ValueError(f"Example {i} missing required field '{INPUT_FIELDS.TEXT}'")

    @property
    def examples(self) -> List[Dict[str, Any]]:
        """Return all examples in the dataset."""
        return self._examples

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self._examples)

    def __iter__(self):
        """Iterate over examples."""
        return iter(self._examples)

    def description(self) -> str:
        """Return a description of the dataset."""
        return f"{self._name}: {len(self._examples)} examples"

    def spec(self) -> Dict[str, Any]:
        """Return the spec describing the dataset fields.

        This tells LIT what fields each example contains and their types.

        Returns:
            Dictionary mapping field names to LIT type specs.
        """
        return {
            INPUT_FIELDS.TEXT: lit_types.TextSegment(),  # type: ignore[union-attr]
        }

    @classmethod
    def from_strings(
        cls,
        texts: Sequence[str],
        name: str = "TextDataset",
    ) -> "SimpleTextDataset":
        """Create a dataset from a list of strings.

        Args:
            texts: Sequence of text strings.
            name: Dataset name.

        Returns:
            SimpleTextDataset instance.

        Example:
            >>> dataset = SimpleTextDataset.from_strings([  # doctest: +SKIP
            ...     "First example",
            ...     "Second example",
            ... ])
        """
        examples = [{INPUT_FIELDS.TEXT: text} for text in texts]
        return cls(examples, name=name)

    @classmethod
    def from_file(
        cls,
        filepath: Union[str, Path],
        name: Optional[str] = None,
        max_examples: Optional[int] = None,
    ) -> "SimpleTextDataset":
        """Load a dataset from a text file.

        Each line in the file becomes one example.

        Args:
            filepath: Path to the text file.
            name: Optional dataset name (defaults to filename).
            max_examples: Maximum number of examples to load.

        Returns:
            SimpleTextDataset instance.
        """
        filepath = Path(filepath)

        if name is None:
            name = filepath.stem

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if max_examples is not None:
            lines = lines[:max_examples]

        texts = [line.strip() for line in lines if line.strip()]
        return cls.from_strings(texts, name=name)


class PromptCompletionDataset(_LITDatasetBase):  # type: ignore[misc, valid-type]
    """Dataset with prompt-completion pairs for generation analysis.

    This dataset type is useful for analyzing model generation behavior,
    where each example has a prompt and an expected completion.

    Example:
        >>> dataset = PromptCompletionDataset([  # doctest: +SKIP
        ...     {"prompt": "The capital of France is", "completion": " Paris"},
        ...     {"prompt": "2 + 2 =", "completion": " 4"},
        ... ])
    """

    # Field names for this dataset type
    PROMPT_FIELD = "prompt"
    COMPLETION_FIELD = "completion"
    FULL_TEXT_FIELD = "text"

    def __init__(
        self,
        examples: Optional[List[Dict[str, Any]]] = None,
        name: str = "PromptCompletionDataset",
    ):
        """Initialize the dataset.

        Args:
            examples: List of example dictionaries with prompt/completion.
            name: Name for the dataset.
        """
        _ensure_lit_available()

        self._name = name
        self._examples: List[Dict[str, Any]] = []

        if examples:
            for ex in examples:
                self._add_example(ex)

    def _add_example(self, example: Dict[str, Any]) -> None:
        """Add and validate an example.

        Args:
            example: Example dictionary.
        """
        if self.PROMPT_FIELD not in example:
            raise ValueError(f"Example missing required field '{self.PROMPT_FIELD}'")

        # Ensure completion field exists (can be empty)
        if self.COMPLETION_FIELD not in example:
            example[self.COMPLETION_FIELD] = ""

        # Create full text field
        example[self.FULL_TEXT_FIELD] = example[self.PROMPT_FIELD] + example[self.COMPLETION_FIELD]

        # Also set as "text" for compatibility with model wrapper
        example[INPUT_FIELDS.TEXT] = example[self.FULL_TEXT_FIELD]

        self._examples.append(example)

    @property
    def examples(self) -> List[Dict[str, Any]]:
        """Return all examples."""
        return self._examples

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self._examples)

    def __iter__(self):
        """Iterate over examples."""
        return iter(self._examples)

    def description(self) -> str:
        """Return a description of the dataset."""
        return f"{self._name}: {len(self._examples)} prompt-completion pairs"

    def spec(self) -> Dict[str, Any]:
        """Return the spec describing the dataset fields."""
        return {
            self.PROMPT_FIELD: lit_types.TextSegment(),  # type: ignore[union-attr]
            self.COMPLETION_FIELD: lit_types.TextSegment(),  # type: ignore[union-attr]
            self.FULL_TEXT_FIELD: lit_types.TextSegment(),  # type: ignore[union-attr]
            INPUT_FIELDS.TEXT: lit_types.TextSegment(),  # type: ignore[union-attr]
        }

    @classmethod
    def from_pairs(
        cls,
        pairs: Sequence[tuple],
        name: str = "PromptCompletionDataset",
    ) -> "PromptCompletionDataset":
        """Create a dataset from (prompt, completion) tuples.

        Args:
            pairs: Sequence of (prompt, completion) tuples.
            name: Dataset name.

        Returns:
            PromptCompletionDataset instance.

        Example:
            >>> dataset = PromptCompletionDataset.from_pairs([  # doctest: +SKIP
            ...     ("Hello, my name is", " Alice"),
            ...     ("The weather today is", " sunny"),
            ... ])
        """
        examples = [
            {cls.PROMPT_FIELD: prompt, cls.COMPLETION_FIELD: completion}
            for prompt, completion in pairs
        ]
        return cls(examples, name=name)


class IOIDataset(_LITDatasetBase):  # type: ignore[misc, valid-type]
    """Indirect Object Identification (IOI) dataset.

    This dataset contains examples for the Indirect Object Identification
    task, commonly used in mechanistic interpretability research.

    Each example has the format:
    "When {name1} and {name2} went to the {place}, {name1} gave a {object} to"

    The model should complete with name2 (the indirect object).

    Reference:
        Wang et al. "Interpretability in the Wild: a Circuit for Indirect
        Object Identification in GPT-2 small"
        https://arxiv.org/abs/2211.00593
    """

    # Common names for IOI examples
    NAMES = [
        "Mary",
        "John",
        "Alice",
        "Bob",
        "Charlie",
        "Diana",
        "Emma",
        "Frank",
        "Grace",
        "Henry",
        "Ivy",
        "Jack",
    ]

    # Common places
    PLACES = [
        "store",
        "park",
        "beach",
        "restaurant",
        "library",
        "museum",
        "cafe",
        "market",
        "school",
        "hospital",
    ]

    # Common objects
    OBJECTS = [
        "book",
        "gift",
        "letter",
        "key",
        "phone",
        "drink",
        "flower",
        "card",
        "ticket",
        "bag",
    ]

    TEMPLATE = "When {name1} and {name2} went to the {place}, {name1} gave a {object} to"

    def __init__(
        self,
        examples: Optional[List[Dict[str, Any]]] = None,
        name: str = "IOI Dataset",
    ):
        """Initialize the IOI dataset.

        Args:
            examples: Optional pre-defined examples.
            name: Dataset name.
        """
        _ensure_lit_available()

        self._name = name
        self._examples = examples or []

    @property
    def examples(self) -> List[Dict[str, Any]]:
        """Return all examples."""
        return self._examples

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self._examples)

    def __iter__(self):
        """Iterate over examples."""
        return iter(self._examples)

    def description(self) -> str:
        """Return a description of the dataset."""
        return f"{self._name}: {len(self._examples)} IOI examples"

    def spec(self) -> Dict[str, Any]:
        """Return the spec describing the dataset fields."""
        return {
            INPUT_FIELDS.TEXT: lit_types.TextSegment(),  # type: ignore[union-attr]
            "name1": lit_types.CategoryLabel(),  # type: ignore[union-attr]
            "name2": lit_types.CategoryLabel(),  # type: ignore[union-attr]
            "place": lit_types.CategoryLabel(),  # type: ignore[union-attr]
            "object": lit_types.CategoryLabel(),  # type: ignore[union-attr]
            "answer": lit_types.CategoryLabel(),  # type: ignore[union-attr]
        }

    def add_example(
        self,
        name1: str,
        name2: str,
        place: str,
        obj: str,
    ) -> None:
        """Add a single IOI example.

        Args:
            name1: Subject name (gives the object).
            name2: Indirect object name (receives the object).
            place: Location.
            obj: Object being given.
        """
        text = self.TEMPLATE.format(
            name1=name1,
            name2=name2,
            place=place,
            object=obj,
        )
        self._examples.append(
            {
                INPUT_FIELDS.TEXT: text,
                "name1": name1,
                "name2": name2,
                "place": place,
                "object": obj,
                "answer": name2,  # The correct completion
            }
        )

    @classmethod
    def generate(
        cls,
        n_examples: int = 100,
        seed: int = 42,
        name: str = "IOI Dataset",
    ) -> "IOIDataset":
        """Generate random IOI examples.

        Args:
            n_examples: Number of examples to generate.
            seed: Random seed for reproducibility.
            name: Dataset name.

        Returns:
            IOIDataset with generated examples.
        """
        import random

        random.seed(seed)

        dataset = cls(name=name)

        for _ in range(n_examples):
            # Select two different names
            name1, name2 = random.sample(cls.NAMES, 2)
            place = random.choice(cls.PLACES)
            obj = random.choice(cls.OBJECTS)

            dataset.add_example(name1, name2, place, obj)

        return dataset


class InductionDataset(_LITDatasetBase):  # type: ignore[misc, valid-type]
    """Dataset for induction head analysis.

    Induction heads are attention heads that perform pattern matching
    of the form [A][B] ... [A] -> [B]. This dataset provides examples
    designed to trigger induction behavior.

    Example pattern:
    "The cat sat on the mat. The cat sat on the" -> " mat"

    Reference:
        Olsson et al. "In-context Learning and Induction Heads"
        https://arxiv.org/abs/2209.11895
    """

    def __init__(
        self,
        examples: Optional[List[Dict[str, Any]]] = None,
        name: str = "Induction Dataset",
    ):
        """Initialize the induction dataset.

        Args:
            examples: Optional pre-defined examples.
            name: Dataset name.
        """
        _ensure_lit_available()

        self._name = name
        self._examples = examples or []

    @property
    def examples(self) -> List[Dict[str, Any]]:
        """Return all examples."""
        return self._examples

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self._examples)

    def __iter__(self):
        """Iterate over examples."""
        return iter(self._examples)

    def description(self) -> str:
        """Return a description of the dataset."""
        return f"{self._name}: {len(self._examples)} induction examples"

    def spec(self) -> Dict[str, Any]:
        """Return the spec describing the dataset fields."""
        return {
            INPUT_FIELDS.TEXT: lit_types.TextSegment(),  # type: ignore[union-attr]
            "pattern": lit_types.TextSegment(),  # type: ignore[union-attr]
            "expected_completion": lit_types.TextSegment(),  # type: ignore[union-attr]
        }

    def add_example(
        self,
        pattern: str,
        repeated_text: str,
        completion: str,
    ) -> None:
        """Add an induction example.

        Args:
            pattern: The pattern that is repeated.
            repeated_text: The text before the second occurrence.
            completion: The expected completion.
        """
        # Create the full text: pattern + separator + repeated start
        text = f"{pattern} {repeated_text} {pattern.split()[0]}"

        self._examples.append(
            {
                INPUT_FIELDS.TEXT: text,
                "pattern": pattern,
                "expected_completion": completion,
            }
        )

    @classmethod
    def generate_simple(
        cls,
        n_examples: int = 50,
        seed: int = 42,
        name: str = "Induction Dataset",
    ) -> "InductionDataset":
        """Generate simple induction examples.

        Args:
            n_examples: Number of examples to generate.
            seed: Random seed.
            name: Dataset name.

        Returns:
            InductionDataset with generated examples.
        """
        import random

        random.seed(seed)

        # Simple word pairs
        patterns = [
            ("The cat sat", "on the mat"),
            ("Hello my name", "is Alice"),
            ("The quick brown", "fox jumps"),
            ("Once upon a", "time there"),
            ("In the beginning", "was the"),
            ("To be or", "not to"),
            ("The sun rises", "in the"),
            ("Water flows down", "the hill"),
        ]

        dataset = cls(name=name)

        for i in range(n_examples):
            pattern_start, pattern_end = patterns[i % len(patterns)]
            full_pattern = f"{pattern_start} {pattern_end}"

            # Add some random connecting text
            connectors = ["Then, later,", "After that,", "Subsequently,", "Next,"]
            connector = random.choice(connectors)

            dataset.add_example(
                pattern=full_pattern,
                repeated_text=connector,
                completion=pattern_end,
            )

        return dataset


# Wrapper to make datasets LIT-compatible if LIT is available
if _LIT_AVAILABLE:

    class LITDatasetWrapper(lit_dataset.Dataset):  # type: ignore[union-attr]
        """Wrapper to make our datasets inherit from lit_dataset.Dataset.

        This wrapper takes TransformerLens dataset classes and makes them
        compatible with LIT's Dataset interface.
        """

        def __init__(self, examples: List[Dict[str, Any]], spec_dict: Dict[str, Any], name: str):
            """Create a LIT-compatible dataset.

            Args:
                examples: List of example dictionaries.
                spec_dict: The spec dictionary describing the fields.
                name: Name/description of the dataset.
            """
            super().__init__()
            self._examples = examples
            self._spec_dict = spec_dict
            self._name = name

        @classmethod
        def init_spec(cls) -> None:
            """Return None to indicate this dataset is not UI-configurable."""
            return None

        def spec(self) -> Dict[str, Any]:
            return self._spec_dict

        def description(self) -> str:
            return self._name

        @property
        def examples(self) -> List[Dict[str, Any]]:
            """Return the examples list."""
            return self._examples

        def __len__(self) -> int:
            """Return the number of examples."""
            return len(self._examples)

        def __iter__(self):
            """Iterate over examples."""
            return iter(self._examples)

    def wrap_for_lit(dataset: Any) -> LITDatasetWrapper:
        """Wrap a dataset for use with LIT.

        Args:
            dataset: One of our dataset classes (SimpleTextDataset,
                PromptCompletionDataset, IOIDataset, or InductionDataset).

        Returns:
            LIT-compatible dataset.
        """
        return LITDatasetWrapper(
            examples=list(dataset.examples),
            spec_dict=dataset.spec(),
            name=dataset.description(),
        )

else:
    # Define wrap_for_lit when LIT is not available
    def wrap_for_lit(dataset: Any) -> Any:  # type: ignore[misc]
        """Placeholder when LIT is not available."""
        raise ImportError(
            "LIT (lit-nlp) is not installed. " "Please install it with: pip install lit-nlp"
        )
