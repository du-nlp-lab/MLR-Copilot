"""Import DatasetProcessor classes."""
from autoresearch.prompt2model.dataset_processor.base import BaseProcessor
from autoresearch.prompt2model.dataset_processor.mock import MockProcessor
from autoresearch.prompt2model.dataset_processor.textualize import TextualizeProcessor

__all__ = ("BaseProcessor", "TextualizeProcessor", "MockProcessor")
