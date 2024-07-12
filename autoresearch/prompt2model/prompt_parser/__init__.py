"""Import PromptSpec classes."""
from autoresearch.prompt2model.prompt_parser.base import PromptSpec, TaskType
from autoresearch.prompt2model.prompt_parser.instr_parser import PromptBasedInstructionParser
from autoresearch.prompt2model.prompt_parser.mock import MockPromptSpec

__all__ = (
    "PromptSpec",
    "TaskType",
    "MockPromptSpec",
    "PromptBasedInstructionParser",
)
