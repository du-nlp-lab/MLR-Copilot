"""Import DemoCreator functions."""

from autoresearch.prompt2model.demo_creator.create import create_gradio
from autoresearch.prompt2model.demo_creator.mock import mock_gradio_create

__all__ = (
    "mock_gradio_create",
    "create_gradio",
)
