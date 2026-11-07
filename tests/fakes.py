"""Shared stand-ins of the tests: a parameter count and functions that fail if a test calls them.

The guard tests replace the file reads, the memory check and the model loads with these
functions, so they read no weights file and load no model (F21 of
specs/05-all-sizes/verify.html).
"""

from collections.abc import Callable

# The parameters of ViT-7B/16 from its safetensors headers, with the mask token.
VIT7B16_PARAMETER_COUNT = 6_716_035_072


def fail_if_called(name: str) -> Callable[..., None]:
    """Return a function that raises AssertionError with the name if a test calls it.

    Args:
        name: The name of the replaced function, for example "parameter_count".
    """

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"{name} was called")

    return fail


def fail_load(*args: object, **kwargs: object) -> None:
    """Replace a model loader in a dry run: a call raises AssertionError."""
    raise AssertionError("the dry run loaded a model")
