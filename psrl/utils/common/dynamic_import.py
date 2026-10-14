import importlib
import inspect
from typing import Any


def import_class_from_string(class_string: str) -> type:
    """
    Import a class from a string in the format 'module.path:ClassName'.

    Args:
        class_string: String in format 'module.path:ClassName'

    Returns:
        The imported class

    Raises:
        ImportError: If the module or class cannot be imported
        ValueError: If the string format is invalid
    """
    if ":" not in class_string:
        raise ValueError(f"Class string must be in format 'module.path:ClassName', got: {class_string}")

    module_path, class_name = class_string.split(":", 1)

    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}': {e}") from e
    except AttributeError as e:
        raise ImportError(f"Could not find class '{class_name}' in module '{module_path}': {e}") from e


def lazy_import_to_globals(
    module_path: str,
    name: str,
    alias: str | None = None,
    target_globals: dict[str, Any] | None = None,
) -> Any:
    """
    Lazily import an object from a module and store it in the global namespace.

    Equivalent to:
    - from module_path import name
    - from module_path import name as alias
    """
    if target_globals is None:
        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame is not None else None
        target_globals = caller_frame.f_globals if caller_frame is not None else globals()

    key = alias or name
    obj = target_globals.get(key)
    if obj is not None:
        return obj

    mod = importlib.import_module(module_path)
    obj = getattr(mod, name)
    target_globals[key] = obj
    return obj


def lazy_import_many_to_globals(module_path: str, names: list[str]) -> None:
    """
    Lazily import multiple objects from a module and store them in the caller globals.
    """
    frame = inspect.currentframe()
    caller_frame = frame.f_back if frame is not None else None
    target_globals = caller_frame.f_globals if caller_frame is not None else globals()

    for name in names:
        lazy_import_to_globals(module_path, name, target_globals=target_globals)
