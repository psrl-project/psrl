import base64
import pickle
from typing import Any


def b64_dumps(obj: Any) -> str:
    """Serialize a Python object to a base64 string.

    Note: this is intended for trusted, in-cluster traffic only. Pickle is not
    safe for untrusted inputs.
    """
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.b64encode(payload).decode("ascii")


def b64_loads(payload_b64: str) -> Any:
    """Deserialize a base64 string back to a Python object."""
    payload = base64.b64decode(payload_b64.encode("ascii"))
    return pickle.loads(payload)


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
    import importlib

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
