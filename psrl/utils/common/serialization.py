import base64
import pickle
from typing import Any

import numpy as np


def json_encode_default(obj: Any) -> Any:
    """Coerce numpy scalars and array-likes into JSON-encodable Python values.

    Rollout records carry numpy scalars from the dataloader and tensors from the
    agent loop, neither of which `json.dumps` accepts. Pass as its `default=`.

    Args:
        obj (Any): The value `json.dumps` could not encode.

    Returns:
        Any: A JSON-encodable equivalent.

    Raises:
        TypeError: If the value has no known JSON equivalent.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def b64_dumps(obj: Any) -> str:
    """
    Serialize a Python object to a base64 string.

    Note: this is intended for trusted, in-cluster traffic only. Pickle is not
    safe for untrusted inputs.
    """
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.b64encode(payload).decode("ascii")


def b64_loads(payload_b64: str) -> Any:
    """
    Deserialize a base64 string back to a Python object.
    """
    payload = base64.b64decode(payload_b64.encode("ascii"))
    return pickle.loads(payload)
