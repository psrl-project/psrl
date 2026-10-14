import base64
import pickle
from typing import Any


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
