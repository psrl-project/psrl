import enum


class KVCacheBackend(enum.Enum):
    """
    Backend for KV cache offloading.
    """

    CPU = "cpu"
    DISK = "disk"
    REMOTE = "remote"
