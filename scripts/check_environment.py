#!/usr/bin/env python3
"""
Check installed dependencies and the pinned NCCL suspend API override.
"""

import ctypes
import subprocess
import sys
from importlib.metadata import distribution, version


def main() -> None:
    """
    Accept only the explicitly configured NCCL wheel metadata exception.
    """
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    torch_version = version("torch")
    nccl_version = version("nvidia-nccl-cu12")
    if torch_version != "2.11.0+cu128" or nccl_version != "2.29.7":
        raise RuntimeError(
            f"Expected PyTorch 2.11.0+cu128 and NCCL 2.29.7, got {torch_version!r} and {nccl_version!r}."
        )
    expected = (
        f"torch {torch_version} has requirement nvidia-nccl-cu12==2.28.9"
        '; platform_system == "Linux", but you have nvidia-nccl-cu12 2.29.7.'
    )
    if result.returncode != 0 and result.stdout.strip() != expected:
        raise SystemExit(result.returncode)

    nccl = distribution("nvidia-nccl-cu12")
    library = ctypes.CDLL(str(nccl.locate_file("nvidia/nccl/lib/libnccl.so.2")))
    for symbol in ("ncclCommSuspend", "ncclCommResume"):
        getattr(library, symbol)
    library.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    library.ncclGetVersion.restype = ctypes.c_int
    native_version = ctypes.c_int()
    status = library.ncclGetVersion(ctypes.byref(native_version))
    if status != 0 or native_version.value != 22907:
        raise RuntimeError(f"Expected native NCCL 22907, got {native_version.value!r} with status {status!r}.")
    print("Accepted the pinned NCCL 2.29.7 override. Communicator suspend and resume symbols are available.")


if __name__ == "__main__":
    main()
