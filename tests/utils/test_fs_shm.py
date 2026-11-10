from pathlib import Path
from types import SimpleNamespace

import pytest
from verl.utils import fs

pytestmark = pytest.mark.cpu_test


def test_copy_to_shm_fails_before_copy_when_space_is_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights").write_bytes(b"weights")
    shm_root = tmp_path / "shm"
    lock_root = tmp_path / "locks"
    monkeypatch.setattr(fs, "_SHM_MODEL_ROOT", str(shm_root))
    monkeypatch.setattr(fs, "_SHM_LOCK_ROOT", str(lock_root))
    monkeypatch.setattr(
        fs.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=1, used=0, free=1),
    )

    with pytest.raises(RuntimeError, match="model.use_shm=False"):
        fs.copy_to_shm(str(source))
    assert not any(".tmp." in path.name for path in shm_root.rglob("*"))


def test_copy_to_shm_publishes_complete_copy_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights").write_bytes(b"weights")
    shm_root = tmp_path / "shm"
    lock_root = tmp_path / "locks"
    monkeypatch.setattr(fs, "_SHM_MODEL_ROOT", str(shm_root))
    monkeypatch.setattr(fs, "_SHM_LOCK_ROOT", str(lock_root))
    monkeypatch.setenv("VERL_SHM_FREE_SPACE_RESERVE_BYTES", "0")

    destination = Path(fs.copy_to_shm(str(source)))
    assert (destination / "weights").read_bytes() == b"weights"
    assert fs.copy_to_shm(str(source)) == str(destination)
