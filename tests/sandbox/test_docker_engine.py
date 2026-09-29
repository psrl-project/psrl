from psrl.sandbox.backends.docker_engine import DockerEngineClient


def _frame(stream: int, payload: bytes) -> bytes:
    return bytes([stream, 0, 0, 0]) + len(payload).to_bytes(4, "big") + payload


def test_docker_exec_stream_demultiplexing() -> None:
    body = _frame(1, b"out-1") + _frame(2, b"err") + _frame(1, b"out-2")

    stdout, stderr = DockerEngineClient._demultiplex_exec(body)

    assert stdout == b"out-1out-2"
    assert stderr == b"err"
