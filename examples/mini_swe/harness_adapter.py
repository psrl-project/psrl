"""Adapter between MiniSWEAgent's protocol and PSRL sandbox sessions."""

import platform
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

from psrl.sandbox import SyncSandboxSession


class RunnerCancelled(RuntimeError):
    """Signal cooperative cancellation of a synchronous MiniSWE rollout."""


@dataclass
class MiniSWEAgentConfig:
    """Configuration fields consumed by the third-party MiniSWEAgent."""

    image: str
    cwd: str = "/"
    env: dict[str, str] = field(default_factory=dict)
    forward_env: list[str] = field(default_factory=list)
    timeout: int = 30

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Match the Pydantic interface expected by MiniSWE templates."""
        return asdict(self)


class MiniSWEAgentAdapter:
    """Translate MiniSWEAgent calls into the generic sandbox data plane."""

    def __init__(
        self,
        session: SyncSandboxSession,
        config: MiniSWEAgentConfig,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self._session = session
        self.config = config
        self._cancel_event = cancel_event
        self._closed = False

    @property
    def session(self) -> SyncSandboxSession:
        """Return the backend-neutral session for orchestration operations."""
        return self._session

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute one MiniSWE shell action."""
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise RunnerCancelled("MiniSWE runner was cancelled.")
        from minisweagent.exceptions import Submitted

        try:
            result = self._session.exec(
                action.get("command", ""),
                cwd=cwd or self.config.cwd,
                timeout_s=timeout or self.config.timeout,
            )
            output = {
                "output": result.stdout + result.stderr,
                "returncode": result.exit_code,
                "exception_info": "",
            }
        except Exception as exc:
            output = {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {exc}",
                "extra": {"exception_type": type(exc).__name__, "exception": str(exc)},
            }

        if self._cancel_event is not None and self._cancel_event.is_set():
            raise RunnerCancelled("MiniSWE runner was cancelled.")

        lines = output["output"].lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )
        return output

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        """Return variables consumed by MiniSWE prompt templates."""
        from minisweagent.utils.serialize import recursive_merge

        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), kwargs)

    def serialize(self) -> dict[str, Any]:
        """Serialize adapter metadata into a MiniSWE trajectory."""
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def cleanup(self) -> None:
        """Close the sandbox session once."""
        if self._closed:
            return
        self._session.close()
        self._closed = True
