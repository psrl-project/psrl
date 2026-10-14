import asyncio
import enum
import logging
import os
from enum import Enum
from typing import Any

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class CommandType(Enum):
    """
    Enum for different command types used in the server.
    STOP: Stop the server.
    RESUME: Resume the server.
    SHUTDOWN: Shutdown the server gracefully.
    SYNC: Synchronize the server state with the latest weights.
    ABORT: Abort requests running in the server.
    CHECK_AND_SYNC: Check if the server needs to sync with the latest weights and do so if necessary.
    ENGINE_STATUS: Send engine status information.
    SLEEP: Sleep the server.
    WAKE_UP: Wake up the server.
    """

    STOP = enum.auto()
    RESUME = enum.auto()
    SHUTDOWN = enum.auto()
    SYNC = enum.auto()
    ABORT = enum.auto()
    CHECK_AND_SYNC = enum.auto()
    ENGINE_STATUS = enum.auto()
    SLEEP = enum.auto()
    WAKE_UP = enum.auto()


class Command:
    """
    A command that can be sent to the server.

    Attributes:
        type (CommandType): The type of the command.
        _args (dict): Arguments for the command.
        _kwargs (dict): Keyword arguments for the command.

    Methods:
        get_args(): Returns the arguments of the command.
        get_kwargs(): Returns the keyword arguments of the command.
    """

    __slots__ = ("type", "_args", "_kwargs")

    def __init__(self, type, **kwargs):
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "_args", {})
        object.__setattr__(self, "_kwargs", {})

        for key, value in kwargs.items():
            if key == "_kwargs":
                if not isinstance(value, dict):
                    raise ValueError(f"_kwargs must be a dict, got {type(value)}")
                self._kwargs = value
            else:
                self._args[key] = value

    def __repr__(self):
        return f"RolloutCommand(type={self.type}, args={self._args}, kwargs={self._kwargs})"

    def get_args(self):
        """Get the arguments of the command."""
        return self._args

    def get_kwargs(self):
        """Get the keyword arguments of the command."""
        return self._kwargs

    def __getattr__(self, item):
        if item in {"type", "_args", "_kwargs"}:
            return object.__getattribute__(self, item)

        args = object.__getattribute__(self, "_args")
        if item in args:
            return args[item]

        meta = object.__getattribute__(self, "_kwargs")
        if item in meta:
            return meta[item]

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        if key in {"type", "_args", "_kwargs"}:
            object.__setattr__(self, key, value)
            return

        args = object.__getattribute__(self, "_args")
        if key in args:
            args[key] = value
        else:
            meta = object.__getattribute__(self, "_kwargs")
            meta[key] = value

    def __getstate__(self):
        return {"type": self.type, "_args": self._args, "_kwargs": self._kwargs}

    def __setstate__(self, state):
        object.__setattr__(self, "type", state["type"])
        object.__setattr__(self, "_args", state["_args"])
        object.__setattr__(self, "_kwargs", state["_kwargs"])


class CommandEvent:
    """
    A class to handle command events using asyncio.Event.

    Attributes:
        command_id (int): The ID of the command.
        event (asyncio.Event): The asyncio event associated with the command.
    """

    def __init__(self, command_id: int, event: asyncio.Event | None = None):
        self.command_id = command_id
        self.event = event if event is not None else asyncio.Event()

    async def wait(self, timeout=None):
        """Wait for the command event to be set."""
        if timeout is not None:
            try:
                await asyncio.wait_for(self.event.wait(), timeout=timeout)
                return True
            except asyncio.TimeoutError:
                return False
        else:
            await self.event.wait()
            return True

    def set(self):
        """Set the command event."""
        self.event.set()

    def clear(self):
        """Clear the command event."""
        self.event.clear()

    def is_set(self):
        """Check if the command event is set."""
        return self.event.is_set()

    def __repr__(self):
        return f"CommandEvent(command_id={self.command_id}, event={self.event})"


class CommandExtension:
    def __init__(self):
        """
        Initialize the CommandExtension with asyncio queue to handle command results and events.
        """
        self._command_results = {}
        self._command_events = {}
        self._command_status = {}
        self._command_errors: dict[int, str] = {}
        self._timed_out_command_ids: set[int] = set()
        self._command_counter = 0
        self.command_queue = asyncio.Queue()  # For async commands like abort

    async def exec_command(self, command: Command, timeout=None, blocking=True):
        """
        Add a command to the command queue and wait for its completion within the specified timeout.

        Args:
            command (Command): The command to be executed.
            timeout (float, optional): The maximum time to wait for the command to complete when `blocking` is True.
                                       If None, waits indefinitely.
            blocking (bool): If True, blocks until the command is executed. If False, returns immediately.

        Returns:
            Union[int, dict[int, DataProto], None]:
                If `blocking` is True, returns a CommandEvent or the command result.
                If `blocking` is False, returns the command ID immediately.
                You can use `synchronize_command` to wait for the command result later.
        """
        psrl_logger.debug(
            f"Received command type={command.type}, args={command.get_args()}, "
            f"kwargs={command.get_kwargs()}, blocking={blocking}, timeout={timeout}."
        )

        command_id = self._command_counter
        self._command_counter += 1

        command_event = CommandEvent(command_id)
        self._command_events[command_id] = command_event
        self._command_results[command_id] = None
        self._command_status[command_id] = "QUEUED"

        command._kwargs["id"] = command_id
        self.command_queue.put_nowait(command)

        if not blocking:
            psrl_logger.debug(f"Command type={command.type}, id={command_id} is non-blocking. Returning command ID.")
            return command_id

        success = await command_event.wait(timeout=timeout)
        if not success:
            # Timeout is not cancellation. Keep the event/result so callers can
            # synchronize with the original operation instead of issuing a duplicate.
            self._timed_out_command_ids.add(command_id)
            return {
                "status": self._command_status.get(command_id, "PENDING"),
                "command_id": command_id,
            }

        error = self._command_errors.pop(command_id, None)
        result = self._command_results.get(command_id, None)
        psrl_logger.debug(f"Command id={command_id} completed with result={result}.")
        if command_id in self._command_results:
            del self._command_results[command_id]
        if command_id in self._command_events:
            del self._command_events[command_id]
        self._command_status.pop(command_id, None)
        self._timed_out_command_ids.discard(command_id)

        if error is not None:
            raise RuntimeError(error)
        return result

    async def synchronize_command(self, command_id: int, timeout=None):
        """
        Wait for a specific command event to be set and return its result.

        Args:
            command_id (int): The command event to wait for.
            timeout (float, optional): The maximum time to wait for the command event to be set.

        Returns:
            Any: The result of the command if it was completed, otherwise None.
        """
        assert command_id in self._command_events, f"Command id={command_id} not found in command events."

        command_event = self._command_events[command_id]

        success = await command_event.wait(timeout=timeout)
        if not success:
            psrl_logger.warning(f"Command event id={command_id} timed out after seconds={timeout}.")
            return None

        error = self._command_errors.pop(command_id, None)
        result = self._command_results.get(command_id, None)
        if command_id in self._command_results:
            del self._command_results[command_id]
        if command_id in self._command_events:
            del self._command_events[command_id]
        self._command_status.pop(command_id, None)
        self._timed_out_command_ids.discard(command_id)

        if error is not None:
            raise RuntimeError(error)
        return result

    def get_command_status(self, command_id: int) -> dict[str, Any]:
        """Return a non-destructive command lifecycle snapshot."""
        command_id = int(command_id)
        return {
            "command_id": command_id,
            "status": self._command_status.get(command_id, "UNKNOWN"),
            "result": self._command_results.get(command_id),
            "error": self._command_errors.get(command_id),
        }

    def _start_command(self, command_id: int) -> None:
        if command_id in self._command_status:
            self._command_status[command_id] = "RUNNING"

    def _fail_command(self, command_id: int, error: Exception) -> None:
        """
        Wake the original command waiter with the execution error.
        """
        self._command_errors[command_id] = f"Command {command_id} failed: {error!r}."
        self._command_status[command_id] = "FAILED"
        self._command_events[command_id].set()

    def _complete_command(self, command_id: int, result: Any):
        """Set the command result, mark it as completed and notify the event waiter."""
        if command_id not in self._command_results:
            # Completion requires an outstanding command record.
            psrl_logger.warning(
                f"Dropping completion without an outstanding waiter: command_id={command_id}, result={result!r}."
            )
            return
        self._command_results[command_id] = result
        self._command_status[command_id] = "SUCCEEDED" if result is not False else "FAILED"
        psrl_logger.debug(f"Command ID {command_id} completed with result: {result}")
        if command_id in self._command_events:
            self._command_events[command_id].set()
