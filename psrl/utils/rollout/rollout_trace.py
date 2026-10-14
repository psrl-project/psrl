# Adapted from verl/verl/utils/rollout_trace.py
import contextlib
import dataclasses
import functools
import inspect
import json
import os
from contextvars import ContextVar
from enum import Enum
from typing import Optional

from pydantic import BaseModel
from verl.utils.ray_utils import get_event_loop

_trace_enabled: ContextVar[bool] = ContextVar("_trace_enabled", default=True)
_trace_attributes: ContextVar[dict | None] = ContextVar("_trace_attributes", default=None)


class RolloutTraceConfig:
    """Configuration for rollout tracing with various backends.

    Singleton configuration class for managing rollout trace settings across different
            tracing backends like Weave, MLflow, and Trackio.

    Args:
        backend (Optional[str]): Tracing backend to use ('weave', 'mlflow', or None).
        client (Optional[object]): Client instance for the selected backend.
        token2text (bool): Whether to convert tokens to text in traces. Defaults to False.
        project_name (str): Name of the project for tracing.
        experiment_name (str): Name of the experiment for tracing.
        max_samples_per_step_per_worker (Optional[int]): Maximum number of unique samples to trace
            per worker per step. If None, all samples are traced. If set, each worker will randomly
            select up to this many unique samples to trace (including all their rollouts for GRPO).
            Total traces = max_samples_per_step_per_worker * num_workers * n_rollouts_per_sample.
    """

    _instance: Optional["RolloutTraceConfig"] = None
    backend: str | None = None
    client: object | None = None
    token2text: bool = False
    _initialized: bool = False
    project_name: str = None
    experiment_name: str = None
    max_samples_per_step_per_worker: int | None = None

    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern: only one instance exists.

        Returns:
            RolloutTraceConfig: The singleton instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "RolloutTraceConfig":
        """Get the singleton instance of RolloutTraceConfig.

        Returns:
            RolloutTraceConfig: The singleton configuration instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def init(
        cls,
        project_name: str,
        experiment_name: str,
        backend: str,
        token2text: bool = False,
        max_samples_per_step_per_worker: int | None = None,
    ):
        """Initialize the tracing configuration with the specified backend.

        Sets up the tracing backend (Weave or MLflow) and initializes the client.
        This method is idempotent. Calling it multiple times has no effect after
        the first initialization.

        Args:
            project_name: Name of the project for organizing traces
            experiment_name: Name of the experiment within the project
            backend: Tracing backend to use ('weave', 'mlflow', or None to disable)
            token2text: Whether to convert token IDs to text in traces (default: False)
        """
        config = cls.get_instance()
        if config._initialized:
            return

        config.backend = backend
        config.token2text = token2text
        config.project_name = project_name
        config.experiment_name = experiment_name
        config.max_samples_per_step_per_worker = max_samples_per_step_per_worker

        # Initialize backend-specific client
        if backend == "weave":
            import weave

            config.client = weave.init(project_name)
        elif backend == "mlflow":
            import mlflow

            mlflow.config.enable_async_logging()
            config.client = mlflow

            # Configure MLflow tracking URI from environment or use default
            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            mlflow.set_experiment(project_name)
        elif backend == "trackio":
            import trackio
            from trackio import context_vars

            if context_vars.current_run.get() is None:
                trackio.init(project=project_name, name=experiment_name, config={"framework": "psrl"})
            config.client = trackio
        else:
            # No tracing backend configured
            config.client = None

        config._initialized = True

    @classmethod
    def get_backend(cls) -> str | None:
        """Get the configured tracing backend name.

        Returns:
            str | None: Backend name ('weave', 'mlflow', or None).
        """
        return cls.get_instance().backend

    @classmethod
    def get_client(cls) -> object | None:
        """Get the tracing client instance.

        Returns:
            object | None: The client instance for the configured backend, or None.
        """
        return cls.get_instance().client

    @classmethod
    def enable_token2text(cls) -> bool | None:
        """Check if token-to-text conversion is enabled.

        Returns:
            bool | None: True if token-to-text conversion is enabled, False otherwise.
        """
        return cls.get_instance().token2text

    @classmethod
    def reset(cls):
        """Reset the singleton instance, clearing all configuration.

        Useful for testing or re-initialization scenarios.
        """
        cls._instance = None


@contextlib.contextmanager
def rollout_trace_attr(
    prompt_index=None,
    request_index=None,
    step=None,
    name="rollout_trace",
    validate=False,
    trace: bool = True,
    **extra_attributes,
):
    """A context manager to add attributes to a trace for the configured backend.

    Args:
        prompt_index: PSRL prompt id / parent id.
        request_index: PSRL request id / rollout uid.
        step: Training step number.
        name: Name for the trace span (used by mlflow backend).
        validate: Whether this is a validation run.
        trace: If False, disables tracing for the duration of the context.
        extra_attributes: Additional trace attributes.
    """
    backend = RolloutTraceConfig.get_backend()

    should_skip = backend is not None and not trace

    if should_skip:
        token = _trace_enabled.set(False)
        try:
            yield
        finally:
            _trace_enabled.reset(token)
        return

    # Build attributes for the trace
    attributes = {}

    # Collect trace attributes if backend is configured
    if backend:
        if prompt_index is not None:
            attributes["prompt_index"] = prompt_index
        if request_index is not None:
            attributes["request_index"] = request_index
        if step is not None:
            attributes["step"] = step
        if name is not None:
            attributes["name"] = name
        attributes["validate"] = validate
        attributes["experiment_name"] = RolloutTraceConfig.get_instance().experiment_name
        attributes.update(extra_attributes)

    # If no backend or no attributes, just yield without tracing
    if not attributes or backend is None:
        yield
        return

    # Add attributes to the appropriate backend
    token = _trace_attributes.set(attributes)
    if backend == "weave":
        import weave

        try:
            with weave.attributes(attributes):
                yield
        finally:
            _trace_attributes.reset(token)
    elif backend == "mlflow":
        import mlflow

        try:
            with mlflow.start_span(name=name) as span:
                trace_id = span.trace_id
                for key, value in attributes.items():
                    mlflow.set_trace_tag(trace_id, str(key), str(value))
                yield
        finally:
            _trace_attributes.reset(token)
    else:
        try:
            yield
        finally:
            _trace_attributes.reset(token)


def _json_trace_content(value):
    value = _json_trace_metadata(value)
    return json.dumps(value, default=str, ensure_ascii=False)


def _dataclass_to_dict(value):
    return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}


def _json_trace_metadata(value):
    if isinstance(value, BaseModel):
        value = value.model_dump()
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = _dataclass_to_dict(value)
    elif isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): _json_trace_metadata(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_trace_metadata(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _json_trace_metadata(value.tolist())
        except Exception:
            pass
    return str(value)


def _trackio_message_dict(message):
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if not isinstance(role, str):
        return None
    return {str(key): _json_trace_metadata(value) for key, value in message.items()}


def _trackio_output_dict(output):
    output = _normalize_trace_output(output)
    if output is None:
        return None
    if isinstance(output, dict):
        return output
    return {"value": output}


def _normalize_trace_output(output):
    if isinstance(output, BaseModel):
        return _normalize_trace_output(output.model_dump())
    if dataclasses.is_dataclass(output) and not isinstance(output, type):
        return {key: _normalize_trace_output(value) for key, value in _dataclass_to_dict(output).items()}
    if isinstance(output, dict):
        return {str(key): _normalize_trace_output(value) for key, value in output.items()}
    if isinstance(output, tuple):
        if len(output) == 2 and _looks_like_terminate_reason(output[1]):
            return {
                "output": _normalize_trace_output(output[0]),
                "terminate_reason": _json_trace_metadata(output[1]),
            }
        return [_normalize_trace_output(value) for value in output]
    if isinstance(output, list):
        return [_normalize_trace_output(value) for value in output]
    if isinstance(output, Enum):
        return output.value
    if output is None or isinstance(output, str | int | float | bool):
        return output
    if hasattr(output, "tolist"):
        try:
            return _normalize_trace_output(output.tolist())
        except Exception:
            pass
    if hasattr(output, "__dict__"):
        return _normalize_trace_output(dict(vars(output)))
    return str(output)


def _looks_like_terminate_reason(value):
    return isinstance(value, Enum) or value.__class__.__name__ == "TerminateReason"


def _extract_messages_from_inputs(inputs):
    input_messages = inputs.get("messages") if isinstance(inputs, dict) else None
    if isinstance(input_messages, list):
        return input_messages

    request = inputs.get("request") if isinstance(inputs, dict) else None
    if isinstance(request, dict):
        raw_prompt = request.get("raw_prompt")
        if isinstance(raw_prompt, list):
            return raw_prompt

    observation = inputs.get("observation") if isinstance(inputs, dict) else None
    if isinstance(observation, list):
        return observation
    return None


def _extract_response_text(output):
    if output is None:
        return None
    if isinstance(output, dict):
        for key in ("response_text", "answer", "text"):
            value = output.get(key)
            if value:
                return str(value)
        nested = output.get("output")
        if isinstance(nested, dict):
            value = nested.get("text")
            if value:
                return str(value)
        for key in ("output", "value", "items"):
            value = output.get(key)
            response_text = _extract_response_text(value)
            if response_text:
                return response_text
    if isinstance(output, list | tuple):
        for item in output:
            response_text = _extract_response_text(item)
            if response_text:
                return response_text
    return None


def _trackio_trace_key(op_name):
    return "rollout_trace/" + "".join(char if char.isalnum() or char in "._-" else "_" for char in op_name)


def _trackio_trace_step(attributes):
    step = attributes.get("step")
    if step is None:
        return None
    try:
        return int(step)
    except (TypeError, ValueError):
        return None


def _log_trackio_trace(op_name, inputs, output=None, exception=None):
    trackio = RolloutTraceConfig.get_client()
    if trackio is None:
        return
    attributes = _current_trace_attributes()
    metadata_inputs = {key: value for key, value in inputs.items() if key != "messages"}
    output_dict = _trackio_output_dict(output)
    metadata = {
        "op": op_name,
        "backend": "trackio",
        "experiment_name": RolloutTraceConfig.get_instance().experiment_name,
        "inputs": _json_trace_metadata(metadata_inputs),
        **{key: _json_trace_metadata(value) for key, value in attributes.items()},
    }
    if exception is not None:
        metadata["status"] = "error"
        metadata["exception_type"] = type(exception).__name__
    else:
        metadata["status"] = "success"
        metadata["output"] = _json_trace_metadata(output_dict if output_dict is not None else output)

    messages = []
    input_messages = _extract_messages_from_inputs(inputs)
    if isinstance(input_messages, list):
        messages = [
            message
            for message in (_trackio_message_dict(message) for message in input_messages)
            if message is not None
        ]

    if not messages:
        messages = [
            {"role": "system", "content": f"PSRL rollout trace operation: {op_name}"},
            {"role": "user", "content": _json_trace_content({"inputs": inputs})},
        ]

    if exception is not None:
        messages.append(
            {
                "role": "assistant",
                "content": _json_trace_content(
                    {
                        "exception_type": type(exception).__name__,
                        "exception": str(exception),
                    }
                ),
            }
        )
    else:
        response_text = _extract_response_text(output_dict)
        if response_text:
            messages.append({"role": "assistant", "content": response_text})
        else:
            messages.append({"role": "assistant", "content": _json_trace_content({"output": output_dict})})

    trackio.log(
        {_trackio_trace_key(op_name): trackio.Trace(messages=messages, metadata=metadata)},
        step=_trackio_trace_step(attributes),
    )


def _current_trace_attributes():
    backend = RolloutTraceConfig.get_backend()
    if backend == "weave":
        from weave.trace.context import call_context

        return {**(call_context.call_attributes.get() or {})}
    return {**(_trace_attributes.get() or {})}


async def _decode_token_ids(owner, token_ids):
    tokenizer = getattr(owner, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        return None
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids is None:
        return None
    loop = get_event_loop()
    return await loop.run_in_executor(None, tokenizer.decode, token_ids)


async def _add_token2text(owner, result):
    if isinstance(result, tuple):
        if len(result) == 2 and _looks_like_terminate_reason(result[1]):
            return {
                "output": await _add_token2text(owner, result[0]),
                "terminate_reason": _json_trace_metadata(result[1]),
            }
        return [await _add_token2text(owner, item) for item in result]
    if isinstance(result, list):
        return [await _add_token2text(owner, item) for item in result]
    if isinstance(result, dict):
        return {str(key): await _add_token2text(owner, value) for key, value in result.items()}

    result_dict = _normalize_trace_output(result)
    if not isinstance(result_dict, dict):
        return result_dict

    prompt_ids = getattr(result, "prompt_ids", result_dict.get("prompt_ids"))
    response_ids = getattr(result, "response_ids", result_dict.get("response_ids"))
    if prompt_ids is None and response_ids is None:
        return result_dict

    try:
        prompt_text = await _decode_token_ids(owner, prompt_ids)
        if prompt_text is not None:
            result_dict["prompt_text"] = prompt_text

        response_text = await _decode_token_ids(owner, response_ids)
        if response_text is not None:
            result_dict["response_text"] = response_text
    except Exception as exc:
        result_dict["token2text_error"] = f"{type(exc).__name__}: {exc}"
    return result_dict


def rollout_trace_op(func):
    """Decorator for tracing function/method calls during rollout.

    This decorator automatically traces function calls with their inputs and outputs,
    integrating with the configured tracing backend (Weave or MLflow). It handles
    both synchronous and asynchronous functions.

    For async functions, the decorator captures:
    - Function inputs (arguments and keyword arguments)
    - Function outputs (return values)
    - Exceptions (if any occur)
    - Optional token-to-text conversion for DataProto outputs

    Args:
        func: The function or method to be traced

    Returns:
        Wrapped function with tracing capabilities

    Example:
        @rollout_trace_op
        async def process_batch(self, batch: DataProto):
            # Function implementation
            return result
    """

    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        if not _trace_enabled.get():
            return await func(self, *args, **kwargs)

        backend = RolloutTraceConfig.get_backend()
        enable_token2text = RolloutTraceConfig.enable_token2text()
        if backend is None:
            # No tracing configured, execute function directly
            return await func(self, *args, **kwargs)

        # Extract function arguments for tracing
        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)
        del inputs["self"]

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()

            cur_attributes = _current_trace_attributes()
            call = tracer.create_call(op=func.__qualname__, inputs=inputs, attributes=cur_attributes)
            try:
                result = await func(self, *args, **kwargs)

                if enable_token2text:
                    _result = await _add_token2text(self, result)
                    tracer.finish_call(call, output=_result)
                else:
                    tracer.finish_call(call, output=_normalize_trace_output(result))

                return result

            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            with mlflow.start_span(name=func.__qualname__) as span:
                span.set_inputs(inputs)
                result = await func(self, *args, **kwargs)
                if enable_token2text:
                    _result = await _add_token2text(self, result)
                    span.set_outputs(_result)
                else:
                    span.set_outputs(_normalize_trace_output(result))

            return result
        elif backend == "trackio":
            try:
                result = await func(self, *args, **kwargs)
                if enable_token2text:
                    _result = await _add_token2text(self, result)
                    _log_trackio_trace(func.__qualname__, inputs, output=_result)
                else:
                    _log_trackio_trace(func.__qualname__, inputs, output=result)
                return result
            except Exception as e:
                _log_trackio_trace(func.__qualname__, inputs, exception=e)
                raise e

        else:
            return await func(self, *args, **kwargs)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not _trace_enabled.get():
            return func(self, *args, **kwargs)

        backend = RolloutTraceConfig.get_backend()
        if backend is None:
            return func(self, *args, **kwargs)

        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)
        del inputs["self"]

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()

            cur_attributes = _current_trace_attributes()
            call = tracer.create_call(op=func.__qualname__, inputs=inputs, attributes=cur_attributes)
            try:
                result = func(self, *args, **kwargs)
                tracer.finish_call(call, output=_normalize_trace_output(result))
                return result
            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            return mlflow.trace(func)(self, *args, **kwargs)
        elif backend == "trackio":
            try:
                result = func(self, *args, **kwargs)
                _log_trackio_trace(func.__qualname__, inputs, output=result)
                return result
            except Exception as e:
                _log_trackio_trace(func.__qualname__, inputs, exception=e)
                raise e
        else:
            return func(self, *args, **kwargs)

    return async_wrapper if inspect.iscoroutinefunction(func) else wrapper
