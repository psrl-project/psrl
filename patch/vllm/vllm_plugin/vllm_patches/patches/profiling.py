import logging
import os

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def apply_profiling_patches() -> None:
    """
    Apply all patches required for per-trajectory profiling.

    Patches applied (all idempotent):

    1. Extend `EngineCoreEventType` with two new members:
       - `FIRST_TOKEN = 4`: marks the boundary between prefill completion
         and the first decode token.
       - `LAST_TOKEN = 5`: marks decode completion, providing a reliable
         monotonic end timestamp for `decode_duration_s`.

    2. Wrap `Scheduler._update_request_with_output` to record:
       - `FIRST_TOKEN` before the first output token is appended
         (detected via `num_output_tokens == 0`).
       - `LAST_TOKEN` when the request is stopped (`stopped=True`).

    3. Add an `events` field to `RequestOutput` so profiling events
       survive the journey from the engine core to the rollout worker.

    4. Add `accumulated_events` to `RequestState` and patch
       `RequestState._new_request_output` to drain the list into each
       emitted `RequestOutput`, then clear it. This means each
       `RequestOutput` carries only the events produced since the
       previous output was emitted — no duplication.

    5. Patch `OutputProcessor.process_outputs` to copy events from each
       `EngineCoreOutput` onto the corresponding
       `RequestState.accumulated_events` before `_new_request_output`
       is called (so the drain in patch 4 sees them).

    Usage:
        Call once at process startup, before any vLLM engine objects are
        created::

            from vllm_patches.patches.profiling import apply_profiling_patches

            apply_profiling_patches()
    """
    _patch_engine_core_event_type()
    _patch_scheduler_update_request_with_output()
    _patch_request_output_events()
    _patch_request_state_event_accumulation()
    _patch_output_processor_accumulate_events()


# ----------------------------------------
# Patch 1: Extend EngineCoreEventType enum
# ----------------------------------------


def _patch_engine_core_event_type() -> None:
    """
    Add `FIRST_TOKEN = 4` and `LAST_TOKEN = 5` to `EngineCoreEventType`.

    Both members are skipped individually if already present, so the patch
    is safe to call multiple times.
    """
    from vllm.v1.engine import EngineCoreEventType

    new_members = [("FIRST_TOKEN", 4), ("LAST_TOKEN", 5)]

    for name, value in new_members:
        if name in EngineCoreEventType._member_names_:
            psrl_logger.debug(f"EngineCoreEventType.{name} already present, skipping.")
            continue

        # NOTE(claude): `IntEnum` stores membership in three mirrored structures
        # that all must be updated to keep the enum consistent.
        # `type.__setattr__` bypasses `EnumType.__setattr__`, which raises
        # `AttributeError` when the name is already in `_member_map_`. The class
        # attribute must be set first (before `_member_map_` is populated) to
        # avoid that guard.
        new_member = int.__new__(EngineCoreEventType, value)
        new_member._name_ = name
        new_member._value_ = value

        type.__setattr__(EngineCoreEventType, name, new_member)
        EngineCoreEventType._member_names_.append(name)
        EngineCoreEventType._member_map_[name] = new_member
        EngineCoreEventType._value2member_map_[value] = new_member

        psrl_logger.info(f"Patched EngineCoreEventType: added {name} = {value}.")


# -----------------------------------------------------------------------
# Patch 2: Wrap Scheduler._update_request_with_output (FIRST/LAST_TOKEN)
# -----------------------------------------------------------------------


def _patch_scheduler_update_request_with_output() -> None:
    """
    Wrap `Scheduler._update_request_with_output` to record profiling events.

    - `FIRST_TOKEN` is recorded *before* the original call when this is the
      first output token of the *current scheduling cycle*. The cycle baseline
      is tracked via `request._psrl_cycle_output_token_baseline`, which is set
      to the current `num_output_tokens` each time a request is preempted and
      re-queued (see `RolloutScheduler._preempt_request`). For brand-new
      requests the attribute is absent, so the baseline defaults to 0.
      This correctly handles internal preemption where `_output_token_ids` is
      not cleared and `num_output_tokens` is non-zero from a prior decode
      segment.
    - `LAST_TOKEN` is recorded *after* the original call when `stopped` is
      True **and** `new_token_ids_out` is non-empty, providing a monotonic
      decode-end timestamp. Guarding on `new_token_ids_out` prevents a spurious
      `LAST_TOKEN` when a request is aborted mid-prefill with no output tokens.

    Both events are only recorded when `self.log_stats` is enabled.
    """
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine import EngineCoreEventType

    _SENTINEL = "_psrl_first_last_token_patched"

    if getattr(Scheduler, _SENTINEL, False):
        psrl_logger.debug("Scheduler._update_request_with_output already patched, skipping.")
        return

    _original = Scheduler._update_request_with_output

    def _patched(self, request, new_token_ids):
        if new_token_ids and self.log_stats:
            # Fire FIRST_TOKEN when output tokens first exceed the per-cycle
            # baseline. For new requests the baseline is absent → defaults to 0.
            # For resumed requests the baseline was set in _preempt_request to
            # `num_output_tokens` at the moment of preemption, so FIRST_TOKEN
            # fires on the first new token of the current scheduling cycle.
            baseline = getattr(request, "_psrl_cycle_output_token_baseline", 0)
            if request.num_output_tokens == baseline:
                request.record_event(EngineCoreEventType.FIRST_TOKEN)

        result = _original(self, request, new_token_ids)

        # Record LAST_TOKEN only when there are actual output tokens — an abort
        # with no tokens (e.g. preempt before any decode) must not emit
        # LAST_TOKEN, as there is no valid FIRST_TOKEN to pair it with.
        new_token_ids_out, stopped = result
        if stopped and new_token_ids_out and self.log_stats:
            request.record_event(EngineCoreEventType.LAST_TOKEN)

        return result

    Scheduler._update_request_with_output = _patched
    setattr(Scheduler, _SENTINEL, True)

    psrl_logger.info("Patched Scheduler._update_request_with_output to record FIRST_TOKEN and LAST_TOKEN.")


# -------------------------------------------------------
# Patch 3: Add events field to RequestOutput
# -------------------------------------------------------


def _patch_request_output_events() -> None:
    """
    Add an `events` kwarg and field to `RequestOutput.__init__`.

    Defaults to `None` so existing call sites are unaffected. The rollout
    worker reads this field to obtain the profiling event stream.
    """
    from vllm.outputs import RequestOutput

    _SENTINEL = "_psrl_events_field_patched"
    if getattr(RequestOutput, _SENTINEL, False):
        psrl_logger.debug("RequestOutput.events field already patched, skipping.")
        return

    _original_init = RequestOutput.__init__

    def _patched_init(self, *args, events=None, **kwargs):
        _original_init(self, *args, **kwargs)
        self.events = events

    RequestOutput.__init__ = _patched_init
    setattr(RequestOutput, _SENTINEL, True)

    psrl_logger.info("Patched RequestOutput.__init__ to accept and store events.")


# -------------------------------------------------------
# Patch 4: RequestState event accumulation and draining
# -------------------------------------------------------


def _patch_request_state_event_accumulation() -> None:
    """
    Patch `RequestState` to accumulate and forward profiling events.

    Sub-patch (a) — `__init__`:
        Adds `self.accumulated_events = []` to hold events received from
        `EngineCoreOutput` across multiple engine steps.

    Sub-patch (b) — `_new_request_output`:
        Drains `accumulated_events` into the emitted `RequestOutput` and
        clears the list, so each output carries only the events produced since
        the previous one was emitted (no duplication across streaming chunks).
        `PoolingRequestOutput` objects are left untouched.
    """
    from vllm.v1.engine.output_processor import RequestState

    _SENTINEL = "_psrl_event_accum_patched"
    if getattr(RequestState, _SENTINEL, False):
        psrl_logger.debug("RequestState event accumulation already patched, skipping.")
        return

    # --- sub-patch (a): __init__ ---

    _original_init = RequestState.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        self.accumulated_events: list = []

    RequestState.__init__ = _patched_init

    # --- sub-patch (b): _new_request_output ---

    _original_new_request_output = RequestState._new_request_output

    def _patched_new_request_output(self, request_id, outputs, finished, kv_transfer_params=None):
        request_output = _original_new_request_output(self, request_id, outputs, finished, kv_transfer_params)
        from vllm.outputs import RequestOutput

        if isinstance(request_output, RequestOutput) and self.accumulated_events:
            request_output.events = list(self.accumulated_events)
            self.accumulated_events.clear()
        return request_output

    RequestState._new_request_output = _patched_new_request_output
    setattr(RequestState, _SENTINEL, True)

    psrl_logger.info("Patched RequestState.__init__ and _new_request_output for event accumulation.")


# -----------------------------------------------------------
# Patch 5: OutputProcessor.process_outputs event accumulation
# -----------------------------------------------------------


def _patch_output_processor_accumulate_events() -> None:
    """
    Pre-wrap `OutputProcessor.process_outputs` to copy events from each
    `EngineCoreOutput` onto the matching `RequestState.accumulated_events`.

    Must run *before* the original method so that events are present on
    `accumulated_events` when patch 4's `_new_request_output` drains them.

    Implemented as a wrapper (not an in-place edit) to stay decoupled from
    the exact internal structure of `process_outputs`.
    """
    from vllm.v1.engine.output_processor import OutputProcessor

    _SENTINEL = "_psrl_process_outputs_patched"
    if getattr(OutputProcessor, _SENTINEL, False):
        psrl_logger.debug("OutputProcessor.process_outputs already patched, skipping.")
        return

    _original_process_outputs = OutputProcessor.process_outputs

    def _patched_process_outputs(self, engine_core_outputs, *args, **kwargs):
        for engine_core_output in engine_core_outputs:
            req_state = self.request_states.get(engine_core_output.request_id)
            if req_state is None:
                continue
            if engine_core_output.events and hasattr(req_state, "accumulated_events"):
                req_state.accumulated_events.extend(engine_core_output.events)

        return _original_process_outputs(self, engine_core_outputs, *args, **kwargs)

    OutputProcessor.process_outputs = _patched_process_outputs
    setattr(OutputProcessor, _SENTINEL, True)

    psrl_logger.info("Patched OutputProcessor.process_outputs to accumulate events onto RequestState.")
