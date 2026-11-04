"""
Represent sampler distributions as compact response-aligned sparse rows.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np

MAX_SAMPLER_CAPTURE_WIDTH = 1 << 20


def capture_width(mode: str, max_tokens: int) -> int:
    """
    Per-token wire width the gateway returns for a capture mode.

    The request asks the engine for exactly this many log-prob slots per token and the gateway
    filters the engine's ``rank <= width`` entries, so a returned row has exactly ``width`` columns:

    - ``topk``: the ``max_tokens``-wide head.
    - ``support``: ``max_tokens`` support slots plus one sentinel slot witnessing completeness.

    The engine returns one column more than it is asked for (the sampled token is prepended), which
    is why the gateway's ``rank <= width`` filter, not the request value alone, defines the wire
    width. Kept here so the request, the buffer and the trainer padding share one definition.
    """
    if mode not in ("disabled", "topk", "support") or not 1 <= max_tokens:
        raise ValueError("Invalid sampler distribution mode or sampler_max_tokens budget.")
    width = 0 if mode == "disabled" else max_tokens + (mode == "support")
    if width > MAX_SAMPLER_CAPTURE_WIDTH:
        raise ValueError(
            f"sampler_max_tokens resolves to width {width}, above the protocol limit {MAX_SAMPLER_CAPTURE_WIDTH}."
        )
    return width


@dataclass(frozen=True)
class SamplerCapture:
    """
    Resolved capture configuration shared by the rollout request and the training batch.

    Bundles the mode, the per-token budget and the compact-transport flag so the request width, the
    TITO validation and the trainer padding all derive from one object instead of parallel fields.
    """

    mode: str = "disabled"
    budget: int = 32
    compact_support: bool = True

    @property
    def enabled(self) -> bool:
        """Whether any sampler capture is requested."""
        return self.mode != "disabled"

    @property
    def width(self) -> int:
        """Per-token wire width, see :func:`capture_width`."""
        return capture_width(self.mode, self.budget)

    @classmethod
    def from_rollout_config(cls, rollout_config) -> SamplerCapture:
        """Build the capture config from a (structured or OmegaConf) rollout config."""
        return cls(
            mode=rollout_config.get("sampler_distribution", "disabled"),
            budget=int(rollout_config.get("sampler_max_tokens", 32)),
            compact_support=bool(rollout_config.get("sampler_compact_support", True)),
        )


def resolve_capture_mode(mode: str, *, top_p: float, top_k: int) -> str:
    """
    Resolve ``auto`` from the sampling parameters.

    Truncated sampling (``top_p < 1`` or ``top_k > 0``) draws only from the realized support, so the
    capture must be exact and the support mode applies. Untruncated sampling uses the top-k head.
    """
    if mode == "disabled":
        return "disabled"
    if mode == "auto":
        return "support" if float(top_p) < 1.0 or int(top_k) > 0 else "topk"
    return mode


def configure_capture(body: dict, capture: SamplerCapture, *, protocol: str) -> None:
    """
    Apply the sampler capture contract to a request body.

    ``protocol`` names the wire protocol the body speaks. The OpenAI ``logprobs`` and
    ``top_logprobs`` pair is written for chat and Responses. Anthropic Messages lowers to the
    canonical chat IR inside SMG and only carries ``sampler_top_logprobs``.

    A Responses request that asks for ``top_logprobs`` must also name the logprobs include.
    The upstream rejects the request otherwise, so the field is added here rather than left to
    the caller.
    """
    if protocol not in ("chat", "responses", "messages"):
        raise ValueError(f"Unknown request protocol {protocol!r}.")
    if protocol in ("chat", "responses"):
        body["logprobs"] = True
        body["top_logprobs"] = 0
        if protocol == "responses":
            include = body.get("include")
            items = list(include) if isinstance(include, (list, tuple)) else []
            if "message.output_text.logprobs" not in items:
                items.append("message.output_text.logprobs")
            body["include"] = items
    body["sampler_top_logprobs"] = capture.width
    if not capture.enabled:
        return
    if float(body.get("temperature", 1.0)) <= 0:
        raise ValueError("Sampler distribution capture requires positive temperature.")
    if float(body.get("min_p", 0.0)) > 0:
        raise ValueError("Sampler distribution capture requires min_p=0.")
    if capture.mode == "topk" and (float(body.get("top_p", 1.0)) < 1 or int(body.get("top_k", -1)) > 0):
        raise ValueError("Truncated sampling requires sampler_distribution=support.")


@dataclass(frozen=True)
class SamplerDistribution:
    """
    Store token IDs and log probabilities with one offset per response boundary.

    Empty rows represent environment tokens. `complete` distinguishes a captured
    support from an approximate head without inferring completeness from mass.
    """

    offsets: np.ndarray
    token_ids: np.ndarray
    log_probs: np.ndarray
    complete: np.ndarray

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def prefix(self, length: int) -> SamplerDistribution:
        """
        Truncate response rows without copying the entry buffers.
        """
        if not 0 <= length <= len(self):
            raise ValueError(f"Invalid sampler distribution prefix length {length!r}.")
        end = int(self.offsets[length])
        return SamplerDistribution(
            self.offsets[: length + 1], self.token_ids[:end], self.log_probs[:end], self.complete[:length]
        )

    @classmethod
    def empty(cls, rows: int) -> SamplerDistribution:
        """
        Construct rows for masked environment tokens.
        """
        return cls(
            np.zeros(rows + 1, dtype=np.int64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.zeros(rows, dtype=np.bool_),
        )

    @classmethod
    def from_wire(cls, payload: dict, sampled: list, *, strict: bool = True) -> SamplerDistribution:
        """
        Decode and validate a TITO distribution against its sampled tokens.

        ``strict`` enables the per-row checks that sort each row (duplicate ids, probability mass,
        sampled-token agreement). Those are the expensive part over long responses, so normal training
        skips them and relies on the structural checks plus the sampler's own fail-closed capture. Set
        ``strict`` in CI-style validation runs.
        """
        offsets = np.asarray(payload["offsets"], dtype=np.int64)
        ids = np.frombuffer(base64.b64decode(payload["token_ids"], validate=True), dtype="<i4")
        log_probs = np.frombuffer(base64.b64decode(payload["log_probs"], validate=True), dtype="<f4")
        complete = np.asarray(payload["complete"], dtype=np.bool_)
        rows = len(sampled)
        if (
            offsets.ndim != 1
            or len(offsets) != rows + 1
            or complete.shape != (rows,)
            or offsets[0] != 0
            or offsets[-1] != len(ids)
            or len(ids) != len(log_probs)
            or np.any(np.diff(offsets) <= 0)
        ):
            raise ValueError("Sampler distribution offsets do not align with sampled tokens.")
        if np.any(ids < 0) or not np.isfinite(log_probs).all() or np.any(log_probs > 1e-6):
            raise ValueError("Sampler distribution contains invalid IDs or log probabilities.")
        distribution = cls(offsets, ids, log_probs, complete)
        if not strict:
            return distribution
        distribution._validate_sampled(sampled)
        return distribution

    def _validate_sampled(self, sampled: list) -> None:
        """
        Validate row identity, probability mass, and sampled-token agreement.
        """
        if len(sampled) != len(self):
            raise ValueError("Sampler distribution row count does not match sampled tokens.")
        for row, (sample_logprob, sample_id) in enumerate(sampled):
            if not np.isfinite(sample_logprob) or sample_logprob > 1e-6 or sample_id < 0:
                raise ValueError(f"Invalid sampled token or log probability at row {row}.")
            start, end = self.offsets[row : row + 2]
            row_ids, row_lp = self.token_ids[start:end], self.log_probs[start:end]
            if len(np.unique(row_ids)) != len(row_ids):
                raise ValueError(f"Duplicate sampler token IDs at row {row}.")
            mass = np.exp(row_lp.astype(np.float64)).sum()
            if mass > 1 + 2e-5 or (self.complete[row] and abs(mass - 1) > 2e-5):
                raise ValueError(f"Invalid sampler probability mass {mass!r} at row {row}.")
            selected = row_ids == sample_id
            if self.complete[row] and not selected.any():
                raise ValueError(f"Complete sampler support omits the sampled token at row {row}.")
            if selected.any() and not np.isclose(row_lp[selected][0], sample_logprob, rtol=1e-5, atol=1e-5):
                raise ValueError(f"Sampler head and sampled log probability disagree at row {row}.")

    @classmethod
    def from_topk_rows(cls, token_ids, log_probs, sampled: list, *, strict: bool = True) -> SamplerDistribution:
        """
        Build a distribution from a fixed-width head decoded from the /generate response.

        Finite entries form the sampler support. An ``-inf`` entry is the completeness sentinel for
        its row and its id is dropped. The /generate endpoint has no environment turns, so every row
        is present and offsets follow the per-row finite counts.

        ``strict`` runs the per-row sampled-token checks, an O(rows) Python pass over every response.
        Normal training leaves it off and relies on the structural checks plus the engine's
        fail-closed capture. Enable it for CI-style validation runs.
        """
        ids = np.asarray(token_ids)
        logps = np.asarray(log_probs, dtype=np.float32)
        if ids.ndim != 2 or ids.shape != logps.shape or ids.shape[0] < 1 or ids.shape[1] < 1:
            raise ValueError(f"topk rows must be a shared [R, k] shape, got {ids.shape} and {logps.shape}.")
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Sampler token IDs must be integers.")
        finite = np.isfinite(logps)
        sentinel = np.isneginf(logps)
        if np.any((~finite) & (~sentinel)):
            raise ValueError("Sampler rows contain NaN or positive infinity.")
        if np.any(logps[finite] > 1e-6) or np.any(ids[finite] < 0) or np.any(ids[finite] > np.iinfo(np.int32).max):
            raise ValueError("Sampler rows contain invalid IDs or log probabilities.")
        if np.any(np.maximum.accumulate(sentinel, axis=-1) & finite):
            raise ValueError("Finite sampler entries must precede the support sentinel.")
        complete = sentinel.any(-1)
        counts = finite.sum(-1)
        offsets = np.zeros(len(ids) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        distribution = cls(
            offsets=offsets,
            token_ids=ids[finite].astype(np.int32, copy=False),
            log_probs=logps[finite].astype(np.float32, copy=False),
            complete=complete,
        )
        if strict:
            distribution._validate_sampled(sampled)
        return distribution

    @classmethod
    def concatenate(cls, segments: list[SamplerDistribution]) -> SamplerDistribution:
        """
        Join turns and environment rows with one allocation per entry buffer.
        """
        if not segments:
            return cls.empty(0)
        counts = np.concatenate([np.diff(segment.offsets) for segment in segments])
        offsets = np.empty(len(counts) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        return cls(
            offsets,
            np.concatenate([segment.token_ids for segment in segments]),
            np.concatenate([segment.log_probs for segment in segments]),
            np.concatenate([segment.complete for segment in segments]),
        )

    def validate_capture(self, capture: SamplerCapture, response_mask: list[int]) -> None:
        """
        Reject an incomplete support or misaligned rows before training.

        The row-alignment and support-completeness checks read the mode and budget from ``capture``,
        so the request width and the validation cannot drift apart.
        """
        active = np.asarray(response_mask, dtype=np.bool_)
        if active.shape != (len(self),):
            raise ValueError("Sampler distribution and response mask lengths differ.")
        if not capture.enabled:
            raise ValueError("Sampler distribution present while capture is disabled.")
        counts = np.diff(self.offsets)
        if np.any(counts[active] == 0) or np.any(counts[~active] != 0):
            raise ValueError("Sampler distribution rows do not match the response mask.")
        if capture.mode == "support" and (not self.complete[active].all() or np.any(counts > capture.budget)):
            raise ValueError("Sampler support exceeds the capture budget. Increase sampler_max_tokens.")
