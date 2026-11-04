import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from psrl.utils.common.http_utils import SESSION_TURN_BUDGET_HEADER, delete, get, post
from psrl.utils.rollout.vision_utils import normalize_messages
from psrl.utils.sampler_distribution import SamplerCapture, configure_capture
from psrl.utils.tito.training_data import build_training_data
from psrl.workers.agent_loop.agent_data import AgentData
from psrl.workers.agent_loop.context import AgentLoopContext
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.gen.smg_adapter import get_trajectory_id_strategy
from psrl.workers.gen.utils import TokenOutput

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass
class SessionAgentResult:
    """Business result returned by an agent using a session-scoped API URL."""

    extra_fields: dict = field(default_factory=dict)
    terminate_reason: TerminateReason = TerminateReason.FINISHED


class SessionAgentLoop(AgentLoopBase):
    """Provide SessionRouter/TITO support and a URL-only external-agent contract."""

    def __init__(
        self,
        context: AgentLoopContext,
    ):
        super().__init__(context=context)
        self.session_router_url = context.session_router_url.rstrip("/")
        self.trajectory_id_strategy = get_trajectory_id_strategy(context.config)
        self.max_turns = context.config.gen_actor_rollout_ref.rollout.multi_turn.max_turns
        self.session_snapshot = None  # TITO session snapshot for offline analysis
        self._capture_sessions: set[str] = set()

    def get_generate_fields(self) -> list[str]:
        fields = super().get_generate_fields()
        fields.extend(["env_class", "data_class", "seed"])
        return fields

    def build_session_headers(self, request: dict) -> dict[str, str]:
        """Build immutable routing metadata for one session."""
        request_id = request["uid"]
        headers = {
            "x-request-id": str(request_id),
            "x-prompt-id": str(request.get("parent_id", request_id)),
            "x-version-tag": str(request.get("version_tag", 0)),
            "x-is-validate": str(request.get("validate", False)).lower(),
            "x-is-sticky": str(
                bool(self.config.psrl.rollout_coordination.routing_strategy.enable_trajectory_sticky)
            ).lower(),
        }
        rollout_instance_id = request.get("rollout_instance_id")
        if rollout_instance_id is not None:
            headers["x-base-worker-id"] = str(rollout_instance_id[0])
            headers["x-target-dp-rank"] = str(rollout_instance_id[1])
        return headers

    def session_turn_budget(self) -> int | None:
        """Return the model-call budget this session declares, or None when it declares none.

        Only a loop whose harness CLI takes no turn cap of its own declares one: for such a
        CLI the router is the only component that sees every model call. A loop with a cap
        in the CLI declares nothing, because that CLI already counts the parent agent's
        turns and a second counter would have to guess at the same number.
        """
        return None

    async def create_session(self, request: dict) -> str:
        """Create one TITO session and bind its routing metadata."""
        headers = self.build_session_headers(request)
        # Optional session-scoped prompt-too-long budget. SMG returns an Anthropic
        # `prompt_too_long` error so Claude Code reactively compacts within the budget.
        limit = getattr(self, "prompt_too_long_limit", None)
        if limit is not None:
            headers["x-smg-prompt-too-long-limit"] = str(limit)
        budget = self.session_turn_budget()
        if budget is not None:
            headers[SESSION_TURN_BUDGET_HEADER] = str(budget)
        response = await post(
            f"{self.session_router_url}/sessions",
            payload={},
            headers=headers,
        )
        session_id = response["session_id"]
        if (
            not request.get("validate", False)
            and self.rollout_config.get("sampler_distribution", "disabled") != "disabled"
        ):
            self._capture_sessions.add(session_id)
        psrl_logger.debug("Created TITO session %r.", session_id)
        return session_id

    async def delete_session(self, session_id: str) -> None:
        """Delete one TITO session after its requests have drained."""
        self._capture_sessions.discard(session_id)
        response = await delete(f"{self.session_router_url}/sessions/{session_id}")
        psrl_logger.debug("Deleted TITO session %r with status %r.", session_id, response.status)

    async def get_session_data(self, session_id: str) -> dict:
        """Fetch one TITO session snapshot."""
        return await get(f"{self.session_router_url}/sessions/{session_id}")

    @asynccontextmanager
    async def session_scope(self, request: dict) -> AsyncIterator[str]:
        """Create and reliably clean up one session for an external agent."""
        session_id = await self.create_session(request)
        try:
            yield session_id
        finally:
            await self.delete_session(session_id)

    def session_api_url(self, session_id: str) -> str:
        """Return the OpenAI-compatible API base for one session."""
        return f"{self.session_root_url(session_id)}/v1"

    def session_root_url(self, session_id: str, base_url: str | None = None) -> str:
        """Return the protocol-neutral root for one session."""
        root = (base_url or self.session_router_url).rstrip("/")
        return f"{root}/sessions/{session_id}"

    def attach_tito_tree_metadata(self, training_data: list[dict], outputs: list["TokenOutput"]) -> None:
        """Attach SMG TITO tree/leaf metadata to each generated output.

        Reads the snapshot cached by :meth:`get_training_data`. When the SMG
        gateway exports the ``tree`` block, each ``TokenOutput`` gets:

        - ``tito_tree``: the full session prefix-tree (nodes with id/hash/parent/
          finish_reason/truncated/num_tokens/trajectory_ids, leaves with a
          root-to-leaf path of node ids).
        - ``tito_leaf``: the leaf descriptor of this trajectory (node_id, parent,
          path_node_ids) enriched with the leaf node's finish_reason/truncated.

        The generic trajectory dumper then persists ``{uid}.tree.json`` next to the
        per-trajectory ``{uid}_{idx}.txt`` files for offline lineage analysis.
        """
        payload = self.session_snapshot or {}
        tree = payload.get("tree") if isinstance(payload, dict) else None
        if not tree:
            return
        nodes = {str(n.get("id")): n for n in tree.get("nodes", []) or []}
        leaves = {str(leaf.get("trajectory_id")): leaf for leaf in tree.get("leaves", []) or []}
        for item, output in zip(training_data, outputs):
            leaf = leaves.get(str(item.get("trajectory_id")))
            if leaf:
                leaf = dict(leaf)
                node = nodes.get(str(leaf.get("node_id"))) or {}
                leaf.setdefault("finish_reason", node.get("finish_reason"))
                leaf.setdefault("truncated", bool(node.get("truncated")))
                leaf.setdefault("num_tokens", node.get("num_tokens"))
            output.extra_fields.setdefault("tito_leaf", leaf)
            output.extra_fields.setdefault("tito_tree", tree)

    async def run_session(
        self,
        request: dict,
        api_base_url: str,
    ) -> SessionAgentResult:
        """Run an external agent after replacing its normal API base URL."""
        raise NotImplementedError(f"{type(self).__name__} must implement run_session().")

    async def run(
        self,
        request: dict,
    ) -> tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
        """Run a URL-configured agent and collect its TITO trajectories."""
        async with self.session_scope(request) as session_id:
            result = await self.run_session(request, self.session_api_url(session_id))
            training_data = await self.get_training_data(session_id)
            if not training_data or any(item["num_turns"] == 0 or not item["response_ids"] for item in training_data):
                return None, TerminateReason.ROLLOUT_ERROR

            outputs = [self.build_token_output(item, extra_fields=result.extra_fields) for item in training_data]
            output: TokenOutput | list[TokenOutput] = outputs[0] if len(outputs) == 1 else outputs
            self.attach_tito_tree_metadata(training_data, outputs)
            scored_output = await self.compute_reward_score(output, **request)
            if scored_output is None:
                return None, TerminateReason.ABORTED
            return scored_output, result.terminate_reason

    def get_session_sampling_params(self, request: dict) -> dict:
        """Build per-turn sampling parameters and the TITO training contract."""
        config = self.rollout_config
        if request.get("validate", False):
            config = self.config.train_actor_rollout_ref.rollout.val_kwargs
        params = {
            "temperature": float(config.temperature),
            "top_p": float(config.top_p),
            "top_k": int(config.top_k),
            "repetition_penalty": float(self.rollout_config.get("repetition_penalty", 1.0)),
            "ignore_eos": self.rollout_config.get("ignore_eos", False),
            "max_tokens": int(self.rollout_config.response_length),
            "logprobs": True,
            "top_logprobs": 1,
            "stream": False,
        }
        if request.get("seed") is not None:
            params["seed"] = int(request["seed"])
        capture = SamplerCapture() if request.get("validate", False) else self.sampler_capture
        configure_capture(params, capture, protocol="chat")
        return params

    async def chat_completion(
        self,
        session_id: str,
        messages: list[dict],
        sampling_params: dict,
        *,
        tools: list[dict] | None = None,
        chat_template_kwargs: dict | None = None,
        multi_modal_data: dict | None = None,
        trajectory_id: int | str | None = None,
    ) -> dict:
        """
        Send one chat completion request through a session.

        The `trajectory_id` request header preserves independent model contexts
        while retaining session routing and version pinning.
        """
        messages = await normalize_messages(
            messages,
            mm_data=multi_modal_data,
        )
        payload = {
            "model": self.model_config.path,
            "messages": messages,
            **sampling_params,
        }
        if tools:
            payload["tools"] = tools
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        headers = None
        if self.trajectory_id_strategy == "manual" and trajectory_id is not None:
            headers = {"x-smg-tito-trajectory-id": str(trajectory_id)}
        with self.timer.generation():
            return await post(
                f"{self.session_api_url(session_id)}/chat/completions",
                payload=payload,
                headers=headers,
            )

    async def get_training_data(self, session_id: str) -> list[dict]:
        """Fetch and convert every trajectory in a TITO session snapshot."""
        session_data = await self.get_session_data(session_id)
        self.session_snapshot = session_data
        trajectories = session_data.get("trajectories")
        if not isinstance(trajectories, list):
            raise RuntimeError(
                f"TITO session {session_id!r} returned an invalid snapshot: expected a trajectories list."
            )
        training_data = [
            self.build_training_data_from_trajectory(trajectory, session_data) for trajectory in trajectories
        ]
        if session_id in self._capture_sessions and any(
            item["response_ids"] and item.get("sampler_distribution") is None for item in training_data
        ):
            raise ValueError("Sampler capture was requested but TITO returned no distribution.")
        return training_data

    async def get_primary_training_data(self, session_id: str) -> dict:
        """Collect the single linear trajectory expected by native session loops."""
        training_data = await self.get_training_data(session_id)
        if len(training_data) != 1:
            raise RuntimeError(
                f"TITO session {session_id!r} produced {len(training_data)} trajectories; "
                "expected one linear trajectory."
            )
        return training_data[0]

    def build_training_data_from_trajectory(
        self,
        trajectory: dict,
        session_data: dict,
    ) -> dict:
        """Convert one trajectory from an already-fetched session snapshot."""
        records = trajectory.get("records", [])
        if self.config.psrl.rollout_gateway.tito_debug:
            self._validate_records(records)
        training_data = build_training_data(
            trajectory.get("accumulated_token_ids", []),
            records,
            max_trim_tokens=session_data.get("max_trim_tokens", 0),
            strict_validation=self.rollout_config.get("sampler_strict_validation", False),
        )
        distribution = training_data.get("sampler_distribution")
        if distribution is not None:
            distribution.validate_capture(self.sampler_capture, training_data["response_mask"])
        training_data["trajectory_id"] = trajectory["trajectory_id"]
        header_info = session_data.get("header_info", {})
        if header_info.get("base_worker_id") is not None and header_info.get("target_dp_rank") is not None:
            training_data["rollout_instance_id"] = (
                header_info["base_worker_id"],
                int(header_info["target_dp_rank"]),
            )
        training_data["finish_reason"] = records[-1].get("finish_reason") if records else None
        return training_data

    def build_token_output(self, training_data: dict, *, extra_fields: dict | None = None) -> TokenOutput:
        """Convert one TITO trajectory into the canonical rollout output.

        The response is clamped to the trainable context budget left after the
        prompt, not to `rollout.response_length`. A multi-turn trajectory is one
        sample whose real limit is the total context, so clamping at
        `response_length` would drop the final turns of a long valid episode.
        Without any clamp an over-long sample trips `rearrange_micro_batches`'
        `max_token_len >= max_seq_len` assertion instead of being truncated.
        """
        prompt_ids = training_data["prompt_ids"]
        response_budget = max(0, self.rollout_budget - len(prompt_ids))
        response_ids = training_data["response_ids"][:response_budget]
        response_mask = training_data["response_mask"][:response_budget]
        logprobs = training_data["logprobs"]
        response_log_probs = logprobs[:response_budget] if logprobs else None
        routed_experts = training_data["routed_experts"]
        if routed_experts is not None:
            routed_experts = routed_experts[: len(prompt_ids) + response_budget]

        trajectory_fields = dict(extra_fields or {})
        trajectory_fields["trajectory_id"] = training_data["trajectory_id"]
        distribution = training_data.get("sampler_distribution")
        if distribution is not None:
            distribution = distribution.prefix(len(response_ids))
            distribution.validate_capture(self.sampler_capture, response_mask)
        return TokenOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_log_probs=response_log_probs,
            routed_experts=routed_experts,
            sampler_distribution=distribution,
            sampler_capture=self.sampler_capture,
            stop_reason=training_data.get("finish_reason"),
            num_turns=training_data["num_turns"],
            rollout_instance_id=training_data.get("rollout_instance_id"),
            extra_fields=trajectory_fields,
        )

    @staticmethod
    def attach_training_data(
        agent_data: AgentData,
        training_data: dict,
        *,
        update_turn_counts: bool = False,
    ) -> None:
        """Attach canonical TITO training data and optional turn counts."""
        trajectory = agent_data.session_data.trajectories[-1]
        trajectory.prompt_ids = training_data["prompt_ids"]
        trajectory.response_ids = training_data["response_ids"]
        trajectory.response_mask = training_data["response_mask"]
        trajectory.response_logprobs = training_data["logprobs"]
        trajectory.sampler_distribution = training_data.get("sampler_distribution")
        trajectory.response_length = len(trajectory.response_ids)
        if update_turn_counts:
            trajectory.assistant_turns = training_data["num_turns"]
            trajectory.user_turns = max(0, training_data["num_turns"] - 1)
        if training_data["routed_experts"] is not None:
            trajectory.routed_experts = training_data["routed_experts"]

        agent_data.session_data.response_length = trajectory.response_length
        if update_turn_counts:
            agent_data.session_data.assistant_turns = trajectory.assistant_turns
            agent_data.session_data.user_turns = trajectory.user_turns
        agent_data.session_data.curr_rollout_instance_id = training_data.get("rollout_instance_id")

    @staticmethod
    def _validate_records(records: list[dict]) -> None:
        for turn, record in enumerate(records):
            for mismatch in record.get("mismatch_report", []):
                # The TITO re-tokenization comparator is a DEBUG diagnostic. Its mismatches fire on
                # truncated generations and mid-stream special tokens, so warn without aborting.
                psrl_logger.warning("TITO token mismatch at turn %d: %r", turn, mismatch)
