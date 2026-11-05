"""Codex CLI harness adapter."""

import functools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from psrl.workers.agent_loop.harness.base import Harness, HarnessRuntime
from psrl.workers.agent_loop.harness.runtime import executable_path

# Codex reads its tool catalog (including `apply_patch`) from the model catalog,
# not from config: `apply_patch` is exposed to the model only when the model's
# `apply_patch_tool_type` is `freeform`. A custom model slug would otherwise hit
# Codex's built-in fallback metadata (apply_patch disabled), so the adapter writes
# a static catalog entry and points Codex at it with `model_catalog_json`.
_BASE_INSTRUCTIONS_PATH = Path(__file__).parent / "assets" / "codex_base_instructions.md"

# Codex's own reasoning levels, in the shape the catalog expects. The configured
# effort is validated against these.
_SUPPORTED_REASONING_LEVELS = (
    ("low", "Fast responses with lighter reasoning"),
    ("medium", "Balances speed and reasoning depth for everyday tasks"),
    ("high", "Greater reasoning depth for complex problems"),
    ("xhigh", "Extra high reasoning depth for complex problems"),
    ("max", "Maximum reasoning depth for the hardest problems"),
)


@functools.lru_cache(maxsize=1)
def _base_instructions() -> str:
    """The Codex stock system prompt, shipped verbatim as a pinned asset.

    Codex's catalog deserializer rejects an entry that carries neither
    `base_instructions` nor `model_messages.instructions_template`, so the model
    catalog cannot be minimal. Keeping the stock prompt also preserves the
    behavior the harness is meant to elicit (this mirrors how the Claude Code
    recipe keeps its stock system prompt).
    """
    return _BASE_INSTRUCTIONS_PATH.read_text(encoding="utf-8")


class CodexHarness(Harness):
    """Run Codex against a TITO session-scoped Responses endpoint."""

    def config_dir(self) -> str:
        return str(PurePosixPath(self.config.home_dir) / ".codex")

    async def _prepare(self, runtime: HarnessRuntime) -> None:
        codex_dir = PurePosixPath(self.config_dir())
        model = self.config.model or runtime.model
        effort = self.config.reasoning_effort or "medium"
        levels = {name for name, _ in _SUPPORTED_REASONING_LEVELS}
        if effort not in levels:
            raise ValueError(
                f"Codex reasoning_effort {effort!r} is not one of: {', '.join(sorted(levels))}."
            )

        # Choose a synthetic window that makes a requested auto-compact token
        # threshold effective: Codex clamps the trigger to 90% of the resolved
        # window, so the window must be at least ceil(limit / 0.9).
        catalog_window: int | None = runtime.context_window_tokens
        if runtime.context_window_tokens and runtime.compaction_token_limit:
            catalog_window = max(
                runtime.context_window_tokens,
                math.ceil(runtime.compaction_token_limit / 0.9),
            )

        # Top-level keys must be emitted before the first table header, otherwise
        # TOML would nest them under the preceding table.
        config_lines = [
            f"model = {json.dumps(model)}",
            'model_provider = "psrl"',
            'approval_policy = "never"',
            'sandbox_mode = "danger-full-access"',
            f"model_reasoning_effort = {json.dumps(effort)}",
            'model_reasoning_summary = "auto"',
            f"model_catalog_json = {json.dumps(str(codex_dir / 'model_catalog.json'))}",
            # Codex enables its hosted web_search tool by default (Feature::WebSearchCached),
            # but this gateway does not serve hosted tools, and the task forbids fetching the
            # solution from the web. Disabling it also keeps Codex from sending a tool whose
            # schema the gateway rejects.
            'web_search = "disabled"',
        ]
        if runtime.context_window_tokens and runtime.compaction_token_limit:
            config_lines.extend(
                (
                    f"model_context_window = {catalog_window}",
                    f"model_auto_compact_token_limit = {runtime.compaction_token_limit}",
                )
            )
        if self.config.subagents_enabled:
            # The catalog's `multi_agent_version` is what a custom provider uses to
            # expose the tools; this feature flag is Codex's older fallback for the
            # same surface and is harmless when the catalog already names a version.
            config_lines.extend(("", "[features]", "multi_agent = true"))
        config_lines.extend(
            (
                "",
                "[analytics]",
                "enabled = false",
                "",
                "[feedback]",
                "enabled = false",
                "",
                "[otel]",
                'exporter = "none"',
                'trace_exporter = "none"',
                'metrics_exporter = "none"',
                "",
                "[model_providers.psrl]",
                'name = "PSRL TITO"',
                f"base_url = {json.dumps(runtime.session_root_url.rstrip('/') + '/v1')}",
                'env_key = "OPENAI_API_KEY"',
                'wire_api = "responses"',
                "requires_openai_auth = false",
                "supports_websockets = false",
                "",
            )
        )
        config = "\n".join(config_lines)
        await self.sandbox.write_bytes(str(codex_dir / "config.toml"), config.encode())
        await self.sandbox.write_bytes(
            str(codex_dir / "model_catalog.json"),
            self._model_catalog(model, effort, catalog_window),
        )

    def _model_catalog(self, model: str, effort: str, context_window: int | None) -> bytes:
        """Build the static model catalog Codex loads via `model_catalog_json`.

        Every field is written explicitly because Codex's catalog deserializer
        requires the non-optional `ModelInfo` fields to be present. The catalog is
        authoritative for the subagent tool surface: `multi_agent_version` is the
        only switch that enables Codex's native subagent tools on a custom provider.
        """
        entry: dict = {
            "slug": model,
            "display_name": model,
            "description": None,
            "default_reasoning_level": effort,
            "supported_reasoning_levels": [
                {"effort": name, "description": description}
                for name, description in _SUPPORTED_REASONING_LEVELS
            ],
            "shell_type": "unified_exec",
            "visibility": "list",
            "supported_in_api": True,
            "priority": 0,
            "additional_speed_tiers": [],
            "service_tiers": [],
            "default_service_tier": None,
            "available_access_programs": None,
            "availability_nux": None,
            "upgrade": None,
            "model_messages": {"instructions_template": _base_instructions()},
            "include_skills_usage_instructions": False,
            "include_plugin_usage_instructions": False,
            "include_apps_usage_instructions": False,
            "supports_reasoning_summary_parameter": True,
            "default_reasoning_summary": "auto",
            "support_verbosity": False,
            "default_verbosity": None,
            # The whole point of shipping a catalog: enable Codex's native patch tool.
            "apply_patch_tool_type": "freeform",
            "web_search_tool_type": "text",
            "truncation_policy": {"mode": "tokens", "limit": 10000},
            "supports_image_detail_original": False,
            "context_window": context_window,
            "max_context_window": context_window,
            "auto_compact_token_limit": None,
            "comp_hash": None,
            "effective_context_window_percent": 95,
            "experimental_supported_tools": [],
            "input_modalities": ["text"],
            "supports_search_tool": False,
            "supports_experimental_context": False,
            "use_responses_lite": False,
            "supports_reasoning_effort_updates": False,
            "node_repl_auto_review_required": False,
            "node_repl_disabled": False,
            "auto_review_model_override": None,
            "model_specialty": None,
            "tool_mode": None,
            # Codex 0.164 supports the namespaced v1 subagent tools. Select v1 because
            # its spawn surface is the stable one for a custom provider; v2's flat tool
            # names are emitted once the CLI pins that surface by default.
            "multi_agent_version": "v1" if self.config.subagents_enabled else None,
            "multi_agent_reasoning_effort": None,
        }
        return json.dumps({"models": [entry]}, ensure_ascii=False).encode()

    def build_command(self, prompt: str, runtime: HarnessRuntime) -> Sequence[str]:
        return (
            executable_path(self.config.runtime_mount, self.config.executable),
            "exec",
            "--skip-git-repo-check",
            "--json",
            *self.config.args,
            prompt,
        )

    def build_env(self, runtime: HarnessRuntime) -> Mapping[str, str]:
        return self.callback_no_proxy(
            runtime,
            {
                **self.inherited_proxy_env(),
                **self.config.env,
                "HOME": self.config.home_dir,
                "OPENAI_API_KEY": runtime.session_id,
                "CODEX_HOME": self.config_dir(),
            },
        )
