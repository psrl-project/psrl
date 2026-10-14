import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psrl.tools.mcp_clients.schema import MCPToolSchema
from psrl.utils.concurrency import TokenBucket

psrl_logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    name: str
    url: str
    auth_token: str | None = None


class MCPClientManager:
    """Manage MCP clients, tool discovery and tool calls.

    Design goals:
    - Keep a single shared manager per process (cheap to reuse).
    - Reuse long-lived clients and map tool_name -> client.
    - Centralized rate limiting.
    """

    root_server_name = "mcpServers"

    def __init__(
        self,
        servers_config_path: str,
        rate_limit: float = 120.0,
        timeout: float = 30.0,
    ):
        self.servers_config_path = servers_config_path
        self.timeout = timeout

        self._initialized = False
        self._clients: list[Any] = []
        self._tool_client_mapping: dict[str, Any] = {}
        self._rate_limiter = TokenBucket(rate_limit)

    async def initialize(self) -> None:
        """Initialize MCP clients from config."""
        if self._initialized:
            return

        from fastmcp import Client
        from fastmcp.client.transports import SSETransport

        # Load MCP servers from config.
        servers = self._load_servers(self.servers_config_path)

        # `fastmcp` accepts a dictionary for non-SSE servers and explicit transports for SSE servers.
        non_sse_config = {self.root_server_name: {}}

        for s in servers:
            if s.auth_token:
                transport = SSETransport(url=s.url, headers={"Authorization": f"Bearer {s.auth_token}"})
                self._clients.append(Client(transport))
            else:
                non_sse_config[self.root_server_name][s.name] = {"url": s.url}

        if non_sse_config[self.root_server_name]:
            self._clients.append(Client(non_sse_config))

        self._initialized = True

    async def fetch_tool_schemas(self, tool_selected_list: list[str] | None = None) -> list[MCPToolSchema]:
        """Fetch available MCP tool schemas from all clients.

        Args:
            tool_selected_list: Optional list of tool names to filter.
                If provided, only tools in this list are returned.
                If not provided, all discovered tools are returned.

        Returns:
            List of MCPToolSchema objects for the available tools.
        """
        await self.initialize()
        schemas: list[MCPToolSchema] = []

        for client in self._clients:
            async with client:
                tools = await client.list_tools_mcp()
                for mcp_tool in tools.tools:
                    name = mcp_tool.name
                    description = mcp_tool.description
                    params = mcp_tool.inputSchema

                    if tool_selected_list and name not in tool_selected_list:
                        continue

                    # map tool -> client
                    self._tool_client_mapping[name] = client
                    schemas.append(MCPToolSchema(name=name, description=description, parameters=params))

        return schemas

    async def call_tool(self, tool_name: str, parameters: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Call an MCP tool and return (text, metadata).

        Args:
            tool_name: Name of the tool to call.
            parameters: Parameters to pass to the tool.

        Returns:
            A tuple of (text content, metadata dict).
        """
        # rate limit (token bucket)
        await self._rate_limiter.async_acquire()

        if tool_name not in self._tool_client_mapping:
            raise KeyError(f"Tool '{tool_name}' not found in MCP tool mapping. Did you fetch schemas first?")

        client = self._tool_client_mapping[tool_name]
        async with client:
            result = await client.call_tool_mcp(tool_name, parameters)

        # Extract text content from result
        text_parts: list[str] = []
        if hasattr(result, "content"):
            if hasattr(result.content, "text"):
                content_str = result.content.text
            elif isinstance(result.content, list) and hasattr(result.content[0], "text"):
                content_str = result.content[0].text
            else:
                content_str = str(result.content)
        else:
            content_str = str(result)
        text_parts.append(content_str)

        return " ".join(text_parts).strip(), {}

    def _load_servers(self, config_path: str) -> list[MCPServerConfig]:
        """Load MCP server configs from a JSON file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"MCP servers config not found: {config_path}")

        with path.open(encoding="utf-8") as f:
            raw = json.load(f)

        servers_root = raw.get(self.root_server_name, {})
        servers: list[MCPServerConfig] = []
        for name, cfg in servers_root.items():
            if not isinstance(cfg, dict):
                continue
            url = cfg.get("url")
            if not url:
                raise ValueError(f"MCP server '{name}' missing URL in config: {config_path}")

            servers.append(MCPServerConfig(name=name, url=url, auth_token=cfg.get("auth_token")))

        if not servers:
            raise ValueError(f"No MCP servers found in config: {config_path}")

        return servers
