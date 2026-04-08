"""
Agent Orchestrator
==================
Connects to the Weather and News MCP servers via stdio subprocesses,
collects their tools, and runs a Claude-powered agent loop that calls
those tools on demand.

Usage (from async code):
    orchestrator = WeatherNewsOrchestrator(api_key="sk-ant-...")
    reply = await orchestrator.process_query("What's the weather in Paris?")
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import anthropic

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_SERVERS_DIR = _HERE.parent / "mcp_servers"
WEATHER_SERVER = str(_SERVERS_DIR / "weather_server.py")
NEWS_SERVER = str(_SERVERS_DIR / "news_server.py")

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a knowledgeable and friendly AI assistant that specialises in:
1. **Current weather** — real-time conditions and multi-day forecasts for any city worldwide.
2. **Latest news** — top stories and topic-specific searches from GNews.io.

Guidelines:
- Always use the provided tools to fetch live data before answering.
- Present weather data clearly: include temperature (°C), description, humidity, wind, and feels-like.
- Present news as a concise bulleted list with title, points, summary, and URL.
- If asked about both weather and news in one query, address both.
- Be friendly, concise, and factually accurate based on the tool output.
- If a location is not found, politely say so and suggest trying a different name.
- Units: temperature in °F, wind speed in miles/h, precipitation in mm.
"""

# ── Tool conversion helpers ────────────────────────────────────────────────────

def _sanitize_schema(raw) -> dict:
    """
    Return a clean JSON Schema dict that Anthropic's API accepts.

    FastMCP emits extra fields (title, $defs, additionalProperties, etc.)
    that are valid JSON Schema but cause Anthropic's tool validator to
    silently drop or reject the tool definition.  We keep only the three
    fields the API actually uses: type, properties, required.
    """
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(exclude_none=True)
    if not isinstance(raw, dict):
        return {"type": "object", "properties": {}}

    schema: dict = {"type": "object"}

    props = raw.get("properties", {})
    if props:
        # Strip per-property 'title' fields too — they're noise to the API
        schema["properties"] = {
            k: {pk: pv for pk, pv in v.items() if pk != "title"}
            if isinstance(v, dict) else v
            for k, v in props.items()
        }
    else:
        schema["properties"] = {}

    required = raw.get("required", [])
    if required:
        schema["required"] = required

    return schema


def _mcp_tool_to_anthropic(tool) -> dict:
    """Convert an MCP Tool object to Anthropic API tool format."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": _sanitize_schema(tool.inputSchema),
    }


# ── Main orchestrator class ────────────────────────────────────────────────────

class WeatherNewsOrchestrator:
    """
    Agent that orchestrates two MCP servers (weather + news) using Claude.

    Each call to `process_query` spawns fresh MCP server sub-processes,
    runs the full agent loop (potentially multiple tool calls), then
    terminates the servers.  This is safe for multi-user Streamlit apps.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        gnews_api_key: str = "",
    ):
        self.api_key = api_key
        self.model = model
        self.gnews_api_key = gnews_api_key

    # ── Public API ─────────────────────────────────────────────────────────────

    async def process_query(
        self,
        query: str,
        history: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Process a user query using the agent loop.

        Returns a dict with keys:
          - "response" (str): The final text answer.
          - "tool_calls" (list): Log of every tool call made.
          - "model" (str): Model used.
          - "error" (str | None): Any fatal error message.
        """
        if not MCP_AVAILABLE:
            return {
                "response": "MCP library is not installed. Please run: pip install mcp",
                "tool_calls": [],
                "model": self.model,
                "error": "MCP not available",
            }

        tool_call_log: list[dict] = []
        error: str | None = None
        response_text = ""

        try:
            weather_params = StdioServerParameters(
                command=sys.executable, args=[WEATHER_SERVER]
            )
            # Inject the GNews API key into the news server's environment.
            # The subprocess reads it via os.environ — nothing touches disk.
            news_env = {**os.environ, "GNEWS_API_KEY": self.gnews_api_key}
            news_params = StdioServerParameters(
                command=sys.executable, args=[NEWS_SERVER], env=news_env
            )

            async with stdio_client(weather_params) as (wr, ww):
                async with ClientSession(wr, ww) as weather_session:
                    await weather_session.initialize()

                    async with stdio_client(news_params) as (nr, nw):
                        async with ClientSession(nr, nw) as news_session:
                            await news_session.initialize()

                            response_text, tool_call_log = await self._agent_loop(
                                query=query,
                                history=history or [],
                                weather_session=weather_session,
                                news_session=news_session,
                            )

        except Exception as exc:
            error = str(exc)
            response_text = (
                f"An error occurred while processing your request: {error}\n\n"
                "Please check your API key and ensure the MCP servers can start."
            )

        return {
            "response": response_text,
            "tool_calls": tool_call_log,
            "model": self.model,
            "error": error,
        }

    # ── Internal agent loop ────────────────────────────────────────────────────

    async def _agent_loop(
        self,
        query: str,
        history: list[dict],
        weather_session: "ClientSession",
        news_session: "ClientSession",
        max_iterations: int = 10,
    ) -> tuple[str, list[dict]]:
        """Run the Claude tool-use loop until a final text response is returned."""

        # ── Fix 2: validate server capabilities before calling list_tools ──────
        # The MCP spec requires clients to check that the server advertised
        # 'tools' support in its initialize response before issuing list_tools.
        # We read server_capabilities defensively: different versions of the
        # mcp library may store it under different attribute names or not at
        # all.  If we cannot determine capabilities we proceed optimistically
        # (both our servers are known to support tools).
        anthropic_tools: list[dict] = []
        tool_session_map: dict[str, "ClientSession"] = {}

        for label, session in [("weather", weather_session), ("news", news_session)]:
            # Try known attribute names across mcp library versions
            caps = (
                getattr(session, "server_capabilities", None)
                or getattr(session, "_server_capabilities", None)
            )
            if caps is not None:
                # Capabilities found — honour the spec: skip if tools not declared
                if getattr(caps, "tools", None) is None:
                    continue
            # caps is None → library version doesn't expose them; proceed anyway
            try:
                tools_resp = await session.list_tools()
            except Exception:
                # Server truly doesn't support list_tools — skip
                continue
            for t in tools_resp.tools:
                anthropic_tools.append(_mcp_tool_to_anthropic(t))
                tool_session_map[t.name] = session

        client = anthropic.Anthropic(api_key=self.api_key)

        # Build message history (shallow copy so we don't mutate caller's list)
        messages: list[dict] = list(history)
        messages.append({"role": "user", "content": query})

        tool_call_log: list[dict] = []

        for _iteration in range(max_iterations):
            resp = client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=anthropic_tools,
            )

            if resp.stop_reason == "end_turn":
                text_parts = [
                    block.text
                    for block in resp.content
                    if hasattr(block, "text")
                ]
                return "\n".join(text_parts), tool_call_log

            # ── Fix 3: handle max_tokens explicitly ───────────────────────────
            if resp.stop_reason == "max_tokens":
                text_parts = [
                    block.text
                    for block in resp.content
                    if hasattr(block, "text")
                ]
                partial = "\n".join(text_parts)
                return (
                    partial + "\n\n_(Note: the response was truncated because "
                    "it reached the maximum length. Try asking a more specific "
                    "question for a complete answer.)_"
                ), tool_call_log

            if resp.stop_reason == "tool_use":
                messages.append(
                    {"role": "assistant", "content": resp.content}
                )

                tool_use_blocks = [b for b in resp.content if b.type == "tool_use"]

                # ── Fix 4: run all tool calls in this turn concurrently ────────
                async def _invoke(block) -> tuple[dict, dict]:
                    """Call one MCP tool and return (log_entry, tool_result)."""
                    tool_name: str = block.name
                    tool_input: dict = block.input
                    session = tool_session_map.get(tool_name)

                    log_entry = {
                        "tool": tool_name,
                        "input": tool_input,
                        "output": None,
                        "error": None,
                    }

                    if session is None:
                        err = f"Unknown tool: {tool_name}"
                        log_entry["error"] = err
                        result_text = json.dumps({"error": err})
                        is_error = True
                    else:
                        try:
                            mcp_result = await session.call_tool(tool_name, tool_input)

                            # ── Fix 1: check isError on the MCP result ────────
                            is_error = getattr(mcp_result, "isError", False)
                            if mcp_result.content:
                                result_text = "".join(
                                    c.text
                                    for c in mcp_result.content
                                    if hasattr(c, "text")
                                )
                            else:
                                result_text = json.dumps({"result": "empty"})

                            if is_error:
                                log_entry["error"] = result_text
                            else:
                                log_entry["output"] = result_text

                        except Exception as exc:
                            is_error = True
                            result_text = json.dumps({"error": str(exc)})
                            log_entry["error"] = str(exc)

                    tool_result = {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                        # Propagate isError so Claude knows the tool failed
                        **({"is_error": True} if is_error else {}),
                    }
                    return log_entry, tool_result

                # Dispatch all tool calls concurrently
                outcomes = await asyncio.gather(*[_invoke(b) for b in tool_use_blocks])

                for log_entry, tool_result in outcomes:
                    tool_call_log.append(log_entry)

                tool_results = [tr for _, tr in outcomes]
                messages.append({"role": "user", "content": tool_results})

            else:
                # Unrecognised stop reason — surface it rather than silently drop
                break

        return (
            "I reached the maximum number of steps without a final answer. "
            "Please try a simpler question.",
            tool_call_log,
        )

    # ── Utility ────────────────────────────────────────────────────────────────

    def get_tool_list(self) -> list[str]:
        """Return a static list of known tools (for UI display)."""
        return [
            "get_current_weather",
            "get_weather_forecast",
            "get_top_news",
            "get_recent_news",
            "search_news",
            "get_news_by_topic",
        ]
