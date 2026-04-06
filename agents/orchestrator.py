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

def _mcp_tool_to_anthropic(tool) -> dict:
    """Convert an MCP Tool object to Anthropic API tool format."""
    schema = tool.inputSchema
    # Ensure it's a proper dict
    if hasattr(schema, "model_dump"):
        schema = schema.model_dump(exclude_none=True)
    elif not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}, "required": []}
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": schema,
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

        # Collect tools from both servers
        weather_tools_resp = await weather_session.list_tools()
        news_tools_resp = await news_session.list_tools()

        anthropic_tools: list[dict] = []
        tool_session_map: dict[str, "ClientSession"] = {}

        for t in weather_tools_resp.tools:
            anthropic_tools.append(_mcp_tool_to_anthropic(t))
            tool_session_map[t.name] = weather_session

        for t in news_tools_resp.tools:
            anthropic_tools.append(_mcp_tool_to_anthropic(t))
            tool_session_map[t.name] = news_session

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
                # Extract final text
                text_parts = [
                    block.text
                    for block in resp.content
                    if hasattr(block, "text")
                ]
                return "\n".join(text_parts), tool_call_log

            if resp.stop_reason == "tool_use":
                # Append assistant message (with tool_use blocks)
                messages.append(
                    {"role": "assistant", "content": resp.content}
                )

                # Process each tool call
                tool_results = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue

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
                        result_text = json.dumps(
                            {"error": f"Unknown tool: {tool_name}"}
                        )
                        log_entry["error"] = f"Unknown tool: {tool_name}"
                    else:
                        try:
                            mcp_result = await session.call_tool(
                                tool_name, tool_input
                            )
                            # Extract text from MCP result content
                            if mcp_result.content:
                                result_text = "".join(
                                    c.text
                                    for c in mcp_result.content
                                    if hasattr(c, "text")
                                )
                            else:
                                result_text = json.dumps({"result": "empty"})
                            log_entry["output"] = result_text
                        except Exception as exc:
                            result_text = json.dumps({"error": str(exc)})
                            log_entry["error"] = str(exc)

                    tool_call_log.append(log_entry)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        }
                    )

                messages.append({"role": "user", "content": tool_results})

            else:
                # Unexpected stop reason
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
