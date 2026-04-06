"""
Weather & News AI Assistant — Streamlit App
==========================================
A multi-tab Streamlit application powered by:
  • Claude (Anthropic) as the agent brain
  • Open-Meteo MCP Server  — free weather data, no API key
  • HackerNews MCP Server  — free news via Algolia, no API key

Tabs
----
  💬 Chat          — interactive conversation with the assistant
  📊 Evaluation    — run the built-in eval suite and view metrics
  🔧 Debug         — inspect tool calls for the last query
  ℹ️  About         — architecture and usage notes
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import sys
import threading
from pathlib import Path

import streamlit as st

# Propagate Streamlit's ScriptRunContext into background threads so that
# UI calls (st.progress, st.empty, etc.) work without the "missing
# ScriptRunContext" warning on Streamlit Community Cloud.
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
    _HAS_ST_CTX = True
except ImportError:
    _HAS_ST_CTX = False

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestrator import WeatherNewsOrchestrator
from evaluation.evaluator import (
    Evaluator,
    EvalResult,
    load_eval_dataset,
    run_evaluation_async,
    summarise_results,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Weather & News AI",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom CSS ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 2rem; color: #cba6f7; }
    .metric-card p  { margin: 4px 0 0; font-size: 0.85rem; color: #a6adc8; }

    .tool-call-box {
        background: #181825;
        border-left: 3px solid #89b4fa;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 10px;
        font-size: 0.82rem;
    }
    .chat-info { font-size: 0.78rem; color: #6c7086; margin-top: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state helpers ──────────────────────────────────────────────────────

def init_state():
    defaults = {
        "messages": [],           # chat history [{role, content}]
        "last_tool_calls": [],    # debug: tool calls from last query
        "last_model": "",
        "eval_results": [],       # list[EvalResult]
        "eval_summary": {},
        "eval_running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Weather_icon_-_sunny.svg/240px-Weather_icon_-_sunny.svg.png",
        width=60,
    )
    st.title("Weather & News AI")
    st.caption("Powered by Claude + MCP Servers")

    st.divider()
    st.subheader("⚙️ Configuration")

    # API key — always entered explicitly by the user
    api_key = st.text_input(
        "Anthropic API Key",
        value="",
        type="password",
        help="Get your free key at console.anthropic.com",
        placeholder="sk-ant-...",
    )

    # Model selector
    model = st.selectbox(
        "Claude Model",
        options=[
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
        ],
        index=0,
        help="Haiku 4.5 is fastest and cheapest; Sonnet 4.6 is more capable; Opus 4.6 is most powerful.",
    )

    st.divider()
    st.subheader("🛠️ Available Tools")
    tools = [
        ("🌡️", "get_current_weather", "Open-Meteo"),
        ("📅", "get_weather_forecast", "Open-Meteo"),
        ("📰", "get_top_news", "HackerNews"),
        ("🆕", "get_recent_news", "HackerNews"),
        ("🔍", "search_news", "HackerNews"),
        ("🏷️", "get_news_by_topic", "HackerNews"),
    ]
    for icon, name, source in tools:
        st.markdown(f"{icon} `{name}`  \n<span style='font-size:0.75rem;color:#888'>{source}</span>", unsafe_allow_html=True)

    st.divider()
    st.caption("No news API key required — uses HackerNews via Algolia Search API.")
    st.caption("No weather API key required — uses Open-Meteo.")


# ── Helper: run async code safely regardless of the host event loop ───────────

def _run_in_thread(coro):
    """
    Execute *coro* in a brand-new thread that owns its own event loop.

    Two problems solved here:
    1. uvloop incompatibility — Streamlit on macOS/Linux uses uvloop which
       nest_asyncio cannot patch. A plain new thread always gets a standard
       asyncio loop that asyncio.run() manages without any patching.
    2. Missing ScriptRunContext — Streamlit UI calls (st.progress, st.empty,
       etc.) made from a background thread emit a warning and may silently
       fail on Streamlit Community Cloud. We capture the context on the main
       thread and inject it into the worker thread before the coroutine runs.
    """
    ctx = get_script_run_ctx() if _HAS_ST_CTX else None

    def _target():
        if _HAS_ST_CTX and ctx is not None:
            add_script_run_ctx(threading.current_thread(), ctx)
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_target)
        return future.result()  # re-raises any exception from the thread


def run_query(query: str, api_key: str, model: str, history: list) -> dict:
    """Synchronous wrapper around the async orchestrator."""
    orchestrator = WeatherNewsOrchestrator(api_key=api_key, model=model)
    return _run_in_thread(orchestrator.process_query(query, history=history))


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_chat, tab_eval, tab_debug, tab_about = st.tabs(
    ["💬 Chat", "📊 Evaluation", "🔧 Debug", "ℹ️ About"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_chat:
    st.header("💬 Ask about Weather or News")

    # Quick-start example queries
    with st.expander("✨ Example queries — click to try"):
        examples = [
            "What's the weather in Tokyo right now?",
            "Give me the 5-day forecast for London.",
            "What are the top tech news stories today?",
            "Find me news about artificial intelligence.",
            "What's the weather in Paris and give me the latest news about climate change?",
            "Is it going to rain in New York this week?",
            "What are the latest startup funding news?",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state["_pending_query"] = ex

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_summary"):
                st.markdown(
                    f"<div class='chat-info'>🔧 Tools called: {msg['tool_summary']}</div>",
                    unsafe_allow_html=True,
                )

    # Chat input
    pending = st.session_state.pop("_pending_query", None)
    user_input = st.chat_input("Ask about weather or news…") or pending

    if user_input:
        if not api_key:
            st.error("⛔ Please enter your Anthropic API key in the sidebar.")
            st.stop()

        # Display user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build history for Claude (exclude the current message we just added)
        history_for_agent = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state["messages"][:-1]
            if m["role"] in ("user", "assistant")
        ]

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking… (fetching live data via MCP servers)"):
                try:
                    result = run_query(user_input, api_key, model, history_for_agent)
                    response = result["response"]
                    tool_calls = result.get("tool_calls", [])
                    error = result.get("error")

                    st.markdown(response)

                    tool_names = [tc["tool"] for tc in tool_calls]
                    tool_summary = ", ".join(f"`{n}`" for n in tool_names) if tool_names else None
                    if tool_summary:
                        st.markdown(
                            f"<div class='chat-info'>🔧 Tools called: {tool_summary}</div>",
                            unsafe_allow_html=True,
                        )

                    # Store for debug tab
                    st.session_state["last_tool_calls"] = tool_calls
                    st.session_state["last_model"] = result.get("model", model)

                    # Persist message
                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": response,
                            "tool_summary": tool_summary,
                        }
                    )

                    if error:
                        st.warning(f"⚠️ Agent encountered an error: {error}")

                except Exception as exc:
                    err_msg = f"❌ Unexpected error: {exc}"
                    st.error(err_msg)
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": err_msg}
                    )

    # Clear chat button
    if st.session_state["messages"]:
        if st.button("🗑️ Clear conversation", key="clear_chat"):
            st.session_state["messages"] = []
            st.session_state["last_tool_calls"] = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

with tab_eval:
    st.header("📊 Evaluation Dashboard")
    st.markdown(
        """
        Run the built-in evaluation suite against **10 test cases** covering
        weather queries, news queries, mixed queries, and edge cases.

        ### Metrics Explained
        | Metric | Description | Scale |
        |--------|-------------|-------|
        | **Keyword Presence** | Expected keywords appear; unexpected ones absent | 0 – 1 |
        | **Length Adequacy** | Response is long enough to be useful | 0 – 1 |
        | **Criteria Coverage** | Each test's named criteria are satisfied (heuristic) | 0 – 1 |
        | **LLM Judge — Relevance** | Claude rates how well the response addresses the query | 1 – 5 |
        | **LLM Judge — Accuracy** | Claude rates factual plausibility | 1 – 5 |
        | **LLM Judge — Completeness** | Claude rates thoroughness | 1 – 5 |
        | **LLM Judge — Clarity** | Claude rates readability | 1 – 5 |
        | **Composite Score** | Weighted combination (see weights below) | 0 – 100 |

        **Weights:** Keyword Presence 20 % · Length Adequacy 10 % · Criteria Coverage 30 % · LLM Judge 40 %
        """
    )

    st.divider()

    # Show dataset preview
    with st.expander("📋 View evaluation dataset (10 test cases)"):
        dataset = load_eval_dataset()
        for tc in dataset:
            st.markdown(f"**`{tc['id']}`** ({tc['category']}) — _{tc['query']}_")

    col_run, col_opts = st.columns([2, 1])
    with col_opts:
        run_llm_judge = st.checkbox(
            "Include LLM Judge", value=True,
            help="Uses Claude to score Relevance/Accuracy/Completeness/Clarity. "
                 "Costs extra tokens but gives richer metrics."
        )
        subset_options = {
            "All 10 cases": None,
            "Weather only (5)": "weather",
            "News only (3)": "news",
            "Mixed (2)": "mixed",
        }
        subset_label = st.selectbox("Run subset", list(subset_options.keys()))
        category_filter = subset_options[subset_label]

    with col_run:
        run_eval = st.button(
            "🚀 Run Evaluation Suite",
            type="primary",
            use_container_width=True,
            disabled=not api_key,
        )
        if not api_key:
            st.caption("⛔ Enter your API key in the sidebar to run evaluation.")

    if run_eval:
        dataset = load_eval_dataset()
        if category_filter:
            dataset = [tc for tc in dataset if tc["category"] == category_filter]

        orchestrator = WeatherNewsOrchestrator(api_key=api_key, model=model)
        evaluator = Evaluator(api_key=api_key if run_llm_judge else None, model=model)

        progress_bar = st.progress(0, text="Starting evaluation…")
        status_text = st.empty()
        results_container = st.empty()

        collected: list[EvalResult] = []

        def update_progress(current: int, total: int, test_id: str):
            if total > 0:
                pct = current / total
                progress_bar.progress(pct, text=f"Running {test_id} ({current}/{total})…")

        async def _run():
            return await run_evaluation_async(
                orchestrator=orchestrator,
                evaluator=evaluator,
                test_cases=dataset,
                run_llm_judge=run_llm_judge,
                progress_callback=update_progress,
            )

        with st.spinner("Evaluation in progress… this may take 1–3 minutes."):
            try:
                eval_results = _run_in_thread(_run())
                st.session_state["eval_results"] = eval_results
                st.session_state["eval_summary"] = summarise_results(eval_results)
                progress_bar.progress(1.0, text="✅ Evaluation complete!")
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                eval_results = []

    # ── Display results ────────────────────────────────────────────────────────

    if st.session_state["eval_results"]:
        results = st.session_state["eval_results"]
        summary = st.session_state["eval_summary"]

        st.divider()
        st.subheader("📈 Results Summary")

        # Top-level KPI cards
        c1, c2, c3, c4, c5 = st.columns(5)
        metrics = [
            (c1, f"{summary.get('avg_composite_score', 0):.1f}", "Composite Score (0–100)"),
            (c2, f"{summary.get('avg_keyword_presence', 0):.2f}", "Keyword Presence (0–1)"),
            (c3, f"{summary.get('avg_criteria_coverage', 0):.2f}", "Criteria Coverage (0–1)"),
            (c4, f"{summary.get('avg_llm_relevance', 0):.1f}", "LLM Relevance (1–5)"),
            (c5, f"{summary.get('avg_latency_seconds', 0):.1f}s", "Avg Latency"),
        ]
        for col, value, label in metrics:
            col.markdown(
                f"<div class='metric-card'><h3>{value}</h3><p>{label}</p></div>",
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Per-category breakdown
        if summary.get("by_category"):
            st.subheader("By Category")
            cat_cols = st.columns(len(summary["by_category"]))
            for i, (cat, stats) in enumerate(summary["by_category"].items()):
                cat_cols[i].metric(
                    f"{cat.title()} ({stats['count']} cases)",
                    f"{stats['avg_composite']:.1f} / 100",
                )

        # Detailed per-test table
        st.subheader("Per-Test Results")

        import pandas as pd
        rows = [r.to_dict() for r in results]
        df = pd.DataFrame(rows)

        # Colour-code composite score
        def colour_score(val):
            if isinstance(val, float):
                if val >= 75:
                    return "background-color: #1e4d2b; color: #a6e3a1"
                elif val >= 50:
                    return "background-color: #4d3a00; color: #f9e2af"
                else:
                    return "background-color: #4d1e20; color: #f38ba8"
            return ""

        display_cols = [
            "test_id", "category", "query",
            "composite_score", "keyword_presence_score", "criteria_coverage_score",
            "llm_relevance", "llm_accuracy", "llm_completeness", "llm_clarity",
            "latency_seconds",
        ]
        display_df = df[[c for c in display_cols if c in df.columns]]
        # .map() replaces the deprecated .applymap() from pandas 2.1+
        styler = display_df.style.map(colour_score, subset=["composite_score"])
        st.dataframe(styler, use_container_width=True, height=350)

        # Per-criteria breakdown
        st.subheader("Criteria Coverage Detail")
        for r in results:
            if not r.criteria_results:
                continue
            with st.expander(f"**{r.test_id}** — {r.query[:60]}…"):
                for cr in r.criteria_results:
                    icon = "✅" if cr.passed else "❌"
                    st.markdown(f"{icon} **{cr.name}**: {cr.description}")
                if r.llm_judge_raw:
                    st.code(r.llm_judge_raw, language="json")

        # Download results
        st.download_button(
            "⬇️ Download Results (JSON)",
            data=json.dumps([r.to_dict() for r in results], indent=2),
            file_name="eval_results.json",
            mime="application/json",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DEBUG
# ═══════════════════════════════════════════════════════════════════════════════

with tab_debug:
    st.header("🔧 Debug — Last Query Tool Calls")

    tool_calls = st.session_state.get("last_tool_calls", [])
    last_model = st.session_state.get("last_model", "—")

    if not tool_calls:
        st.info("No tool calls yet. Send a query in the Chat tab first.")
    else:
        st.caption(f"Model used: `{last_model}`  |  Tool calls: **{len(tool_calls)}**")
        for i, tc in enumerate(tool_calls, 1):
            status = "✅" if not tc.get("error") else "❌"
            with st.expander(f"{status} Call {i}: `{tc['tool']}`"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Input")
                    st.json(tc.get("input", {}))
                with col2:
                    st.subheader("Output")
                    raw_output = tc.get("output") or tc.get("error") or "—"
                    try:
                        parsed = json.loads(raw_output)
                        st.json(parsed)
                    except (json.JSONDecodeError, TypeError):
                        st.code(str(raw_output)[:1000])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_about:
    st.header("ℹ️ About This Application")

    st.markdown(
        """
## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Streamlit Frontend                   │
│   Chat Tab │ Evaluation Tab │ Debug Tab │ About Tab  │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│            Agent Orchestrator (Claude)               │
│  • Receives user query + conversation history        │
│  • Spawns MCP server subprocesses                    │
│  • Runs tool-use loop until final answer             │
│  Model: claude-haiku-4-5 / claude-sonnet-4-6         │
└──────────┬──────────────────────┬───────────────────┘
           │ stdio MCP            │ stdio MCP
           ▼                      ▼
┌──────────────────┐   ┌──────────────────────────────┐
│ Weather MCP      │   │  News MCP Server              │
│ Server           │   │  (HackerNews via Algolia)     │
│ (Open-Meteo)     │   │                               │
│                  │   │  Tools:                       │
│ Tools:           │   │  • get_top_news               │
│ • get_current_   │   │  • get_recent_news            │
│   weather        │   │  • search_news                │
│ • get_weather_   │   │  • get_news_by_topic          │
│   forecast       │   │                               │
└────────┬─────────┘   └───────────────┬──────────────┘
         │                             │
         ▼                             ▼
  open-meteo.com             hn.algolia.com
  (free, no API key)         (free, no API key)
```

## MCP Servers

### 🌤️ Open-Meteo Weather Server (`mcp_servers/weather_server.py`)
- **Protocol:** Model Context Protocol (MCP) over stdio
- **Library:** `mcp` (FastMCP)
- **Data source:** [Open-Meteo](https://open-meteo.com/) — free, no API key
- **Tools:**
  - `get_current_weather(city)` — temperature, humidity, wind, pressure, visibility
  - `get_weather_forecast(city, days=7)` — up to 16-day daily forecast

### 📰 HackerNews News Server (`mcp_servers/news_server.py`)
- **Protocol:** Model Context Protocol (MCP) over stdio
- **Library:** `mcp` (FastMCP)
- **Data source:** [Algolia HN Search API](https://hn.algolia.com/api) — free, no API key
- **Tools:**
  - `get_top_news(limit)` — front-page quality stories
  - `get_recent_news(limit)` — latest submissions by time
  - `search_news(query, limit)` — full-text search
  - `get_news_by_topic(topic, limit)` — date-sorted topic search

## Evaluation Metrics

| Metric | Type | Weight |
|--------|------|--------|
| Keyword Presence Score | Automated | 20% |
| Length Adequacy Score | Automated | 10% |
| Criteria Coverage Score | Automated heuristic | 30% |
| LLM Judge (Relevance + Accuracy + Completeness + Clarity) | AI-based (Claude) | 40% |

**Composite Score** = weighted sum, normalised to 0–100.

## Requirements

```
anthropic>=0.25.0   — Claude agent + LLM judge
mcp>=1.0.0          — MCP server/client (FastMCP)
streamlit>=1.32.0   — UI framework
httpx>=0.27.0       — Async HTTP for API calls
nest-asyncio>=1.6.0 — Allows asyncio.run() in Streamlit
python-dotenv>=1.0.0 — .env file support
pandas>=2.0.0       — Eval results table
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...
# or create a .env file with ANTHROPIC_API_KEY=...

# 3. Run the app
streamlit run app.py
```
        """
    )
