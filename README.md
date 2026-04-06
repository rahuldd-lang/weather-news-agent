# 🌤️ Weather & News AI Assistant

A Python + Streamlit application that answers questions about **current weather**
and **latest news** using an **agent orchestrator** and two **MCP servers** —
all with **zero third-party API keys** for the data sources.

## Architecture at a glance

```
Streamlit UI
    └── Agent Orchestrator (Claude, tool-use loop)
            ├── Weather MCP Server  (Open-Meteo — free, no key)
            │       tools: get_current_weather · get_weather_forecast
            └── News MCP Server    (HackerNews via Algolia — free, no key)
                    tools: get_top_news · get_recent_news
                           search_news · get_news_by_topic
```

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/weather-news-agent.git
cd weather-news-agent

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

When the app opens, **enter your Anthropic API key in the sidebar** to get started.
Get a free key at [console.anthropic.com](https://console.anthropic.com).

## What you need

| Dependency | Why | API key? |
|------------|-----|----------|
| [Anthropic Claude](https://console.anthropic.com) | LLM agent brain | **Yes — entered in the app sidebar** |
| [Open-Meteo](https://open-meteo.com) | Weather data | No |
| [HackerNews / Algolia](https://hn.algolia.com/api) | News data | No |

## Deploying to Streamlit Community Cloud

1. Push the repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repo, set **Main file path** to `app.py`, and click **Deploy**.
4. Users enter their own Anthropic API key directly in the app sidebar — no server-side secrets needed.

## Models supported

| Model | Speed | Cost |
|-------|-------|------|
| `claude-haiku-4-5-20251001` | Fastest | Lowest |
| `claude-sonnet-4-6` | Balanced | Medium |
| `claude-opus-4-6` | Most capable | Highest |

Select the model from the sidebar dropdown at runtime.

## Evaluation

The app ships with a **10-case evaluation suite** (see `evaluation/`) that measures:

| Metric | Description | Scale |
|--------|-------------|-------|
| Keyword Presence | Expected terms appear; error terms absent | 0–1 |
| Length Adequacy | Response is sufficiently detailed | 0–1 |
| Criteria Coverage | Named test criteria satisfied (heuristic) | 0–1 |
| LLM Judge × 4 | Claude rates Relevance, Accuracy, Completeness, Clarity | 1–5 each |
| **Composite** | Weighted combination | **0–100** |

Run it from the **📊 Evaluation** tab in the UI.

## File structure

```
weather-news-agent/
├── app.py                        # Streamlit entry point
├── requirements.txt
├── .gitignore
├── agents/
│   └── orchestrator.py           # Claude agent loop + MCP client management
├── mcp_servers/
│   ├── weather_server.py         # Open-Meteo MCP server (FastMCP)
│   └── news_server.py            # HackerNews MCP server (FastMCP)
└── evaluation/
    ├── eval_dataset.json          # 10 test cases (weather, news, mixed, edge)
    └── evaluator.py               # Metric computation + async runner
```
