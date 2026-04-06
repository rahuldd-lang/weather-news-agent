#!/usr/bin/env python3
"""
GNews.io News MCP Server
No hardcoded credentials — reads GNEWS_API_KEY from the environment,
injected at subprocess-spawn time by the orchestrator.

Free tier: 100 req/day, up to 10 articles per request.
Register at https://gnews.io/register

Tools exposed:
  - get_top_news(limit)
  - get_recent_news(limit)
  - search_news(query, limit)
  - get_news_by_topic(topic, limit)
"""

import json
import os
import sys

import certifi
import httpx

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("mcp library not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# ── SSL & base config ─────────────────────────────────────────────────────────
_SSL_VERIFY = certifi.where()
_BASE_URL = "https://gnews.io/api/v4"

# GNews supported topics
_TOPIC_MAP: dict[str, str] = {
    "breaking": "breaking-news",
    "breaking-news": "breaking-news",
    "world": "world",
    "international": "world",
    "nation": "nation",
    "national": "nation",
    "business": "business",
    "finance": "business",
    "economy": "business",
    "technology": "technology",
    "tech": "technology",
    "science": "science",
    "health": "health",
    "medical": "health",
    "sports": "sports",
    "sport": "sports",
    "entertainment": "entertainment",
    "culture": "entertainment",
}
_VALID_TOPICS = sorted(set(_TOPIC_MAP.values()))

mcp = FastMCP(
    "GNews News Server",
    instructions=(
        "Provides mainstream news from hundreds of global publishers via the "
        "GNews.io API. Covers breaking news, world affairs, business, technology, "
        "science, health, sports, and entertainment. "
        f"Supported topics: {', '.join(_VALID_TOPICS)}."
    ),
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_api_key() -> str | None:
    return os.environ.get("GNEWS_API_KEY", "").strip() or None


def _clean_article(article: dict) -> dict:
    return {
        "title": article.get("title", "No title"),
        "description": article.get("description", ""),
        "url": article.get("url", ""),
        "source": article.get("source", {}).get("name", "Unknown"),
        "published_at": article.get("publishedAt", "Unknown"),
        "image_url": article.get("image", ""),
    }


async def _gnews_request(endpoint: str, params: dict) -> dict:
    api_key = _get_api_key()
    if not api_key:
        return {"error": "GNEWS_API_KEY is not set. Add it to your Streamlit secrets."}

    params = {**params, "token": api_key, "lang": "en"}
    async with httpx.AsyncClient(timeout=15, verify=_SSL_VERIFY) as client:
        resp = await client.get(f"{_BASE_URL}/{endpoint}", params=params)
        resp.raise_for_status()
        return resp.json()


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def get_top_news(limit: int = 10) -> str:
    """
    Get the current top headlines from mainstream news sources worldwide
    via GNews.io.

    Args:
        limit: Number of articles to return (1–10, default 10).

    Returns:
        JSON string with top headlines including title, description, URL,
        source name, and published date.
    """
    limit = max(1, min(10, limit))
    data = await _gnews_request("top-headlines", {"max": limit})
    if "error" in data:
        return json.dumps(data)

    articles = [_clean_article(a) for a in data.get("articles", [])]
    return json.dumps({
        "source": "GNews.io",
        "type": "top_headlines",
        "total_results": data.get("totalArticles", len(articles)),
        "count": len(articles),
        "articles": articles,
    }, indent=2)


@mcp.tool()
async def get_recent_news(limit: int = 10) -> str:
    """
    Get the most recently published news articles from GNews.io,
    sorted by publication date (newest first).

    Args:
        limit: Number of articles to return (1–10, default 10).

    Returns:
        JSON string with the latest articles including title, description,
        URL, source name, and published date.
    """
    limit = max(1, min(10, limit))
    data = await _gnews_request(
        "top-headlines", {"max": limit, "sortby": "publishedAt"}
    )
    if "error" in data:
        return json.dumps(data)

    articles = [_clean_article(a) for a in data.get("articles", [])]
    return json.dumps({
        "source": "GNews.io",
        "type": "recent_news",
        "count": len(articles),
        "articles": articles,
    }, indent=2)


@mcp.tool()
async def search_news(query: str, limit: int = 10) -> str:
    """
    Search for news articles matching a query using GNews.io.
    Returns results from mainstream global publishers.

    Args:
        query: Search term (e.g. "artificial intelligence", "climate change",
               "US economy", "Premier League").
        limit: Number of results to return (1–10, default 10).

    Returns:
        JSON string with matching articles sorted by relevance, including
        title, description, URL, source name, and published date.
    """
    limit = max(1, min(10, limit))
    data = await _gnews_request("search", {"q": query, "max": limit, "sortby": "relevance"})
    if "error" in data:
        return json.dumps(data)

    articles = [_clean_article(a) for a in data.get("articles", [])]
    return json.dumps({
        "source": "GNews.io",
        "query": query,
        "type": "search_results",
        "total_results": data.get("totalArticles", len(articles)),
        "count": len(articles),
        "articles": articles,
    }, indent=2)


@mcp.tool()
async def get_news_by_topic(topic: str, limit: int = 10) -> str:
    """
    Get top headlines for a specific news topic from GNews.io.

    Supported topics: breaking-news, world, nation, business, technology,
    science, health, sports, entertainment.

    Args:
        topic: Topic name (e.g. "technology", "business", "health", "sports").
               Partial matches are supported (e.g. "tech" → technology).
        limit: Number of articles to return (1–10, default 10).

    Returns:
        JSON string with topic headlines including title, description,
        URL, source name, and published date.
    """
    limit = max(1, min(10, limit))
    normalised = _TOPIC_MAP.get(topic.lower().strip())
    if not normalised:
        # Fall back to search if topic isn't a known GNews section
        return await search_news(topic, limit)

    data = await _gnews_request("top-headlines", {"topic": normalised, "max": limit})
    if "error" in data:
        return json.dumps(data)

    articles = [_clean_article(a) for a in data.get("articles", [])]
    return json.dumps({
        "source": "GNews.io",
        "topic": normalised,
        "type": "topic_headlines",
        "total_results": data.get("totalArticles", len(articles)),
        "count": len(articles),
        "articles": articles,
    }, indent=2)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
