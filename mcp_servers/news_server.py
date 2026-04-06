#!/usr/bin/env python3
"""
HackerNews News MCP Server  (no API key required)
Uses the Algolia HackerNews Search API — completely free, zero credentials.

Tools exposed:
  - get_top_news(limit)
  - search_news(query, limit)
  - get_news_by_topic(topic, limit)
"""

import asyncio
import json
import sys
from datetime import datetime, timezone

import certifi
import httpx

# Use certifi's CA bundle so macOS Python.org installs don't fail SSL verification
_SSL_VERIFY = certifi.where()

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("mcp library not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"
HN_SEARCH_DATE_URL = "https://hn.algolia.com/api/v1/search_by_date"
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{id}.json"
HN_TOP_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"

TOPIC_TAGS = {
    "tech": "story",
    "technology": "story",
    "ask": "ask_hn",
    "show": "show_hn",
    "jobs": "job",
    "science": "story",
    "ai": "story",
    "machine learning": "story",
    "security": "story",
    "programming": "story",
    "startup": "story",
}

mcp = FastMCP(
    "HackerNews News Server",
    instructions=(
        "Provides real-time technology and startup news from Hacker News "
        "via the Algolia search API. No API key required. "
        "Stories are ranked by popularity (points) and recency."
    ),
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_time(ts: int | None) -> str:
    if ts is None:
        return "Unknown"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _clean_hit(hit: dict) -> dict:
    """Extract the useful fields from an Algolia hit."""
    return {
        "title": hit.get("title", "No title"),
        "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
        "author": hit.get("author", "unknown"),
        "points": hit.get("points", 0),
        "comments": hit.get("num_comments", 0),
        "published_at": _fmt_time(hit.get("created_at_i")),
        "hn_id": hit.get("objectID"),
        "hn_link": f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
    }


async def _algolia_search(
    params: dict,
    base_url: str = HN_SEARCH_URL,
    limit: int = 10,
) -> list[dict]:
    """Run an Algolia HN search and return cleaned hits."""
    params.setdefault("hitsPerPage", min(limit, 30))
    async with httpx.AsyncClient(timeout=10, verify=_SSL_VERIFY) as client:
        resp = await client.get(base_url, params=params)
        resp.raise_for_status()
        data = resp.json()
    return [_clean_hit(h) for h in data.get("hits", [])][:limit]


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def get_top_news(limit: int = 10) -> str:
    """
    Get the current top stories from Hacker News (front page quality).
    No API key required.

    Args:
        limit: Number of stories to return (1–30, default 10).

    Returns:
        JSON string with a list of top stories including title, URL,
        author, points, comment count, and publish time.
    """
    limit = max(1, min(30, limit))
    stories = await _algolia_search(
        {"tags": "front_page", "hitsPerPage": limit},
        base_url=HN_SEARCH_URL,
        limit=limit,
    )
    result = {
        "source": "Hacker News (via Algolia)",
        "type": "top_stories",
        "count": len(stories),
        "stories": stories,
    }
    return json.dumps(result, indent=2)


@mcp.tool()
async def search_news(query: str, limit: int = 10) -> str:
    """
    Search Hacker News for stories matching a query.
    No API key required.

    Args:
        query: Search term (e.g. "Python AI", "climate change", "GPT-4").
        limit: Number of results to return (1–30, default 10).

    Returns:
        JSON string with matching stories sorted by relevance, including
        title, URL, author, points, comment count, and publish time.
    """
    limit = max(1, min(30, limit))
    stories = await _algolia_search(
        {"query": query, "tags": "story", "hitsPerPage": limit},
        base_url=HN_SEARCH_URL,
        limit=limit,
    )
    result = {
        "source": "Hacker News (via Algolia)",
        "query": query,
        "type": "search_results",
        "count": len(stories),
        "stories": stories,
    }
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_news_by_topic(topic: str, limit: int = 10) -> str:
    """
    Get the latest Hacker News stories for a specific topic.
    No API key required. Good topics: technology, AI, science, security,
    programming, startups, climate, finance, health, space.

    Args:
        topic: Topic keyword (e.g. "artificial intelligence", "cybersecurity",
               "climate change", "Python", "startups").
        limit: Number of stories to return (1–30, default 10).

    Returns:
        JSON string with the most recent relevant stories, including title,
        URL, author, points, comment count, and publish time.
    """
    limit = max(1, min(30, limit))
    # Use date-sorted endpoint for "latest" feel, filtered by topic query
    stories = await _algolia_search(
        {"query": topic, "tags": "story", "hitsPerPage": limit},
        base_url=HN_SEARCH_DATE_URL,
        limit=limit,
    )
    result = {
        "source": "Hacker News (via Algolia)",
        "topic": topic,
        "type": "topic_news",
        "count": len(stories),
        "stories": stories,
    }
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_recent_news(limit: int = 10) -> str:
    """
    Get the most recently submitted stories from Hacker News.
    No API key required. Unlike get_top_news (sorted by popularity),
    this returns the absolute latest submissions by time.

    Args:
        limit: Number of stories to return (1–30, default 10).

    Returns:
        JSON string with the newest stories including title, URL,
        author, points, comment count, and publish time.
    """
    limit = max(1, min(30, limit))
    stories = await _algolia_search(
        {"tags": "story", "hitsPerPage": limit},
        base_url=HN_SEARCH_DATE_URL,
        limit=limit,
    )
    result = {
        "source": "Hacker News (via Algolia)",
        "type": "recent_stories",
        "count": len(stories),
        "stories": stories,
    }
    return json.dumps(result, indent=2)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()  # stdio transport by default
