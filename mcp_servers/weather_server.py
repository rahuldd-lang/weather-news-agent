#!/usr/bin/env python3
"""
Open-Meteo Weather MCP Server
No API key required — uses the fully free Open-Meteo API.

Tools exposed:
  - get_current_weather(city)
  - get_weather_forecast(city, days)
"""

import asyncio
import json
import sys

import certifi
import httpx

# Use certifi's CA bundle so macOS Python.org installs don't fail SSL verification
_SSL_VERIFY = certifi.where()

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("mcp library not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# ── WMO Weather Interpretation Codes ──────────────────────────────────────────
WMO_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

mcp = FastMCP(
    "Open-Meteo Weather Server",
    instructions=(
        "Provides real-time weather data powered by the Open-Meteo API. "
        "No API key required. Data is refreshed frequently."
    ),
)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _geocode(city: str) -> dict | None:
    """Return lat/lon/name/country/timezone for *city*, or None if not found."""
    async with httpx.AsyncClient(timeout=10, verify=_SSL_VERIFY) as client:
        resp = await client.get(
            GEOCODE_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
        )
        resp.raise_for_status()
        data = resp.json()

    if not data.get("results"):
        return None
    r = data["results"][0]
    return {
        "latitude": r["latitude"],
        "longitude": r["longitude"],
        "name": r.get("name", city),
        "country": r.get("country", ""),
        "timezone": r.get("timezone", "auto"),
    }


def _wmo_description(code: int) -> str:
    return WMO_CODES.get(code, f"Unknown (code {code})")


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def get_current_weather(city: str) -> str:
    """
    Get current weather conditions for a city using the Open-Meteo API.

    Args:
        city: City name (e.g. "London", "New York", "Tokyo", "Sydney").

    Returns:
        JSON string with temperature, humidity, wind speed, weather description,
        pressure, precipitation, cloud cover, and visibility.
    """
    loc = await _geocode(city)
    if loc is None:
        return json.dumps({"error": f"Location '{city}' not found."})

    async with httpx.AsyncClient(timeout=10, verify=_SSL_VERIFY) as client:
        resp = await client.get(
            FORECAST_URL,
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "current": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "precipitation",
                    "weather_code",
                    "cloud_cover",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "surface_pressure",
                    "visibility",
                ],
                "timezone": loc["timezone"],
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
                "precipitation_unit": "mm",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    cur = data.get("current", {})
    wc = cur.get("weather_code", 0)

    result = {
        "location": f"{loc['name']}, {loc['country']}",
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "time_utc": cur.get("time"),
        "weather_description": _wmo_description(wc),
        "weather_code": wc,
        "temperature_celsius": cur.get("temperature_2m"),
        "feels_like_celsius": cur.get("apparent_temperature"),
        "humidity_percent": cur.get("relative_humidity_2m"),
        "precipitation_mm": cur.get("precipitation"),
        "cloud_cover_percent": cur.get("cloud_cover"),
        "wind_speed_kmh": cur.get("wind_speed_10m"),
        "wind_direction_degrees": cur.get("wind_direction_10m"),
        "surface_pressure_hpa": cur.get("surface_pressure"),
        "visibility_m": cur.get("visibility"),
    }
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_weather_forecast(city: str, days: int = 7) -> str:
    """
    Get a daily weather forecast for a city for the next N days.

    Args:
        city: City name (e.g. "London", "New York", "Tokyo").
        days: Number of days to forecast (1–16, default 7).

    Returns:
        JSON string with per-day high/low temperature, precipitation,
        wind speed, weather description, sunrise and sunset times.
    """
    days = max(1, min(16, days))
    loc = await _geocode(city)
    if loc is None:
        return json.dumps({"error": f"Location '{city}' not found."})

    async with httpx.AsyncClient(timeout=10, verify=_SSL_VERIFY) as client:
        resp = await client.get(
            FORECAST_URL,
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "daily": [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "precipitation_probability_max",
                    "weather_code",
                    "wind_speed_10m_max",
                    "sunrise",
                    "sunset",
                ],
                "timezone": loc["timezone"],
                "forecast_days": days,
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
                "precipitation_unit": "mm",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])

    def _safe(key: str, i: int):
        arr = daily.get(key, [])
        return arr[i] if i < len(arr) else None

    forecast = []
    for i, date in enumerate(dates):
        wc = _safe("weather_code", i) or 0
        forecast.append(
            {
                "date": date,
                "weather_description": _wmo_description(wc),
                "max_temp_celsius": _safe("temperature_2m_max", i),
                "min_temp_celsius": _safe("temperature_2m_min", i),
                "precipitation_sum_mm": _safe("precipitation_sum", i),
                "precipitation_probability_percent": _safe(
                    "precipitation_probability_max", i
                ),
                "max_wind_speed_kmh": _safe("wind_speed_10m_max", i),
                "sunrise": _safe("sunrise", i),
                "sunset": _safe("sunset", i),
            }
        )

    result = {
        "location": f"{loc['name']}, {loc['country']}",
        "forecast_days": days,
        "daily_forecast": forecast,
    }
    return json.dumps(result, indent=2)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()  # stdio transport by default
