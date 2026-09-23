"""Structured, keyless weather primitives.

The harness uses these as the deterministic first-line weather source.  They are
kept narrow so recipes can compose location resolution and forecast retrieval,
while web_search/browse_url remain an independent fallback/provenance path.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import requests

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _bounded_int(value: Any, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = low
    return max(low, min(parsed, high))


def geocode_location(query: str = "", count: int = 5, language: str = "en") -> str:
    """Resolve a human place name to a small list of latitude/longitude candidates."""
    query = " ".join(str(query or "").split()).strip()
    if not query:
        return "Error: Missing required 'query' parameter."
    if len(query) > 240:
        return "Error: Location query is limited to 240 characters."
    params = {
        "name": query,
        "count": _bounded_int(count, 1, 10),
        "language": (str(language or "en").strip() or "en")[:12],
        "format": "json",
    }
    try:
        response = requests.get(_GEOCODE_URL, params=params, timeout=(3.05, 8))
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return f"Error: geocoding failed: {exc}"

    rows = []
    for item in list(payload.get("results") or [])[: params["count"]]:
        if not isinstance(item, dict):
            continue
        try:
            latitude = float(item["latitude"])
            longitude = float(item["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({
            "name": str(item.get("name") or "")[:160],
            "latitude": latitude,
            "longitude": longitude,
            "country": str(item.get("country") or "")[:120],
            "country_code": str(item.get("country_code") or "")[:12],
            "admin1": str(item.get("admin1") or "")[:120],
            "admin2": str(item.get("admin2") or "")[:120],
            "timezone": str(item.get("timezone") or "")[:80],
            "population": item.get("population"),
        })
    if not rows:
        return "Error: no matching location found."
    return _json(rows)


def weather_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 8,
    timezone_name: str = "auto",
) -> str:
    """Fetch structured current conditions and a bounded daily forecast for coordinates."""
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return "Error: latitude and longitude must be numeric."
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return "Error: latitude or longitude is outside the valid range."

    days = _bounded_int(forecast_days, 1, 16)
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": (str(timezone_name or "auto").strip() or "auto")[:80],
        "forecast_days": days,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "current": ",".join([
            "temperature_2m", "apparent_temperature", "precipitation", "rain", "snowfall",
            "weather_code", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
        ]),
        "daily": ",".join([
            "weather_code", "temperature_2m_max", "temperature_2m_min",
            "apparent_temperature_max", "apparent_temperature_min",
            "precipitation_sum", "precipitation_probability_max", "rain_sum", "snowfall_sum",
            "wind_speed_10m_max", "wind_gusts_10m_max", "sunrise", "sunset",
        ]),
    }
    try:
        response = requests.get(_FORECAST_URL, params=params, timeout=(3.05, 10))
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return f"Error: weather forecast failed: {exc}"

    if not isinstance(payload, dict) or not isinstance(payload.get("daily"), dict):
        return "Error: weather provider returned an invalid forecast payload."

    result = {
        "provider": "Open-Meteo",
        "source_url": _FORECAST_URL,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latitude": payload.get("latitude", lat),
        "longitude": payload.get("longitude", lon),
        "timezone": payload.get("timezone", params["timezone"]),
        "timezone_abbreviation": payload.get("timezone_abbreviation", ""),
        "utc_offset_seconds": payload.get("utc_offset_seconds"),
        "elevation": payload.get("elevation"),
        "forecast_days": days,
        "current": payload.get("current") or {},
        "current_units": payload.get("current_units") or {},
        "daily": payload.get("daily") or {},
        "daily_units": payload.get("daily_units") or {},
    }
    return _json(result)

_WMO_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    56: "Light freezing drizzle",
    57: "Freezing drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Freezing rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Light rain showers",
    81: "Rain showers",
    82: "Heavy rain showers",
    85: "Light snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorms",
    96: "Thunderstorms with hail",
    99: "Severe thunderstorms with hail",
}




_CURRENT_WEATHER_RE = re.compile(
    r"\b(?:right\s+now|now|currently|current\s+(?:weather|conditions?))\b|\bat\s+the\s+moment\b",
    re.I,
)


def _is_current_weather_request(user_request: str) -> bool:
    return bool(_CURRENT_WEATHER_RE.search(str(user_request or "")))


def _wind_direction_label(value: Any) -> str:
    try:
        degrees = float(value) % 360.0
    except (TypeError, ValueError):
        return ""
    labels = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
    return labels[int((degrees + 11.25) // 22.5) % 16]


def _canonical_place(place: Any, fallback: str = "") -> str:
    if isinstance(place, dict):
        parts = [
            str(place.get("name") or "").strip(),
            str(place.get("admin1") or "").strip(),
            str(place.get("country") or "").strip(),
        ]
        canonical = ", ".join(part for index, part in enumerate(parts) if part and part not in parts[:index])
        if canonical:
            return canonical
    return str(fallback or "").strip()


def _format_current_weather(forecast: dict[str, Any], location: str) -> str:
    current = forecast.get("current") or {}
    if not isinstance(current, dict) or not current:
        return ""
    try:
        code = int(current.get("weather_code"))
    except (TypeError, ValueError):
        code = -1
    condition = _WMO_DESCRIPTIONS.get(code, f"Weather code {code}" if code >= 0 else "Current conditions")
    temperature = _fmt_number(current.get("temperature_2m"), " °C", 1)
    apparent = _fmt_number(current.get("apparent_temperature"), " °C", 1)
    precip = _fmt_number(current.get("precipitation"), " mm", 1)
    wind = _fmt_number(current.get("wind_speed_10m"), " km/h", 1)
    gusts = _fmt_number(current.get("wind_gusts_10m"), " km/h", 1)
    direction = _wind_direction_label(current.get("wind_direction_10m"))
    cloud = _fmt_number(current.get("cloud_cover"), "%")
    observed = str(current.get("time") or "").replace("T", " ")
    retrieved = str(forecast.get("retrieved_at") or "").replace("T", " ")[:25]

    lines = [f"**Current weather for {location or 'the requested location'}**", "", f"**{condition}, {temperature}**"]
    if apparent != "—":
        lines[-1] += f" (feels like {apparent})"
    lines.extend([
        f"- **Precipitation:** {precip}",
        f"- **Wind:** {wind}{f' {direction}' if direction else ''}{f', gusting to {gusts}' if gusts != '—' else ''}",
        f"- **Cloud cover:** {cloud}",
    ])
    if observed:
        lines.append(f"- **Observed:** {observed} {str(forecast.get('timezone_abbreviation') or '').strip()}".rstrip())
    lines.extend(["", f"Source: Open-Meteo. Retrieved {retrieved} UTC.".rstrip()])
    return "\n".join(lines)

def _fmt_number(value: Any, suffix: str = "", digits: int = 0) -> str:
    if value is None or value == "":
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    text = f"{number:.{digits}f}" if digits else f"{number:.0f}"
    return f"{text}{suffix}"


def _daily_rows(forecast: dict[str, Any]) -> list[dict[str, Any]]:
    daily = forecast.get("daily") or {}
    if not isinstance(daily, dict):
        return []
    dates = list(daily.get("time") or [])
    rows: list[dict[str, Any]] = []
    keys = (
        "weather_code", "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "precipitation_probability_max",
        "wind_speed_10m_max", "wind_gusts_10m_max",
    )
    for index, date in enumerate(dates):
        row: dict[str, Any] = {"date": str(date)}
        for key in keys:
            values = daily.get(key) or []
            row[key] = values[index] if isinstance(values, list) and index < len(values) else None
        rows.append(row)
    return rows


def _requested_daily_rows(rows: list[dict[str, Any]], user_request: str) -> list[dict[str, Any]]:
    """Select only dates explicitly covered by the user's requested horizon."""
    text = str(user_request or "").lower()
    if not rows:
        return []
    if re.search(r"\b(?:today|tonight|this (?:morning|afternoon|evening))\b", text):
        return rows[:1]
    if re.search(r"\btomorrow\b", text):
        return rows[1:2] if len(rows) > 1 else rows[:1]
    match = re.search(r"\bnext\s+(\d{1,2})\s+days?\b", text)
    if match:
        count = max(1, min(int(match.group(1)), 15))
        return rows[1:1 + count] if len(rows) > 1 else rows[:count]
    if re.search(r"\bnext week\b|\b(?:7|seven)[ -]?day\b|\bweek(?:ly)? forecast\b", text):
        return rows[1:8] if len(rows) > 1 else rows[:7]
    return rows[:7]


def is_simple_weather_request(user_request: str) -> bool:
    """Return whether a request is a plain forecast display suitable for deterministic rendering."""
    text = " ".join(str(user_request or "").lower().split())
    if not re.search(r"\b(?:weather|forecast)\b", text):
        return False
    return not re.search(
        r"\b(?:why|compare|versus|vs\.?|recommend|should i|safe|risk|explain|analy[sz]e|plan|best|worst|umbrella|wear|drive|travel)\b",
        text,
    )


def format_weather_recovery(result: dict[str, Any], user_request: str) -> str:
    """Render provider-backed weather without asking a small model to reshape data.

    Structured Open-Meteo data is preferred. If that path failed but the
    independent search+browse fallback was positively weather-grounded, render
    its bounded verified source excerpt instead of silently dropping weather
    from a compound answer.
    """
    payload = result.get("result") if isinstance(result, dict) else None
    if not isinstance(payload, dict):
        return ""
    forecast = payload.get("forecast")
    place = payload.get("place") or {}
    recovery = result.get("grounding_recovery") if isinstance(result, dict) else {}
    recovery = recovery if isinstance(recovery, dict) else {}
    location = str(payload.get("location") or recovery.get("location") or "").strip()
    if not isinstance(forecast, dict):
        verification = str(payload.get("verification") or "").strip()
        if not verification:
            return ""
        # browse_url emits URL/Content-Type/Extraction headers followed by the
        # bounded source passages. Keep the useful evidence compact and do not
        # ask the generation model to paraphrase untrusted page text.
        lines = [line.strip() for line in verification.splitlines() if line.strip()]
        source_url = next((line[4:].strip() for line in lines if line.startswith("URL:")), "")
        body = [
            line for line in lines
            if not line.startswith(("URL:", "Content-Type:", "Extraction:"))
        ]
        excerpt = " ".join(body)
        excerpt = re.sub(r"\s+", " ", excerpt).strip()[:1200]
        if not excerpt:
            return ""
        title = f"**Current weather for {location or 'the requested location'}**"
        rendered = [title, "", excerpt]
        if source_url:
            rendered.extend(["", f"Source: {source_url} (verified web fallback)."])
        return "\n".join(rendered)
    location = _canonical_place(place, location)
    if _is_current_weather_request(user_request):
        current_rendered = _format_current_weather(forecast, location)
        if current_rendered:
            return current_rendered

    rows = _requested_daily_rows(_daily_rows(forecast), user_request)
    if not rows:
        return ""
    title = f"**Weather forecast for {location or 'the requested location'}**"
    lines = [title, "", "| Date | Conditions | High | Low | Precip. chance | Precip. | Wind | Gusts |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        try:
            code = int(row.get("weather_code"))
        except (TypeError, ValueError):
            code = -1
        lines.append(
            "| {date} | {conditions} | {high} | {low} | {chance} | {precip} | {wind} | {gust} |".format(
                date=row.get("date") or "—",
                conditions=_WMO_DESCRIPTIONS.get(code, f"Weather code {code}" if code >= 0 else "—"),
                high=_fmt_number(row.get("temperature_2m_max"), " °C"),
                low=_fmt_number(row.get("temperature_2m_min"), " °C"),
                chance=_fmt_number(row.get("precipitation_probability_max"), "%"),
                precip=_fmt_number(row.get("precipitation_sum"), " mm", 1),
                wind=_fmt_number(row.get("wind_speed_10m_max"), " km/h"),
                gust=_fmt_number(row.get("wind_gusts_10m_max"), " km/h"),
            )
        )
    lines.extend(["", f"Source: Open-Meteo. Forecast retrieved {str(forecast.get('retrieved_at') or '').replace('T', ' ')[:25]} UTC.".rstrip()])
    return "\n".join(lines)
