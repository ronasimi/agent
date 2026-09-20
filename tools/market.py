"""Structured current market quote helpers for common assets and futures."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote


_ASSETS: dict[str, dict[str, str]] = {
    "wti": {"symbol": "CL=F", "name": "WTI Crude Oil Futures", "unit": "USD/barrel"},
    "brent": {"symbol": "BZ=F", "name": "Brent Crude Oil Futures", "unit": "USD/barrel"},
    "gold": {"symbol": "GC=F", "name": "Gold Futures", "unit": "USD/troy oz"},
    "silver": {"symbol": "SI=F", "name": "Silver Futures", "unit": "USD/troy oz"},
    "natural gas": {"symbol": "NG=F", "name": "Natural Gas Futures", "unit": "USD/MMBtu"},
    "copper": {"symbol": "HG=F", "name": "Copper Futures", "unit": "USD/lb"},
}

_ALIAS_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:wti|west\s+texas\s+intermediate|cl\s*=\s*f)\b", re.I), "wti"),
    (re.compile(r"\b(?:brent(?:\s+crude(?:\s+oil)?)?|bz\s*=\s*f)\b", re.I), "brent"),
    (re.compile(r"\bgold(?:\s+futures?)?\b|\bgc\s*=\s*f\b", re.I), "gold"),
    (re.compile(r"\bsilver(?:\s+futures?)?\b|\bsi\s*=\s*f\b", re.I), "silver"),
    (re.compile(r"\bnatural\s+gas(?:\s+futures?)?\b|\bng\s*=\s*f\b", re.I), "natural gas"),
    (re.compile(r"\bcopper(?:\s+futures?)?\b|\bhg\s*=\s*f\b", re.I), "copper"),
)


def extract_market_instruments(text: str) -> list[str]:
    """Return canonical market instruments mentioned in a request, in source order."""
    raw = str(text or "")
    found: list[tuple[int, str]] = []
    for pattern, canonical in _ALIAS_PATTERNS:
        match = pattern.search(raw)
        if match:
            found.append((match.start(), canonical))
    found.sort(key=lambda item: item[0])
    result: list[str] = []
    for _offset, canonical in found:
        if canonical not in result:
            result.append(canonical)
    return result


def is_market_price_request(text: str) -> bool:
    """Conservatively identify a live/latest quote request for a known market instrument."""
    raw = " ".join(str(text or "").strip().split())
    if not raw or not extract_market_instruments(raw):
        return False
    lower = raw.lower()
    explicit_price = bool(re.search(r"\b(?:price|prices|quote|quotes|trading\s+at|worth)\b", lower))
    current_cue = bool(re.search(r"\b(?:current|currently|latest|now|right\s+now|today|live)\b", lower))
    question_cue = bool(re.search(r"^(?:what|how much|give|show|get|check|tell me|where)\b", lower) or raw.endswith("?"))
    return bool(explicit_price and (current_cue or question_cue))


def _resolve_instrument(value: str) -> tuple[str, dict[str, str]] | None:
    raw = " ".join(str(value or "").strip().split())
    if not raw:
        return None
    canonical = raw.lower()
    if canonical in _ASSETS:
        return canonical, _ASSETS[canonical]
    extracted = extract_market_instruments(raw)
    if extracted:
        canonical = extracted[0]
        return canonical, _ASSETS[canonical]
    # Permit explicit Yahoo-compatible symbols without guessing names.
    if re.fullmatch(r"[A-Za-z0-9.^=-]{1,20}", raw):
        return raw.upper(), {"symbol": raw.upper(), "name": raw.upper(), "unit": ""}
    return None


def _latest_close(payload: dict[str, Any]) -> tuple[float | None, int | None]:
    try:
        result = payload["chart"]["result"][0]
    except (KeyError, IndexError, TypeError):
        return None, None
    meta = result.get("meta") or {}
    value = meta.get("regularMarketPrice")
    timestamp = meta.get("regularMarketTime")
    if isinstance(value, (int, float)):
        return float(value), int(timestamp) if isinstance(timestamp, (int, float)) else None
    timestamps = result.get("timestamp") or []
    closes = (((result.get("indicators") or {}).get("quote") or [{}])[0].get("close") or [])
    for index in range(min(len(closes), len(timestamps)) - 1, -1, -1):
        candidate = closes[index]
        if isinstance(candidate, (int, float)):
            return float(candidate), int(timestamps[index])
    return None, None


def market_quote(instruments: list[str]) -> str:
    """Fetch latest available quotes for requested market instruments from Yahoo Finance chart data."""
    requested = [str(item).strip() for item in (instruments or []) if str(item).strip()]
    if not requested:
        return "Error: Missing required 'instruments' parameter."
    if len(requested) > 8:
        return "Error: A maximum of 8 instruments may be requested at once."

    try:
        import requests
    except Exception as exc:
        return f"Error: market quote dependency unavailable: {exc}"

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    headers = {"User-Agent": "Mozilla/5.0 LocalAgent/1.0", "Accept": "application/json"}
    for raw in requested:
        resolved = _resolve_instrument(raw)
        if resolved is None:
            errors.append({"instrument": raw, "error": "unsupported_or_ambiguous_instrument"})
            continue
        canonical, spec = resolved
        symbol = spec["symbol"]
        response_payload: dict[str, Any] | None = None
        final_url = ""
        last_error = ""
        for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
            url = f"https://{host}/v8/finance/chart/{quote(symbol, safe='')}?interval=1m&range=1d"
            try:
                response = requests.get(url, headers=headers, timeout=8)
                response.raise_for_status()
                candidate = response.json()
                if isinstance(candidate, dict) and (candidate.get("chart") or {}).get("result"):
                    response_payload = candidate
                    final_url = url
                    break
                last_error = "provider returned no chart result"
            except Exception as exc:
                last_error = str(exc)[:240]
        if response_payload is None:
            errors.append({"instrument": raw, "symbol": symbol, "error": last_error or "quote_fetch_failed"})
            continue
        result = response_payload["chart"]["result"][0]
        meta = result.get("meta") or {}
        price, timestamp = _latest_close(response_payload)
        if price is None:
            errors.append({"instrument": raw, "symbol": symbol, "error": "provider returned no numeric price"})
            continue
        if timestamp is not None:
            observed = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        else:
            observed = datetime.now(timezone.utc).isoformat()
        rows.append({
            "instrument": canonical,
            "name": str(spec.get("name") or meta.get("shortName") or symbol),
            "symbol": symbol,
            "price": price,
            "currency": str(meta.get("currency") or "USD"),
            "unit": str(spec.get("unit") or ""),
            "exchange": str(meta.get("fullExchangeName") or meta.get("exchangeName") or ""),
            "as_of": observed,
            "previous_close": meta.get("chartPreviousClose") if isinstance(meta.get("chartPreviousClose"), (int, float)) else None,
            "source": "Yahoo Finance chart",
            "source_url": final_url,
            "note": "Latest available provider quote; exchange data may be delayed.",
        })
    return json.dumps({"quotes": rows, "errors": errors}, ensure_ascii=False, indent=2)


def format_market_quotes(content: str) -> str:
    """Render structured market quotes without allowing the language model to alter prices or timestamps."""
    try:
        payload = json.loads(str(content or ""))
    except (TypeError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    quotes = [row for row in (payload.get("quotes") or []) if isinstance(row, dict) and isinstance(row.get("price"), (int, float))]
    if not quotes:
        return ""
    lines = ["**Latest available market quotes**", "", "| Instrument | Price | As of |", "|---|---:|---|"]
    for row in quotes:
        name = str(row.get("name") or row.get("instrument") or row.get("symbol") or "")
        price = float(row["price"])
        unit = str(row.get("unit") or row.get("currency") or "").strip()
        as_of = str(row.get("as_of") or "").replace("+00:00", " UTC")
        price_text = f"{price:,.2f} {unit}".strip()
        lines.append(f"| {name} | {price_text} | {as_of} |")
    lines.extend(["", "Source: Yahoo Finance chart data. Quotes may be delayed by the exchange/provider."])
    errors = [row for row in (payload.get("errors") or []) if isinstance(row, dict)]
    if errors:
        missing = ", ".join(str(row.get("instrument") or row.get("symbol") or "unknown") for row in errors)
        lines.append(f"Unavailable: {missing}.")
    return "\n".join(lines)


def is_simple_market_price_request(user_request: str) -> bool:
    """Return whether the request only asks for current/latest quote values."""
    text = " ".join(str(user_request or "").lower().split())
    if not is_market_price_request(user_request):
        return False
    return not re.search(r"\b(?:why|explain|analy[sz]e|compare|history|historical|chart|forecast|predict|outlook|impact|trend)\b", text)
