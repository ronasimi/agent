import json

from tools.grounding import execute_weather_grounding_recovery, make_observation, validate_fact_grounding


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_weather_recovery_uses_declared_location_and_structured_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_RECIPE_DB", str(tmp_path / "recipes.db"))

    def fake_get(url, params=None, timeout=None):
        if "geocoding-api" in url:
            assert params["name"] == "London, Ontario, Canada"
            return _Response({
                "results": [{
                    "name": "London",
                    "latitude": 42.9834,
                    "longitude": -81.233,
                    "country": "Canada",
                    "country_code": "CA",
                    "admin1": "Ontario",
                    "timezone": "America/Toronto",
                }]
            })
        assert "api.open-meteo.com" in url
        assert params["forecast_days"] == 8
        return _Response({
            "latitude": 42.98,
            "longitude": -81.23,
            "timezone": "America/Toronto",
            "timezone_abbreviation": "EDT",
            "utc_offset_seconds": -14400,
            "elevation": 250,
            "current": {"temperature_2m": 20.0, "weather_code": 1, "wind_speed_10m": 9.0},
            "current_units": {"temperature_2m": "°C", "wind_speed_10m": "km/h"},
            "daily": {
                "time": ["2026-09-19", "2026-09-20"],
                "temperature_2m_max": [22.0, 21.0],
                "temperature_2m_min": [12.0, 11.0],
                "precipitation_probability_max": [10, 30],
                "weather_code": [1, 2],
            },
            "daily_units": {"temperature_2m_max": "°C", "temperature_2m_min": "°C"},
        })

    import tools.weather
    monkeypatch.setattr(tools.weather.requests, "get", fake_get)

    context = """### User Context
**Location**: London, Ontario, Canada

[{"topic":"user_location","fact":"London, Ontario, Canada"}]
"""
    result = execute_weather_grounding_recovery("What is the weather for the next week?", context)
    assert result["ok"] is True
    assert [stage["tool"] for stage in result["stages"]] == ["geocode_location", "weather_forecast", "compose_object"]
    assert result["grounding_recovery"]["location"] == "London, Ontario, Canada"
    assert result["grounding_recovery"]["forecast_days"] == 8

    raw = json.dumps(result)
    observation = make_observation(
        "recipe:weather.current_forecast",
        raw,
        turn_id=1,
        arguments={"request": "weather next week"},
    )
    report = validate_fact_grounding("weather for the next week", [observation], current_turn_id=1)
    assert report["grounded"] is True


def test_weather_fallback_renderer_surfaces_verified_web_evidence():
    from tools.weather import format_weather_recovery

    result = {
        "ok": True,
        "result": {
            "query": "London Ontario weather current",
            "verification": (
                "URL: https://weather.example/london\n"
                "Content-Type: text/html\n"
                "Extraction: current temperature conditions feels like humidity wind precipitation\n\n"
                "London, Ontario: 14 C, partly cloudy. Feels like 13 C. "
                "Humidity 70%. Wind W 12 km/h."
            ),
        },
        "grounding_recovery": {
            "fact_type": "weather",
            "location": "London, Ontario, Canada",
            "source": "web_verification_fallback",
        },
    }

    rendered = format_weather_recovery(result, "Get the current weather for London, Ontario.")
    assert "Current weather for London, Ontario, Canada" in rendered
    assert "14 C" in rendered
    assert "weather.example/london" in rendered
