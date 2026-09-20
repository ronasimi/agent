import json

from tools.grounding import make_observation, requested_fact_types, validate_fact_grounding
from tools.weather import format_weather_recovery, is_simple_weather_request
from tools.web import format_news_results, is_simple_headline_request


def test_latest_headlines_require_news_grounding():
    req = "what are the latest headlines for London ON"
    assert requested_fact_types(req) == {"news"}
    report = validate_fact_grounding(req, [], current_turn_id=1)
    assert report["grounded"] is False
    assert report["missing_fact_types"] == ["news"]


def test_news_search_observation_satisfies_latest_headlines():
    req = "what are the latest headlines for London ON"
    content = json.dumps([{
        "date": "2026-09-19T19:00:00+00:00",
        "title": "Local headline",
        "url": "https://example.com/story",
        "snippet": "Story summary",
        "source": "Example News",
    }])
    obs = make_observation("news_search", content, turn_id=7, arguments={"query": req})
    report = validate_fact_grounding(req, [obs], current_turn_id=7)
    assert report["grounded"] is True
    assert report["evidence"]["news"] == ["news_search"]


def test_news_renderer_only_uses_returned_rows():
    content = json.dumps([
        {"title": "One", "url": "https://example.com/1", "source": "A", "date": "2026-09-19"},
        {"title": "Two", "url": "https://example.com/2", "source": "B", "date": "2026-09-19"},
    ])
    rendered = format_news_results(content)
    assert "One" in rendered and "Two" in rendered
    assert "example.com/1" in rendered and "example.com/2" in rendered
    assert is_simple_headline_request("latest headlines for London ON") is True


def test_weather_renderer_enforces_next_week_horizon_and_provider_fields_only():
    result = {
        "ok": True,
        "result": {
            "location": "London, Ontario, Canada",
            "place": {"name": "London", "admin1": "Ontario", "country": "Canada"},
            "forecast": {
                "retrieved_at": "2026-09-19T21:00:00+00:00",
                "daily": {
                    "time": [f"2026-09-{day:02d}" for day in range(19, 28)],
                    "weather_code": [1] * 9,
                    "temperature_2m_max": list(range(20, 29)),
                    "temperature_2m_min": list(range(10, 19)),
                    "precipitation_probability_max": [10] * 9,
                    "precipitation_sum": [0.0] * 9,
                    "wind_speed_10m_max": [12] * 9,
                    "wind_gusts_10m_max": [20] * 9,
                },
            },
        },
    }
    rendered = format_weather_recovery(result, "What is the weather for the next week?")
    # Sep 19 is provider 'today'; next week is the next seven future dates only.
    assert "| 2026-09-19 |" not in rendered
    for day in range(20, 27):
        assert f"2026-09-{day:02d}" in rendered
    assert "2026-09-27" not in rendered
    assert "Humidity" not in rendered
    assert rendered.count("\n|") == 9  # header + separator + seven data rows
    assert is_simple_weather_request("What is the weather for the next week?") is True


def test_weather_right_now_uses_current_conditions_and_not_daily_table():
    result = {
        "ok": True,
        "result": {
            "location": "London, Ontario, Canada",
            "place": {"name": "London", "admin1": "Ontario", "country": "Canada"},
            "forecast": {
                "retrieved_at": "2026-09-19T23:56:20+00:00",
                "timezone_abbreviation": "EDT",
                "current": {
                    "time": "2026-09-19T19:55",
                    "temperature_2m": 18.4,
                    "apparent_temperature": 17.8,
                    "precipitation": 0.0,
                    "weather_code": 3,
                    "cloud_cover": 91,
                    "wind_speed_10m": 8.2,
                    "wind_direction_10m": 45,
                    "wind_gusts_10m": 14.8,
                },
                "daily": {
                    "time": ["2026-09-19", "2026-09-20"],
                    "weather_code": [3, 2],
                    "temperature_2m_max": [23, 24],
                    "temperature_2m_min": [13, 14],
                },
            },
        },
    }
    rendered = format_weather_recovery(result, "What is the weather right now?")
    assert "Current weather for London, Ontario, Canada" in rendered
    assert "Overcast, 18.4 °C" in rendered
    assert "feels like 17.8 °C" in rendered
    assert "8.2 km/h NE" in rendered
    assert "| Date |" not in rendered
    assert "2026-09-20" not in rendered
