from datetime import datetime, timedelta, timezone

from tools.grounding import (
    classify_fact_types,
    make_observation,
    requested_fact_types,
    validate_fact_grounding,
)


def _at(dt):
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def test_weather_request_is_fact_grounded_but_unrelated_request_is_not_gated():
    assert requested_fact_types("What's the weather in London Ontario now?") == {"weather"}
    assert requested_fact_types("Explain how HTTP caching works") == set()


def test_current_time_only_is_missing_weather_evidence():
    now = datetime(2026, 9, 19, 17, 48, tzinfo=timezone.utc)
    observations = [make_observation(
        "current_time",
        '{"utc":"2026-09-19T17:48:00+00:00","local":"2026-09-19T13:48:00-04:00","timezone":"America/Toronto"}',
        at=_at(now),
        turn_id=11,
    )]
    report = validate_fact_grounding(
        "What is the weather in London Ontario now?",
        observations,
        current_turn_id=11,
        now=now,
    )
    assert report["status"] == "missing_evidence"
    assert report["grounded"] is False
    assert report["missing_fact_types"] == ["weather"]
    assert report["observed_fact_types"] == ["current_time"]


def test_current_weather_requires_search_and_verified_page_when_using_web_path():
    now = datetime(2026, 9, 19, 17, 48, tzinfo=timezone.utc)
    search = make_observation(
        "web_search",
        '[{"title":"London, ON - 7 Day Forecast","url":"https://weather.gc.ca/x","snippet":"Weather forecast for London"}]',
        at=_at(now), turn_id=4,
    )
    report = validate_fact_grounding("weather in London today", [search], current_turn_id=4, now=now)
    assert report["status"] == "missing_evidence"

    browse = make_observation(
        "browse_url",
        "URL: https://weather.gc.ca/x\nCurrent Conditions\nTemperature 20 C\nWind SW 10 km/h\nForecast Tonight: cloudy",
        at=_at(now), turn_id=4,
    )
    report = validate_fact_grounding("weather in London today", [search, browse], current_turn_id=4, now=now)
    assert report["status"] == "grounded"
    assert report["evidence"]["weather"] == ["web_search", "browse_url"]


def test_weather_recipe_observation_satisfies_grounding_by_itself():
    now = datetime(2026, 9, 19, 17, 48, tzinfo=timezone.utc)
    recipe_result = '''{
      "ok": true,
      "stages": [
        {"id":"search","tool":"web_search","ok":true},
        {"id":"verify","tool":"browse_url","ok":true},
        {"id":"result","tool":"compose_object","ok":true}
      ],
      "result": {"verification":"Current Conditions Temperature 19 C Wind W 12 km/h Forecast cloudy"},
      "grounding_recovery": {"fact_type":"weather","recipe":"weather.current_forecast"}
    }'''
    obs = make_observation("recipe:weather.current_forecast", recipe_result, at=_at(now), turn_id=9)
    assert "weather" in classify_fact_types(obs["tool"], obs["evidence_preview"])
    report = validate_fact_grounding("weather now", [obs], current_turn_id=9, now=now)
    assert report["grounded"] is True


def test_fresh_stored_verified_weather_can_be_reused_but_stale_weather_cannot():
    now = datetime(2026, 9, 19, 17, 48, tzinfo=timezone.utc)
    content = "Current Conditions Temperature 18 C Humidity 70 percent Wind 8 km/h Forecast clear"
    fresh = make_observation("browse_url", content, at=_at(now - timedelta(minutes=30)), turn_id=20)
    fresh_report = validate_fact_grounding(
        "show the weather forecast",
        [fresh],
        current_turn_id=21,
        weather_max_age_seconds=3600,
        now=now,
    )
    assert fresh_report["grounded"] is True
    assert fresh_report["evidence"]["weather"][0].startswith("stored:")

    stale = make_observation("browse_url", content, at=_at(now - timedelta(hours=3)), turn_id=20)
    stale_report = validate_fact_grounding(
        "show the weather forecast",
        [stale],
        current_turn_id=21,
        weather_max_age_seconds=3600,
        now=now,
    )
    assert stale_report["grounded"] is False


def test_weather_keyword_in_code_edit_request_does_not_trigger_fact_retrieval_gate():
    assert requested_fact_types("Refactor the weather validator and update its tests") == set()


def test_weather_recipe_does_not_pass_when_verified_page_is_not_weather_bearing():
    now = datetime(2026, 9, 19, 17, 48, tzinfo=timezone.utc)
    recipe_result = '''{
      "ok": true,
      "stages": [
        {"id":"search","tool":"web_search","ok":true},
        {"id":"verify","tool":"browse_url","ok":true}
      ],
      "result": {
        "discovery":[{"title":"London Weather Forecast","url":"https://example.com"}],
        "verification":"Example Domain. This domain is for documentation examples."
      }
    }'''
    obs = make_observation("recipe:weather.current_forecast", recipe_result, at=_at(now), turn_id=9)
    report = validate_fact_grounding("weather in London today", [obs], current_turn_id=9, now=now)
    assert report["status"] == "missing_evidence"
    assert report["missing_fact_types"] == ["weather"]


def test_weather_api_observation_can_satisfy_grounding_without_web_pair():
    now = datetime(2026, 9, 19, 17, 48, tzinfo=timezone.utc)
    api = make_observation(
        "weather_api",
        '{"temperature_2m":19.2,"wind_speed_10m":12.0,"precipitation_probability":20}',
        at=_at(now), turn_id=30,
    )
    report = validate_fact_grounding("weather in London now", [api], current_turn_id=30, now=now)
    assert report["grounded"] is True
    assert report["evidence"]["weather"] == ["weather_api"]


def test_weather_domain_error_page_is_not_weather_evidence_by_hostname_alone():
    now = datetime(2026, 9, 19, 17, 48, tzinfo=timezone.utc)
    browse = make_observation(
        "browse_url",
        "URL: https://weather.gc.ca/missing\n404 Not Found\nThe requested page could not be located.",
        at=_at(now), turn_id=31,
    )
    assert "weather" not in browse["fact_types"]


def test_persisted_recipe_metadata_survives_clipped_evidence_preview():
    now = datetime(2026, 9, 19, 17, 48, tzinfo=timezone.utc)
    obs = {
        "tool": "recipe:weather.current_forecast",
        "status": "ok",
        "evidence_preview": "{clipped before the verified payload}",
        "fact_types": ["weather"],
        "source_tools": ["web_search", "browse_url", "compose_object"],
        "weather_verified": True,
        "at": _at(now),
        "turn_id": 32,
    }
    report = validate_fact_grounding("weather in London now", [obs], current_turn_id=32, now=now)
    assert report["grounded"] is True
    assert report["evidence"]["weather"] == ["recipe:weather.current_forecast"]


def test_structured_weather_recipe_observation_satisfies_grounding():
    now = datetime(2026, 9, 19, 17, 48, tzinfo=timezone.utc)
    recipe_result = '''{
      "ok": true,
      "stages": [
        {"id":"place","tool":"geocode_location","ok":true},
        {"id":"forecast","tool":"weather_forecast","ok":true},
        {"id":"result","tool":"compose_object","ok":true}
      ],
      "result": {
        "location":"London, Ontario, Canada",
        "forecast": {
          "provider":"Open-Meteo",
          "daily":{"temperature_2m_max":[21,19],"temperature_2m_min":[10,9],"precipitation_probability_max":[20,40]}
        }
      },
      "grounding_recovery": {"fact_type":"weather","query":"weather London Ontario next week","location":"London, Ontario, Canada"}
    }'''
    obs = make_observation(
        "recipe:weather.current_forecast",
        recipe_result,
        at=_at(now),
        turn_id=40,
        arguments={"request": "weather London Ontario next week"},
    )
    assert obs["weather_verified"] is True
    assert "weather_forecast" in obs["source_tools"]
    report = validate_fact_grounding(
        "weather in London Ontario next week", [obs], current_turn_id=40, now=now
    )
    assert report["grounded"] is True


def test_simple_encyclopedic_definition_requires_current_turn_wikipedia_evidence():
    import json
    from tools.grounding import encyclopedic_lookup_query

    req = "What is a shoggoth?"
    assert encyclopedic_lookup_query(req) == "shoggoth"
    assert requested_fact_types(req) == {"encyclopedic"}
    assert requested_fact_types("Is God real?") == set()
    assert requested_fact_types("What is love?") == set()

    content = json.dumps({
        "title": "Shoggoth",
        "url": "https://en.wikipedia.org/wiki/Shoggoth",
        "summary": "A shoggoth is a fictional monster in the Cthulhu Mythos.",
    })
    old = make_observation("wiki_search", content, turn_id=10, arguments={"query": "shoggoth"})
    assert validate_fact_grounding(req, [old], current_turn_id=11)["grounded"] is False

    current = make_observation("wiki_search", content, turn_id=11, arguments={"query": "shoggoth"})
    report = validate_fact_grounding(req, [current], current_turn_id=11)
    assert report["grounded"] is True
    assert report["evidence"]["encyclopedic"] == ["wiki_search"]
