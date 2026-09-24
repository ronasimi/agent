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
    assert requested_fact_types("Check network status") == {"network_state"}


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


def test_current_time_requires_current_turn_and_requested_timezone_scope():
    import json

    request = "What time is it in Tokyo?"
    frame = __import__("tools.task_requirements", fromlist=["derive_task_frame"]).derive_task_frame(request)
    stale = make_observation(
        "current_time",
        json.dumps({"utc": "2026-09-21T04:00:00+00:00", "local": "2026-09-21T13:00:00+09:00", "timezone": "Asia/Tokyo"}),
        turn_id=40,
        arguments={"timezone_name": "Asia/Tokyo"},
    )
    assert validate_fact_grounding(request, [stale], current_turn_id=41, task_frame=frame)["grounded"] is False

    wrong = make_observation(
        "current_time",
        json.dumps({"utc": "2026-09-21T04:00:00+00:00", "local": "2026-09-21T00:00:00-04:00", "timezone": "America/Toronto"}),
        turn_id=41,
        arguments={"timezone_name": "America/Toronto"},
    )
    assert validate_fact_grounding(request, [wrong], current_turn_id=41, task_frame=frame)["grounded"] is False

    geocode = make_observation(
        "geocode_location",
        json.dumps([{"name": "Tokyo", "country": "Japan", "latitude": 35.6762, "longitude": 139.6503, "timezone": "Asia/Tokyo"}]),
        turn_id=41,
        arguments={"query": "Tokyo", "count": 1},
    )
    current = make_observation(
        "current_time",
        json.dumps({"utc": "2026-09-21T04:00:00+00:00", "local": "2026-09-21T13:00:00+09:00", "timezone": "Asia/Tokyo"}),
        turn_id=41,
        arguments={"timezone_name": "Asia/Tokyo"},
    )
    assert validate_fact_grounding(request, [geocode, current], current_turn_id=41, task_frame=frame)["grounded"] is True


def test_encyclopedic_lookup_does_not_pass_from_incidental_summary_mention():
    import json

    request = "What is a Cthulhu?"
    wrong = make_observation(
        "wiki_search",
        json.dumps({
            "title": "Shoggoth",
            "url": "https://en.wikipedia.org/wiki/Shoggoth",
            "summary": "A shoggoth is a fictional creature in the Cthulhu Mythos.",
        }),
        turn_id=50,
        arguments={"query": "shoggoth"},
    )
    report = validate_fact_grounding(request, [wrong], current_turn_id=50)
    assert report["grounded"] is False
    assert report["missing_fact_types"] == ["encyclopedic"]


def test_weather_browse_scope_proof_survives_location_after_preview_clip(tmp_path, monkeypatch):
    import json
    from tools import working_state
    from tools.task_requirements import derive_task_frame

    request = "What is the weather in London Ontario today?"
    frame = derive_task_frame(request)
    monkeypatch.setattr(working_state, "DB_PATH", str(tmp_path / "weather-state.db"))
    store = working_state.WorkingStateStore(limits={"evidence_preview_chars": 180})
    store.begin_turn(
        turn_id=60, objective=request, rolling_summary="", recalled_context="", recent_messages=[],
        policy_note="", tool_schemas=[], task_frame=frame,
    )
    url = "https://weather.example/london"
    search_payload = json.dumps([{"title": "Forecast", "url": url, "snippet": "London Ontario Canada weather today"}])
    store.record_tool_result(
        tool_name="web_search", arguments={"query": "London Ontario Canada weather today"}, status="ok",
        reason="ok", result_text=search_payload, fingerprint="search-60",
    )
    filler = " ".join(f"token{i}" for i in range(1000))
    page = f"URL: {url}\n{filler}\nLondon Ontario Canada Current Conditions Temperature 19 C Wind 10 km/h Forecast cloudy"
    store.record_tool_result(
        tool_name="browse_url", arguments={"url": url}, status="ok", reason="ok",
        result_text=page, fingerprint="browse-60",
    )
    observations = store.load()["verified_observations"]
    browse = observations[-1]
    assert "London" not in browse["evidence_preview"]
    assert browse["grounding_proof"]["scope_entity"] == frame["entity"]
    report = validate_fact_grounding(request, observations, current_turn_id=60, task_frame=frame)
    assert report["grounded"] is True


def test_structured_fact_tools_do_not_ground_arbitrary_nonempty_text():
    assert classify_fact_types("current_time", "timezone UTC at noon") == set()
    assert classify_fact_types("host_snapshot", "CPU looks healthy") == set()
    assert classify_fact_types("network_snapshot", "interfaces are up") == set()
    assert classify_fact_types("repo_status", "working tree clean") == set()

    assert "host_state" in classify_fact_types("host_snapshot", '{"cpu_count":8}')
    assert "host_state" not in classify_fact_types("filesystem_snapshot", '{"filesystems":[]}')
    assert "host_state" not in classify_fact_types("process_snapshot", '{"processes":[]}')
    assert "host_state" not in classify_fact_types("pressure_snapshot", '{"cpu":[]}')
    assert "network_state" in classify_fact_types("neighbor_snapshot", "[]")
    assert "repository_state" in classify_fact_types("repo_status", '{"git_repo":false}')


def test_weather_shell_with_no_data_and_navigation_numbers_is_not_current_weather_evidence():
    from tools.grounding import is_weather_data_evidence

    shell = (
        "Hourly Today's Conditions -- Sunrise -- Sunset -- No Data Available "
        "Wind -- Gust: -- No Data Available Pressure -- No Data Available "
        "Humidity -- No Data Available Visibility -- No Data Available Ceiling -- "
        "No Data Available Yesterday -- No Data Available 7 Days 14 Days Radar Map "
        "News Welcome to fall, Canada! 2:00"
    )
    assert is_weather_data_evidence(shell) is False




def test_weather_shell_with_article_durations_and_pressure_prose_is_not_evidence():
    from tools.grounding import is_weather_data_evidence

    shell = (
        "Hourly Today’s Conditions -- Sunrise -- Sunset -- No Data Available "
        "Wind -- Gust: -- No Data Available Pressure -- No Data Available "
        "Humidity -- No Data Available Visibility -- No Data Available Ceiling -- "
        "No Data Available Yesterday -- No Data Available 7 Days 14 Days Radar Map News "
        "Welcome to fall, Canada! Here's your next 3 months of weather 2:00 "
        "Canada officially welcomes fall with early signs of...snow? 1:03 Could the fall "
        "equinox enhance aurora chances Wednesday night? Category 5 Hurricane Polo 1:07. "
        "Coldest air in months: Unusually strong high pressure grips Ontario, Quebec 1:33."
    )
    assert is_weather_data_evidence(shell) is False

def test_weather_numeric_values_still_qualify_when_an_unrelated_field_is_unavailable():
    from tools.grounding import is_weather_data_evidence

    text = (
        "Current Conditions. Temperature 14 C. Feels like 13 C. Humidity 70%. "
        "Wind W 12 km/h. Visibility No Data Available."
    )
    assert is_weather_data_evidence(text) is True


def test_malformed_generic_weather_entity_cannot_match_carried_evidence():
    now = datetime(2026, 9, 24, 2, 40, tzinfo=timezone.utc)
    content = "Current Conditions Temperature 18 C Humidity 70 percent Wind 8 km/h Forecast clear"
    observation = make_observation(
        "browse_url",
        content,
        at=_at(now - timedelta(minutes=5)),
        turn_id=20,
        arguments={"url": "https://weather.example/london"},
    )
    report = validate_fact_grounding(
        "what is the current weather?",
        [observation],
        current_turn_id=21,
        weather_max_age_seconds=3600,
        now=now,
        task_frame={"intent": "weather", "entity": "what is the", "time_scope": "current"},
    )
    assert report["grounded"] is False
    assert report["missing_fact_types"] == ["weather"]


def test_grounding_records_selected_evidence_details_for_supported_fact_paths():
    now = datetime(2026, 9, 24, 2, 40, tzinfo=timezone.utc)

    time_obs = make_observation(
        "current_time",
        '{"utc":"2026-09-24T02:40:00+00:00","local":"2026-09-23T22:40:00-04:00","timezone":"America/Toronto"}',
        at=_at(now), turn_id=31,
    )
    time_obs["evidence_ref"] = "time-ref"
    time_report = validate_fact_grounding("What time is it?", [time_obs], current_turn_id=31, now=now)
    assert time_report["evidence_details"]["current_time"]["evidence_ref"] == "time-ref"

    news_obs = make_observation(
        "news_search",
        '[{"title":"Headline","url":"https://example.com/story","source":"Example"}]',
        at=_at(now), turn_id=31, arguments={"query": "latest news", "location": ""},
    )
    news_obs["evidence_ref"] = "news-ref"
    news_report = validate_fact_grounding("latest news", [news_obs], current_turn_id=31, now=now)
    assert news_report["evidence_details"]["news"]["evidence_ref"] == "news-ref"

    market_obs = make_observation(
        "market_quote",
        '{"quotes":[{"instrument":"brent","symbol":"BZ=F","price":74.2}]}',
        at=_at(now), turn_id=31, arguments={"instruments": ["brent"]},
    )
    market_obs["evidence_ref"] = "market-ref"
    market_report = validate_fact_grounding(
        "What is the current price of Brent crude?", [market_obs], current_turn_id=31, now=now,
    )
    assert market_report["evidence_details"]["market_price"]["evidence_ref"] == "market-ref"


def test_fact_requirement_closure_keeps_grounding_provenance():
    from tools.task_requirements import TaskRequirementLedger

    ledger = TaskRequirementLedger.from_request("What is the weather?")
    ledger.mark_fact_satisfied(
        "weather",
        evidence={
            "tool": "recipe:weather.current_forecast",
            "evidence_ref": "weather-ref",
            "evidence_preview": "Open-Meteo structured weather",
        },
    )
    row = next(item for item in ledger.as_list() if item["tool"] == "weather_forecast")
    assert row["status"] == "satisfied"
    assert row["evidence"]
    assert row["evidence"][-1]["evidence_ref"] == "weather-ref"


def test_fresh_stored_weather_exposes_exact_selected_evidence_reference():
    now = datetime(2026, 9, 24, 2, 40, tzinfo=timezone.utc)
    content = '''{
      "ok": true,
      "stages": [{"id":"forecast","tool":"weather_forecast","ok":true}],
      "result": {"forecast":{"current":{"temperature_2m":12.5},"daily":{"time":["2026-09-24"]}}},
      "grounding_recovery": {"fact_type":"weather","query":"London Ontario weather current","location":"London, Ontario, Canada"}
    }'''
    observation = make_observation(
        "recipe:weather.current_forecast",
        content,
        at=_at(now - timedelta(minutes=5)),
        turn_id=20,
        task_frame={"intent": "weather", "entity": "London, Ontario, Canada", "time_scope": "current"},
    )
    observation["evidence_ref"] = "weather-observation-ref"
    report = validate_fact_grounding(
        "what is the current weather?",
        [observation],
        current_turn_id=21,
        weather_max_age_seconds=3600,
        now=now,
        task_frame={"intent": "weather", "entity": "London, Ontario, Canada", "time_scope": "current"},
        fact_frames={"weather": {"intent": "weather", "entity": "London, Ontario, Canada", "time_scope": "current"}},
    )
    assert report["grounded"] is True
    assert report["evidence_details"]["weather"]["source"] == "stored"
    assert report["evidence_details"]["weather"]["evidence_ref"] == "weather-observation-ref"


def test_every_grounded_fact_path_has_selected_evidence_metadata():
    now = datetime(2026, 9, 24, 2, 40, tzinfo=timezone.utc)
    cases = []

    time_obs = make_observation(
        "current_time",
        '{"utc":"2026-09-24T02:40:00+00:00","local":"2026-09-23T22:40:00-04:00","timezone":"America/Toronto"}',
        at=_at(now), turn_id=31,
    )
    cases.append(("What time is it?", [time_obs], {"current_time"}))

    host_obs = make_observation("host_snapshot", '{"cpu":{"load":0.1}}', at=_at(now), turn_id=31)
    cases.append(("Check host health", [host_obs], {"host_state"}))

    news_obs = make_observation(
        "news_search",
        '[{"title":"Headline","url":"https://example.com/story","source":"Example"}]',
        at=_at(now), turn_id=31, arguments={"query": "latest news", "location": ""},
    )
    cases.append(("latest news", [news_obs], {"news"}))

    market_obs = make_observation(
        "market_quote",
        '{"quotes":[{"instrument":"brent","symbol":"BZ=F","price":74.2}]}',
        at=_at(now), turn_id=31, arguments={"instruments": ["brent"]},
    )
    cases.append(("current price of Brent crude", [market_obs], {"market_price"}))

    for request, observations, expected in cases:
        report = validate_fact_grounding(request, observations, current_turn_id=31, now=now)
        assert report["grounded"] is True, (request, report)
        assert expected.issubset(set(report["evidence_details"])), (request, report)
        assert set(report["evidence"]).issubset(set(report["evidence_details"])), (request, report)


def test_all_grounded_fact_types_expose_selected_evidence_metadata():
    """Every grounding success must identify the concrete observation that closed it."""
    import json

    now = datetime(2026, 9, 24, 2, 40, tzinfo=timezone.utc)
    turn = 44
    cases = []

    cases.append((
        "Who is Ada Lovelace?",
        [make_observation(
            "wiki_search",
            json.dumps({"title": "Ada Lovelace", "summary": "English mathematician and writer."}),
            at=_at(now), turn_id=turn, arguments={"query": "Ada Lovelace"},
        )],
        "encyclopedic",
    ))
    cases.append((
        "Check host health",
        [make_observation("host_snapshot", json.dumps({"cpu": {"load": 0.1}}), at=_at(now), turn_id=turn)],
        "host_state",
    ))
    cases.append((
        "Check network status",
        [make_observation("network_snapshot", json.dumps({"interfaces": []}), at=_at(now), turn_id=turn)],
        "network_state",
    ))
    cases.append((
        "Check repo status",
        [make_observation("repo_status", json.dumps({"branch": "main", "dirty": False}), at=_at(now), turn_id=turn)],
        "repository_state",
    ))
    cases.append((
        "latest news",
        [make_observation(
            "news_search",
            json.dumps([{"title": "Headline", "url": "https://example.com/story", "source": "Example"}]),
            at=_at(now), turn_id=turn, arguments={"query": "latest news", "location": ""},
        )],
        "news",
    ))
    cases.append((
        "current price of Brent crude",
        [make_observation(
            "market_quote",
            json.dumps({"quotes": [{"instrument": "brent", "symbol": "BZ=F", "price": 74.2}]}),
            at=_at(now), turn_id=turn, arguments={"instruments": ["brent"]},
        )],
        "market_price",
    ))

    search = make_observation(
        "web_search",
        json.dumps([{"title": "Python release", "url": "https://example.com/python", "snippet": "current Python release"}]),
        at=_at(now), turn_id=turn, arguments={"query": "current Python release"},
    )
    browse = make_observation(
        "browse_url",
        "URL: https://example.com/python\nCurrent Python release information.",
        at=_at(now), turn_id=turn, arguments={"url": "https://example.com/python"},
    )
    cases.append(("Search the web for current Python release", [search, browse], "web_fact"))

    weather = make_observation(
        "recipe:weather.current_forecast",
        json.dumps({
            "ok": True,
            "stages": [{"id": "forecast", "tool": "weather_forecast", "ok": True}],
            "result": {
                "location": "London, Ontario, Canada",
                "forecast": {
                    "current": {"temperature_2m": 12.5, "weather_code": 1},
                    "daily": {"time": ["2026-09-24"], "weather_code": [1]},
                },
            },
            "grounding_recovery": {
                "fact_type": "weather", "query": "London Ontario weather now",
                "location": "London, Ontario, Canada",
            },
        }),
        at=_at(now), turn_id=turn,
        task_frame={"intent": "weather", "entity": "London, Ontario, Canada", "time_scope": "now"},
    )
    cases.append(("weather in London, Ontario, Canada now", [weather], "weather"))

    for request, observations, fact_type in cases:
        report = validate_fact_grounding(request, observations, current_turn_id=turn, now=now)
        assert report["grounded"] is True, (fact_type, report)
        assert fact_type in report["evidence"], (fact_type, report)
        detail = report["evidence_details"].get(fact_type)
        assert isinstance(detail, dict) and detail.get("tool"), (fact_type, report)
