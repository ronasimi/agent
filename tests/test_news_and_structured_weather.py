import json

from tools.grounding import make_observation, requested_fact_types, validate_fact_grounding
from tools.weather import format_weather_recovery, is_simple_weather_request
from tools.task_requirements import build_news_query, derive_task_frame
from tools.web import (
    _location_scoped_news_rows, format_news_no_results, format_news_results,
    is_simple_headline_request, news_search, news_search_is_empty,
)


def test_latest_headlines_require_news_grounding():
    req = "what are the latest headlines for London ON"
    assert requested_fact_types(req) == {"news"}
    report = validate_fact_grounding(req, [], current_turn_id=1)
    assert report["grounded"] is False
    assert report["missing_fact_types"] == ["news"]


def test_news_search_observation_satisfies_latest_headlines():
    req = "what are the latest headlines for London ON"
    frame = derive_task_frame(req)
    query = build_news_query(req, frame)
    content = json.dumps([{
        "date": "2026-09-19T19:00:00+00:00",
        "title": "Local headline",
        "url": "https://example.com/story",
        "snippet": "Story summary",
        "source": "Example News",
    }])
    obs = make_observation(
        "news_search", content, turn_id=7,
        arguments={"query": query, "location": frame["entity"]},
    )
    report = validate_fact_grounding(req, [obs], current_turn_id=7, task_frame=frame)
    assert report["grounded"] is True
    assert report["evidence"]["news"] == ["news_search"]




def test_news_search_no_results_exception_becomes_one_bounded_empty_observation(monkeypatch):
    import sys
    import types
    import tools.web as web_tools

    calls = []

    class FakeDDGS:
        def news(self, **kwargs):
            calls.append(dict(kwargs))
            raise RuntimeError("No results found.")

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=FakeDDGS))
    monkeypatch.setattr(web_tools, "_google_news_rss_rows", lambda *args, **kwargs: [])
    content = news_search(
        query="London, Ontario, Canada local latest news",
        location="London, Ontario, Canada",
        timelimit="d",
        region="ca-en",
        max_results=8,
    )
    assert json.loads(content) == []
    assert news_search_is_empty(content) is True
    # Exactly one DDGS lookup; provider fallback stays internal and bounded.
    assert len(calls) == 1
    assert calls[0]["timelimit"] == "d"
    assert calls[0]["query"].lower().count("london") == 1


def test_news_search_non_empty_provider_failure_remains_explicit_error(monkeypatch):
    import sys
    import types
    import tools.web as web_tools

    class FakeDDGS:
        def news(self, **_kwargs):
            raise TimeoutError("provider timed out")

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=FakeDDGS))
    monkeypatch.setattr(web_tools, "_google_news_rss_rows", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("rss timed out")))
    content = news_search(query="local headlines", location="London, Ontario, Canada")
    assert content.startswith("Error: news search failed:")
    assert "provider timed out" in content


def test_location_scoped_news_filter_rejects_london_england_false_positive():
    rows = [{
        "date": "2026-09-16T11:02:00+00:00",
        "title": "Cricket is growing in Canada. How these girls from Ontario are making that happen",
        "url": "https://www.cbc.ca/kidsnews/post/cricket-is-growing-in-canada",
        "snippet": "A group of girls from Ontario visited London, England, to learn more about cricket.",
        "source": "CBC.ca",
    }]
    assert _location_scoped_news_rows(rows, "London, Ontario, Canada") == []


def test_news_empty_renderer_reports_retrieval_miss_without_claiming_no_news_exists():
    rendered = format_news_no_results(location="London ON")
    assert "London, Ontario, Canada" in rendered
    assert "couldn't find" in rendered
    assert "no local news exists" not in rendered.lower()


def test_news_renderer_only_uses_returned_rows():
    content = json.dumps([
        {"title": "One", "url": "https://example.com/1", "source": "A", "date": "2026-09-19"},
        {"title": "Two", "url": "https://example.com/2", "source": "B", "date": "2026-09-19"},
    ])
    rendered = format_news_results(content)
    assert "One" in rendered and "Two" in rendered
    assert "example.com/1" in rendered and "example.com/2" in rendered
    assert is_simple_headline_request("latest headlines for London ON") is True
    assert is_simple_headline_request("Fix the latest-headlines formatter") is False


def test_news_grounding_rejects_wrong_location_query():
    req = "what are the latest headlines for London ON"
    frame = derive_task_frame(req)
    content = json.dumps([{
        "title": "London headline", "url": "https://example.co.uk/story",
        "source": "London News", "snippet": "London story",
    }])
    wrong = make_observation(
        "news_search", content, turn_id=7,
        arguments={"query": "London UK latest news", "location": "London, United Kingdom"},
    )
    report = validate_fact_grounding(req, [wrong], current_turn_id=7, task_frame=frame)
    assert report["grounded"] is False
    assert report["missing_fact_types"] == ["news"]


def test_location_scoped_news_filter_rejects_similar_city_noise():
    rows = [
        {
            "title": "Man Utd XI vs Fulham", "url": "https://standard.co.uk/sport/story",
            "source": "London Evening Standard", "snippet": "",
        },
        {
            "title": "Game scheduled in London", "url": "https://sports.yahoo.com/story",
            "source": "Yahoo Sports", "snippet": "",
        },
        {
            "title": "Council approves housing plan", "url": "https://lfpress.com/news/local/council",
            "source": "London Free Press", "snippet": "",
        },
        {
            "title": "London police issue update", "url": "https://cbc.ca/news/canada/london/update",
            "source": "CBC News", "snippet": "Ontario investigation",
        },
    ]
    filtered = _location_scoped_news_rows(rows, "London ON")
    assert [row["source"] for row in filtered] == ["CBC News", "London Free Press"]


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


def test_encyclopedia_renderer_uses_only_structured_lookup_fields():
    from tools.web import format_encyclopedia_result, is_simple_encyclopedic_request

    content = json.dumps({
        "title": "Shoggoth",
        "url": "https://en.wikipedia.org/wiki/Shoggoth",
        "summary": "Shoggoths are fictional creatures in the Cthulhu Mythos.",
    })
    rendered = format_encyclopedia_result(content)
    assert "fictional creatures" in rendered
    assert "Wikipedia" in rendered
    assert "https://en.wikipedia.org/wiki/Shoggoth" in rendered
    assert "Shadow over Mitten" not in rendered
    assert is_simple_encyclopedic_request("What is a shoggoth?") is True
    assert is_simple_encyclopedic_request("Is God real?") is False


def test_news_search_uses_independent_rss_fallback_without_model_retry(monkeypatch):
    import sys
    import types
    import tools.web as web_tools

    ddgs_calls = []
    rss_calls = []

    class FakeDDGS:
        def news(self, **kwargs):
            ddgs_calls.append(dict(kwargs))
            raise RuntimeError("No results found.")

    def fake_rss(query, **kwargs):
        rss_calls.append((query, dict(kwargs)))
        return [{
            "date": "Sun, 21 Sep 2026 10:00:00 GMT",
            "title": "London council approves housing plan - CTV News London",
            "url": "https://news.google.com/rss/articles/example",
            "snippet": "London, Ontario council approved the plan.",
            "source": "CTV News London",
        }]

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=FakeDDGS))
    monkeypatch.setattr(web_tools, "_google_news_rss_rows", fake_rss)
    content = news_search(
        query="London, Ontario, Canada local latest news",
        location="London, Ontario, Canada",
        timelimit="d",
        region="ca-en",
    )
    rows = json.loads(content)
    assert len(rows) == 1
    assert rows[0]["source"] == "CTV News London"
    assert len(ddgs_calls) == 1
    assert len(rss_calls) == 1
    assert rss_calls[0][1]["timelimit"] == "w"


def test_google_news_rss_parser_returns_structured_rows(monkeypatch):
    import tools.web as web_tools

    xml = """<?xml version='1.0' encoding='UTF-8'?>
    <rss><channel><item>
      <title>Example headline - Example News</title>
      <link>https://news.google.com/rss/articles/abc</link>
      <pubDate>Sun, 21 Sep 2026 10:00:00 GMT</pubDate>
      <source url='https://example.com'>Example News</source>
      <description><![CDATA[<p>Example summary</p>]]></description>
    </item></channel></rss>"""

    monkeypatch.setattr(web_tools, "fetch_text", lambda *args, **kwargs: (args[0], "application/xml", xml))
    rows = web_tools._google_news_rss_rows("latest news", region="ca-en", timelimit="d")
    assert rows == [{
        "date": "Sun, 21 Sep 2026 10:00:00 GMT",
        "title": "Example headline - Example News",
        "url": "https://news.google.com/rss/articles/abc",
        "snippet": "Example summary",
        "source": "Example News",
    }]


def test_general_headlines_do_not_inherit_previous_local_news_scope():
    from tools.task_requirements import is_task_continuation

    previous = derive_task_frame(
        "what are the latest local headlines?",
        default_location="London, Ontario, Canada",
    )
    request = "what are the latest headlines?"
    assert is_task_continuation(request, previous) is False
    frame = derive_task_frame(request, {}, default_location="London, Ontario, Canada")
    assert frame == {"intent": "news", "time_scope": "latest"}
    assert build_news_query(request, frame, "London, Ontario, Canada") == "latest news"


def test_news_topic_is_not_misclassified_as_location():
    frame = derive_task_frame("latest AI headlines", default_location="London, Ontario, Canada")
    assert frame == {"intent": "news", "time_scope": "latest"}
    assert build_news_query("latest AI headlines", frame, "London, Ontario, Canada") == "ai latest news"


def test_generic_news_empty_renderer_is_not_localized():
    rendered = format_news_no_results()
    assert "current headlines" in rendered
    assert "local headlines" not in rendered
    assert "London" not in rendered
