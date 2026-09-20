import json

from tools.grounding import make_observation, requested_fact_types, validate_fact_grounding
from tools.market import extract_market_instruments, format_market_quotes, is_market_price_request, is_simple_market_price_request
from tools.task_requirements import derive_requirements, derive_task_frame


def test_current_brent_wti_request_routes_to_market_price():
    req = "What is the current price of brent crude and WTI?"
    assert is_market_price_request(req) is True
    assert extract_market_instruments(req) == ["brent", "wti"]
    frame = derive_task_frame(req)
    assert frame["intent"] == "market_price"
    assert frame["instruments"] == ["brent", "wti"]
    assert requested_fact_types(req, task_frame=frame) == {"market_price"}
    requirements = derive_requirements(req)
    assert any(row.tool == "market_quote" for row in requirements)


def test_market_quote_observation_must_cover_all_requested_instruments():
    req = "What is the current price of brent crude and WTI?"
    frame = derive_task_frame(req)
    brent_only = json.dumps({"quotes": [{"instrument": "brent", "symbol": "BZ=F", "price": 74.2}]})
    obs = make_observation("market_quote", brent_only, turn_id=12, arguments={"instruments": ["brent"]})
    report = validate_fact_grounding(req, [obs], current_turn_id=12, task_frame=frame)
    assert report["grounded"] is False
    assert report["missing_fact_types"] == ["market_price"]

    both = json.dumps({"quotes": [
        {"instrument": "brent", "symbol": "BZ=F", "price": 74.2},
        {"instrument": "wti", "symbol": "CL=F", "price": 70.1},
    ]})
    obs2 = make_observation("market_quote", both, turn_id=12, arguments={"instruments": ["brent", "wti"]})
    report2 = validate_fact_grounding(req, [obs2], current_turn_id=12, task_frame=frame)
    assert report2["grounded"] is True
    assert report2["evidence"]["market_price"] == ["market_quote"]


def test_market_renderer_is_deterministic_and_preserves_provider_values():
    payload = json.dumps({"quotes": [
        {
            "instrument": "brent", "name": "Brent Crude Oil Futures", "symbol": "BZ=F",
            "price": 74.217, "currency": "USD", "unit": "USD/barrel",
            "as_of": "2026-09-20T12:30:00+00:00",
        },
        {
            "instrument": "wti", "name": "WTI Crude Oil Futures", "symbol": "CL=F",
            "price": 70.013, "currency": "USD", "unit": "USD/barrel",
            "as_of": "2026-09-20T12:30:00+00:00",
        },
    ], "errors": []})
    rendered = format_market_quotes(payload)
    assert "74.22 USD/barrel" in rendered
    assert "70.01 USD/barrel" in rendered
    assert "2026-09-20T12:30:00 UTC" in rendered
    assert "Yahoo Finance" in rendered
    assert is_simple_market_price_request("What is the current price of Brent crude and WTI?") is True
    assert is_simple_market_price_request("Why did the current price of Brent crude rise?") is False


def test_price_word_in_implementation_request_does_not_route_to_live_market_data():
    req = "Fix the current-price parser for Brent crude in the harness"
    assert requested_fact_types(req) == set()


def test_market_quote_fetches_yahoo_chart_and_maps_crude_symbols(monkeypatch):
    from tools.market import market_quote

    class Response:
        def __init__(self, symbol):
            self.symbol = symbol
        def raise_for_status(self):
            return None
        def json(self):
            price = 75.25 if "BZ%3DF" in self.symbol else 71.5
            return {"chart": {"result": [{
                "meta": {
                    "regularMarketPrice": price,
                    "regularMarketTime": 1789900000,
                    "currency": "USD",
                    "exchangeName": "NYM",
                    "chartPreviousClose": price - 1,
                },
                "timestamp": [1789900000],
                "indicators": {"quote": [{"close": [price]}]},
            }], "error": None}}

    def fake_get(url, headers=None, timeout=None):
        return Response(url)

    monkeypatch.setattr("requests.get", fake_get)
    payload = json.loads(market_quote(["brent", "wti"]))
    assert [row["symbol"] for row in payload["quotes"]] == ["BZ=F", "CL=F"]
    assert [row["price"] for row in payload["quotes"]] == [75.25, 71.5]
    assert all(row["unit"] == "USD/barrel" for row in payload["quotes"])
