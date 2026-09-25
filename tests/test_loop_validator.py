import json
from tools.loop_validator import classify_tool_outcome

def test_tool_outcome_classifier_recognizes_hard_and_soft_failures():
    from tools.loop_validator import classify_tool_outcome

    assert classify_tool_outcome("Error: missing file")["success"] is False
    assert classify_tool_outcome("Error reading file 'x': missing")["success"] is False
    assert classify_tool_outcome("No search results found for query: x")["success"] is False
    assert classify_tool_outcome('{"error":"offline"}')["success"] is False
    assert classify_tool_outcome("normal useful data")["success"] is True


def test_tool_aware_empty_search_and_negative_diagnostics_are_distinguished():
    from tools.loop_validator import classify_tool_outcome

    assert classify_tool_outcome("[]", tool_name="web_search")["success"] is False
    news_empty = classify_tool_outcome("[]", tool_name="news_search")
    assert news_empty["success"] is True
    assert news_empty["reason"] == "empty_result"
    unreachable = '[{"target":"https://example.invalid","ok":false,"error":"timeout"}]'
    assert classify_tool_outcome(unreachable, tool_name="network_reachability")["success"] is True
    endpoint = '{"host":"example.invalid","port":443,"ok":false,"stage":"dns","error":"not found"}'
    outcome = classify_tool_outcome(endpoint, tool_name="endpoint_probe")
    assert outcome["success"] is True
    assert outcome["reason"] == "diagnostic_negative"
    assert classify_tool_outcome("No logs found.", tool_name="read_host_journal")["success"] is True
    blank_page = "URL: https://example.com\n\nThe page returned no readable text content."
    assert classify_tool_outcome(blank_page, tool_name="browse_url")["success"] is False


def test_unavailable_reminder_backend_is_not_treated_as_retryable_arguments():
    from tools.loop_validator import classify_tool_outcome

    outcome = classify_tool_outcome(
        "Error: reminder backend unavailable: Failed to connect to user scope bus"
    )
    assert outcome["success"] is False
    assert outcome["reason"] == "tool_unavailable"


def test_grounding_sensitive_structured_tools_reject_malformed_success_payloads():
    from tools.loop_validator import classify_tool_outcome

    malformed = (
        ("current_time", "not-json"),
        ("market_quote", "{not json"),
        ("weather_forecast", '{"latitude":42.9,"longitude":-81.2}'),
        ("host_snapshot", "CPU looks fine"),
        ("network_snapshot", "interfaces look fine"),
        ("repo_status", "clean"),
    )
    for tool_name, content in malformed:
        outcome = classify_tool_outcome(content, tool_name=tool_name)
        assert outcome["success"] is False, (tool_name, outcome)
        assert outcome["reason"] == "malformed_structured_result", (tool_name, outcome)

    no_location = classify_tool_outcome("[]", tool_name="geocode_location")
    assert no_location["success"] is False
    assert no_location["reason"] == "no_progress_result"


def test_valid_empty_structured_diagnostics_remain_successful():
    from tools.loop_validator import classify_tool_outcome

    assert classify_tool_outcome("[]", tool_name="neighbor_snapshot")["success"] is True
    assert classify_tool_outcome(
        '{"subnets":[],"count":0}', tool_name="local_subnets"
    )["success"] is True


def test_structured_tool_explicit_error_is_not_mislabeled_malformed():
    from tools.loop_validator import classify_tool_outcome

    outcome = classify_tool_outcome(
        "Error: news search failed: provider timeout", tool_name="news_search"
    )
    assert outcome["success"] is False
    assert outcome["reason"] == "tool_reported_error"
