from al_agent.model_roles import should_start_with_reasoning, validator_requests_reasoning


def test_simple_direct_request_stays_on_executor():
    assert not should_start_with_reasoning("Summarize this sentence briefly.")


def test_explicit_think_uses_reasoning():
    assert should_start_with_reasoning("hello", thinking_enabled=True)


def test_structured_plan_final_synthesis_uses_reasoning():
    assert should_start_with_reasoning("summarize completed steps", scheduler_finalizing=True)


def test_complex_direct_analysis_uses_reasoning():
    request = (
        "Analyze the root cause of this runtime architecture failure and compare the trade-offs "
        "between two scheduler designs. Review the schema and algorithm, explain correctness risks, "
        "and propose a bounded recovery strategy. " * 3
    )
    assert should_start_with_reasoning(request, tool_count=0, min_complex_chars=280)


def test_tool_bearing_complex_request_starts_on_executor():
    request = (
        "Analyze this runtime architecture and debug the tool schema routing failure in detail. " * 8
    )
    assert not should_start_with_reasoning(request, tool_count=2, min_complex_chars=280)


def test_low_confidence_validator_requests_reasoning():
    report = {"decision": "retry", "diagnosis": "unknown", "confidence": "low"}
    assert validator_requests_reasoning(report)


def test_medium_confidence_wrong_tool_requests_reasoning():
    report = {"decision": "switch_tool", "diagnosis": "wrong_tool", "confidence": "medium"}
    assert validator_requests_reasoning(report)


def test_high_confidence_wrong_tool_stays_on_executor():
    report = {"decision": "switch_tool", "diagnosis": "wrong_tool", "confidence": "high"}
    assert not validator_requests_reasoning(report)


def test_blocked_validator_does_not_request_reasoning():
    report = {"decision": "blocked", "diagnosis": "tool_unavailable", "confidence": "low"}
    assert not validator_requests_reasoning(report)
