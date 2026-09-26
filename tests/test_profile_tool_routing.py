from __future__ import annotations


def test_profile_read_tool_is_lexically_routable():
    from al_agent.deterministic_router import DeterministicToolRouter
    from tools.tool_registry import function_schema
    from tools.user_profile import get_user_profile

    schema = function_schema(get_user_profile)
    router = DeterministicToolRouter(candidate_count=8, min_candidate_score=0.18)
    decision = router.decide("Read my profile", [schema], {})
    assert decision.selected == ("get_user_profile",)
    assert decision.confidence > 0
