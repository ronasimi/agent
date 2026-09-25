from __future__ import annotations

import sqlite3

from al_agent.routing_decision import RoutingFeedbackStore, _context_key


def test_feedback_store_persists_across_instances(tmp_path):
    db = tmp_path / "routing.db"
    key = _context_key("current local time")
    first = RoutingFeedbackStore(str(db))
    before = first.scores("current_time", key)
    for _ in range(5):
        first.record("current_time", key, 1.0, event_type="task_success")

    restarted = RoutingFeedbackStore(str(db))
    after = restarted.scores("current_time", key)
    assert after[0] > before[0]
    assert after[1] > before[1]
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT attempts FROM tool_routing_stats WHERE tool_name='current_time'"
        ).fetchone()[0] == 5


def test_infrastructure_event_is_persisted_without_changing_confidence(tmp_path):
    db = tmp_path / "routing.db"
    key = _context_key("current local time")
    store = RoutingFeedbackStore(str(db))
    store.record("current_time", key, 1.0, event_type="task_success")
    before = store.scores("current_time", key)
    store.record(
        "current_time", key, None,
        event_type="infrastructure_failure", detail="model timed out",
    )
    restarted = RoutingFeedbackStore(str(db))
    after = restarted.scores("current_time", key)
    assert abs(after[0] - before[0]) < 1e-6
    assert abs(after[1] - before[1]) < 1e-6
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT outcome,event_type FROM tool_routing_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row == (None, "infrastructure_failure")
