"""Regression coverage for the 2026-09-24 optimization recommendation review.

These tests cover only changes that remain applicable to the current harness.
Existing project_snapshot()/semantic_diff()/incremental snapshot behavior is
covered by the established browser UI tests and intentionally is not replaced.
"""
from __future__ import annotations

from tools.browser_ui import (
    _BROWSER_INIT_JS,
    _find_replacement_ref,
    _history_replacement_ref,
    _token_metrics,
    BrowserSession,
)
from tools.grounding import requested_fact_types
from tools.market import extract_market_instruments, is_market_price_request


def test_ref_counter_persists_and_stable_ref_fingerprint_is_installed():
    assert "sessionStorage.getItem('__agentRefCounter')" in _BROWSER_INIT_JS
    assert "sessionStorage.setItem('__agentRefCounter'" in _BROWSER_INIT_JS
    assert "window.__agentStableRefFor" in _BROWSER_INIT_JS
    assert "window.__agentStableRefIndex" in _BROWSER_INIT_JS


def test_stable_ref_recovery_prefers_unique_semantic_match():
    old = {"ref": "e12", "stable_ref": "sabc", "role": "button", "name": "Continue"}
    current = {
        "elements": [
            {"ref": "e1042", "stable_ref": "sabc", "role": "button", "name": "Continue"},
            {"ref": "e1043", "stable_ref": "sother", "role": "button", "name": "Continue"},
        ]
    }
    assert _find_replacement_ref(old, current) == "e1042"


def test_stable_ref_recovery_refuses_ambiguous_fingerprint():
    old = {"ref": "e12", "stable_ref": "sabc", "role": "button", "name": "Continue"}
    current = {
        "elements": [
            {"ref": "e1042", "stable_ref": "sabc", "role": "button", "name": "Continue"},
            {"ref": "e1043", "stable_ref": "sabc", "role": "button", "name": "Continue"},
        ]
    }
    # Role/name is also ambiguous, so deterministic recovery must decline.
    assert _find_replacement_ref(old, current) == ""


def test_ref_history_can_reacquire_after_ephemeral_ref_changes():
    session = BrowserSession(context=None, page=None)
    session.ref_history["e9"] = "sstable"
    current = {"elements": [{"ref": "e1009", "stable_ref": "sstable"}]}
    assert _history_replacement_ref(session, "e9", current) == "e1009"


def test_observation_accounting_separates_canonical_projection_and_delta():
    full = {
        "elements": [
            {"ref": f"e{i}", "name": "button", "role": "button", "value": "x" * 20}
            for i in range(100)
        ],
        "text": "x" * 4000,
    }
    projection = {
        "elements": full["elements"][:20],
        "candidate_count_total": 100,
        "candidate_count_exposed": 20,
        "candidate_pruned": 80,
        "text": "x" * 1000,
    }
    delta = {"full": False, "changed": [{"ref": "e0", "disabled": False}]}
    metrics = _token_metrics(full, projection, delta)
    assert metrics["full_snapshot_tokens"] > metrics["projected_snapshot_tokens"] > metrics["observation_tokens"]
    assert metrics["element_count_full"] == 100
    assert metrics["element_count_projected"] == 20
    assert metrics["candidates_pruned"] == 80
    assert metrics["tokens_saved_by_projection"] > 0
    assert metrics["tokens_saved_by_delta"] > 0


def test_targeted_fact_phrase_expansion_without_implementation_false_positives():
    assert requested_fact_types("What are the conditions outside?") == {"weather"}
    assert requested_fact_types("Tell me the current clock time.") == {"current_time"}
    assert requested_fact_types("What's the weather, top news, and BTC price?") == {"weather", "news", "market_price"}
    assert requested_fact_types("Refactor the weather conditions parser and market price router") == set()
    assert requested_fact_types("What is the price of this API call?") == set()


def test_market_detection_supports_named_crypto_and_explicit_ticker_quotes():
    assert extract_market_instruments("How much is Bitcoin?") == ["bitcoin"]
    assert is_market_price_request("How much is Bitcoin?") is True
    assert extract_market_instruments("What's the trading price of AAPL?") == ["AAPL"]
    assert is_market_price_request("What's the trading price of AAPL?") is True
    # A generic category is intentionally not guessed into a specific asset.
    assert extract_market_instruments("Tell me crypto prices.") == []
