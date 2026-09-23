# Multi-Fact Grounding Refactor — 2026-09-22

> **Historical status:** This is a point-in-time engineering/review record and is intentionally preserved as written. Model roles, tool counts, test totals, limits, and runtime behavior may have changed since this revision. For the current harness use `CURRENT_STATE.md`, `README.md`, and `ARCHITECTURE.md`.


## Architectural rule

A turn may have one primary conversational intent, but any number of independently scoped factual requirements.

The legacy `task_frame` remains as a compatibility projection for older routing code. New grounding, recovery, and fact-tool exposure use `fact_frames`, keyed by factual domain such as `weather`, `news`, `market_price`, and `current_time`.

## Parsing order

Compound requests are handled in this order:

1. Detect required fact types from the complete user utterance.
2. Locate fact-specific spans without blindly splitting every conjunction.
3. Extract time, location, and topic modifiers.
4. Assign modifiers to local, shared, or global scope.
5. Produce exactly one frame for every required fact type.
6. Select one primary frame only for backward-compatible `task_frame` consumers.

This prevents `weather in London and Windsor` from being mistaken for two intents while allowing `headlines and weather` to become two independent factual frames.

For example, `What are the current headlines and weather?` with a configured default location produces the equivalent of:

```python
{
    "news": {
        "intent": "news",
        "entity": "",
        "time_scope": "current",
        "source_text": "What are the current headlines",
    },
    "weather": {
        "intent": "weather",
        "entity": "London, Ontario, Canada",
        "time_scope": "current",
        "source_text": "weather?",
    },
}
```

The resulting news query is `latest news`; the weather frame no longer contaminates the news topic or location.

## Modifier inheritance

Shared leading time modifiers may apply to multiple facts:

- `today's headlines and weather` -> `today`/`today's` applies to both.
- `headlines today and weather tomorrow` -> each frame keeps its own local time scope.
- `headlines about AI and weather in London` -> `AI` remains a news topic while `London` remains a weather location.
- `weather as well as local news` -> both facts are preserved and local scope is inherited where appropriate.

Fact spans are established before modifier assignment so overlapping text cannot silently delete a secondary requirement.

## Independent grounding state

`FactGroundingLedger` tracks each fact independently. A successful observation stays satisfied even if another fact fails or requires recovery.

```python
ledger = FactGroundingLedger.from_fact_types({"weather", "news"})
ledger.apply_report({
    "required_fact_types": ["weather", "news"],
    "missing_fact_types": ["news"],
    "evidence": {"weather": ["weather_forecast"]},
})

assert ledger.requirements["weather"].satisfied
assert not ledger.requirements["news"].satisfied
```

Only unresolved fact types are recovered on later iterations. This prevents a failed news lookup from forcing a second weather lookup.

## Requirement ledger alignment

Weather is now represented in the normal `TaskRequirementLedger`. It can be satisfied by any qualifying weather evidence path, including structured provider output or the verified weather recipe. Once satisfied, the matching tool schema can be pruned before model generation.

Tool pruning now considers the complete set of active `fact_frames`, rather than the single compatibility `task_frame`. Thus a compound weather/news turn keeps both tools available until each requirement is satisfied.

## Grounding and finalization

Grounding metadata, observation validation, and deterministic recovery select the frame associated with the fact type being evaluated. Finalization requires all requested fact types to have qualifying evidence under their own scope.

A sparse frame is deliberately retained for any detected fact type that cannot be fully scoped. The invariant is:

```python
set(fact_frames) == set(required_fact_types)
```

This makes missing secondary-intent parsing visible instead of silently dropping it.

## Regression coverage

`tests/test_multi_fact_frames.py` covers:

- independent news/weather scopes;
- shared and local temporal modifiers;
- topic/location isolation;
- `as well as` conjunctions;
- same-domain entity conjunctions;
- frame completeness;
- independent grounding status transitions;
- compound requirement creation;
- fact-aware tool pruning;
- end-to-end reproduction of `What are the current headlines and weather?`.

The end-to-end regression verifies that weather and news are both grounded before model generation, the news query is `latest news` rather than a weather-derived query, and already-satisfied fact tools are pruned from the model schema.

## Validation

The refactor was validated with the project test suite and architecture checks:

- `431 passed, 1 skipped`;
- architecture checker passes with the same temporary external dependency stubs used for repository QA;
- builtin manifest remains current at `223` tools;
- Python compilation succeeds.
