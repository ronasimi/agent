# Compound Stress-Prompt Parser Fix — 2026-09-22

> **Historical status:** This is a point-in-time engineering/review record and is intentionally preserved as written. Model roles, tool counts, test totals, limits, and runtime behavior may have changed since this revision. For the current harness use `CURRENT_STATE.md`, `README.md`, and `ARCHITECTURE.md`.


This revision fixes a compound-task failure exposed by the evaluation prompt that requests weather, local news, Brent crude, HTTP reachability, and a file summary.

## Root causes

1. The global no-tools policy regex interpreted the phrase `without tool evidence` as `without tools`, blocking the entire tool surface.
2. Fact-frame parsing collapsed newlines before clause extraction, destroying numbered-list boundaries and allowing adjacent tasks to contaminate weather/news/market spans and entities.
3. Natural-language HTTP reachability checks and explicit path reads were not represented as completion requirements unless the user named the underlying tools directly.
4. Fact-specific recovery prompts appended the complete compound request even when a clean `source_text` span existed, polluting deterministic weather/search recovery.

## Changes

- Global no-tools detection now excludes phrases such as `without tool evidence`, `without tool output`, and related evidence nouns while preserving explicit `without using tools`/`do not use tools` constraints.
- Fact parsing preserves line boundaries and treats numbered/bulleted task items as authoritative clause boundaries before conjunction splitting.
- Local-news extraction supports forms such as `latest 3 local London, Ontario headlines`.
- `http_probe` requirements recognize natural requests to check whether an HTTP(S) endpoint is reachable/available/up/responding.
- `read_file` requirements recognize explicit filesystem paths paired with read/summarize/inspect intent and carry the path as requirement scope.
- Fact-specific recovery uses the scoped frame's `source_text` instead of re-appending unrelated compound instructions.

## Validation

- Focused parser/policy suite: 43 passed.
- Full suite: 455 passed, 1 skipped.
- Architecture checker: passed.
- Python compilation: passed.

The validation environment did not provide the real `ollama`/`ddgs` packages; import-only external stubs were used outside the repository to allow tests to collect. They are not included in this build and do not simulate inference or web retrieval.
