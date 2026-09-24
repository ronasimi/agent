# Execution and Web Grounding Hardening — 2026-09-22

> **Historical status:** This is a point-in-time engineering/review record and is intentionally preserved as written. Model roles, tool counts, test totals, limits, and runtime behavior may have changed since this revision. For the current harness use `CURRENT_STATE.md`, `README.md`, and `ARCHITECTURE.md`.


This revision tightens three agent-loop contracts:

1. **Truncated observations must be read before summarization.** The stable system policy now explicitly requires `read_observation` whenever the harness emits a `middle truncated` warning. The bounded-result marker includes the first omitted offset, `read_observation` is automatically exposed, and candidate final answers are blocked until a successful read retrieves data from the omitted region.
2. **`browse_url` returns main content instead of site chrome.** HTML pages are parsed with BeautifulSoup, obvious navigation/advertising/recommendation containers are removed, and semantic article/main/content containers are scored by prose density and link density. Sparse pages fall back safely to cleaned body text.
3. **Completion claims require execution evidence.** The stable system policy prohibits confirming completion without a successful corresponding tool observation. The runtime additionally rejects positive completion claims when explicit tool requirements remain failed or blocked, while still allowing accurate failure/blocker reporting.

## Validation

- `451 passed, 1 skipped`
- `diagnostics/check_architecture.py`: passed
- Python compilation: passed

The validation environment does not provide the real `ollama` package, so external import-only stubs were used to permit test collection. They are not included in this repository and do not simulate inference success.
