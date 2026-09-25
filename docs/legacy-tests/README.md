# Superseded architecture tests

These `.py.txt` snapshots preserve tests from the supplied baseline. They are
reference material, not collected by pytest. Tests asserting keyword-selected
schemas, no-model execution, hardcoded weather/profile/stress-test paths,
separate model roles, forced validator recipes, and sidecar vision were retired
because this refactor deliberately removes those contracts.

Independent tool, persistence, security, browser, pipeline, research and UI
regressions remain in `tests/`. Replacement runtime coverage lives in
`tests/test_autonomous_loop.py` and `tests/test_single_model_integration.py`.
No live-model behavior is inferred from scripted transport tests.
