import importlib.util
from pathlib import Path
import sys
import types


def _load_benchmark(monkeypatch):
    if "ollama" not in sys.modules:
        stub = types.ModuleType("ollama")
        stub.Client = object
        monkeypatch.setitem(sys.modules, "ollama", stub)
    path = Path(__file__).resolve().parents[1] / "diagnostics" / "benchmarks" / "benchmark_model_roles.py"
    spec = importlib.util.spec_from_file_location("benchmark_model_roles_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeClient:
    def chat(self, *, stream=False, messages=None, **kwargs):
        if stream:
            return iter([{"message": {"content": "socket"}}])
        return {"message": {"content": ""}, "eval_count": 8, "eval_duration": 1_000_000_000}

    def list(self):
        return {"models": []}

    def ps(self):
        return {"models": []}


def test_summary_emits_individual_samples(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    result = benchmark._summary([1.0, 2.0, 3.0])
    assert result["samples_ms"] == [1.0, 2.0, 3.0]
    assert result["median_ms"] == 2.0


def test_executor_and_decision_validator_separate_cold_from_warm(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    client = FakeClient()

    main = benchmark._main_latency(client, "executor:2b", {"num_ctx": 16384}, 3)
    validator = benchmark._fast_validator(client, "decision:0.8b", {"num_ctx": 8192}, -1, 4)

    assert main["cold"]["error"] is None
    assert main["warm"]["runs"] == 3
    assert len(main["samples_ms"]) == 3
    assert validator["cold"]["error"] is None
    assert validator["warm"]["runs"] == 4
    assert len(validator["samples_ms"]) == 4


def test_decision_contention_probe_uses_separate_background_client(monkeypatch):
    benchmark = _load_benchmark(monkeypatch)
    clients = []

    def factory(*args, **kwargs):
        client = FakeClient()
        clients.append(client)
        return client

    monkeypatch.setattr(benchmark, "Client", factory)
    result = benchmark._foreground_during_decision_prewarm(
        FakeClient(), "http://ollama", "executor:2b", {"num_ctx": 16384},
        "decision:0.8b", {"num_ctx": 8192}, -1, 0,
    )

    assert len(clients) == 1
    assert result["baseline_error"] is None
    assert result["foreground_error"] is None
    assert result["prewarm_error"] is None
    assert result["prewarm_finished"] is True
