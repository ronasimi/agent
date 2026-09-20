from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


_original_ollama = sys.modules.get("ollama")
fake_ollama = types.ModuleType("ollama")
fake_ollama.Client = object
sys.modules["ollama"] = fake_ollama
try:
    ROOT = Path(__file__).resolve().parents[1]
    SPEC = importlib.util.spec_from_file_location("benchmark_model_roles", ROOT / "scripts" / "benchmark_model_roles.py")
    assert SPEC and SPEC.loader
    bench = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(bench)
finally:
    if _original_ollama is None:
        sys.modules.pop("ollama", None)
    else:
        sys.modules["ollama"] = _original_ollama


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([
                {"message": {"content": "A socket is an endpoint."}},
            ])
        return {"message": {"content": '{"decision":"recover"}'}, "eval_count": 1, "eval_duration": 1_000_000_000}

    def show(self, model):
        return {"model": model}

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        return {"embeddings": [[0.1, 0.2]]}

    def ps(self):
        return {
            "models": [
                {
                    "name": "main",
                    "model": "main",
                    "context_length": 16384,
                    "size": 100,
                    "size_vram": 50,
                }
            ]
        }


def test_main_ttft_mirrors_runtime_thinking_flag():
    client = FakeClient()
    result = bench._main_ttft(client, "main", {"num_ctx": 16384}, 1, thinking_enabled=False)
    assert result["runs"] == 1
    assert result["failures"] == 0
    assert client.calls[0]["think"] is False


def test_fast_validator_disables_thinking():
    client = FakeClient()
    result = bench._fast_validator(client, "fast", {"num_ctx": 4096}, "2m", 1)
    assert result["runs"] == 1
    assert client.calls[0]["think"] is False
    assert client.calls[0]["options"]["num_ctx"] == 4096


def test_disabled_embedding_reports_install_state_without_latency_failure():
    client = FakeClient()
    result = bench._embedding_latency(client, "nomic-embed-text", 5, enabled=False)
    assert result["enabled"] is False
    assert result["installed"] is True
    assert result["failures"] == 0
    assert result["runs"] == 0


class MissingEmbeddingClient(FakeClient):
    def show(self, model):
        raise RuntimeError(f'model "{model}" not found (status code: 404)')


def test_missing_disabled_embedding_is_actionable_but_not_runtime_failure():
    result = bench._embedding_latency(MissingEmbeddingClient(), "nomic-embed-text", 5, enabled=False)
    assert result["installed"] is False
    assert result["failures"] == 0
    assert result["remediation"] == "ollama pull nomic-embed-text"


def test_ps_snapshot_records_context_and_memory_placement():
    result = bench._ps_snapshot(FakeClient())
    assert result["available"] is True
    assert result["models"][0]["context_length"] == 16384
    assert result["models"][0]["vram_percent"] == 50.0


def test_residency_benchmark_includes_true_warm_switches_and_main_only_report_restore():
    client = FakeClient()
    result = bench._load_and_swap(client, [
        ("main", "main", {"num_ctx": 16384}, -1),
        ("fast", "fast", {"num_ctx": 4096}, "2m"),
        ("report", "report", {"num_ctx": 8192}, "10m"),
    ])
    transitions = result["transition_ms"]
    assert "main->fast_cold_beside_main" in transitions
    assert "warm_fast->main" in transitions
    assert "warm_main->fast" in transitions
    assert "report->main_restore" in transitions
    assert "main->fast_lazy_after_report" in transitions
    assert "report->interactive" not in transitions
    assert "main_only" in result["residency_snapshots"]
    assert "after_report_main_restore" in result["residency_snapshots"]
