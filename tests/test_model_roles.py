from pathlib import Path

from tools.config import load_config


def test_runtime_uses_three_generation_roles_plus_embedding_model():
    cfg = load_config()["agent"]
    assert cfg["model"] == "agent-main:4b"
    assert cfg["fast_model"] == "agent-fast:2b"
    assert cfg["report_model"] == "agent-report:9b"
    assert cfg["embed_model"] == "nomic-embed-text"


def test_alias_helper_contains_only_generation_role_aliases():
    text = (Path(__file__).resolve().parents[1] / "scripts" / "create_ollama_aliases.sh").read_text(encoding="utf-8")
    assert "qwen3.5:4b" in text
    assert "qwen3.5:2b" in text
    assert "qwen3.5:9b" in text
