from pathlib import Path

from tools.config import load_config


def test_runtime_uses_explicit_generation_roles_plus_embedding_model():
    cfg = load_config()["agent"]
    assert cfg["model"] == "agent-main:4b"
    assert cfg["fast_model"] == "agent-main:2b"
    assert cfg["vision_model"] == "agent-main:4b"
    assert cfg["report_model"] == "agent-report:9b"
    assert cfg["embed_model"] == "nomic-embed-text"


def test_alias_helper_contains_only_generation_role_aliases():
    text = (Path(__file__).resolve().parents[1] / "scripts" / "create_ollama_aliases.sh").read_text(encoding="utf-8")
    assert "hf.co/empero-ai/Qwen3.8-2B-Distill-GGUF:Q8_0" in text
    assert 'ollama cp "$main_src" agent-main:4b' in text
    assert 'ollama cp "$fast_src" agent-main:2b' in text
    assert "agent-fast:2b" not in text


def test_vision_role_reuses_main_runner_by_default():
    from al_agent import state

    assert state.VISION_MODEL == state.MODEL == "agent-main:4b"
    assert state.VISION_OPTIONS["num_ctx"] == state.MAIN_OPTIONS["num_ctx"]
    assert state.VISION_MODEL_KEEP_ALIVE == -1
