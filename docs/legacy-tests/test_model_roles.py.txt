from pathlib import Path

from tools.config import load_config


def test_runtime_uses_explicit_decision_executor_reasoning_roles_plus_embedding_model():
    cfg = load_config()["agent"]
    assert cfg["executor_model"] == "agent-main"
    assert cfg["decision_model"] == "agent-micro"
    assert cfg["reasoning_model"] == "agent-reasoning"
    assert cfg["model"] == cfg["executor_model"]
    assert cfg["fast_model"] == cfg["executor_model"]
    assert cfg["vision_model"] == "qwen3.5:4b"
    assert cfg["report_model"] == "agent-research"
    assert cfg["embed_model"] == "nomic-embed-text"


def test_alias_helper_contains_executor_reasoning_aliases_and_decision_pull():
    text = (Path(__file__).resolve().parents[1] / "scripts" / "create_ollama_aliases.sh").read_text(encoding="utf-8")
    assert 'decision_src="qwen2.5-coder:0.5b"' in text
    assert 'executor_src="qwen2.5-coder:1.5b"' in text
    assert 'ollama cp "$decision_src" agent-micro' in text
    assert 'ollama cp "$executor_src" agent-main' in text
    assert 'ollama cp "$reasoning_src" agent-reasoning' in text
    assert 'ollama cp "$report_src" agent-research' in text
    assert 'vision_src="qwen3.5:4b"' in text
    assert "qwen3.5:9b" in text


def test_vision_role_stays_on_multimodal_4b_not_text_only_reasoning_gguf():
    from al_agent import state

    assert state.MODEL == state.EXECUTOR_MODEL == "agent-main"
    assert state.DECISION_MODEL == "agent-micro"
    assert state.REASONING_MODEL == "agent-reasoning"
    assert state.VISION_MODEL == "qwen3.5:4b"
    assert state.VISION_MODEL != state.REASONING_MODEL
    assert state.VISION_MODEL_KEEP_ALIVE == "2m"


def test_decision_context_is_small_and_executor_reasoning_contexts_stay_full():
    from al_agent import state

    assert state.DECISION_OPTIONS["num_ctx"] == 8192
    assert state.MAIN_OPTIONS["num_ctx"] == 16384
    assert state.REASONING_OPTIONS["num_ctx"] == 16384
    assert state.LOOP_VALIDATOR_OPTIONS["num_ctx"] == 8192
