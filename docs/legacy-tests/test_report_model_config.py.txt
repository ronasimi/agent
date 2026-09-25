from tools.config import load_config


def test_dedicated_report_model_defaults_to_configured_9b_writer():
    cfg = load_config()
    agent = cfg["agent"]
    report = cfg["research"]["report"]
    assert agent["report_model"] == "agent-research"
    assert agent["report_options"]["temperature"] == 0.6
    assert report["factuality"]["enabled"] is True
    assert report["factuality"]["max_repair_passes"] >= 1


def test_role_models_use_decision_executor_reasoning_sampling_defaults():
    cfg = load_config()
    agent = cfg["agent"]
    assert agent["executor_model"] == "agent-main"
    assert agent["decision_model"] == "agent-micro"
    assert agent["reasoning_model"] == "agent-reasoning"
    assert agent["report_model"] == "agent-research"
    assert agent["main_options"]["temperature"] == 0.3
    assert agent["decision_options"]["temperature"] == 0.0
    assert agent["reasoning_options"]["temperature"] == 0.4
    assert agent["report_options"]["temperature"] == 0.6
    assert agent["fast_options"]["temperature"] == 0.1
    assert agent["main_options"]["num_ctx"] == 16384
    assert agent["decision_options"]["num_ctx"] == 8192
    assert agent["reasoning_options"]["num_ctx"] == 16384
    validator = agent["tool_loop_validator"]["options"]
    assert validator["num_ctx"] == 8192
    assert validator["temperature"] == 0.1
