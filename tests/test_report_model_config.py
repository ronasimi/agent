from tools.config import load_config


def test_dedicated_report_model_defaults_to_configured_9b_writer():
    cfg = load_config()
    agent = cfg["agent"]
    report = cfg["research"]["report"]
    assert agent["report_model"] == "agent-report:9b"
    assert agent["report_options"]["temperature"] == 0.6
    assert report["factuality"]["enabled"] is True
    assert report["factuality"]["max_repair_passes"] >= 1


def test_role_models_use_local_aliases_and_sampling_defaults():
    cfg = load_config()
    agent = cfg["agent"]
    assert agent["model"] == "agent-main:4b"
    assert agent["fast_model"] == "agent-fast:2b"
    assert agent["report_model"] == "agent-report:9b"
    for key in ("main_options", "fast_options", "report_options"):
        opts = agent[key]
        assert opts["temperature"] == 0.6
        assert opts["top_p"] == 0.95
        assert opts["top_k"] == 20
    validator = cfg["agent"]["tool_loop_validator"]["options"]
    assert agent["fast_options"]["num_ctx"] == 4096
    assert validator["num_ctx"] == 4096
    assert validator["temperature"] == 0.6
    assert validator["top_p"] == 0.95
    assert validator["top_k"] == 20
