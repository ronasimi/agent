from tools.config import load_config


def test_dedicated_report_model_defaults_to_configured_9b_writer():
    cfg = load_config()
    agent = cfg["agent"]
    report = cfg["research"]["report"]
    assert agent["report_model"] == "tobestyledintro/qwen3.8-9b-distill:latest"
    assert agent["report_options"]["temperature"] <= 0.2
    assert report["factuality"]["enabled"] is True
    assert report["factuality"]["max_repair_passes"] >= 1
