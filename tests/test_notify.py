from tools.notify import format_monitor_notification


def test_temperature_notification_is_human_readable():
    message = format_monitor_notification(
        "host_high_temperature",
        {
            "sensors": [
                {"sensor": "acpitz", "label": "", "current": 90.0},
                {"sensor": "thinkpad", "label": "CPU", "current": 90},
                {"sensor": "k10temp", "label": "Tctl", "current": 90.375},
            ]
        },
    )
    assert message == "acpitz: 90.0 °C\nthinkpad (CPU): 90.0 °C\nk10temp (Tctl): 90.4 °C"
    assert "{" not in message
    assert '"sensors"' not in message


def test_resource_notifications_use_labels_and_percentages():
    assert format_monitor_notification("host_high_memory", {"used_percent": 91.25}) == "Memory usage: 91.2%"
    assert format_monitor_notification("host_high_disk", {"used_percent": 95}) == "Disk usage: 95.0%"


def test_monitor_notification_fallback_does_not_dump_nested_json():
    message = format_monitor_notification("unknown", {"nested": {"secret": "value"}})
    assert message == "A monitor threshold was crossed."
