from datetime import datetime
from zoneinfo import ZoneInfo

from tools.reminders import _calendar_expression, _slug


def test_calendar_and_slug():
    dt = datetime(2026, 9, 18, 9, 30, tzinfo=ZoneInfo("America/Toronto"))
    assert _calendar_expression(dt, "once") == "2026-09-18 09:30:00"
    assert _calendar_expression(dt, "daily") == "*-*-* 09:30:00"
    assert _slug("Hello, World!") == "hello-world"


def test_unit_files_are_safe_and_use_runtime(monkeypatch, tmp_path):
    import tools.reminders as reminders

    monkeypatch.setattr(reminders, "TIMER_DIR", tmp_path)
    service, timer = reminders._unit_files(
        "agent-reminder-demo",
        'Title "quoted" % test',
        "hello\nworld",
        "2026-09-18 09:30:00",
        "once",
    )
    service_text = service.read_text()
    timer_text = timer.read_text()
    assert service.exists() and timer.exists()
    assert "notify-send" in service_text
    assert "XDG_RUNTIME_DIR=" in service_text
    assert "quoted" in service_text
    assert "%% test" in service_text
    assert "hello world" in service_text
    assert "OnCalendar=2026-09-18 09:30:00" in timer_text
