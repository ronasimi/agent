import tempfile
from pathlib import Path


def _use_temp_db(monkeypatch, directory: str):
    from tools import memory, runtime
    db = str(Path(directory) / "memory.db")
    monkeypatch.setattr(runtime, "DB_PATH", db)
    monkeypatch.setattr(memory, "DB_PATH", db)
    memory.init_db()
    return memory


def test_compaction_watermark_filters_reloaded_history(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        memory = _use_temp_db(monkeypatch, td)
        memory._save_message_to_db({"role": "user", "content": "old"})
        second = memory._save_message_to_db({"role": "assistant", "content": "old answer"})
        third = memory._save_message_to_db({"role": "user", "content": "recent"})
        assert memory.apply_conversation_compaction("summary", second)
        history = memory._load_chat_history_from_db(limit=20)
        assert [message["_db_id"] for message in history] == [third]
        assert memory.get_compacted_through_id() == second


def test_large_observation_can_be_read_in_slices(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        memory = _use_temp_db(monkeypatch, td)
        observation_id = memory.store_tool_observation("read_file", "abcdefghij" * 100)
        result = memory.read_observation(observation_id, offset=10, length=100)
        assert observation_id in result
        assert '"has_more": true' in result
        assert "abcdefghij" in result
