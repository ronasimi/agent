from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "webui" / "static" / "app.js").read_text(encoding="utf-8")
STYLE_CSS = (ROOT / "webui" / "static" / "style.css").read_text(encoding="utf-8")


def test_streamed_assistant_unhides_empty_placeholder() -> None:
    """The streaming bubble starts empty, so it must drop attachment-only once text arrives."""
    assert ".bubble.attachment-only{display:none}" in STYLE_CSS
    assert "function paintAssistantStream()" in APP_JS
    assert "assistantNode.classList.remove('attachment-only');" in APP_JS


def test_stream_updates_are_frame_batched_and_finalized_as_markdown() -> None:
    assert "requestAnimationFrame(paintAssistantStream)" in APP_JS
    assert "shell.answer.textContent=assistantStreamBuffer;" in APP_JS
    assert "function finalizeAssistantStream(content='')" in APP_JS
    assert "shell.answer.innerHTML=renderMarkdown(parsed.text);" in APP_JS
    assert "else if(e.type==='assistant_final'){finalizeThinkingStream();finalizeAssistantStream(e.content||'');" in APP_JS


def test_turn_end_flushes_pending_stream_before_clearing_state() -> None:
    finish = APP_JS.index("function finishTurn(){")
    flush = APP_JS.index("finalizeAssistantStream();", finish)
    clear = APP_JS.index("activeTurn=null", finish)
    assert flush < clear


def test_thinking_stream_is_separate_frame_batched_and_flushed() -> None:
    assert "function appendThinking(content)" in APP_JS
    assert "requestAnimationFrame(paintThinkingStream)" in APP_JS
    assert "function finalizeThinkingStream" in APP_JS
    assert "else if(e.type==='thinking_delta')appendThinking(e.content||'');" in APP_JS
    finish = APP_JS.index("function finishTurn(){")
    think_flush = APP_JS.index("finalizeThinkingStream();", finish)
    answer_flush = APP_JS.index("finalizeAssistantStream();", finish)
    assert think_flush < answer_flush
