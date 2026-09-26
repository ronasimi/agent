from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "webui" / "static" / "app.js").read_text(encoding="utf-8")
STYLE_CSS = (ROOT / "webui" / "static" / "style.css").read_text(encoding="utf-8")


def test_streamed_assistant_unhides_empty_placeholder() -> None:
    """The streaming bubble starts empty, so it must drop attachment-only once text arrives."""
    assert ".bubble.attachment-only{display:none}" in STYLE_CSS
    assert "function paintAssistantStream()" in APP_JS
    assert "assistantNode.classList.remove('attachment-only');" in APP_JS


def test_stream_updates_are_hybrid_buffered_and_finalized_as_markdown() -> None:
    assert "requestAnimationFrame(paintAssistantStream)" in APP_JS
    assert "RichOutput.streamingMarkdownSnapshot(assistantStreamBuffer)" in APP_JS
    assert "shell.answer.innerHTML=renderMarkdown(snapshot.visible);" in APP_JS
    assert "shell.answer.dataset.pendingFormat=snapshot.kind||'';" in APP_JS
    assert "function finalizeAssistantStream(content='')" in APP_JS
    assert "shell.answer.innerHTML=renderMarkdown(parsed.text);" in APP_JS
    assert "else if(e.type==='assistant_final'){finalizeThinkingStream({label:'done'});finalizeAssistantStream(e.content||'');" in APP_JS


def test_turn_end_flushes_pending_stream_before_clearing_state() -> None:
    finish = APP_JS.index("function finishTurn(){")
    flush = APP_JS.index("finalizeAssistantStream();", finish)
    clear = APP_JS.index("activeTurn=null", finish)
    assert flush < clear


def test_parser_retraction_does_not_erase_live_thinking_stream() -> None:
    assert "else if(e.type==='assistant_reset'){resetAssistantStream();}" in APP_JS
    assert "else if(e.type==='assistant_reset'){resetAssistantStream();resetThinkingStream();}" not in APP_JS


def test_thinking_stream_is_separate_frame_batched_and_flushed() -> None:
    assert "function appendThinking(content)" in APP_JS
    assert "requestAnimationFrame(paintThinkingStream)" in APP_JS
    assert "function finalizeThinkingStream" in APP_JS
    assert "else if(e.type==='thinking_delta')appendThinking(e.content||'');" in APP_JS
    finish = APP_JS.index("function finishTurn(){")
    think_flush = APP_JS.index("finalizeThinkingStream({collapse:true,label:'done'});", finish)
    answer_flush = APP_JS.index("finalizeAssistantStream();", finish)
    assert think_flush < answer_flush


def test_thinking_container_is_lazy_composer_scoped_and_disabled_turns_ignore_reasoning_deltas() -> None:
    start = APP_JS.index("function ensureAssistantComposite(){")
    end = APP_JS.index("function paintAssistantStream(){", start)
    composite = APP_JS[start:end]
    assert "createElement('details')" not in composite
    assert "assistant-thinking" not in composite

    thinking_start = APP_JS.index("function ensureThinkingStream(){")
    thinking_end = APP_JS.index("function paintThinkingStream(){", thinking_start)
    thinking = APP_JS[thinking_start:thinking_end]
    assert "if(!activeThinkingEnabled)return null;" in thinking
    assert "$('#thinkingStreamHost')" in thinking
    assert "createElement('details')" in thinking
    assert "composer-thinking" in thinking
    assert "assistantNode" not in thinking

    append_start = APP_JS.index("function appendThinking(content){")
    append_end = APP_JS.index("function finalizeThinkingStream", append_start)
    append = APP_JS[append_start:append_end]
    assert "if(!activeThinkingEnabled)return;" in append


def test_reasoning_paints_do_not_schedule_transcript_scroll_work() -> None:
    start = APP_JS.index("function ensureThinkingStream(){")
    end = APP_JS.index("function appendThinking(content){", start)
    implementation = APP_JS[start:end]
    assert "scrollBottom" not in implementation
    assert "pre.scrollTop=pre.scrollHeight" in implementation


def test_backend_foreground_loop_emits_assistant_deltas() -> None:
    loop_py = (ROOT / "al_agent" / "agent_loop.py").read_text(encoding="utf-8")
    assert "content_stream_allowed=True" in loop_py
    assert '_VisibleContentRouter' in loop_py
    assert 'self.emit("assistant_delta", content=' in loop_py
    assert 'self.emit("activity_progress", content=label)' in loop_py
    assert 'emit("assistant_reset")' in loop_py


def test_activity_progress_renders_inside_activity_group() -> None:
    assert "function addProgress(e)" in APP_JS
    assert "e.type==='activity_progress'" in APP_JS
    assert "activity-progress" in APP_JS
    assert ".activity-progress" in STYLE_CSS
