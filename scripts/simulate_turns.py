#!/usr/bin/env python3
"""Offline turn simulator: exercise the harness without an Ollama server.

The Ollama transport is replaced by a scripted client whose behavior is a
callable over the real request, so the fake "model" can only call tools that
were actually supplied in that request.  That makes the traces useful for
reviewing harness behavior rather than its malformed-call path.

Two things are reported:

* a per-turn trace (events, tools executed, request count, final answer), and
* a prompt-prefix analysis, which approximates how much of each request Ollama
  can serve from its KV cache.  Chat templates render the tool schemas and the
  system prompt at the top of the prompt, so the reusable prefix ends at the
  first byte that differs from the previous request.  Anything after that is
  re-prefilled, and on a small local model that prefill dominates
  time-to-first-token for every iteration after the first.

Usage:
    python scripts/simulate_turns.py            # trace + prefix analysis
    python scripts/simulate_turns.py --prefix   # prefix analysis only
    python scripts/simulate_turns.py --prompts  # include rendered prompts
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import traceback
from collections.abc import Callable
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Never touch real durable storage.
_SANDBOX = tempfile.mkdtemp(prefix="agent-sim-")
os.environ.setdefault("AGENT_DB_PATH", os.path.join(_SANDBOX, "knowledge.db"))
os.environ.setdefault("AGENT_INFERENCE_LOCK", os.path.join(_SANDBOX, "inference.lock"))
os.environ.setdefault("AGENT_CONFIG", os.path.join(REPO, "config", "config.yaml"))

from ollama._types import ChatResponse, Message

import agent
from al_agent import events as _events
from al_agent import turn_support as _turn_support

ToolCall = Message.ToolCall


def tool_call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(function=ToolCall.Function(name=name, arguments=arguments))


class ScriptedClient:
    """Fake Ollama client driven by ``policy(turn_index, request) -> reply``."""

    def __init__(self, policy: Callable[[int, dict[str, Any]], dict[str, Any]]):
        self.policy = policy
        self.requests: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any):
        request = {
            "model": kwargs.get("model"),
            "messages": list(kwargs.get("messages") or []),
            "tools": [t.get("function", {}).get("name") for t in (kwargs.get("tools") or [])],
            "stream": bool(kwargs.get("stream")),
            "keep_alive": kwargs.get("keep_alive"),
        }
        self.requests.append(request)
        reply = self.policy(len(self.requests), request) or {}
        content = str(reply.get("content") or "")
        calls = list(reply.get("tool_calls") or [])
        thinking = str(reply.get("thinking") or "")
        if not kwargs.get("stream"):
            return ChatResponse(
                model=kwargs.get("model"), done=True,
                message=Message(role="assistant", content=content, tool_calls=calls or None),
            )
        return self._stream(kwargs.get("model"), content, calls, thinking)

    def _stream(self, model, content, calls, thinking):
        if thinking:
            yield ChatResponse(model=model, done=False, message=Message(role="assistant", thinking=thinking))
        for one in calls:
            yield ChatResponse(model=model, done=False, message=Message(role="assistant", content="", tool_calls=[one]))
        for index in range(0, len(content), 24):
            yield ChatResponse(model=model, done=False, message=Message(role="assistant", content=content[index:index + 24]))
        yield ChatResponse(
            model=model, done=True, message=Message(role="assistant", content=""),
            prompt_eval_count=900, prompt_eval_duration=120_000_000,
            eval_count=40, eval_duration=400_000_000, load_duration=5_000_000,
        )


class Run:
    def __init__(self, label: str):
        self.label = label
        self.events: list[dict] = []
        self.client: ScriptedClient | None = None
        self.error = ""

    def final(self) -> str:
        finals = [e for e in self.events if e["type"] == "assistant_final"]
        return finals[-1].get("content", "") if finals else ""


def run_turn(label: str, prompt: str, policy, *, thinking: bool = False) -> Run:
    run = Run(label)
    client = ScriptedClient(policy)
    run.client = client
    agent.OLLAMA = client
    agent.LOOP_VALIDATOR_CLIENT = client
    _turn_support.OLLAMA = client
    messages = [{"role": "system", "content": "system"}]
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                _events.frontend_event_context(sink=run.events.append):
            agent.handle_user_turn(messages, prompt, thinking)
    except Exception as exc:
        run.error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return run


def cooperative(preferred: list[str], final_text: str = "Here is the summary."):
    """Call each preferred tool once, in order, whenever it is supplied."""
    used: set[str] = set()

    def policy(index, request):
        for name in preferred:
            if name in set(request["tools"]) and name not in used:
                used.add(name)
                return {"tool_calls": [tool_call(name)]}
        return {"content": final_text}

    return policy


SCENARIOS: list[tuple[str, str, Any, dict]] = [
    ("plain-chat", "Explain in two sentences what an agent harness does.",
     lambda i, r: {"content": "It coordinates a model with typed tools."}, {}),
    ("single-tool", "What time is it right now?",
     cooperative(["current_time"], "That is the current local time."), {}),
    ("multi-requirement", "Check host memory, disk usage, network status and the git repo status.",
     cooperative(["host_snapshot", "filesystem_snapshot", "network_snapshot", "repo_status"],
                 "All four checks are complete."), {}),
    ("grounded-weather", "What is the weather in London Ontario right now?",
     cooperative(["web_search", "browse_url"], "It is sunny and 20C."), {}),
    ("ungrounded-weather", "What is the weather in London Ontario right now?",
     lambda i, r: {"content": "It is sunny and 20C."}, {}),
    ("repeated-failure", "Read the file missing.txt and tell me what is in it.",
     lambda i, r: ({"tool_calls": [tool_call("read_file", filename="missing.txt")]}
                   if "read_file" in r["tools"] else {"content": "I could not read it."}), {}),
    ("empty-responses", "Say hello.",
     lambda i, r: {"content": ""} if i < 3 else {"content": "Hello."}, {}),
    ("parallel-readonly", "List the local subnets and tell me the time.",
     lambda i, r: ({"tool_calls": [tool_call(n) for n in ("local_subnets", "current_time") if n in r["tools"]]}
                   if i == 1 else {"content": "Both done."}), {}),
    ("thinking", "Think it through: what is 17*3?",
     lambda i, r: {"thinking": "17*3 = 51", "content": "51"}, {"thinking": True}),
]


def _rendered_blocks(request: dict[str, Any]) -> list[str]:
    """Approximate the template-rendered prompt, block by block."""
    blocks = ["TOOLS:" + json.dumps(request["tools"], ensure_ascii=False)]
    for message in request["messages"]:
        if isinstance(message, dict):
            role, content = message.get("role"), str(message.get("content") or "")
            calls = json.dumps(message.get("tool_calls") or [], default=str)
        else:
            role, content, calls = getattr(message, "role", ""), str(getattr(message, "content", "") or ""), ""
        blocks.append(f"{role}:{content}{calls}")
    return blocks


def _reusable_prefix(previous: list[str], current: list[str]) -> int:
    total = 0
    for left, right in zip(previous, current):
        if left == right:
            total += len(left)
            continue
        index = 0
        while index < min(len(left), len(right)) and left[index] == right[index]:
            index += 1
        return total + index
    return total


def describe(run: Run, *, show_prompts: bool) -> None:
    print("=" * 78)
    print("RUN:", run.label)
    if run.error:
        print("!! EXCEPTION\n" + run.error)
    print("events:", " ".join(event["type"] for event in run.events))
    print("tools run:", [(e.get("name"), e.get("status")) for e in run.events if e["type"] == "tool_result"])
    print("model requests:", len(run.client.requests))
    if show_prompts:
        for index, request in enumerate(run.client.requests, 1):
            print(f"  req{index} tools={request['tools']}")
            for message in request["messages"]:
                role = message.get("role") if isinstance(message, dict) else getattr(message, "role", "?")
                content = str((message.get("content") if isinstance(message, dict) else getattr(message, "content", "")) or "")
                print(f"      [{role}] {content[:150].replace(chr(10), ' ')}")
    print("final:", json.dumps(run.final()[:200]))


def analyze(run: Run) -> None:
    print("-" * 78)
    print("PREFIX REUSE:", run.label)
    previous = None
    for index, request in enumerate(run.client.requests, 1):
        blocks = _rendered_blocks(request)
        size = sum(len(block) for block in blocks)
        if previous is None:
            print(f"  req{index}: {size:>7} chars")
        else:
            shared = _reusable_prefix(previous, blocks)
            print(f"  req{index}: {size:>7} chars | reusable {shared:>7} "
                  f"({100.0 * shared / max(size, 1):5.1f}%) | re-prefilled {size - shared:>7}")
        previous = blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", action="store_true", help="only report prompt-prefix reuse")
    parser.add_argument("--prompts", action="store_true", help="include rendered prompt messages")
    parser.add_argument("--only", default="", help="run a single scenario by name")
    args = parser.parse_args()

    runs = []
    for label, prompt, policy, options in SCENARIOS:
        if args.only and args.only != label:
            continue
        runs.append(run_turn(label, prompt, policy, **options))

    for run in runs:
        if not args.prefix:
            describe(run, show_prompts=args.prompts)
        analyze(run)

    print("=" * 78)
    failed = [run.label for run in runs if run.error]
    print("EXCEPTIONS:", ", ".join(failed) if failed else "none")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
