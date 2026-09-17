# ==========================================
# FILE: worker.py
# ==========================================
"""Durable background worker for research jobs and low-cost host/network monitoring."""
from __future__ import annotations

import json
import os
import socket
import time
import traceback
from pathlib import Path

from ollama import Client
from tools.config import load_config

from tools.deep_research import (
    deep_search_and_scrape,
    evaluate_research,
    plan_research_queries,
    read_research_buffer,
)
from tools.host_tools import gpu_snapshot_dict, host_snapshot, ollama_runtime_snapshot
from tools.notify import notify_desktop
from tools.pdf_generator import generate_pdf_report
from tools.runtime import (
    DB_PATH,
    cancel_job,
    claim_next_job,
    complete_job,
    fail_job,
    get_job,
    get_monitor_state,
    heartbeat_job,
    init_runtime_db,
    record_monitor_event,
    record_monitor_state,
    recover_stale_jobs,
    save_checkpoint,
    utc_now,
)

CONFIG = load_config()

AGENT_CFG = CONFIG.get("agent", {})
RESEARCH_CFG = CONFIG.get("research", {})
WORKER_CFG = CONFIG.get("worker", {})
MONITOR_CFG = CONFIG.get("host_monitor", {})
MODEL = AGENT_CFG.get("model", "qwen3.5:4b")
MAIN_OPTIONS = AGENT_CFG.get("main_options", {"num_ctx": 16384, "temperature": 0.4})
OLLAMA_HOST = AGENT_CFG.get("host", os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
os.environ["OLLAMA_HOST"] = OLLAMA_HOST
POLL_SECONDS = float(WORKER_CFG.get("poll_interval_seconds", 3))
HEARTBEAT_SECONDS = float(WORKER_CFG.get("heartbeat_seconds", 15))
STALE_SECONDS = int(WORKER_CFG.get("stale_job_seconds", 180))
INTERACTIVE_COOLDOWN = float(WORKER_CFG.get("interactive_cooldown_seconds", 10))
MIN_AVAILABLE_RAM_MB = int(WORKER_CFG.get("min_available_memory_mb", 900))
MAX_AGENT_VRAM_MB = int(WORKER_CFG.get("max_agent_vram_mb", 7200))
MONITOR_INTERVAL = float(MONITOR_CFG.get("interval_seconds", 30))


def _memory_available_mb() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().available / 1024**2)
    except Exception:
        return None


def _gpu_vram_in_use_mb() -> float | None:
    data = gpu_snapshot_dict()
    gpus = data.get("gpus", []) if isinstance(data, dict) else []
    values = [float(g.get("vram_used_mb", 0)) for g in gpus if isinstance(g, dict)]
    return sum(values) if values else None


def _ollama_vram_in_use_mb() -> float | None:
    try:
        raw = json.loads(ollama_runtime_snapshot())
        values = []
        for model in raw.get("models", []) if isinstance(raw, dict) else []:
            value = model.get("size_vram")
            if isinstance(value, (int, float)):
                values.append(float(value) / 1024**2)
        return sum(values) if values else None
    except Exception:
        return None


def resources_available() -> tuple[bool, str]:
    available = _memory_available_mb()
    if available is not None and available < MIN_AVAILABLE_RAM_MB:
        return False, f"Host available RAM is only {available} MiB (minimum {MIN_AVAILABLE_RAM_MB} MiB)."

    gpu_used = _gpu_vram_in_use_mb()
    if gpu_used is not None and gpu_used > MAX_AGENT_VRAM_MB:
        return False, f"Detected GPU VRAM usage of {gpu_used:.0f} MiB, over configured limit {MAX_AGENT_VRAM_MB} MiB."

    ollama_used = _ollama_vram_in_use_mb()
    if ollama_used is not None and ollama_used > MAX_AGENT_VRAM_MB:
        return False, f"Ollama reports {ollama_used:.0f} MiB VRAM currently resident, over configured limit {MAX_AGENT_VRAM_MB} MiB."
    return True, "resources available"


def _interactive_recent() -> bool:
    stamp = get_monitor_state("agent.last_interaction")
    if not stamp:
        return False
    try:
        from datetime import datetime, timezone
        last = datetime.fromisoformat(stamp)
        return (datetime.now(timezone.utc) - last).total_seconds() < INTERACTIVE_COOLDOWN
    except Exception:
        return False


def _notify(title: str, message: str) -> None:
    try:
        notify_desktop(title, message)
    except Exception:
        pass


def _safe_filename(topic: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", topic).strip("._")[:80] or "research"


def _synthesize(topic: str, evidence: str, job_id: str) -> str:
    client = Client(host=os.environ.get("OLLAMA_HOST", OLLAMA_HOST))
    prompt = [
        {
            "role": "system",
            "content": (
                "You are the final research synthesizer. Use ONLY the supplied evidence. "
                "Write a structured Markdown report with an executive summary, findings, caveats, "
                "and a References section. Cite source IDs like [S12] in the prose and include the source URLs in References. "
                "Do not invent sources or claims."
            ),
        },
        {"role": "user", "content": f"Target: {topic}\n\nEvidence:\n{evidence}"},
    ]
    stream = client.chat(model=MODEL, messages=prompt, options=MAIN_OPTIONS, keep_alive=-1, stream=True)
    output = []
    for chunk in stream:
        msg = chunk.get("message", {}) if isinstance(chunk, dict) else getattr(chunk, "message", {})
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if content:
            output.append(content)
    text = "".join(output).strip()
    if not text:
        raise RuntimeError("Main model returned an empty research report.")
    return text


def run_research_job(job_id: str, worker_id: str) -> str:
    """Resume a research job from its persisted state until completion."""
    job = get_job(job_id)
    if not job:
        raise RuntimeError(f"Job {job_id} was not found.")
    payload = job.get("payload", {})
    topic = str(payload.get("topic") or job.get("title") or "").strip()
    if not topic:
        raise RuntimeError("Research job has no topic.")

    state = job.get("state") or {}
    if not state:
        state = {
            "phase": "plan",
            "round": 0,
            "queries": [],
            "completed_queries": [],
            "evaluations": [],
            "report_path": None,
        }

    max_rounds = max(1, int(RESEARCH_CFG.get("max_rounds", 5)))
    max_queries = max(1, int(RESEARCH_CFG.get("max_queries_per_round", 3)))
    max_sources = max(1, int(RESEARCH_CFG.get("max_total_sources", 30)))

    while True:
        if (current := get_job(job_id)) and current.get("status") == "cancelled":
            raise RuntimeError("Research job was cancelled.")

        heartbeat_job(job_id, worker_id, state)
        save_checkpoint(job_id, state)
        phase = state.get("phase", "plan")
        round_num = int(state.get("round", 0))

        if phase == "plan":
            if round_num >= max_rounds:
                state["phase"] = "synthesize"
                continue
            queries = plan_research_queries(topic, max_queries=max_queries)
            state["queries"] = queries
            state["completed_queries"] = []
            state["phase"] = "search"
            state["round"] = round_num + 1
            save_checkpoint(job_id, state)
            continue

        if phase == "search":
            completed = set(state.get("completed_queries", []))
            for query in state.get("queries", []):
                if query in completed:
                    continue
                ok, reason = resources_available()
                if not ok:
                    heartbeat_job(job_id, worker_id, state)
                    time.sleep(min(POLL_SECONDS * 2, 10))
                    continue
                if _interactive_recent():
                    time.sleep(min(POLL_SECONDS * 2, 5))
                    continue
                deep_search_and_scrape(job_id, query, max_results=int(RESEARCH_CFG.get("max_results_per_query", 3)))
                completed.add(query)
                evidence_now = read_research_buffer(job_id, max_chars=120000)
                source_count = evidence_now.count("### Source S")
                if source_count >= max_sources:
                    state["source_cap_reached"] = True
                    state["phase"] = "evaluate"
                    heartbeat_job(job_id, worker_id, state)
                    save_checkpoint(job_id, state)
                    break
                state["completed_queries"] = sorted(completed)
                heartbeat_job(job_id, worker_id, state)
                save_checkpoint(job_id, state)
                break
            else:
                state["phase"] = "evaluate"
                save_checkpoint(job_id, state)
            continue

        if phase == "evaluate":
            evaluation = evaluate_research(job_id, topic)
            state.setdefault("evaluations", []).append(evaluation)
            status = evaluation.get("status", "insufficient")
            if status == "complete" or round_num >= max_rounds:
                state["phase"] = "synthesize"
            else:
                gaps = [str(q).strip() for q in evaluation.get("gap_queries", []) if str(q).strip()]
                if not gaps:
                    gaps = [topic]
                state["queries"] = gaps[:max_queries]
                state["completed_queries"] = []
                state["phase"] = "search"
            save_checkpoint(job_id, state)
            continue

        if phase == "synthesize":
            evidence = read_research_buffer(job_id, max_chars=int(RESEARCH_CFG.get("max_evidence_chars_for_synthesis", 45000)))
            report = _synthesize(topic, evidence, job_id)
            workspace = Path("/app/workspace/research")
            workspace.mkdir(parents=True, exist_ok=True)
            stem = f"{_safe_filename(topic)}_{job_id[:8]}"
            md_path = workspace / f"{stem}.md"
            md_path.write_text(report + "\n", encoding="utf-8")
            pdf_path = workspace / f"{stem}.pdf"
            try:
                generate_pdf_report(report, output_filename=str(pdf_path))
            except Exception as exc:
                state["pdf_error"] = str(exc)
            state["report_path"] = str(md_path)
            state["phase"] = "complete"
            save_checkpoint(job_id, state)
            complete_job(job_id, result=str(md_path))
            _notify("Deep Research Complete", f"Finished research: {topic}\n{md_path}")
            return report

        if phase == "complete":
            result = state.get("report_path") or "Research already completed."
            complete_job(job_id, result=str(result))
            return str(result)

        raise RuntimeError(f"Unknown research phase: {phase}")


def _transition_event(key: str, condition: bool, event_type: str, summary: str, details: dict) -> None:
    previous = bool(get_monitor_state(key, False))
    if condition and not previous:
        record_monitor_event(event_type, summary, details)
        _notify(summary, json.dumps(details, ensure_ascii=False)[:1200])
    record_monitor_state(key, condition)


def monitor_once() -> None:
    """Record host changes and threshold crossings without network snapshot noise."""
    if not MONITOR_CFG.get("enabled", True):
        return

    try:
        raw_host = json.loads(host_snapshot())
        previous_text = get_monitor_state("host.snapshot")
        normalized_host = json.dumps(raw_host, sort_keys=True, separators=(",", ":"))
        if previous_text is not None and previous_text != normalized_host:
            try:
                old = json.loads(previous_text)
            except Exception:
                old = {}
            old_mem = old.get("memory", {})
            new_mem = raw_host.get("memory", {})
            threshold = float(MONITOR_CFG.get("memory_change_percent", 20))
            if abs(float(new_mem.get("used_percent", 0)) - float(old_mem.get("used_percent", 0))) >= threshold:
                record_monitor_event(
                    "host_memory_change",
                    "Host memory utilization changed materially.",
                    {"before": old_mem, "after": new_mem},
                )
        record_monitor_state("host.snapshot", normalized_host)

        memory_percent = float(raw_host.get("memory", {}).get("used_percent", 0))
        disk_percent = float(raw_host.get("disk", {}).get("used_percent", 0))
        loads = raw_host.get("load_average") or [0]
        cpu_count = max(1, int(raw_host.get("cpu_count") or 1))
        load_per_core = float(loads[0]) / cpu_count
        _transition_event(
            "host.high_memory",
            memory_percent >= float(MONITOR_CFG.get("high_memory_percent", 90)),
            "host_high_memory",
            "High host memory utilization",
            {"used_percent": memory_percent},
        )
        _transition_event(
            "host.high_disk",
            disk_percent >= float(MONITOR_CFG.get("high_disk_percent", 90)),
            "host_high_disk",
            "High host disk utilization",
            {"used_percent": disk_percent},
        )
        record_monitor_state("host.load_per_core", load_per_core)

        temp_limit = float(MONITOR_CFG.get("high_temperature_celsius", 90))
        temperatures = raw_host.get("temperatures", {})
        hot = [
            {"sensor": sensor, "current": entry.get("current")}
            for sensor, entries in temperatures.items()
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("current"), (int, float)) and entry["current"] >= temp_limit
        ]
        _transition_event("host.high_temperature", bool(hot), "host_high_temperature", "High host temperature", {"sensors": hot})
    except Exception:
        pass

    try:
        ollama = json.loads(ollama_runtime_snapshot())
        error = bool(ollama.get("error")) if isinstance(ollama, dict) else True
        _transition_event(
            "ollama.unavailable",
            error,
            "ollama_unavailable",
            "Ollama is unavailable",
            {"snapshot": ollama},
        )
    except Exception:
        pass


def main() -> None:
    init_runtime_db()
    recovered = recover_stale_jobs(STALE_SECONDS)
    if recovered:
        print(f"[worker] recovered {recovered} stale job(s)")
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    last_monitor = 0.0
    last_heartbeat = 0.0
    print(f"[worker] started as {worker_id}; database={DB_PATH}")

    # Warmup / preload main model immediately into VRAM with keep_alive=-1 on worker startup
    print(f"[worker] Preloading main model ({MODEL}) into VRAM...")
    try:
        Client(host=os.environ.get("OLLAMA_HOST", OLLAMA_HOST)).chat(
            model=MODEL,
            messages=[{"role": "user", "content": "warmup"}],
            options=MAIN_OPTIONS,
            keep_alive=-1,
            think=False,
        )
        print(f"[worker] Main model successfully loaded and pinned in VRAM.")
    except Exception as exc:
        print(f"[worker] Warning - failed to preload main model: {exc}")

    while True:
        now = time.monotonic()
        if now - last_monitor >= MONITOR_INTERVAL:
            monitor_once()
            last_monitor = now

        job = None
        if now - last_heartbeat >= HEARTBEAT_SECONDS:
            last_heartbeat = now
        try:
            job = claim_next_job(worker_id, allowed_types=["research"])
            if job:
                print(f"[worker] claimed {job['id'][:8]}: {job['title']}")
                try:
                    run_research_job(job["id"], worker_id)
                except Exception as exc:
                    detail = f"{exc}\n{traceback.format_exc(limit=5)}"
                    print(f"[worker] job {job['id'][:8]} failed: {exc}")
                    retry = int(job.get("attempts", 1)) < int(job.get("max_attempts", 3))
                    fail_job(job["id"], detail, retry=retry, retry_delay_seconds=min(300, 30 * int(job.get("attempts", 1))))
                    if not retry:
                        _notify("Agent Job Failed", f"{job['title']}: {exc}")
            else:
                time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            print("[worker] stopped")
            return
        except Exception as exc:
            print(f"[worker] loop error: {exc}")
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
