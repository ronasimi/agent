# Bug Report Generation

The Web UI wrench menu exposes **Generate Bug Report**. It replaces the old Working State troubleshooting panel and creates one bounded Markdown artifact intended to be supplied to an LLM together with the matching repository.

## Destination

Reports are written to the repository root with a UTC date/time stamp:

```text
al-agent-bug-report-YYYYMMDD-HHMMSSZ.md
```

The pattern is excluded by `.gitignore` so local reports are not committed accidentally.

The Web UI source bind is writable only because this feature must create the report in the host checkout. Other source mounts remain read-only where they do not require repository writes.

## Contents

A report includes the information that is difficult to reconstruct from source code alone:

- high-signal triage summary and recent failure indicators
- active runtime/model configuration
- effective base system prompt
- current working state and validator history
- rolling conversation summary and compaction watermark
- recent timestamped conversation messages
- recent model-call wire traces and completion metadata
- durable jobs and monitor events
- tool/dependency health
- browser/UI benchmark state
- SQLite/storage health
- Git commit, branch, status, recent commits, diff stat, and bounded diff
- compact repository source map
- Python/platform/package/runtime information

## Safety and size bounds

Known token, password, cookie, authorization, and credential fields are redacted recursively before serialization. Bearer tokens and common secret assignments in text are also replaced.

Large values and sections are bounded so a failed process cannot create an unreasonably large troubleshooting artifact. The report intentionally may contain conversation text and system prompts because both are often required to reproduce routing and model-protocol failures.

Review a report before sharing it outside the trusted local environment.

## Non-blocking behavior

Report collection runs in a worker thread through `asyncio.to_thread`. SQLite reads, Git inspection, tool-health collection, and report serialization therefore do not block FastAPI's event loop or active chat streaming.
