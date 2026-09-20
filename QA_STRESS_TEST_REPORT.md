# QA Stress-Test Report

Target: local AI assistant harness
Date: 2026-09-20

This review used deterministic code-path tracing and regression tests. External `ollama`
and `ddgs` modules were stubbed only in the test runner because the execution environment
does not host the user's live Ollama service. No live-model quality claims are made here.

## Phase 1 — Test Generation

### System / Command Line execution

1. **Mixed stdout/stderr + exit status + non-UTF8 output**

   > Run two read-only commands. First run `printf 'alpha\nbeta\n' | grep beta; printf 'warn\n' >&2`.
   > Then run a command that writes invalid UTF-8 bytes to stdout and stderr. For each command,
   > return stdout, stderr, and the numeric exit status without modifying files.

2. **Timeout must terminate the whole process tree**

   > Run a shell command with a one-second timeout that starts a child process which would write
   > `orphan.txt` two seconds later. After the timeout, verify that no descendant survives and the
   > file is never created.

3. **Remote current-time resolution**

   > What time is it in Tokyo? Give the current local time and timezone exactly once; do not make
   > redundant tool calls.

### File I/O and Parsing

4. **UTF-8 BOM + quoted/multiline CSV**

   > Parse a UTF-8-BOM CSV whose first column is `name`, with quoted commas and a multiline quoted
   > field. Filter for `name=Alice`, then summarize the file without changing the logical header.

5. **Tolerant JSON Lines parsing**

   > Parse `logs/events.jsonl`. Keep valid JSON records, report malformed 1-based line numbers, and
   > do not abort because one line is invalid JSON.

6. **Bounded large-file completeness**

   > Summarize a CSV larger than the harness text-read limit and explicitly tell me whether the
   > reported row count/statistics cover the complete file or only a bounded prefix.

### Context / Memory Retrieval

7. **Negative memory retrieval**

   > What deployment target did I tell you earlier? If that fact is not actually stored, say that
   > you do not know; do not infer it from unrelated memories.

8. **Meta-capability question after a live-data turn**

   Scenario:
   > User: What's the weather in London today?
   > User (next turn): What else can you do?

   The second turn must be treated as a new capability question, not as a weather continuation.

### Malformed inputs and constraints

9. **Contradictory no-tool live-fact request**

   > Without using any tools, tell me the exact current time.

   The harness must not silently violate the explicit no-tool constraint.

10. **Slash command with non-space whitespace**

    > `/job\tdeadbeef`

    Tabs and other normal whitespace separators should parse the same way as spaces, while unknown
    slash commands must remain fail-closed and never reach the model.

## Phase 2 — Harness Evaluation

| # | Trace through existing logic | Failure found |
|---|---|---|
| 1 | `select_tool_schemas` exposes `execute_shell`; `tools.system.execute_shell` used `subprocess.run(..., text=True)` and formatted only STDOUT/STDERR on success. | Successful calls omitted the exit status. Invalid UTF-8 caused `UnicodeDecodeError` and returned a generic execution error instead of usable output. |
| 2 | `execute_shell` used `subprocess.run(..., timeout=N)`. Python terminated the direct shell on timeout, but descendants launched into the same inherited environment were not explicitly killed as a process group. | Reproduced: a background child survived the one-second timeout and created `orphan_probe.txt` later. |
| 3 | `classify_request_intent` correctly selected `current_time`, but `derive_task_frame` required the place regex to end immediately at `$`, so the trailing `?` caused `Tokyo` to be discarded. `current_time()` also had no timezone override, and there was no deterministic time renderer. | Remote-place time scope was lost; simple time requests could proceed into the model/validator loop even after trusted `current_time` evidence was available. |
| 4 | `csv_query`, `csv_summary`, and JSON helpers called `_source_text` / `json.loads` without BOM normalization. | `\ufeffname` became the actual first CSV header, so `column='name'` returned no rows; BOM-prefixed JSON raised `Unexpected UTF-8 BOM`. |
| 5 | Tool selection matched generic JSON/file primitives, but there was no JSONL/NDJSON parser. `json_filter`/`json_query` expect one valid JSON document. | One malformed line makes whole-document JSON tools unusable; the model would need an ad-hoc Python workaround and could not reliably report malformed line numbers. |
| 6 | `_source_text` truncated all structured input at `MAX_TEXT=100000`; `csv_summary` parsed that prefix as if it were a complete CSV. A cutoff in the middle of a record also generated a fake missing field. | Reproduced: a 343 KB CSV was reported as 9,697 rows with no indication that most of the file had never been read. |
| 7 | `search_memory` split the entire conversational query into words and constructed OR clauses for every token. | Common words such as `my`, `you`, and `earlier` matched unrelated memories, injecting irrelevant facts into the prompt and increasing hallucination risk. |
| 8 | `is_followup_request` treated any short `what else...` form as referential. `is_task_continuation` therefore reused the previous weather frame, and `_extract_weather_entity` could turn `What else can you do` into a location-like fragment. | A capability question after weather could trigger another weather task instead of answering about assistant capabilities. |
| 9 | `derive_turn_tool_policy` recognized named/per-tool bans and read-only constraints, but not `without using any tools`. Harness-owned pre-grounding also executed read-only evidence tools without consulting the per-turn policy. | The harness could call `current_time` despite an explicit no-tools constraint, violating user intent. |
| 10 | `parse_slash_command` used `raw.partition(" ")`. | `/job\tdeadbeef` was classified as an unknown command even though the command token and argument were valid. |

Additional cross-cutting issue found during Phase 3 validation: simple current-time requests had no
mechanical fast path after successful pre-grounding, so an empty/poor main-model response could still
enter validator recovery. This was the same class of failure previously observed as repeated
`Fast validator: finish · task_complete` output.

## Phase 3 — Generalized Debugging

### 1. Subprocess execution contract

**Root cause:** `subprocess.run(text=True)` assumes decodable UTF-8 and timeout handling controls the
immediate child, not an entire spawned command tree. The result envelope also omitted return code on
successful execution.

**Refactor:** `tools/system.py` now uses a shared `Popen` byte-mode runner with a separate POSIX process
group. On timeout the entire group is killed, stdout/stderr are decoded with `errors="replace"`, and
every result carries `EXIT_CODE`, `STDOUT`, and `STDERR`. `execute_python` uses the same runner, so the
fix applies to both execution paths.

### 2. Structured-text normalization and completeness

**Root cause:** parsers independently consumed raw bounded strings and had no common BOM/completeness
contract.

**Refactor:** `tools/primitive_modules/common.py` now provides `_source_text_info()`, which normalizes a
leading BOM and returns `(text, truncated)`. JSON loading uses the same normalization. CSV primitives
surface `source_truncated`; `csv_summary` drops an uncertain cutoff record and labels statistics as
partial instead of silently presenting them as whole-file facts.

### 3. Native JSONL support

**Root cause:** JSONL is a record stream, not a single JSON document; routing it through document JSON
primitives is structurally incorrect.

**Refactor:** added `jsonl_summary`, a bounded read-only primitive that independently parses each
nonblank line, keeps valid records, reports malformed 1-based line numbers, distinguishes record-limit
truncation from source truncation, and ignores an incomplete cutoff fragment. The JSONL/NDJSON tool
bundle selects it directly.

### 4. Memory retrieval relevance

**Root cause:** keyword retrieval used OR matching over every conversational token with no stopword
filter or relevance score.

**Refactor:** durable memory search now extracts meaningful terms, ignores conversational stopwords,
requires at least one meaningful hit, weights topic matches more heavily than fact-body matches, and
scores a bounded recent candidate pool before returning results. A negative query now returns no
memories rather than unrelated context.

### 5. Continuation boundary

**Root cause:** `what else` was treated as a continuation without checking whether the user was asking
a meta/capability question.

**Refactor:** added a deterministic meta-capability classifier. Requests such as `What else can you
do?`, `What can you help with?`, and capability/tool questions start a new task epoch; genuinely
referential `What else?` can still continue a prior task.

### 6. Explicit no-tools policy

**Root cause:** the turn policy had no global no-tool form, and deterministic pre-grounding bypassed the
turn policy.

**Refactor:** explicit forms such as `do not use any tools` and `without using tools` now block the full
tool set for that turn. Harness-owned recovery calls consult the same policy and mark the corresponding
requirement blocked instead of executing behind the user's back. The policy note summarizes a global
ban rather than dumping hundreds of blocked tool names.

### 7. Slash command tokenization

**Root cause:** command parsing recognized only the literal ASCII space separator.

**Refactor:** slash parsing now uses normal whitespace splitting (`split(None, 1)`), accepting spaces,
tabs, and newlines consistently while preserving exact command-token matching and fail-closed unknown
commands.

### 8. Deterministic current-time finalization

**Root cause:** place extraction did not tolerate trailing punctuation, `current_time` had no explicit
IANA timezone parameter, and successful clock evidence still depended on another model generation.

**Refactor:** current-time task framing accepts punctuation-safe place phrases. Human place names are
resolved through `geocode_location` to the returned IANA timezone; `current_time(timezone_name=...)`
uses that zone. Once the current-time grounding gate is satisfied, a deterministic formatter returns the
clock/date/timezone directly and skips the main model and fast validator entirely.

## Phase 4 — Validation

New stress regressions: **12 passed**.

Full deterministic suite (with test-only `ollama`/`ddgs` stubs because those external packages/services
are unavailable in this runner): **306 passed, 1 skipped**. The skipped test is the opt-in live-Ollama
conformance test.

Additional checks:

- Architecture checker: passed
- Builtin tool manifest: current, 218 tools
- Python compilation: passed
- Web UI JavaScript syntax: passed
- Ollama alias helper shell syntax: passed
