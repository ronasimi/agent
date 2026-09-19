# Harness Reliability Fixes — 2026-09-19

This pass targets the tool-routing failures visible in the supplied chat transcript and the Web UI status/copy behavior.

## Failure analysis and corrections

| Failure | Root cause | Correction |
|---|---|---|
| Weather turn claimed no web/weather capability even though web tools existed | Generic tool selection did not deterministically expose the two-step weather path | Weather intent now requires/exposes `web_search` + `browse_url`; remembered location is added to weather memory lookup and runtime guidance forbids substituting the timezone city when a stored location exists |
| `what tools are available?` caused `reload_tools` | Lexical tool matching treated registry mutation as discovery | `reload_tools` is hidden unless the request explicitly includes reload; `tool_health` is the discovery path |
| PNG was read with a text primitive and then described | Binary files were not rejected by text readers and image turns still exposed misleading file readers | `read_file`/`read_text` reject likely binary content with a targeted next-tool message; directly attached images hide text/byte readers and are explicitly marked as model-visible media |
| Weather follow-up `display the forecast` re-ran search and switched location | Short presentation follow-ups did not preserve evidence provenance; successful medium-sized observations were not persisted | Presentation follow-ups are recognized as evidence reuse, prior observation handles are loaded, and `read_observation` is preferred instead of launching a new lookup; moderately large tool results are now persisted even when they fit the immediate context |
| Search results included Bing ad/tracker redirects | Search result URLs were accepted without a basic redirect/ad filter | Web search rejects non-HTTP URLs and known Bing/Google ad-click redirect endpoints before ranking the first five results |
| Fast validator emitted repeated `retry · unknown` and consumed the tool budget | Strict `json.loads` rejected small-model preambles/fences; exception fallback always retried | Validator accepts a JSON object after fences/preamble. Stalled-step fallback is deterministic (`blocked` for repeated/tool failures, one bounded retry for model-format failure). Final-edge validator failure now finishes from collected evidence instead of initiating another blind tool call |
| Local network request produced a map but not a useful host list, then validator suggested the wrong reachability tool | `map_network` returned mostly artifact text and there was no focused read-only host-fingerprint primitive | Added `local_subnets` + `scan_subnet`; scans return structured host/service/OS/MAC metadata, while `map_network` reuses the same discovery data and includes it in the result. Public `network_reachability` is removed from the LAN intent bundle |
| Tool schemas increase prompt prefill and distract the 2B/4B models | `read_file` + `current_time` were injected into every turn and deterministic requirements were prioritized only after iteration one | Generic always-on schema set is empty; intent bundles/requirements expose tools. Explicit completion requirements are moved to the front before the first inference. Validator output budget is reduced from 192 to 96 tokens and main temperature from 0.4 to 0.2 |

## Web UI changes

- The top-left `Ready` status indicator beside **Al Agent** is removed.
- Active turn status is transient and rendered inline directly below the latest user prompt (`Sending…`, `Thinking…`, `Running <tool>…`, etc.), then removed at turn completion.
- A **Copy chat** control is beside reload. It calls `/api/history/export` and copies all stored conversation rows, including rows behind the compaction watermark.

## Network primitives

`local_subnets(include_virtual=false)` lists active private/link-local IPv4 subnets. `scan_subnet(network="", max_detail_hosts=32, top_ports=50)` discovers hosts, enriches neighbor/MAC data, performs bounded service/version probing with Nmap when available, and returns structured JSON without creating a file. `map_network(...)` is retained for requests that benefit from a PNG topology artifact and now returns the same structured host inventory alongside the artifact.

Private/link-local scanning is bounded to IPv4 networks of at most 512 addresses. Fingerprinting is bounded to 64 detailed hosts and at most 200 top ports per host.

## Validation performed

- `python -m compileall -q .`
- `node --check webui/static/app.js`
- YAML parse of `config/config.yaml`
- 63 focused reliability/unit tests passed
- 7 selected registry/tool-routing tests passed
- 5 selected Web UI tests passed
- Full `pytest` collection on this external validation host is blocked by missing runtime packages `ollama` and `ddgs`; both are already declared in `requirements.txt` and are installed by the project Docker build.
