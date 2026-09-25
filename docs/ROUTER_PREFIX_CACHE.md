# Router prefix caching

The main model remains `agent-main:4b` at 32768 tokens. Tool routing uses
`qwen2.5:0.5b` at 8192 tokens, temperature zero, and at most four generated tokens.
The larger router context accommodates a compact index of all built-in tools.
Complete JSON schemas are supplied only to the main model after selection.

## Stable prefix and routing

The index contains sorted tool names, stable three-digit IDs, and short
descriptions. Every routing request starts with the exact same index for a given
catalog version. Only shortlist IDs and the current request follow that prefix.
Candidate scoring and learned confidence never reorder the prefix. The router
can select only an ID in the current shortlist; malformed, unknown, or low
confidence results fall back to ordinary discovery.

Catalog definitions are fingerprinted, including argument definitions. Changes
invalidate the prefix. Tool descriptions shrink uniformly when needed to stay
within the configured byte budget and context allowance. If the complete index
cannot fit, routing falls back to `tool_search`/`load_tools`; it never silently
drops tools. The budget is a conservative text estimate, not an exact tokenizer.
Inspect prompt counts on the installed model when adding a large custom catalog.

Each call remains stateless. Clearing turn-local request data does not clear
Ollama's prefix cache. Cached attention state is specific to a model and cannot
be shared with the main model.

## Startup and recovery

The Web UI warms the main model and real router prefix in separate idle inference
slots. Startup deferrals are retried. The router maintenance thread checks
`/api/ps` and the current registry every 30 seconds. It warms only when the prefix
has changed or the model is absent, replaced, or using a different context size.
Foreground turns take priority. Failures back off; repeated eviction is throttled.
Healthy models receive no periodic generation requests. Shutdown signals the
maintenance thread to stop.

A model may unload and reload entirely between polls. The next real routing
request will still send and repopulate the same prefix without an additional
foreground warmup call. Residency and cache reuse are best-effort: memory pressure,
server restarts, other clients, or backend behavior can invalidate cached state.

Apply `ollama.env.example` to the **host Ollama service**, then restart that service
and rebuild the agent containers. Compose cannot configure the host service.
Required residency settings are `OLLAMA_MAX_LOADED_MODELS=2` and
`OLLAMA_KEEP_ALIVE=-1`; retain `OLLAMA_NUM_PARALLEL=1` for this laptop workload.
Merge the new `agent.router` section into existing configurations, particularly
`options.num_ctx: 8192`. A retained 2048-token override cannot fit the full index.
Use `ollama ps` after warmup to confirm both models remain loaded.

## Metrics and benchmarking

Routing and warmup calls are written to the existing `model_calls.jsonl` stream
with role `system1-router` and purposes `tool_routing` / `router_prefix_warmup`.
They include request wall time, inference-lock wait, model request time, prefix
fingerprint, dynamic suffix length, and Ollama's load, prefill, generation, and
cached-token metrics. Ollama duration fields are nanoseconds; harness `*_ms`
fields are milliseconds. Missing metrics on older servers remain null. Bug
reports include current router cache status and routing traces for the selected
conversation; startup warmup traces have no conversation ID.

Run this while the agent is idle:

```bash
docker compose exec webui python diagnostics/benchmarks/benchmark_router.py --runs 20
```

The benchmark uses the same index, options, and varied requests as production. It
reports median/p95 decision time, load/prefill/decode timing, cached tokens,
selected tools, and fallbacks. It performs classification only, without executing
selected tools or training routing feedback. `--cold` explicitly unloads only
the router before the initial warmup; it leaves the main model alone.

Compare initial warmup with subsequent requests. Confirm cached tokens actually
increase on your installed Ollama backend. These measurements establish latency;
an offline mock test cannot establish real model accuracy or cache speedup.
