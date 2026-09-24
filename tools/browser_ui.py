"""Persistent, stateful Playwright environment for grounded UI interaction.

P0 established persistent browser sessions, stable semantic refs, versioned
observations, structured failures, deltas, and machine-verifiable completion.
P1 adds candidate pruning, browser-state caching, state/trajectory persistence,
separate process/outcome scoring, fused action+observation telemetry, optional
coordinate/vision fallbacks, and current-page reuse for extraction tools.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from .browser_state import BrowserStateStore
from .conversation_context import get_active_conversation_id, normalize_conversation_id
from .media import media_result
from .netutil import validate_public_url

try:  # pragma: no cover - dependency availability is environment-specific
    from playwright.async_api import (
        async_playwright,
        Browser,
        BrowserContext,
        Page,
        TimeoutError as PlaywrightTimeoutError,
    )
except ImportError:  # pragma: no cover
    async_playwright = None
    Browser = BrowserContext = Page = Any  # type: ignore
    PlaywrightTimeoutError = TimeoutError  # type: ignore

BrowserOp = Literal[
    "observe", "navigate", "click", "type", "select", "scroll", "key", "back", "verify",
    "new_tab", "list_tabs", "switch_tab", "close_tab", "wait_download"
]

_INTERACTIVE_SELECTOR = ",".join((
    "a[href]", "button", "input", "textarea", "select", "summary",
    "[role]", "[contenteditable='true']", "[tabindex]",
))

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "at", "by",
    "from", "this", "that", "it", "is", "are", "be", "as", "my", "your", "page", "webpage",
    "website", "open", "click", "type", "select", "fill", "button", "field", "form",
}

_BROWSER_INIT_JS = r"""
(() => {
  if (window.__agentInstrumentationInstalled) return;
  window.__agentInstrumentationInstalled = true;
  window.__agentUiRevision = 1;
  window.__agentMutationSeq = 0;
  window.__agentMutationLog = [];
  window.__agentRefMap = window.__agentRefMap || new WeakMap();
  window.__agentRefIndex = window.__agentRefIndex || new Map();
  // Keep ephemeral refs monotonic across same-origin hard reloads.  This does
  // not make a stale ref valid on a new document; it prevents a new element
  // from silently reusing the same eN identifier and being mistaken for it.
  let storedRefCounter = 0;
  try {
    storedRefCounter = parseInt(sessionStorage.getItem('__agentRefCounter') || '0', 10) || 0;
  } catch (_) {}
  window.__agentRefCounter = Math.max(Number(window.__agentRefCounter || 1), storedRefCounter > 0 ? storedRefCounter + 1000 : 1);
  window.__agentRefFor = (el) => {
    if (!el || el.nodeType !== 1) return '';
    let ref = window.__agentRefMap.get(el);
    if (!ref) {
      ref = `e${window.__agentRefCounter++}`;
      window.__agentRefMap.set(el, ref);
      window.__agentRefIndex.set(ref, el);
      try { sessionStorage.setItem('__agentRefCounter', String(window.__agentRefCounter)); } catch (_) {}
    }
    return ref;
  };
  // Stable refs are recovery fingerprints, not primary action identifiers.
  // Duplicate semantic controls may share a fingerprint, so Python only uses
  // them when the current snapshot contains one unambiguous match.
  const stableHash = (value) => {
    let hash = 2166136261;
    const text = String(value || '');
    for (let i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    return (hash >>> 0).toString(36);
  };
  window.__agentStableRefFor = (el, semanticName='', semanticRole='') => {
    if (!el || el.nodeType !== 1) return '';
    const identity = [
      el.tagName?.toLowerCase?.() || '',
      semanticRole || el.getAttribute('role') || '',
      String(semanticName || '').replace(/\s+/g, ' ').trim().slice(0, 120),
      el.getAttribute('name') || '',
      el.getAttribute('id') || '',
      el.getAttribute('type') || '',
      el.getAttribute('href') || '',
    ].join('|');
    return `s${stableHash(identity)}`;
  };
  window.__agentStableRefIndex = window.__agentStableRefIndex || new Map();
  const selector = %SELECTOR%;
  const bump = (kind='mutation', target=null, extra={}) => {
    window.__agentUiRevision = (window.__agentUiRevision || 0) + 1;
    window.__agentMutationSeq = (window.__agentMutationSeq || 0) + 1;
    let el = target && target.nodeType === 1 ? target : target?.parentElement;
    let interactive = null;
    try {
      interactive = el?.matches?.(selector) ? el : el?.closest?.(selector);
    } catch (_) {}
    const entry = {
      seq: window.__agentMutationSeq,
      kind,
      ref: interactive ? window.__agentRefFor(interactive) : '',
      topology: !!extra.topology,
      attr: extra.attr || '',
      added_interactive: Number(extra.added_interactive || 0),
      removed_interactive: Number(extra.removed_interactive || 0),
    };
    window.__agentMutationLog.push(entry);
    if (window.__agentMutationLog.length > 160) window.__agentMutationLog.splice(0, window.__agentMutationLog.length - 160);
  };
  const countInteractive = (nodes) => {
    let total = 0;
    for (const node of Array.from(nodes || [])) {
      if (node.nodeType !== 1) continue;
      try {
        if (node.matches(selector)) total += 1;
        total += node.querySelectorAll?.(selector)?.length || 0;
      } catch (_) {}
    }
    return total;
  };
  const observer = new MutationObserver((mutations) => {
    for (const m of mutations) {
      const added = m.type === 'childList' ? countInteractive(m.addedNodes) : 0;
      const removed = m.type === 'childList' ? countInteractive(m.removedNodes) : 0;
      bump(m.type, m.target, {
        topology: m.type === 'childList' && (added > 0 || removed > 0),
        attr: m.attributeName || '',
        added_interactive: added,
        removed_interactive: removed,
      });
    }
  });
  const install = () => {
    const root = document.documentElement || document;
    try { observer.observe(root, {subtree:true, childList:true, attributes:true, characterData:true}); } catch (_) {}
  };
  if (document.documentElement) install(); else document.addEventListener('DOMContentLoaded', install, {once:true});
  for (const eventName of ['input', 'change', 'hashchange', 'popstate']) {
    addEventListener(eventName, (event) => bump(eventName, event.target || document.documentElement), true);
  }
  let scrollTimer = null;
  addEventListener('scroll', (event) => {
    clearTimeout(scrollTimer);
    scrollTimer = setTimeout(() => bump('scroll', event.target || document.documentElement), 24);
  }, true);
  addEventListener('beforeunload', () => {
    try { sessionStorage.setItem('__agentRefCounter', String(window.__agentRefCounter || 1)); } catch (_) {}
  }, {once:true});
})();
""".replace("%SELECTOR%", json.dumps(_INTERACTIVE_SELECTOR))

# JS is kept as one expression so Playwright can evaluate it directly.
_SEMANTIC_SNAPSHOT_JS = r"""
() => {
  if (!window.__agentRefMap) {
    window.__agentRefMap = new WeakMap();
    window.__agentRefIndex = new Map();
    window.__agentRefCounter = 1;
  }
  const refFor = window.__agentRefFor || ((el) => {
    let ref = window.__agentRefMap.get(el);
    if (!ref) {
      ref = `e${window.__agentRefCounter++}`;
      window.__agentRefMap.set(el, ref);
      window.__agentRefIndex?.set(ref, el);
    }
    return ref;
  });
  const textOf = (el) => {
    const labelledBy = el.getAttribute('aria-labelledby');
    if (labelledBy) {
      const labelled = labelledBy.split(/\s+/).map(id => document.getElementById(id)?.innerText || '').join(' ').trim();
      if (labelled) return labelled;
    }
    if (el.labels && el.labels.length) {
      const labels = Array.from(el.labels).map(x => x.innerText || x.textContent || '').join(' ').trim();
      if (labels) return labels;
    }
    return (
      el.getAttribute('aria-label') ||
      el.getAttribute('alt') ||
      el.getAttribute('title') ||
      el.getAttribute('placeholder') ||
      el.innerText || el.textContent || ''
    ).replace(/\s+/g, ' ').trim();
  };
  const regionOf = (el) => {
    const landmark = el.closest('[role="dialog"],dialog,[role="navigation"],nav,main,[role="main"],aside,[role="complementary"],header,[role="banner"],footer,[role="contentinfo"],form,[role="form"]');
    if (!landmark) return 'Page';
    const role = (landmark.getAttribute('role') || '').toLowerCase();
    const tag = landmark.tagName.toLowerCase();
    if (role === 'dialog' || tag === 'dialog') return 'Dialog';
    if (role === 'navigation' || tag === 'nav') return 'Navigation';
    if (role === 'main' || tag === 'main') return 'Main';
    if (role === 'complementary' || tag === 'aside') return 'Sidebar';
    if (role === 'banner' || tag === 'header') return 'Header';
    if (role === 'contentinfo' || tag === 'footer') return 'Footer';
    if (role === 'form' || tag === 'form') return 'Form';
    return 'Page';
  };
  const viewportDistance = (rect) => {
    const dx = rect.right < 0 ? -rect.right : (rect.left > innerWidth ? rect.left - innerWidth : 0);
    const dy = rect.bottom < 0 ? -rect.bottom : (rect.top > innerHeight ? rect.top - innerHeight : 0);
    return Math.round(Math.sqrt(dx*dx + dy*dy));
  };
  const inferRole = (el) => {
    const explicit = el.getAttribute('role');
    if (explicit) return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === 'button') return 'button';
    if (tag === 'a' && el.hasAttribute('href')) return 'link';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'select') return 'combobox';
    if (tag === 'summary') return 'button';
    if (tag === 'input') {
      const type = (el.getAttribute('type') || 'text').toLowerCase();
      if (type === 'checkbox') return 'checkbox';
      if (type === 'radio') return 'radio';
      if (['button','submit','reset','image'].includes(type)) return 'button';
      if (type === 'range') return 'slider';
      return 'textbox';
    }
    return tag;
  };
  const all = Array.from(document.querySelectorAll(%SELECTOR%));
  const elements = [];
  for (const el of all) {
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    const visible = rect.width > 0 && rect.height > 0 && style.display !== 'none' && style.visibility !== 'hidden' && Number(style.opacity || 1) !== 0;
    if (!visible) continue;
    const role = inferRole(el);
    const inputType = el.tagName.toLowerCase() === 'input' ? (el.getAttribute('type') || 'text').toLowerCase() : '';
    const rawValue = ('value' in el && typeof el.value !== 'undefined') ? String(el.value ?? '') : '';
    const value = inputType === 'password' ? '[redacted]' : rawValue;
    const checked = ('checked' in el) ? Boolean(el.checked) : null;
    const disabled = Boolean(el.disabled) || el.getAttribute('aria-disabled') === 'true';
    const selected = el.getAttribute('aria-selected');
    const expanded = el.getAttribute('aria-expanded');
    const name = textOf(el).slice(0, 240);
    const ref = refFor(el);
    const stableRef = window.__agentStableRefFor?.(el, name, role) || '';
    if (stableRef) window.__agentStableRefIndex?.set(stableRef, ref);
    elements.push({
      ref,
      stable_ref: stableRef,
      tag: el.tagName.toLowerCase(),
      input_type: inputType,
      role,
      name,
      value: value.slice(0, 500),
      disabled,
      checked,
      selected: selected === null ? null : selected === 'true',
      expanded: expanded === null ? null : expanded === 'true',
      bbox: {
        x: Math.round(rect.x), y: Math.round(rect.y),
        w: Math.round(rect.width), h: Math.round(rect.height)
      },
      in_viewport: rect.bottom >= 0 && rect.right >= 0 && rect.top <= innerHeight && rect.left <= innerWidth,
      viewport_distance: viewportDistance(rect),
      region: regionOf(el)
    });
  }
  const bodyText = (document.body?.innerText || '').replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
  return {
    url: location.href,
    title: document.title || '',
    revision: Number(window.__agentUiRevision || 0),
    mutation_seq: Number(window.__agentMutationSeq || 0),
    viewport: {width: innerWidth, height: innerHeight, scroll_x: Math.round(scrollX), scroll_y: Math.round(scrollY)},
    focused_ref: document.activeElement ? refFor(document.activeElement) : '',
    elements,
    text: bodyText.slice(0, 5000)
  };
}
""".replace("%SELECTOR%", json.dumps(_INTERACTIVE_SELECTOR))

_LIGHT_STATE_JS = r"""
() => ({
  revision: Number(window.__agentUiRevision || 0),
  mutation_seq: Number(window.__agentMutationSeq || 0),
  mutation_events: (window.__agentMutationLog || []).slice(-40),
  url: location.href,
  title: document.title || '',
  viewport: {width: innerWidth, height: innerHeight, scroll_x: Math.round(scrollX), scroll_y: Math.round(scrollY)},
  ready_state: document.readyState
})
"""


_SEMANTIC_PATCH_JS = r"""
refs => {
  const textOf = (el) => (
    el.getAttribute('aria-label') || el.getAttribute('alt') || el.getAttribute('title') ||
    el.getAttribute('placeholder') || el.innerText || el.textContent || ''
  ).replace(/\s+/g, ' ').trim();
  const inferRole = (el) => {
    const explicit = el.getAttribute('role'); if (explicit) return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === 'button') return 'button';
    if (tag === 'a' && el.hasAttribute('href')) return 'link';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'select') return 'combobox';
    if (tag === 'input') {
      const type = (el.getAttribute('type') || 'text').toLowerCase();
      if (type === 'checkbox') return 'checkbox'; if (type === 'radio') return 'radio';
      if (['button','submit','reset','image'].includes(type)) return 'button'; return 'textbox';
    }
    return tag;
  };
  const regionOf = (el) => {
    const landmark = el.closest('[role="dialog"],dialog,[role="navigation"],nav,main,[role="main"],aside,[role="complementary"],header,[role="banner"],footer,[role="contentinfo"],form,[role="form"]');
    if (!landmark) return 'Page';
    const role = (landmark.getAttribute('role') || '').toLowerCase(), tag = landmark.tagName.toLowerCase();
    if (role === 'dialog' || tag === 'dialog') return 'Dialog'; if (role === 'navigation' || tag === 'nav') return 'Navigation';
    if (role === 'main' || tag === 'main') return 'Main'; if (role === 'complementary' || tag === 'aside') return 'Sidebar';
    if (role === 'banner' || tag === 'header') return 'Header'; if (role === 'contentinfo' || tag === 'footer') return 'Footer';
    if (role === 'form' || tag === 'form') return 'Form'; return 'Page';
  };
  const serialize = (ref, el) => {
    if (!el || !el.isConnected) return {ref, removed:true};
    const rect = el.getBoundingClientRect(), style = getComputedStyle(el);
    const visible = rect.width > 0 && rect.height > 0 && style.display !== 'none' && style.visibility !== 'hidden' && Number(style.opacity || 1) !== 0;
    if (!visible) return {ref, removed:true};
    const inputType = el.tagName.toLowerCase() === 'input' ? (el.getAttribute('type') || 'text').toLowerCase() : '';
    const rawValue = ('value' in el && typeof el.value !== 'undefined') ? String(el.value ?? '') : '';
    const dx = rect.right < 0 ? -rect.right : (rect.left > innerWidth ? rect.left - innerWidth : 0);
    const dy = rect.bottom < 0 ? -rect.bottom : (rect.top > innerHeight ? rect.top - innerHeight : 0);
    const role = inferRole(el), name = textOf(el).slice(0,240);
    const stableRef = window.__agentStableRefFor?.(el, name, role) || '';
    if (stableRef) window.__agentStableRefIndex?.set(stableRef, ref);
    return {
      ref, stable_ref:stableRef, tag:el.tagName.toLowerCase(), input_type:inputType, role, name,
      value:(inputType === 'password' ? '[redacted]' : rawValue).slice(0,500),
      disabled:Boolean(el.disabled) || el.getAttribute('aria-disabled') === 'true',
      checked:('checked' in el) ? Boolean(el.checked) : null,
      selected:el.getAttribute('aria-selected') === null ? null : el.getAttribute('aria-selected') === 'true',
      expanded:el.getAttribute('aria-expanded') === null ? null : el.getAttribute('aria-expanded') === 'true',
      bbox:{x:Math.round(rect.x),y:Math.round(rect.y),w:Math.round(rect.width),h:Math.round(rect.height)},
      in_viewport:rect.bottom >= 0 && rect.right >= 0 && rect.top <= innerHeight && rect.left <= innerWidth,
      viewport_distance:Math.round(Math.sqrt(dx*dx+dy*dy)), region:regionOf(el)
    };
  };
  const elements = (refs || []).map(ref => serialize(ref, window.__agentRefIndex?.get(ref)));
  const bodyText = (document.body?.innerText || '').replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
  return {elements, text:bodyText.slice(0,5000), focused_ref:document.activeElement ? (window.__agentRefFor?.(document.activeElement) || '') : ''};
}
"""


@dataclass
class BrowserSession:
    context: Any
    page: Any
    version: int = 0
    snapshot: dict[str, Any] = field(default_factory=dict)  # full canonical state
    lock: asyncio.Lock | None = None
    pending_requests: int = 0
    downloads: list[dict[str, Any]] = field(default_factory=list)
    popup_events: list[dict[str, Any]] = field(default_factory=list)
    handled_popup_events: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    incremental_updates: int = 0
    last_mutation_seq: int = 0
    step_count: int = 0
    recovery_attempts: int = 0
    session_create_ms: float = 0.0
    # Bounded ephemeral→stable history lets a stale eN ref be reacquired after
    # a same-origin reload or framework re-render without exposing stable hashes
    # as primary action refs to the model.
    ref_history: dict[str, str] = field(default_factory=dict)


class BrowserRuntime:
    """Own one persistent Chromium process and conversation-scoped contexts."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._start_lock = threading.Lock()
        self._playwright: Any = None
        self._browser: Any = None
        self._sessions: dict[str, BrowserSession] = {}

    def is_started(self) -> bool:
        return bool(self._loop and self._thread and self._thread.is_alive() and self._browser is not None)

    def _ensure_started(self) -> None:
        if self.is_started():
            return
        with self._start_lock:
            if self.is_started():
                return
            if async_playwright is None:
                raise RuntimeError("playwright package is not installed")
            self._ready.clear()
            self._thread = threading.Thread(target=self._thread_main, name="browser-runtime", daemon=True)
            self._thread.start()
            if not self._ready.wait(timeout=15):
                raise RuntimeError("browser runtime failed to start")

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._async_start())
        finally:
            self._ready.set()
        if self._browser is not None:
            loop.run_forever()

    async def _async_start(self) -> None:
        try:
            self._playwright = await async_playwright().start()
            launch_kwargs: dict[str, Any] = {
                "headless": True,
                "args": ["--no-sandbox", "--disable-setuid-sandbox"],
            }
            executable = os.getenv("AGENT_BROWSER_EXECUTABLE", "").strip()
            if not executable:
                executable = shutil.which("chromium") or shutil.which("chromium-browser") or ""
            if executable:
                launch_kwargs["executable_path"] = executable
            self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        except Exception:
            self._browser = None
            raise

    def run(self, coro, timeout: float = 35.0):
        self._ensure_started()
        if not self._loop or self._browser is None:
            raise RuntimeError("browser runtime is unavailable")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def run_if_started(self, coro, timeout: float = 10.0):
        if not self.is_started() or not self._loop:
            try:
                coro.close()
            except Exception:
                pass
            return None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    async def session(self, conversation_id: str) -> BrowserSession:
        cid = normalize_conversation_id(conversation_id)
        existing = self._sessions.get(cid)
        if existing and not existing.page.is_closed():
            return existing
        started = time.perf_counter()
        context = await self._browser.new_context(viewport={"width": 1440, "height": 1080}, accept_downloads=True)
        await context.add_init_script(_BROWSER_INIT_JS)

        async def guard_request(route, request):
            url = str(request.url or "")
            if url.startswith(("http://", "https://")):
                try:
                    validate_public_url(url)
                except Exception:
                    await route.abort()
                    return
            await route.continue_()

        # Route at context scope so popups/new tabs inherit the same SSRF guard.
        await context.route("**/*", guard_request)
        page = await context.new_page()
        session = BrowserSession(context=context, page=page, lock=asyncio.Lock())
        session.session_create_ms = (time.perf_counter() - started) * 1000.0

        def request_started(_request) -> None:
            session.pending_requests += 1

        def request_done(_request) -> None:
            session.pending_requests = max(0, session.pending_requests - 1)

        context.on("request", request_started)
        context.on("requestfinished", request_done)
        context.on("requestfailed", request_done)

        download_root = Path(os.environ.get("AGENT_WORKSPACE", "/app/workspace")) / "browser_ui" / hashlib.sha256(cid.encode("utf-8", errors="replace")).hexdigest()[:16] / "downloads"
        download_root.mkdir(parents=True, exist_ok=True)

        async def capture_download(download) -> None:
            download_id = hashlib.sha256(f"{time.time_ns()}:{getattr(download, 'url', '')}".encode()).hexdigest()[:16]
            filename = Path(str(getattr(download, "suggested_filename", "") or "download.bin")).name[:240] or "download.bin"
            target = download_root / f"{download_id}-{filename}"
            row = {
                "id": download_id,
                "filename": filename,
                "url": str(getattr(download, "url", "") or "")[:2000],
                "status": "started",
                "completed": False,
                "path": "",
                "size": 0,
            }
            session.downloads.append(row)
            session.downloads[:] = session.downloads[-50:]
            try:
                await download.save_as(str(target))
                failure = await download.failure()
                if failure:
                    row["status"] = "failed"
                    row["error"] = str(failure)[:300]
                else:
                    row["status"] = "completed"
                    row["completed"] = True
                    row["path"] = str(target)
                    try:
                        row["size"] = target.stat().st_size
                    except OSError:
                        pass
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)[:300]

        def attach_page_handlers(target_page) -> None:
            target_page.on("download", lambda download: asyncio.create_task(capture_download(download)))

        attach_page_handlers(page)

        async def register_new_page(new_page) -> None:
            attach_page_handlers(new_page)
            try:
                opener = await new_page.opener()
                opener_url = str(opener.url or "") if opener else ""
            except Exception:
                opener_url = ""
            session.popup_events.append({
                "index": max(0, len(context.pages) - 1),
                "url": str(new_page.url or "")[:2000],
                "opener_url": opener_url[:2000],
                "detected_at": time.time(),
            })
            session.popup_events[:] = session.popup_events[-50:]

        context.on("page", lambda new_page: asyncio.create_task(register_new_page(new_page)) if new_page is not page else None)
        self._sessions[cid] = session
        return session

    async def existing_session(self, conversation_id: str) -> BrowserSession | None:
        cid = normalize_conversation_id(conversation_id)
        session = self._sessions.get(cid)
        if session is None or session.page.is_closed():
            return None
        return session

    async def close_session(self, conversation_id: str) -> None:
        session = self._sessions.pop(normalize_conversation_id(conversation_id), None)
        if session:
            try:
                await session.context.close()
            except Exception:
                pass


_RUNTIME = BrowserRuntime()


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def _estimate_tokens(value: Any) -> int:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    # Fast deterministic approximation suitable for per-step instrumentation.
    return max(1, (len(text) + 3) // 4)


def _element_map(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("ref")): row
        for row in snapshot.get("elements", [])
        if isinstance(row, dict) and row.get("ref")
    }


def _remember_ref_history(session: BrowserSession, snapshot: dict[str, Any], *, limit: int = 2048) -> None:
    """Retain a bounded mapping from ephemeral refs to semantic fingerprints."""
    for row in snapshot.get("elements", []) if isinstance(snapshot, dict) else []:
        if not isinstance(row, dict):
            continue
        ref = str(row.get("ref") or "")
        stable = str(row.get("stable_ref") or "")
        if ref and stable:
            session.ref_history[ref] = stable
    overflow = len(session.ref_history) - max(128, int(limit))
    if overflow > 0:
        # dict preserves insertion order; discard oldest history first.
        for key in list(session.ref_history)[:overflow]:
            session.ref_history.pop(key, None)


def _stable_replacement_ref(old_row: dict[str, Any], current: dict[str, Any]) -> str:
    stable = str(old_row.get("stable_ref") or "")
    if not stable:
        return ""
    matches = [
        str(row.get("ref") or "")
        for row in current.get("elements", [])
        if isinstance(row, dict) and str(row.get("stable_ref") or "") == stable and row.get("ref")
    ]
    return matches[0] if len(matches) == 1 else ""


def _history_replacement_ref(session: BrowserSession, ref: str, current: dict[str, Any]) -> str:
    stable = str(session.ref_history.get(str(ref or ""), ""))
    return _stable_replacement_ref({"stable_ref": stable}, current) if stable else ""


def _task_tokens(task_hint: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]{2,}", str(task_hint or "").lower())
        if token not in _STOPWORDS
    }


def _element_relevance(row: dict[str, Any], task_tokens: set[str], pinned_refs: set[str]) -> float:
    ref = str(row.get("ref") or "")
    if ref in pinned_refs:
        return 1000.0
    score = 0.0
    if row.get("in_viewport"):
        score += 10.0
    else:
        distance = max(0, int(row.get("viewport_distance") or 0))
        if distance <= 900:
            score += 5.0
        elif distance <= 1800:
            score += 2.0
    role = str(row.get("role") or "").lower()
    if role in {"button", "link", "textbox", "checkbox", "radio", "combobox", "menuitem", "tab", "option"}:
        score += 4.0
    if role in {"alert", "status", "dialog"}:
        score += 7.0
    if not bool(row.get("disabled")):
        score += 1.0
    haystack = " ".join((str(row.get("name") or ""), str(row.get("value") or ""), role)).lower()
    overlap = sum(1 for token in task_tokens if token in haystack)
    score += overlap * 6.0
    # Prefer compact named controls over anonymous layout roles.
    if row.get("name"):
        score += 1.0
    return score


def project_snapshot(
    snapshot: dict[str, Any],
    *,
    task_hint: str = "",
    max_candidates: int = 60,
    include_offscreen: bool = False,
    pinned_refs: set[str] | None = None,
) -> dict[str, Any]:
    """Return a bounded model-facing semantic projection of canonical browser state."""
    max_candidates = max(8, min(int(max_candidates or 60), 200))
    pinned = {str(item) for item in (pinned_refs or set()) if str(item)}
    tokens = _task_tokens(task_hint)
    rows = [dict(row) for row in snapshot.get("elements", []) if isinstance(row, dict)]
    if not include_offscreen:
        visible = [row for row in rows if row.get("in_viewport")]
        # Keep nearby off-screen controls so a small scroll does not force an
        # observation expansion round-trip. More distant controls still need
        # task relevance to survive pruning.
        nearby = [
            row for row in rows
            if not row.get("in_viewport") and 0 <= int(row.get("viewport_distance") or 0) <= 900
        ]
        offscreen_relevant = [
            row for row in rows
            if not row.get("in_viewport")
            and row not in nearby
            and _element_relevance(row, tokens, pinned) >= 10
        ]
        rows = visible + nearby + offscreen_relevant
    rows.sort(key=lambda row: (_element_relevance(row, tokens, pinned), bool(row.get("in_viewport"))), reverse=True)
    exposed = rows[:max_candidates]
    region_counts: dict[str, int] = {}
    region_refs: dict[str, list[str]] = {}
    for row in exposed:
        region = str(row.get("region") or "Page")
        region_counts[region] = region_counts.get(region, 0) + 1
        region_refs.setdefault(region, []).append(str(row.get("ref") or ""))
    regions = [
        {"name": name, "count": region_counts[name], "refs": region_refs[name][:24]}
        for name in ("Dialog", "Navigation", "Header", "Main", "Form", "Sidebar", "Footer", "Page")
        if name in region_counts
    ]
    projected = {
        "url": snapshot.get("url", ""),
        "title": snapshot.get("title", ""),
        "viewport": dict(snapshot.get("viewport") or {}),
        "focused_ref": snapshot.get("focused_ref", ""),
        "auth": dict(snapshot.get("auth") or {}),
        "regions": regions,
        "elements": exposed,
        "text": str(snapshot.get("text") or "")[:2200],
        "tabs": list(snapshot.get("tabs") or [])[:12],
        "popups": list(snapshot.get("popups") or [])[-12:],
        "downloads": list(snapshot.get("downloads") or [])[-12:],
        "candidate_count_total": len(snapshot.get("elements") or []),
        "candidate_count_exposed": len(exposed),
        "candidate_pruned": max(0, len(snapshot.get("elements") or []) - len(exposed)),
    }
    return projected


def semantic_diff(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Return a compact ref-keyed UI delta suitable for model context."""
    if not previous:
        return {"full": True, **current}
    old = _element_map(previous)
    new = _element_map(current)
    added = [new[ref] for ref in new.keys() - old.keys()]
    removed = sorted(old.keys() - new.keys())
    changed: list[dict[str, Any]] = []
    compare_fields = (
        "role", "name", "value", "disabled", "checked", "selected", "expanded", "bbox",
        "in_viewport", "viewport_distance", "region", "input_type"
    )
    for ref in new.keys() & old.keys():
        delta: dict[str, Any] = {"ref": ref}
        for field in compare_fields:
            if old[ref].get(field) != new[ref].get(field):
                delta[field] = new[ref].get(field)
        if len(delta) > 1:
            changed.append(delta)
    page_changed = {
        key: current.get(key)
        for key in ("url", "title", "viewport", "focused_ref", "auth", "regions", "tabs", "popups", "downloads")
        if previous.get(key) != current.get(key)
    }
    text_changed = previous.get("text") != current.get("text")
    churn = len(added) + len(removed) + len(changed)
    if churn > max(80, int(max(len(old), len(new), 1) * 0.65)):
        return {"full": True, **current}
    return {
        "full": False,
        **page_changed,
        "candidate_count_total": current.get("candidate_count_total", len(new)),
        "candidate_count_exposed": current.get("candidate_count_exposed", len(new)),
        "candidate_pruned": current.get("candidate_pruned", 0),
        "added": added,
        "removed": removed,
        "changed": changed,
        "text_changed": text_changed,
        "text": current.get("text", "")[:1800] if text_changed else "",
    }


async def _light_state(session: BrowserSession) -> dict[str, Any]:
    try:
        data = await session.page.evaluate(_LIGHT_STATE_JS)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


async def _tab_state(session: BrowserSession) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, page in enumerate(session.context.pages[:12]):
        if page.is_closed():
            continue
        try:
            title = await page.title()
        except Exception:
            title = ""
        rows.append({
            "index": index,
            "url": str(page.url or "")[:2000],
            "title": str(title or "")[:500],
            "active": page is session.page,
        })
    return rows


def _detect_auth_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Classify authentication state conservatively from semantic page evidence."""
    url = str(snapshot.get("url") or "").lower()
    text = str(snapshot.get("text") or "").lower()
    elements = [row for row in snapshot.get("elements", []) if isinstance(row, dict)]
    names = " ".join(str(row.get("name") or "").lower() for row in elements)
    has_password = any(str(row.get("input_type") or "").lower() == "password" for row in elements)
    mfa_terms = ("verification code", "security code", "one-time code", "one time code", "two-factor", "2fa", "authenticator code")
    expired_terms = ("session expired", "sign in again", "log in again", "login again")
    login_terms = ("sign in", "log in", "login", "continue with google", "continue with microsoft")
    signed_terms = ("sign out", "log out", "logout", "my account", "account settings")
    evidence: list[str] = []
    state = "unknown"
    if any(term in text or term in names for term in mfa_terms):
        state = "mfa_required"; evidence.append("mfa_prompt")
    elif any(term in text for term in expired_terms):
        state = "session_expired"; evidence.append("session_expired_text")
    elif has_password or any(token in url for token in ("/login", "/signin", "/sign-in")):
        state = "login_required"; evidence.append("password_or_login_route")
    elif any(term in names for term in login_terms) and not any(term in names for term in signed_terms):
        state = "login_required"; evidence.append("login_control")
    elif any(term in names or term in text for term in signed_terms):
        state = "signed_in"; evidence.append("account_control")
    return {"state": state, "evidence": evidence[:4]}


def _mutation_events_since(light: dict[str, Any], seq: int) -> list[dict[str, Any]]:
    rows = []
    for row in light.get("mutation_events", []) if isinstance(light, dict) else []:
        if not isinstance(row, dict):
            continue
        try:
            if int(row.get("seq") or 0) > int(seq or 0):
                rows.append(row)
        except (TypeError, ValueError):
            continue
    return rows


async def _incremental_snapshot(session: BrowserSession, light: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Patch cached semantic rows for non-topology mutations without a full DOM scan."""
    if not session.snapshot or not events:
        return None
    if any(bool(row.get("topology")) for row in events):
        return None
    refs = sorted({str(row.get("ref") or "") for row in events if str(row.get("ref") or "")})
    if not refs or len(refs) > 24:
        return None
    if str(light.get("url") or "") != str(session.snapshot.get("url") or ""):
        return None
    if dict(light.get("viewport") or {}) != dict(session.snapshot.get("viewport") or {}):
        return None
    try:
        patch = await session.page.evaluate(_SEMANTIC_PATCH_JS, refs)
    except Exception:
        return None
    if not isinstance(patch, dict):
        return None
    current = dict(session.snapshot)
    mapping = _element_map(current)
    for row in patch.get("elements", []):
        if not isinstance(row, dict):
            continue
        ref = str(row.get("ref") or "")
        if not ref:
            continue
        if row.get("removed"):
            mapping.pop(ref, None)
        else:
            mapping[ref] = row
    current["elements"] = list(mapping.values())
    current["text"] = str(patch.get("text") or current.get("text") or "")[:5000]
    current["focused_ref"] = str(patch.get("focused_ref") or "")
    current["revision"] = int(light.get("revision") or current.get("revision") or 0)
    current["mutation_seq"] = int(light.get("mutation_seq") or current.get("mutation_seq") or 0)
    current["title"] = str(light.get("title") or current.get("title") or "")
    current["tabs"] = await _tab_state(session)
    current["popups"] = [dict(row) for row in session.popup_events[-50:]]
    current["downloads"] = [dict(row) for row in session.downloads[-50:]]
    current["auth"] = _detect_auth_state(current)
    session.incremental_updates += 1
    return current


async def _snapshot(session: BrowserSession, *, force: bool = False) -> dict[str, Any]:
    """Read canonical state using mutation-driven incremental patches when safe."""
    light = await _light_state(session)
    mutation_seq = int(light.get("mutation_seq") or 0) if light else 0
    if not force and session.snapshot and light:
        old = session.snapshot
        same_revision = int(light.get("revision") or 0) == int(old.get("revision") or -1)
        same_location = str(light.get("url") or "") == str(old.get("url") or "")
        same_viewport = dict(light.get("viewport") or {}) == dict(old.get("viewport") or {})
        if same_revision and same_location and same_viewport:
            session.cache_hits += 1
            cached = dict(session.snapshot)
            cached["tabs"] = await _tab_state(session)
            cached["popups"] = [dict(row) for row in session.popup_events[-50:]]
            cached["downloads"] = [dict(row) for row in session.downloads[-50:]]
            cached["auth"] = _detect_auth_state(cached)
            session.last_mutation_seq = max(session.last_mutation_seq, mutation_seq)
            _remember_ref_history(session, cached)
            return cached
        events = _mutation_events_since(light, session.last_mutation_seq)
        patched = await _incremental_snapshot(session, light, events)
        if patched is not None:
            session.cache_hits += 1
            session.last_mutation_seq = max(session.last_mutation_seq, mutation_seq)
            _remember_ref_history(session, patched)
            return patched
    session.cache_misses += 1
    try:
        data = await session.page.evaluate(_SEMANTIC_SNAPSHOT_JS)
    except Exception:
        await session.page.wait_for_load_state("domcontentloaded", timeout=5000)
        data = await session.page.evaluate(_SEMANTIC_SNAPSHOT_JS)
    result = data if isinstance(data, dict) else {}
    result["tabs"] = await _tab_state(session)
    result["popups"] = [dict(row) for row in session.popup_events[-50:]]
    result["downloads"] = [dict(row) for row in session.downloads[-50:]]
    result["auth"] = _detect_auth_state(result)
    session.last_mutation_seq = max(session.last_mutation_seq, int(result.get("mutation_seq") or mutation_seq or 0))
    _remember_ref_history(session, result)
    return result


async def _wait_for_ui_stability(
    session: BrowserSession,
    *,
    timeout_ms: int = 5000,
    quiet_ms: int = 160,
) -> dict[str, Any]:
    """Wait for revision/network quiescence without an unconditional fixed sleep."""
    timeout_ms = min(max(int(timeout_ms), 250), 5000)
    quiet_ms = min(max(int(quiet_ms), 80), 800)
    started = time.perf_counter()
    last_signature: tuple[Any, ...] | None = None
    quiet_since: float | None = None
    samples = 0
    while (time.perf_counter() - started) * 1000.0 < timeout_ms:
        samples += 1
        light = await _light_state(session)
        signature = (
            light.get("revision"), light.get("url"),
            (light.get("viewport") or {}).get("scroll_x"),
            (light.get("viewport") or {}).get("scroll_y"),
            light.get("ready_state"), session.pending_requests,
        )
        now = time.perf_counter()
        if signature == last_signature:
            quiet_since = quiet_since or now
        else:
            last_signature = signature
            quiet_since = now
        pending_ok = session.pending_requests == 0 or (now - started) >= 0.7
        ready_ok = str(light.get("ready_state") or "") in {"interactive", "complete", ""}
        if quiet_since is not None and (now - quiet_since) * 1000.0 >= quiet_ms and pending_ok and ready_ok:
            return {"settled": True, "samples": samples, "pending_requests": session.pending_requests}
        await asyncio.sleep(0.04)
    return {"settled": False, "samples": samples, "pending_requests": session.pending_requests, "timed_out": True}


async def _find_ref(session: BrowserSession, ref: str):
    handle = await session.page.evaluate_handle(
        """ref => {
          const indexed = window.__agentRefIndex?.get(ref);
          if (indexed && indexed.isConnected) return indexed;
          if (!window.__agentRefMap) return null;
          for (const el of document.querySelectorAll('*')) {
            if (window.__agentRefMap.get(el) === ref) { window.__agentRefIndex?.set(ref, el); return el; }
          }
          return null;
        }""",
        ref,
    )
    element = handle.as_element()
    if element is None:
        await handle.dispose()
    return element


def _error(code: str, message: str, *, retryable: bool = True, target: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": str(message)[:600], "retryable": bool(retryable)}
    if target:
        payload["target"] = target
    return payload


def _classify_exception(exc: BaseException, *, target: str = "") -> dict[str, Any]:
    text = str(exc)
    lower = text.lower()
    if isinstance(exc, PlaywrightTimeoutError) or "timeout" in lower:
        return _error("NAVIGATION_TIMEOUT", text, target=target)
    if "not visible" in lower or "visible" in lower and "waiting" in lower:
        return _error("ELEMENT_NOT_VISIBLE", text, target=target)
    if "not enabled" in lower or "disabled" in lower:
        return _error("ELEMENT_DISABLED", text, target=target)
    if "intercepts pointer events" in lower or "another element" in lower:
        return _error("ELEMENT_OBSCURED", text, target=target)
    if "closed" in lower and ("page" in lower or "context" in lower):
        return _error("PAGE_CLOSED", text, retryable=False, target=target)
    return _error("BROWSER_ACTION_FAILED", text, target=target)


async def _resolve_ref(session: BrowserSession, ref: str):
    if not ref:
        return None, _error("ELEMENT_NOT_FOUND", "A semantic element ref is required for this operation.", target=ref)
    element = await _find_ref(session, ref)
    if element is None:
        return None, _error("ELEMENT_NOT_FOUND", f"Element ref {ref!r} is no longer present.", target=ref)
    return element, None


_CONSEQUENTIAL_TERMS = (
    "buy", "purchase", "pay", "place order", "submit order", "confirm order", "checkout",
    "submit", "confirm", "send", "publish", "post", "delete", "remove account", "close account", "transfer",
    "cancel subscription", "confirm payment", "book now", "reserve now", "apply now",
)


def classify_action_safety(op: str, snapshot: dict[str, Any], *, ref: str = "", value: str = "") -> dict[str, Any]:
    """Classify UI operations without relying on model-provided risk labels."""
    op = str(op or "").lower()
    row = _element_map(snapshot).get(str(ref or ""), {})
    name = " ".join((str(row.get("name") or ""), str(row.get("value") or ""))).strip().lower()
    input_type = str(row.get("input_type") or "").lower()
    if op in {"observe", "verify", "list_tabs", "wait_download", "navigate", "back", "scroll", "switch_tab"}:
        level = "read_only"
    elif op in {"type", "select", "new_tab", "close_tab"}:
        level = "reversible"
    elif op == "key" and str(value or "").lower() not in {"enter", "numpadenter"}:
        level = "reversible"
    elif op == "click":
        consequential = any(term in name for term in _CONSEQUENTIAL_TERMS)
        consequential = consequential or input_type == "submit" and any(term in name for term in ("pay", "order", "send", "delete", "submit"))
        level = "consequential" if consequential else "reversible"
    elif op == "key" and str(value or "").lower() in {"enter", "numpadenter"}:
        focused = str(snapshot.get("focused_ref") or "")
        focused_row = _element_map(snapshot).get(focused, {})
        focused_name = str(focused_row.get("name") or "").lower()
        level = "consequential" if any(term in focused_name for term in _CONSEQUENTIAL_TERMS) else "reversible"
    else:
        level = "reversible"
    return {
        "level": level,
        "target_ref": str(ref or ""),
        "target_name": str(row.get("name") or "")[:240],
        "requires_pre_submit_verification": level == "consequential",
    }


def _same_target(before: dict[str, Any], after: dict[str, Any], ref: str) -> bool:
    if not ref:
        return True
    left = _element_map(before).get(ref)
    if not left:
        return False
    right = _element_map(after).get(ref)
    if right is None:
        replacement = _stable_replacement_ref(left, after)
        right = _element_map(after).get(replacement) if replacement else None
    if not right:
        return False
    fields = ("role", "name", "input_type", "disabled")
    return all(left.get(field) == right.get(field) for field in fields)


def _find_replacement_ref(old_row: dict[str, Any], current: dict[str, Any]) -> str:
    # A stable semantic fingerprint is stronger than a plain role/name match,
    # but only when it identifies exactly one current element.
    stable = _stable_replacement_ref(old_row, current)
    if stable:
        return stable
    role = str(old_row.get("role") or "")
    name = str(old_row.get("name") or "").strip().lower()
    candidates = [
        row for row in current.get("elements", [])
        if isinstance(row, dict)
        and (not role or str(row.get("role") or "") == role)
        and (not name or str(row.get("name") or "").strip().lower() == name)
    ]
    return str(candidates[0].get("ref") or "") if len(candidates) == 1 else ""


async def _wait_for_download(session: BrowserSession, needle: str = "", timeout_ms: int = 10000) -> dict[str, Any] | None:
    deadline = time.perf_counter() + max(0.25, min(int(timeout_ms), 15000) / 1000.0)
    needle = str(needle or "").lower()
    while time.perf_counter() < deadline:
        rows = [row for row in session.downloads if not needle or needle in str(row.get("filename") or "").lower()]
        completed = next((row for row in reversed(rows) if row.get("status") == "completed" or row.get("completed")), None)
        if completed:
            return dict(completed)
        failed = next((row for row in reversed(rows) if row.get("status") == "failed"), None)
        if failed:
            return dict(failed)
        await asyncio.sleep(0.05)
    return None


async def _execute_action(
    session: BrowserSession,
    op: str,
    *,
    url: str,
    ref: str,
    value: str,
    direction: str,
    amount: int,
    x: int,
    y: int,
    tab_index: int,
    timeout_ms: int,
) -> dict[str, Any] | None:
    page = session.page
    if op in {"observe", "list_tabs"}:
        return None
    if op == "new_tab":
        new_page = await session.context.new_page()
        session.page = new_page
        return None
    if op == "switch_tab":
        pages = [candidate for candidate in session.context.pages if not candidate.is_closed()]
        if not pages:
            return _error("TAB_NOT_FOUND", "No open browser tabs are available.", retryable=False)
        index = int(tab_index)
        if index < 0 or index >= len(pages):
            return _error("TAB_NOT_FOUND", f"Tab index {index} is outside 0..{len(pages)-1}.", retryable=True)
        session.page = pages[index]
        await session.page.bring_to_front()
        return None
    if op == "close_tab":
        pages = [candidate for candidate in session.context.pages if not candidate.is_closed()]
        if not pages:
            return _error("TAB_NOT_FOUND", "No open browser tabs are available.", retryable=False)
        index = int(tab_index) if int(tab_index) >= 0 else pages.index(session.page)
        if index < 0 or index >= len(pages):
            return _error("TAB_NOT_FOUND", f"Tab index {index} is outside 0..{len(pages)-1}.", retryable=True)
        closing = pages[index]
        await closing.close()
        remaining = [candidate for candidate in session.context.pages if not candidate.is_closed()]
        if not remaining:
            remaining = [await session.context.new_page()]
        session.page = remaining[min(index, len(remaining)-1)]
        await session.page.bring_to_front()
        return None
    if op == "wait_download":
        row = await _wait_for_download(session, value, timeout_ms)
        if row is None:
            return _error("DOWNLOAD_TIMEOUT", f"No completed download matching {value!r} appeared before timeout.", retryable=True)
        if row.get("status") == "failed":
            return _error("DOWNLOAD_FAILED", str(row.get("error") or "Download failed."), retryable=True)
        return None
    if op == "navigate":
        safe_url = validate_public_url(str(url).strip())
        await page.goto(safe_url, wait_until="domcontentloaded", timeout=max(1000, min(int(timeout_ms), 15000)))
        return None
    if op == "back":
        await page.go_back(wait_until="domcontentloaded", timeout=max(1000, min(int(timeout_ms), 15000)))
        return None
    if op == "scroll":
        direction = str(direction or "down").lower()
        pixels = max(100, min(abs(int(amount or 700)), 5000))
        dx, dy = 0, pixels
        if direction == "up": dy = -pixels
        elif direction == "left": dx, dy = -pixels, 0
        elif direction == "right": dx, dy = pixels, 0
        await page.evaluate("([x,y]) => window.scrollBy(x,y)", [dx, dy])
        return None
    if op == "key":
        key = str(value or "").strip()
        if not key:
            return _error("INVALID_ARGUMENT", "key requires value, for example Enter or Escape.", retryable=False)
        await page.keyboard.press(key)
        return None

    # Ref grounding is preferred. Coordinate click exists only as a vision/layout
    # fallback for controls that cannot be represented semantically.
    if op == "click" and not ref and x >= 0 and y >= 0:
        viewport = session.snapshot.get("viewport") or {}
        width = int(viewport.get("width") or 1440)
        height = int(viewport.get("height") or 1080)
        if x >= width or y >= height:
            return _error("INVALID_COORDINATE", f"Coordinate ({x},{y}) is outside the {width}x{height} viewport.", retryable=True)
        await page.mouse.click(int(x), int(y))
        return None

    element, err = await _resolve_ref(session, ref)
    if err:
        return err
    try:
        disabled = await element.evaluate("el => Boolean(el.disabled) || el.getAttribute('aria-disabled') === 'true'")
        if disabled and op in {"click", "type", "select"}:
            return _error("ELEMENT_DISABLED", f"Element {ref} is disabled.", target=ref)
        if not await element.is_visible():
            return _error("ELEMENT_NOT_VISIBLE", f"Element {ref} is not visible.", target=ref)
        if op == "click":
            await element.click(timeout=5000)
        elif op == "type":
            await element.fill(str(value), timeout=5000)
        elif op == "select":
            await element.select_option(str(value), timeout=5000)
        else:
            return _error("INVALID_ARGUMENT", f"Unsupported browser operation: {op}", retryable=False)
    finally:
        try:
            await element.dispose()
        except Exception:
            pass
    return None


async def _execute_action_with_recovery(
    session: BrowserSession,
    op: str,
    *,
    before: dict[str, Any],
    url: str,
    ref: str,
    value: str,
    direction: str,
    amount: int,
    x: int,
    y: int,
    tab_index: int,
    timeout_ms: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Apply bounded deterministic recoveries before asking the model to re-plan."""
    recoveries: list[dict[str, Any]] = []
    pages_before = [page for page in session.context.pages if not page.is_closed()]
    downloads_before = len(session.downloads)
    current_ref = ref

    async def attempt(target_ref: str) -> dict[str, Any] | None:
        return await _execute_action(
            session, op, url=url, ref=target_ref, value=value, direction=direction, amount=amount,
            x=x, y=y, tab_index=tab_index, timeout_ms=timeout_ms,
        )

    error = await attempt(current_ref)
    code = str((error or {}).get("code") or "")

    if error and code == "ELEMENT_NOT_FOUND" and current_ref:
        old_row = _element_map(before).get(current_ref, {})
        fresh = await _snapshot(session, force=True)
        replacement = _find_replacement_ref(old_row, fresh) if old_row else _history_replacement_ref(session, current_ref, fresh)
        if replacement and replacement != current_ref:
            recoveries.append({"kind": "reacquired_detached_element", "from_ref": current_ref, "to_ref": replacement})
            session.recovery_attempts += 1
            current_ref = replacement
            error = await attempt(current_ref)
            code = str((error or {}).get("code") or "")

    if error and code in {"ELEMENT_NOT_VISIBLE", "ELEMENT_OBSCURED"} and current_ref:
        element = await _find_ref(session, current_ref)
        if element is not None:
            try:
                await element.scroll_into_view_if_needed(timeout=2000)
                recoveries.append({"kind": "scroll_target_into_view", "ref": current_ref})
                session.recovery_attempts += 1
                error = await attempt(current_ref)
            except Exception:
                pass
            finally:
                try:
                    await element.dispose()
                except Exception:
                    pass

    # A click that opened exactly one new page is mechanically switched to that
    # page so the next observation is the state produced by the action.
    await asyncio.sleep(0)
    pages_after = [page for page in session.context.pages if not page.is_closed()]
    new_pages = [page for page in pages_after if page not in pages_before]
    if not error and len(new_pages) == 1:
        session.page = new_pages[0]
        try:
            await session.page.bring_to_front()
            await session.page.wait_for_load_state("domcontentloaded", timeout=min(max(timeout_ms, 1000), 5000))
        except Exception:
            pass
        recoveries.append({"kind": "popup_registered_and_switched", "tab_index": pages_after.index(session.page)})
        session.recovery_attempts += 1

    # Downloads are captured asynchronously. If this action started one, wait a
    # short bounded interval for completion so the fused observation can carry
    # usable file evidence without a separate model round-trip.
    if not error:
        for _ in range(10):
            if len(session.downloads) > downloads_before:
                break
            await asyncio.sleep(0.025)
        if len(session.downloads) > downloads_before:
            row = await _wait_for_download(session, "", min(timeout_ms, 5000))
            if row:
                recoveries.append({"kind": "download_waited", "download_id": row.get("id"), "status": row.get("status")})
                session.recovery_attempts += 1

    return error, {"attempted": bool(recoveries), "events": recoveries, "effective_ref": current_ref}


async def _adopt_pending_popup(session: BrowserSession) -> dict[str, Any] | None:
    """Switch to an asynchronously-created popup after UI stabilization.

    Playwright's context ``page`` event may land a few event-loop turns after a
    click has returned. The immediate recovery path catches fast popups; this
    second pass runs after the normal stabilization wait, avoiding an arbitrary
    sleep on every click while still fusing popup adoption into the same agent
    step.
    """
    total = len(session.popup_events)
    if total <= session.handled_popup_events:
        return None
    event = dict(session.popup_events[-1])
    pages = [page for page in session.context.pages if not page.is_closed()]
    target = None
    try:
        index = int(event.get("index", -1))
    except (TypeError, ValueError):
        index = -1
    if 0 <= index < len(pages):
        target = pages[index]
    elif pages:
        target = pages[-1]
    session.handled_popup_events = total
    if target is None or target is session.page:
        return None
    session.page = target
    try:
        await target.bring_to_front()
        await target.wait_for_load_state("domcontentloaded", timeout=3000)
    except Exception:
        pass
    session.recovery_attempts += 1
    return {
        "kind": "popup_registered_and_switched",
        "tab_index": pages.index(target) if target in pages else index,
        "url": str(target.url or "")[:2000],
    }


def _match_named_element(snapshot: dict[str, Any], check: dict[str, Any]) -> dict[str, Any] | None:
    ref = str(check.get("ref") or "")
    role = str(check.get("role") or "").lower()
    name = str(check.get("name") or "").strip().lower()
    for row in snapshot.get("elements", []):
        if not isinstance(row, dict):
            continue
        if ref and str(row.get("ref")) != ref:
            continue
        if role and str(row.get("role") or "").lower() != role:
            continue
        if name and name not in str(row.get("name") or "").lower():
            continue
        return row
    return None


_UI_CHECK_ALIASES = {
    "element_value_equals": "element_value",
    "page_title_contains": "title_contains",
    "page_title_matches": "title_matches",
    "url_matches": "url_matches",
}


def normalize_ui_checks(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in list(checks or [])[:32]:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        kind = str(row.get("type") or "").strip().lower()
        row["type"] = _UI_CHECK_ALIASES.get(kind, kind)
        result.append(row)
    return result


def verify_snapshot(snapshot: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate explicit UI end-state predicates independently of model prose."""
    normalized = normalize_ui_checks(checks)
    results: list[dict[str, Any]] = []
    for check in normalized:
        kind = str(check.get("type") or "").strip().lower()
        expected = check.get("value")
        passed = False
        actual: Any = None
        if kind == "url_equals":
            actual = snapshot.get("url", ""); passed = actual == str(expected or "")
        elif kind == "url_contains":
            actual = snapshot.get("url", ""); passed = str(expected or "") in str(actual)
        elif kind == "url_matches":
            actual = snapshot.get("url", "")
            try: passed = bool(re.search(str(expected or ""), str(actual)))
            except re.error: passed = False
        elif kind == "title_contains":
            actual = snapshot.get("title", ""); passed = str(expected or "").lower() in str(actual).lower()
        elif kind == "title_matches":
            actual = snapshot.get("title", "")
            try: passed = bool(re.search(str(expected or ""), str(actual), re.I))
            except re.error: passed = False
        elif kind == "text_present":
            actual = snapshot.get("text", ""); passed = str(expected or "").lower() in str(actual).lower()
        elif kind == "text_absent":
            actual = snapshot.get("text", ""); passed = str(expected or "").lower() not in str(actual).lower()
        elif kind in {
            "element_visible", "element_not_visible", "element_value", "element_checked",
            "element_disabled", "element_expanded", "element_selected",
        }:
            row = _match_named_element(snapshot, check)
            actual = row
            if kind == "element_not_visible":
                passed = row is None
            elif row is not None:
                if kind == "element_visible": passed = True
                elif kind == "element_value": passed = str(row.get("value") or "") == str(expected or "")
                elif kind == "element_checked": passed = bool(row.get("checked")) is bool(expected)
                elif kind == "element_disabled": passed = bool(row.get("disabled")) is bool(expected)
                elif kind == "element_expanded": passed = bool(row.get("expanded")) is bool(expected)
                elif kind == "element_selected": passed = bool(row.get("selected")) is bool(expected)
        elif kind == "tab_open":
            needle = str(expected or check.get("url") or check.get("title") or "").lower()
            actual = snapshot.get("tabs", [])
            passed = any(
                needle in (str(row.get("url") or "") + " " + str(row.get("title") or "")).lower()
                for row in actual if isinstance(row, dict)
            ) if needle else bool(actual)
        elif kind == "download_exists":
            needle = str(expected or check.get("name") or "").lower()
            actual = snapshot.get("downloads", [])
            passed = any(
                (not needle or needle in str(row.get("filename") or "").lower()) and bool(row.get("completed", True))
                for row in actual if isinstance(row, dict)
            )
        else:
            results.append({"check": check, "passed": False, "error": "unsupported_check_type"})
            continue
        results.append({"check": check, "passed": bool(passed), "actual": actual})
    return {"passed": bool(normalized) and all(row.get("passed") for row in results), "checks": results}


def _task_hint_from_working_state() -> str:
    try:
        from .working_state import WorkingStateStore
        state = WorkingStateStore().load()
        return str(state.get("objective") or "")[:3000]
    except Exception:
        return ""


def _sanitize_action_for_audit(action: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    clean = dict(action)
    value = str(clean.get("value") or "")
    ref = str(clean.get("ref") or "")
    element = _element_map(snapshot).get(ref, {})
    sensitive_name = str(element.get("name") or "").lower()
    sensitive = str(element.get("input_type") or "").lower() == "password" or any(
        token in sensitive_name for token in ("password", "passcode", "secret", "token", "credit card", "cvv")
    )
    if value and sensitive:
        clean["value"] = "[redacted]"
    return clean


def _token_metrics(
    full_snapshot: dict[str, Any],
    projection: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    """Account separately for canonical state, model projection, and delta output.

    The previous metric named ``full_snapshot_tokens`` was actually measuring the
    already-pruned model projection.  Keeping the three stages distinct makes the
    browser diagnostics useful for deciding whether more capture-side pruning is
    worth the state-fidelity tradeoff.
    """
    full_tokens = _estimate_tokens(full_snapshot)
    projected_tokens = _estimate_tokens(projection)
    observation_tokens = _estimate_tokens(observation)
    projection_saved = max(0, full_tokens - projected_tokens)
    delta_saved = max(0, projected_tokens - observation_tokens)
    full_elements = len(full_snapshot.get("elements") or []) if isinstance(full_snapshot, dict) else 0
    projected_elements = len(projection.get("elements") or []) if isinstance(projection, dict) else 0
    return {
        "full_snapshot_tokens": full_tokens,
        "projected_snapshot_tokens": projected_tokens,
        "observation_tokens": observation_tokens,
        "tokens_saved_by_projection": projection_saved,
        "projection_savings_pct": round((projection_saved * 100.0 / full_tokens), 2) if full_tokens else 0.0,
        "tokens_saved_by_delta": delta_saved,
        "delta_savings_pct": round((delta_saved * 100.0 / projected_tokens), 2) if projected_tokens else 0.0,
        "element_count_full": full_elements,
        "element_count_projected": projected_elements,
        "element_pruning_ratio": round((projected_elements / full_elements), 4) if full_elements else 1.0,
        "candidates_total": int(projection.get("candidate_count_total") or full_elements),
        "candidates_exposed": int(projection.get("candidate_count_exposed") or projected_elements),
        "candidates_pruned": int(projection.get("candidate_pruned") or max(0, full_elements - projected_elements)),
    }


async def _capture_optional_screenshot(session: BrowserSession, conversation_id: str) -> str:
    digest = hashlib.sha256(normalize_conversation_id(conversation_id).encode("utf-8", errors="replace")).hexdigest()[:16]
    root = Path(os.environ.get("AGENT_WORKSPACE", "/app/workspace")) / "browser_ui" / digest
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"state-{max(1, session.version):06d}.png"
    await session.page.screenshot(path=str(path), full_page=False)
    return str(path)


def _persist_browser_step(
    conversation_id: str,
    *,
    state_version: int,
    state: dict[str, Any],
    requirements: list[dict[str, Any]] | None,
    operation: str,
    action: dict[str, Any],
    result: dict[str, Any],
    process_success: bool,
    outcome_success: bool | None,
    timings: dict[str, Any],
    token_metrics: dict[str, Any],
) -> None:
    try:
        store = BrowserStateStore(conversation_id)
        store.save_state(
            state_version=state_version,
            state=state,
            requirements=requirements,
            metrics={"timings_ms": timings, "token_metrics": token_metrics, "scores": result.get("scores", {})},
        )
        store.record_step(
            state_version=state_version,
            operation=operation,
            action=action,
            process_success=process_success,
            outcome_success=outcome_success,
            timings=timings,
            token_metrics=token_metrics,
            result=result,
        )
    except Exception:
        # Browser execution must remain available even if durable diagnostics are
        # temporarily unavailable or the DB is read-only.
        pass


async def _browser_step_async(
    conversation_id: str,
    op: str,
    expected_state_version: int,
    url: str,
    ref: str,
    value: str,
    direction: str,
    amount: int,
    checks: list[dict[str, Any]],
    x: int,
    y: int,
    max_candidates: int,
    include_offscreen: bool,
    screenshot: bool,
    tab_index: int,
    timeout_ms: int,
    task_hint: str,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    session_started = time.perf_counter()
    session = await _RUNTIME.session(conversation_id)
    session_get_ms = (time.perf_counter() - session_started) * 1000.0
    assert session.lock is not None
    async with session.lock:
        timings: dict[str, Any] = {
            "session_get_ms": round(session_get_ms, 3),
            "browser_session_create_ms": round(session.session_create_ms if session.step_count == 0 else 0.0, 3),
        }
        cache_hits_before, cache_misses_before = session.cache_hits, session.cache_misses
        incremental_before = session.incremental_updates
        recoveries_before = session.recovery_attempts
        force_full_observation = not bool(session.snapshot) or op in {"navigate", "back", "new_tab", "switch_tab"}
        cached_before = dict(session.snapshot) if session.snapshot else {}
        cached_version = session.version

        pre_started = time.perf_counter()
        try:
            live_before = await _snapshot(session)
            if session.snapshot and live_before != session.snapshot:
                session.version += 1
                session.snapshot = live_before
            elif not session.snapshot:
                session.snapshot = live_before
                if session.version == 0:
                    session.version = 1
        except Exception:
            live_before = session.snapshot
        timings["pre_snapshot_ms"] = round((time.perf_counter() - pre_started) * 1000.0, 3)

        pinned_refs = {ref} if ref else set()
        before_projection = project_snapshot(
            live_before or {}, task_hint=task_hint, max_candidates=max_candidates,
            include_offscreen=include_offscreen, pinned_refs=pinned_refs,
        )
        action = {
            "op": op, "ref": ref, "url": url, "value": value, "direction": direction,
            "amount": amount, "x": x, "y": y, "tab_index": tab_index, "timeout_ms": timeout_ms,
        }
        audit_action = _sanitize_action_for_audit(action, live_before or {})
        normalized_checks = normalize_ui_checks(checks)
        safety = classify_action_safety(op, live_before or {}, ref=ref, value=value)
        recovery_meta: dict[str, Any] = {"attempted": False, "events": []}

        def finish_metrics(canonical: dict[str, Any], projection: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
            token_metrics = _token_metrics(canonical, projection, observation)
            timings["semantic_cache_hits"] = session.cache_hits - cache_hits_before
            timings["semantic_cache_misses"] = session.cache_misses - cache_misses_before
            timings["incremental_semantic_updates"] = session.incremental_updates - incremental_before
            timings["deterministic_recoveries"] = session.recovery_attempts - recoveries_before
            timings["total_ms"] = round((time.perf_counter() - total_started) * 1000.0, 3)
            return token_metrics

        def failure_payload(error: dict[str, Any], *, observation: dict[str, Any] | None = None, state: dict[str, Any] | None = None) -> dict[str, Any]:
            projected = observation or before_projection
            token_metrics = finish_metrics(state or live_before or {}, projected, projected)
            payload = {
                "ok": False,
                "operation": op,
                "state_version": session.version,
                "error": error,
                "observation": projected,
                "safety": safety,
                "recovery": recovery_meta,
                "scores": {"process": {"evaluated": True, "success": False}, "outcome": {"evaluated": False, "success": None}},
                "metrics": {"timings_ms": timings, "token_metrics": token_metrics},
            }
            _persist_browser_step(
                conversation_id, state_version=session.version, state=state or live_before or {}, requirements=None,
                operation=op, action=audit_action, result=payload, process_success=False, outcome_success=None,
                timings=timings, token_metrics=token_metrics,
            )
            session.step_count += 1
            return payload

        # Operations that act on a particular observed state must carry its version.
        versionless_ops = {"observe", "navigate", "verify", "list_tabs", "wait_download", "new_tab"}
        if op not in versionless_ops:
            if expected_state_version < 0:
                return failure_payload(_error(
                    "MISSING_STATE_VERSION",
                    "Interactive actions require expected_state_version from the latest observation.",
                    retryable=True,
                ))
            if expected_state_version != session.version:
                # Mechanical stale recovery is safe when the action was planned on
                # the immediately previous state, its target identity is unchanged,
                # and the action is not consequential. This avoids an LLM round-trip
                # for unrelated timers/badges/DOM churn.
                can_rebase = (
                    expected_state_version == cached_version
                    and bool(cached_before)
                    and safety.get("level") != "consequential"
                    and _same_target(cached_before, live_before or {}, ref)
                    and op not in {"close_tab", "switch_tab"}
                )
                if can_rebase:
                    session.recovery_attempts += 1
                    recovery_meta = {
                        "attempted": True,
                        "events": [{"kind": "stale_state_auto_rebased", "from_version": expected_state_version, "to_version": session.version}],
                    }
                else:
                    return failure_payload(_error(
                        "STALE_OBSERVATION",
                        f"Expected state version {expected_state_version}, current version is {session.version}.",
                        retryable=True,
                    ))

        try:
            if op == "verify":
                verify_started = time.perf_counter()
                current = await _snapshot(session, force=True)
                verification = verify_snapshot(current, normalized_checks)
                timings["verification_ms"] = round((time.perf_counter() - verify_started) * 1000.0, 3)
                session.snapshot = current
                projected = project_snapshot(
                    current, task_hint=task_hint, max_candidates=max_candidates,
                    include_offscreen=include_offscreen, pinned_refs=pinned_refs,
                )
                token_metrics = finish_metrics(current, projected, {"verification": verification})
                payload = {
                    "ok": bool(verification["passed"]),
                    "operation": "verify",
                    "state_version": session.version,
                    "verification": verification,
                    "safety": safety,
                    "scores": {
                        "process": {"evaluated": True, "success": True},
                        "outcome": {"evaluated": True, "success": bool(verification["passed"])},
                    },
                    "metrics": {"timings_ms": timings, "token_metrics": token_metrics},
                    **({} if verification["passed"] else {"error": _error(
                        "COMPLETION_NOT_VERIFIED", "One or more requested UI completion checks failed.", retryable=True
                    )}),
                }
                _persist_browser_step(
                    conversation_id, state_version=session.version, state=current,
                    requirements=verification.get("checks") or normalized_checks,
                    operation=op, action=audit_action, result=payload, process_success=True,
                    outcome_success=bool(verification["passed"]), timings=timings, token_metrics=token_metrics,
                )
                session.step_count += 1
                return payload

            before = session.snapshot or live_before or await _snapshot(session)

            # Consequential controls use a strict just-in-time pre-submit check:
            # refresh state, reject drift, verify caller-specified field/end-state
            # predicates, and only then execute the click/Enter.
            if safety.get("requires_pre_submit_verification"):
                if not normalized_checks:
                    return failure_payload(_error(
                        "PRE_SUBMIT_VERIFICATION_REQUIRED",
                        "Consequential UI actions require machine-verifiable checks for the target and relevant form/task state.",
                        retryable=True,
                        target=ref,
                    ))
                preflight_started = time.perf_counter()
                fresh = await _snapshot(session, force=True)
                timings["pre_submit_snapshot_ms"] = round((time.perf_counter() - preflight_started) * 1000.0, 3)
                if fresh != before:
                    session.version += 1
                    session.snapshot = fresh
                    projected = project_snapshot(
                        fresh, task_hint=task_hint, max_candidates=max_candidates,
                        include_offscreen=include_offscreen, pinned_refs=pinned_refs,
                    )
                    return failure_payload(_error(
                        "STALE_OBSERVATION",
                        "UI state changed during consequential-action preflight; inspect the refreshed state before submitting.",
                        retryable=True,
                        target=ref,
                    ), observation=projected, state=fresh)
                preflight = verify_snapshot(fresh, normalized_checks)
                timings["pre_submit_verification_ms"] = round((time.perf_counter() - preflight_started) * 1000.0, 3)
                if not preflight.get("passed"):
                    projected = project_snapshot(
                        fresh, task_hint=task_hint, max_candidates=max_candidates,
                        include_offscreen=include_offscreen, pinned_refs=pinned_refs,
                    )
                    err = _error(
                        "PRE_SUBMIT_VERIFICATION_FAILED",
                        "Consequential action blocked because one or more pre-submit checks failed.",
                        retryable=True,
                        target=ref,
                    )
                    err["verification"] = preflight
                    return failure_payload(err, observation=projected, state=fresh)
                safety["pre_submit_verified"] = True
                safety["pre_submit_checks"] = len(normalized_checks)

            action_started = time.perf_counter()
            action_error, action_recovery = await _execute_action_with_recovery(
                session, op, before=before, url=url, ref=ref, value=value, direction=direction,
                amount=amount, x=x, y=y, tab_index=tab_index, timeout_ms=timeout_ms,
            )
            if action_recovery.get("attempted"):
                recovery_meta["attempted"] = True
                recovery_meta.setdefault("events", []).extend(action_recovery.get("events") or [])
                recovery_meta["effective_ref"] = action_recovery.get("effective_ref")
            action_ms = (time.perf_counter() - action_started) * 1000.0
            timings["action_ms"] = round(action_ms, 3)
            if op in {"navigate", "back"}:
                timings["navigation_ms"] = round(action_ms, 3)

            if action_error:
                current = await _snapshot(session, force=True)
                session.snapshot = current
                projected = project_snapshot(
                    current, task_hint=task_hint, max_candidates=max_candidates,
                    include_offscreen=include_offscreen, pinned_refs=pinned_refs,
                )
                return failure_payload(action_error, observation=projected, state=current)

            if op not in {"observe", "list_tabs", "wait_download"}:
                settle_started = time.perf_counter()
                stability = await _wait_for_ui_stability(session, timeout_ms=min(timeout_ms, 5000))
                timings["stabilization_ms"] = round((time.perf_counter() - settle_started) * 1000.0, 3)
                timings["stability_samples"] = int(stability.get("samples") or 0)
                timings["stability_settled"] = bool(stability.get("settled"))
                if op not in {"new_tab", "switch_tab", "close_tab"}:
                    late_popup = await _adopt_pending_popup(session)
                    if late_popup:
                        recovery_meta["attempted"] = True
                        recovery_meta.setdefault("events", []).append(late_popup)
                        # The newly active page has its own mutation journal.
                        # Force a fresh snapshot below instead of patching the
                        # previous tab's cached state.
                        force_full_observation = True

            post_started = time.perf_counter()
            # Let mutation-driven incremental patching handle ordinary DOM edits;
            # navigation/topology changes automatically fall back to a full scan.
            current = await _snapshot(session, force=bool(force_full_observation and session.page.url != str(before.get("url") or "")))
            timings["post_snapshot_ms"] = round((time.perf_counter() - post_started) * 1000.0, 3)
            changed = current != before
            if op == "observe":
                if session.version == 0:
                    session.version = 1
            elif changed:
                session.version += 1

            projection_started = time.perf_counter()
            current_projection = project_snapshot(
                current, task_hint=task_hint, max_candidates=max_candidates,
                include_offscreen=include_offscreen, pinned_refs=pinned_refs,
            )
            timings["projection_ms"] = round((time.perf_counter() - projection_started) * 1000.0, 3)
            diff_started = time.perf_counter()
            delta = semantic_diff({} if force_full_observation else before_projection, current_projection)
            timings["diff_ms"] = round((time.perf_counter() - diff_started) * 1000.0, 3)
            session.snapshot = current

            screenshot_path = ""
            if screenshot:
                shot_started = time.perf_counter()
                screenshot_path = await _capture_optional_screenshot(session, conversation_id)
                timings["screenshot_ms"] = round((time.perf_counter() - shot_started) * 1000.0, 3)

            token_metrics = finish_metrics(current, current_projection, delta)
            process_success = True
            target_grounded = (
                bool(ref) or (op == "click" and x >= 0 and y >= 0)
                or op in {"observe", "navigate", "scroll", "key", "back", "new_tab", "list_tabs", "switch_tab", "close_tab", "wait_download"}
            )
            payload = {
                "ok": True,
                "operation": op,
                "state_version": session.version,
                "changed": changed,
                "delta": delta,
                "safety": safety,
                "recovery": recovery_meta,
                "scores": {
                    "process": {
                        "evaluated": True,
                        "success": process_success,
                        "target_grounded": target_grounded,
                        "state_changed": bool(changed),
                    },
                    "outcome": {"evaluated": False, "success": None},
                },
                "metrics": {"timings_ms": timings, "token_metrics": token_metrics},
            }
            if op == "list_tabs":
                payload["tabs"] = current_projection.get("tabs", [])
            if op == "wait_download":
                payload["downloads"] = current_projection.get("downloads", [])
            if screenshot_path:
                payload["screenshot_path"] = screenshot_path
            invalidates_completion = op not in {"observe", "list_tabs", "wait_download"}
            _persist_browser_step(
                conversation_id, state_version=session.version, state=current,
                requirements=[] if invalidates_completion else None,
                operation=op, action=audit_action, result=payload, process_success=process_success,
                outcome_success=None, timings=timings, token_metrics=token_metrics,
            )
            session.step_count += 1
            return payload
        except Exception as exc:
            try:
                current = await _snapshot(session, force=True)
                session.snapshot = current
            except Exception:
                current = {}
            projected = project_snapshot(
                current, task_hint=task_hint, max_candidates=max_candidates,
                include_offscreen=include_offscreen, pinned_refs=pinned_refs,
            )
            return failure_payload(_classify_exception(exc, target=ref), observation=projected, state=current)


def browser_step(
    op: BrowserOp = "observe",
    expected_state_version: int = -1,
    url: str = "",
    ref: str = "",
    value: str = "",
    direction: str = "down",
    amount: int = 700,
    checks: list[dict[str, Any]] = [],
    x: int = -1,
    y: int = -1,
    max_candidates: int = 60,
    include_offscreen: bool = False,
    screenshot: bool = False,
    tab_index: int = -1,
    timeout_ms: int = 10000,
) -> str | dict[str, Any]:
    """Execute one browser action and return its fused post-action semantic state.

    Semantic refs are preferred for actions. ``x``/``y`` are a click-only visual
    fallback. Set ``screenshot=True`` only when pixels materially help; normal UI
    reasoning stays semantic/delta-first to minimize latency and context tokens.
    Increase ``max_candidates`` or enable ``include_offscreen`` to expand a pruned
    observation when the needed control is not visible in the default projection.
    """
    cid = get_active_conversation_id()
    task_hint = _task_hint_from_working_state()
    payload = _RUNTIME.run(
        _browser_step_async(
            cid, str(op), int(expected_state_version), str(url), str(ref), str(value),
            str(direction), int(amount), list(checks or []), int(x), int(y),
            int(max_candidates), bool(include_offscreen), bool(screenshot), int(tab_index),
            max(250, min(int(timeout_ms), 15000)), task_hint,
        ),
        timeout=35,
    )
    screenshot_path = str(payload.get("screenshot_path") or "") if isinstance(payload, dict) else ""
    if screenshot_path:
        return media_result(_json(payload), [screenshot_path])
    return _json(payload)


def close_browser_session() -> str:
    """Close the current conversation's persistent browser context."""
    cid = get_active_conversation_id()
    _RUNTIME.run(_RUNTIME.close_session(cid), timeout=10)
    return _json({"ok": True, "closed": True, "conversation_id": cid})


def _normalized_url(value: str) -> str:
    try:
        parsed = urlsplit(str(value or ""))
        path = parsed.path or "/"
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path.rstrip("/") or "/", parsed.query, ""))
    except Exception:
        return str(value or "").rstrip("#/")


async def _reuse_loaded_page_async(conversation_id: str, requested_url: str, mode: str, limit: int) -> dict[str, Any] | None:
    session = await _RUNTIME.existing_session(conversation_id)
    if session is None:
        return None
    assert session.lock is not None
    async with session.lock:
        if requested_url and _normalized_url(session.page.url) != _normalized_url(requested_url):
            return None
        if mode == "text":
            text = await session.page.locator("body").inner_text(timeout=2500)
            return {
                "source": "browser_session", "url": session.page.url,
                "content_type": "text/html", "text": str(text or "")[:limit],
            }
        if mode == "metadata":
            data = await session.page.evaluate(
                """() => {
                  const meta = {};
                  for (const el of document.querySelectorAll('meta')) {
                    const key = (el.getAttribute('property') || el.getAttribute('name') || el.getAttribute('itemprop') || '').toLowerCase();
                    const content = el.getAttribute('content') || '';
                    if (key && content) meta[key] = content.slice(0, 2000);
                  }
                  const canonical = document.querySelector('link[rel~=canonical]')?.href || '';
                  return {url: location.href, title: document.title || '', canonical, meta};
                }"""
            )
            if isinstance(data, dict):
                data["source"] = "browser_session"
                data["http_status"] = None
                data["content_type"] = "text/html"
                return data
            return None
        if mode == "links":
            data = await session.page.evaluate(
                r"""limit => ({
                  source: location.href,
                  links: Array.from(document.querySelectorAll('a[href]')).slice(0, limit).map(a => ({
                    url: a.href, text: (a.innerText || a.textContent || '').replace(/\s+/g,' ').trim().slice(0,300)
                  }))
                })""",
                min(max(limit, 1), 200),
            )
            if isinstance(data, dict):
                data["source_kind"] = "browser_session"
                return data
            return None
        return None


def reuse_loaded_page(url: str, *, mode: str = "text", limit: int = 50000) -> dict[str, Any] | None:
    """Reuse the already-loaded interactive page without starting a browser/refetching."""
    if not _RUNTIME.is_started():
        return None
    cid = get_active_conversation_id()
    try:
        return _RUNTIME.run_if_started(
            _reuse_loaded_page_async(cid, str(url or ""), str(mode), int(limit)), timeout=8,
        )
    except Exception:
        return None


async def _screenshot_async(conversation_id: str, url: str, output_path: str, task_hint: str = "") -> dict[str, Any]:
    session = await _RUNTIME.session(conversation_id)
    assert session.lock is not None
    async with session.lock:
        safe_url = validate_public_url(url)
        if _normalized_url(session.page.url) != _normalized_url(safe_url):
            await session.page.goto(safe_url, wait_until="domcontentloaded", timeout=15000)
            await _wait_for_ui_stability(session)
        current = await _snapshot(session, force=True)
        if current != session.snapshot:
            session.version += 1
        session.snapshot = current
        await session.page.screenshot(path=output_path, full_page=False)
        projected = project_snapshot(current, task_hint=task_hint, max_candidates=80, include_offscreen=False)
        return {"state_version": session.version, **projected}


def capture_web_screenshot(url: str, output_path: str) -> dict[str, Any]:
    """Internal compatibility helper used by take_web_screenshot."""
    cid = get_active_conversation_id()
    task_hint = _task_hint_from_working_state()
    return _RUNTIME.run(_screenshot_async(cid, str(url), str(output_path), task_hint), timeout=35)
