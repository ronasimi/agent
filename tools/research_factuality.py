"""Evidence-ledger and report-factuality helpers.

This module is intentionally model-agnostic.  The background research worker
owns model calls; these helpers validate that model-produced ledger excerpts
actually occur in persisted source text and provide bounded renderings for
report writing and verification.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse


CLAIM_LEDGER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "claims": {
                        "type": "array",
                        "maxItems": 10,
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim": {"type": "string"},
                                "support_excerpt": {"type": "string"},
                                "support_type": {
                                    "type": "string",
                                    "enum": [
                                        "direct_fact", "organization_statement", "reported_claim",
                                        "analysis_opinion", "forecast_estimate",
                                    ],
                                },
                                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                                "qualification": {"type": "string"},
                            },
                            "required": ["claim", "support_excerpt", "support_type", "confidence", "qualification"],
                        },
                    },
                    "limitations": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                },
                "required": ["source_id", "claims", "limitations"],
            },
        }
    },
    "required": ["sources"],
}

FACTUALITY_GATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["pass", "revise"]},
        "issues": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "reason": {"type": "string"},
                    "issue_type": {
                        "type": "string",
                        "enum": [
                            "unsupported", "citation_mismatch", "overstated_causality",
                            "unattributed_opinion", "invented_specific", "source_scope_error",
                        ],
                    },
                    "supported_source_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                },
                "required": ["text", "reason", "issue_type", "supported_source_ids"],
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["decision", "issues", "notes"],
}

REPORT_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "sections": {
            "type": "array",
            "minItems": 3,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "purpose": {"type": "string"},
                    "source_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8},
                    "media_source_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
                    "target_words": {"type": "integer", "minimum": 250, "maximum": 650},
                },
                "required": ["heading", "purpose", "source_ids", "media_source_ids", "target_words"],
            },
        },
    },
    "required": ["title", "sections"],
}


def normalize_space(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _canonical_source_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", normalize_space(value))
    return (
        text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        .replace("–", "-").replace("—", "-").casefold()
    )


def excerpt_is_verbatim(excerpt: str, source_text: str) -> bool:
    """Require support text to occur in the fetched source after harmless typography normalization."""
    excerpt_n = _canonical_source_text(excerpt)
    source_n = _canonical_source_text(source_text)
    return bool(excerpt_n and len(excerpt_n) >= 12 and excerpt_n in source_n)


def source_authority_hint(url: str) -> str:
    """Return a conservative, non-topic-specific authority hint."""
    host = (urlparse(str(url or "")).hostname or "").lower()
    if host.endswith(".gov") or host in {"sec.gov", "www.sec.gov"}:
        return "government_or_regulator"
    if host.endswith(".edu"):
        return "academic"
    if "wikipedia.org" in host:
        return "tertiary_reference"
    if host:
        return "web_source"
    return "unknown"


def validate_ledger_batch(payload: dict[str, Any], source_map: dict[str, dict[str, Any]], *, max_claims: int = 8) -> list[dict[str, Any]]:
    """Drop claims whose quoted support cannot be found in the raw source."""
    accepted: list[dict[str, Any]] = []
    allowed_types = {"direct_fact", "organization_statement", "reported_claim", "analysis_opinion", "forecast_estimate"}
    allowed_conf = {"high", "medium", "low"}
    for block in payload.get("sources", []) if isinstance(payload, dict) else []:
        if not isinstance(block, dict):
            continue
        source_id = normalize_space(block.get("source_id")).upper()
        source = source_map.get(source_id)
        if not source:
            continue
        raw = str(source.get("raw_content") or "")
        claims = []
        for item in block.get("claims", []) if isinstance(block.get("claims"), list) else []:
            if not isinstance(item, dict):
                continue
            claim = normalize_space(item.get("claim"))
            excerpt = normalize_space(item.get("support_excerpt"))
            if not claim or not excerpt_is_verbatim(excerpt, raw):
                continue
            support_type = normalize_space(item.get("support_type"))
            confidence = normalize_space(item.get("confidence"))
            claims.append({
                "claim": claim,
                "support_excerpt": excerpt,
                "support_type": support_type if support_type in allowed_types else "reported_claim",
                "confidence": confidence if confidence in allowed_conf else "medium",
                "qualification": normalize_space(item.get("qualification")),
            })
            if len(claims) >= max(1, int(max_claims)):
                break
        accepted.append({
            "source_id": source_id,
            "title": normalize_space(source.get("title")) or "Untitled",
            "url": str(source.get("url") or ""),
            "retrieved_at": str(source.get("retrieved_at") or ""),
            "authority_hint": source_authority_hint(str(source.get("url") or "")),
            "claims": claims,
            "limitations": [normalize_space(x) for x in block.get("limitations", []) if normalize_space(x)][:5],
        })
    return accepted


def ledger_source_ids(ledger: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("source_id") or "") for item in ledger if item.get("claims")}


def render_claim_ledger(ledger: list[dict[str, Any]], source_ids: Iterable[object] | None = None, *, max_chars: int = 22000) -> str:
    requested = {normalize_space(x).upper() for x in (source_ids or []) if normalize_space(x)}
    chunks: list[str] = []
    remaining = max(1000, int(max_chars))
    for source in ledger:
        sid = str(source.get("source_id") or "")
        if requested and sid not in requested:
            continue
        claims = source.get("claims") or []
        if not claims:
            continue
        lines = [
            f"### {sid}: {source.get('title', 'Untitled')}",
            f"URL: {source.get('url', '')}",
            f"Authority hint: {source.get('authority_hint', 'unknown')}",
        ]
        for index, claim in enumerate(claims, 1):
            lines.extend([
                f"- Claim {index}: {claim.get('claim', '')}",
                f"  Support type: {claim.get('support_type', '')}; confidence: {claim.get('confidence', '')}",
                f"  Exact support: {claim.get('support_excerpt', '')}",
                f"  Qualification: {claim.get('qualification', '') or 'none'}",
            ])
        limitations = [str(x) for x in source.get("limitations") or [] if str(x).strip()]
        if limitations:
            lines.append("  Source limitations: " + " | ".join(limitations[:3]))
        block = "\n".join(lines) + "\n"
        if len(block) > remaining:
            if not chunks:
                chunks.append(block[:remaining])
            break
        chunks.append(block)
        remaining -= len(block)
    return "\n".join(chunks) or "No verified claim-ledger entries are available."


def normalize_gate_result(payload: dict[str, Any], draft: str) -> dict[str, Any]:
    decision = str(payload.get("decision") or "revise") if isinstance(payload, dict) else "revise"
    issues = []
    for issue in payload.get("issues", []) if isinstance(payload, dict) else []:
        if not isinstance(issue, dict):
            continue
        text = normalize_space(issue.get("text"))
        # Prefer exact spans so later repair/removal can be deterministic. Keep a
        # non-exact issue for diagnostics, but mark whether the span is usable.
        exact = bool(text and text in draft)
        issues.append({
            "text": text,
            "reason": normalize_space(issue.get("reason")),
            "issue_type": normalize_space(issue.get("issue_type")) or "unsupported",
            "supported_source_ids": [normalize_space(x).upper() for x in issue.get("supported_source_ids", []) if normalize_space(x)],
            "exact_span": exact,
        })
    if decision not in {"pass", "revise"}:
        decision = "revise"
    if decision == "pass" and issues:
        decision = "revise"
    return {"decision": decision, "issues": issues, "notes": normalize_space(payload.get("notes")) if isinstance(payload, dict) else ""}


def safe_ledger_fallback(ledger: list[dict[str, Any]], source_ids: Iterable[object], *, max_claims: int = 8) -> str:
    """Last-resort factual section body containing only ledger-approved claims."""
    requested = {normalize_space(x).upper() for x in source_ids if normalize_space(x)}
    lines: list[str] = []
    for source in ledger:
        sid = str(source.get("source_id") or "")
        if requested and sid not in requested:
            continue
        for claim in source.get("claims") or []:
            text = normalize_space(claim.get("claim"))
            qualification = normalize_space(claim.get("qualification"))
            if not text:
                continue
            if qualification:
                text = f"{text} ({qualification})"
            lines.append(f"- {text} [{sid}]")
            if len(lines) >= max(1, int(max_claims)):
                return "\n".join(lines)
    return "\n".join(lines) or "The verified source ledger did not contain enough supported claims to draft this section."
