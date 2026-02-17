# SPDX-License-Identifier: Apache-2.0
"""
Referee-voice summaries derived from GovernorViewModel.

Pure functions — no I/O, no side effects. Translate ViewModel data into
human-readable summaries for the three-screen web API (Now / Why / History).

Referee voice rules:
  - No "I think", no "you should", no personality
  - Lead with verdict: "Blocked:", "Rejected:", "OK:"
  - Show machinery only in `detail` (expandable)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from governor.viewmodel import (
    GovernorViewModel,
    DecisionView,
    ViolationView,
)


# =============================================================================
# Status pill
# =============================================================================


def derive_status_pill(vm: GovernorViewModel) -> str:
    """Traffic-light status: 'ok', 'needs_attention', or 'blocked'.

    Priority order (first match wins):
      blocked       - unstable/ductile regime, QUARANTINE/WARN drift,
                      critical/high unresolved violation, execution blocked items
      needs_attention - warm regime, WATCH drift, medium unresolved violation,
                        recent rejections
      ok            - otherwise
    """
    # --- blocked checks ---
    if vm.regime and vm.regime.name.upper() in ("UNSTABLE", "DUCTILE"):
        return "blocked"

    if vm.stability:
        alert = vm.stability.drift_alert.upper()
        if alert in ("QUARANTINE", "WARN"):
            return "blocked"

    for v in vm.violations:
        if v.severity in ("critical", "high") and v.resolution is None:
            return "blocked"

    if vm.execution:
        if vm.execution.blocked:
            return "blocked"

    # --- needs_attention checks ---
    if vm.regime and vm.regime.name.upper() == "WARM":
        return "needs_attention"

    if vm.stability:
        alert = vm.stability.drift_alert.upper()
        if alert == "WATCH":
            return "needs_attention"

    for v in vm.violations:
        if v.severity == "medium" and v.resolution is None:
            return "needs_attention"

    # Recent rejections
    for d in vm.decisions:
        if d.status == "rejected":
            return "needs_attention"

    return "ok"


# =============================================================================
# One-sentence summary
# =============================================================================


def derive_one_sentence(vm: GovernorViewModel) -> str:
    """One-sentence referee-voice summary of current state."""
    pill = derive_status_pill(vm)

    if pill == "blocked":
        # Find the most salient reason
        if vm.regime and vm.regime.name.upper() in ("UNSTABLE", "DUCTILE"):
            return f"Blocked: regime is {vm.regime.name.lower()}."

        if vm.stability:
            alert = vm.stability.drift_alert.upper()
            if alert in ("QUARANTINE", "WARN"):
                return f"Blocked: drift alert at {alert.lower()} level."

        critical = [v for v in vm.violations if v.severity in ("critical", "high") and v.resolution is None]
        if critical:
            n = len(critical)
            word = "violation" if n == 1 else "violations"
            return f"Blocked: {n} unresolved {word}."

        if vm.execution and vm.execution.blocked:
            n = len(vm.execution.blocked)
            word = "action" if n == 1 else "actions"
            return f"Blocked: {n} execution {word} blocked."

        return "Blocked: governance constraints not satisfied."

    if pill == "needs_attention":
        if vm.regime and vm.regime.name.upper() == "WARM":
            return "Needs attention: regime warming."

        if vm.stability and vm.stability.drift_alert.upper() == "WATCH":
            return "Needs attention: drift watch active."

        medium = [v for v in vm.violations if v.severity == "medium" and v.resolution is None]
        if medium:
            return f"Needs attention: {len(medium)} medium-severity violation(s)."

        rejected = [d for d in vm.decisions if d.status == "rejected"]
        if rejected:
            return f"Needs attention: {len(rejected)} recent rejection(s)."

        return "Needs attention: review recommended."

    return "OK: consistent across last check."


# =============================================================================
# Suggested action
# =============================================================================


def derive_suggested_action(vm: GovernorViewModel) -> str | None:
    """Highest-priority suggested action, or None when OK."""
    pill = derive_status_pill(vm)
    if pill == "ok":
        return None

    # Highest-severity unresolved violation
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    unresolved = [v for v in vm.violations if v.resolution is None]
    if unresolved:
        unresolved.sort(key=lambda v: severity_order.get(v.severity, 99))
        top = unresolved[0]
        if "grounding" in top.rule_breached or "evidence" in top.rule_breached.lower():
            return "Add a source for the ungrounded claim."
        if "premise" in top.rule_breached:
            return "Resolve quarantined premise before proceeding."
        return f"Resolve {top.severity} violation: {top.rule_breached}."

    if vm.regime and vm.regime.name.upper() in ("UNSTABLE", "DUCTILE"):
        return "Wait for regime to stabilize before making changes."

    if vm.stability and vm.stability.drift_alert.upper() in ("QUARANTINE", "WARN"):
        return "Review drift signals and address quarantined premises."

    rejected = [d for d in vm.decisions if d.status == "rejected"]
    if rejected:
        return "Review rejected proposal and provide required evidence."

    return "Review current governance state."


# =============================================================================
# Last event
# =============================================================================


def derive_last_event(vm: GovernorViewModel) -> dict[str, str] | None:
    """Most recent decision or violation as {summary, when}."""
    candidates: list[tuple[str, str]] = []

    for d in vm.decisions:
        title, _ = _referee_voice_decision(d)
        candidates.append((title, d.created_at))

    for v in vm.violations:
        title, _ = _referee_voice_violation(v)
        # Violations don't have created_at; use generated_at from vm
        candidates.append((title, vm.generated_at))

    if not candidates:
        return None

    # Sort by timestamp descending — most recent first
    candidates.sort(key=lambda c: c[1], reverse=True)
    summary, ts = candidates[0]
    return {"summary": summary, "when": _relative_time(ts)}


# =============================================================================
# Why feed
# =============================================================================


def derive_why_feed(
    vm: GovernorViewModel,
    *,
    limit: int = 50,
    severity_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Decision/violation/claim feed items for the Why screen.

    Each item:
        title, summary, type, severity, id, created_at, detail
    """
    items: list[dict[str, Any]] = []

    for d in vm.decisions:
        title, summary = _referee_voice_decision(d)
        sev = "error" if d.status == "rejected" else "info"
        items.append({
            "title": title,
            "summary": summary,
            "type": "decision",
            "severity": sev,
            "id": d.id,
            "created_at": d.created_at,
            "detail": d.to_dict(),
        })

    for v in vm.violations:
        title, summary = _referee_voice_violation(v)
        sev_map = {"critical": "error", "high": "error", "medium": "warning", "low": "info"}
        sev = sev_map.get(v.severity, "info")
        items.append({
            "title": title,
            "summary": summary,
            "type": "violation",
            "severity": sev,
            "id": v.id,
            "created_at": vm.generated_at,
            "detail": v.to_dict(),
        })

    for c in vm.claims:
        if c.state in ("contradicted", "stale"):
            sev = "warning"
            title = f"Contradicted: {_truncate(c.content, 60)}"
            if c.state == "stale":
                title = f"Stale: {_truncate(c.content, 60)}"
            items.append({
                "title": title,
                "summary": f"Claim confidence {c.confidence:.0%}, provenance: {c.provenance}.",
                "type": "claim",
                "severity": sev,
                "id": c.id,
                "created_at": c.created_at,
                "detail": c.to_dict(),
            })

    # Apply severity filter
    if severity_filter:
        items = [i for i in items if i["severity"] == severity_filter]

    # Sort by created_at descending
    items.sort(key=lambda i: i["created_at"], reverse=True)

    return items[:limit]


# =============================================================================
# History (grouped by day)
# =============================================================================


def derive_history_days(
    vm: GovernorViewModel,
    *,
    days: int = 7,
) -> list[dict[str, Any]]:
    """Events grouped by calendar day for the History screen."""
    # Collect all timestamped items
    all_items: list[dict[str, Any]] = []

    for d in vm.decisions:
        title, _ = _referee_voice_decision(d)
        all_items.append({
            "title": title,
            "type": "decision",
            "outcome": d.status,
            "id": d.id,
            "created_at": d.created_at,
        })

    for v in vm.violations:
        title, _ = _referee_voice_violation(v)
        all_items.append({
            "title": title,
            "type": "violation",
            "outcome": v.enforced_outcome,
            "id": v.id,
            "created_at": vm.generated_at,
        })

    # Group by date
    by_date: dict[str, list[dict[str, Any]]] = {}
    for item in all_items:
        try:
            dt = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            date_str = "unknown"
        by_date.setdefault(date_str, []).append(item)

    # Build day summaries, sorted descending
    result: list[dict[str, Any]] = []
    for date_str in sorted(by_date.keys(), reverse=True)[:days]:
        items = by_date[date_str]
        outcomes = [i["outcome"] for i in items]
        result.append({
            "date": date_str,
            "items_count": len(items),
            "rejection_count": sum(1 for o in outcomes if o == "rejected"),
            "contradiction_count": sum(1 for o in outcomes if o in ("contradicted", "quarantined")),
            "outcomes": outcomes,
            "items": items,
        })

    return result


# =============================================================================
# Referee voice helpers
# =============================================================================


def _referee_voice_decision(dec: DecisionView) -> tuple[str, str]:
    """Referee-voice (title, summary) for a decision."""
    if dec.status == "rejected":
        title = f"Rejected: {_truncate(dec.type, 50)}"
        summary = dec.rationale if dec.rationale else "Proposal did not meet governance requirements."
    elif dec.status == "accepted":
        title = f"Accepted: {_truncate(dec.type, 50)}"
        summary = dec.rationale if dec.rationale else "Proposal verified and applied."
    else:
        title = f"Pending: {_truncate(dec.type, 50)}"
        summary = "Awaiting verification."
    return title, summary


def _referee_voice_violation(vio: ViolationView) -> tuple[str, str]:
    """Referee-voice (title, summary) for a violation."""
    title = f"Blocked: {_truncate(vio.rule_breached, 50)}"
    if vio.resolution:
        title = f"Resolved: {_truncate(vio.rule_breached, 50)}"
    summary = vio.detail if vio.detail else f"Enforced outcome: {vio.enforced_outcome}."
    return title, summary


def _relative_time(iso_ts: str) -> str:
    """Convert ISO timestamp to relative time string."""
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - dt

        seconds = int(delta.total_seconds())
        if seconds < 0:
            return "just now"
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            m = seconds // 60
            return f"{m} min ago"
        if seconds < 86400:
            h = seconds // 3600
            return f"{h} hour{'s' if h != 1 else ''} ago"
        d = seconds // 86400
        if d == 1:
            return "yesterday"
        if d < 30:
            return f"{d} days ago"
        return dt.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        return "unknown"


# =============================================================================
# Internal helpers
# =============================================================================


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
