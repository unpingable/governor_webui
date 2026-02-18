# SPDX-License-Identifier: Apache-2.0
"""Tests for gov_webui.summaries — referee-voice ViewModel summaries."""

import pytest
from datetime import datetime, timezone, timedelta

from governor.viewmodel import (
    GovernorViewModel,
    SessionView,
    RegimeView,
    DecisionView,
    ClaimView,
    EvidenceView,
    ViolationView,
    ExecutionView,
    ExecutionActionView,
    StabilityView,
)
from gov_webui.summaries import (
    derive_status_pill,
    derive_one_sentence,
    derive_suggested_action,
    derive_last_event,
    derive_why_feed,
    derive_history_days,
    _referee_voice_decision,
    _referee_voice_violation,
    _relative_time,
)


# =============================================================================
# Helpers — minimal ViewModel construction
# =============================================================================


def _make_vm(**kwargs) -> GovernorViewModel:
    """Create a GovernorViewModel with overrides."""
    defaults = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    defaults.update(kwargs)
    return GovernorViewModel(**defaults)


def _make_regime(name: str = "ELASTIC", **kw) -> RegimeView:
    return RegimeView(
        name=name,
        setpoints=kw.get("setpoints", {}),
        telemetry=kw.get("telemetry", {}),
        boil_mode=kw.get("boil_mode"),
        transitions=kw.get("transitions", []),
    )


def _make_stability(drift_alert: str = "NONE", **kw) -> StabilityView:
    return StabilityView(
        rejection_rate=kw.get("rejection_rate", 0.0),
        claim_churn=kw.get("claim_churn", 0.0),
        contradiction_open_rate=kw.get("contradiction_open_rate", 0.0),
        drift_alert=drift_alert,
        drift_signals=kw.get("drift_signals", {}),
    )


def _make_violation(severity: str = "high", resolution: str | None = None, **kw) -> ViolationView:
    return ViolationView(
        id=kw.get("id", "vio_test_001"),
        rule_breached=kw.get("rule_breached", "grounding_failure"),
        triggering_decision=kw.get("triggering_decision", "dec_abc"),
        severity=severity,
        enforced_outcome=kw.get("enforced_outcome", "block"),
        resolution=resolution,
        source_system=kw.get("source_system", "audit"),
        detail=kw.get("detail", "Missing evidence for claim."),
    )


def _make_decision(status: str = "accepted", **kw) -> DecisionView:
    return DecisionView(
        id=kw.get("id", "dec_test_001"),
        status=status,
        type=kw.get("type", "TESTS_PASS"),
        rationale=kw.get("rationale", ""),
        dependencies=kw.get("dependencies", []),
        violations=kw.get("violations", []),
        source=kw.get("source", "proposal"),
        created_at=kw.get("created_at", datetime.now(timezone.utc).isoformat()),
        raw=kw.get("raw", {}),
    )


def _make_claim(state: str = "proposed", **kw) -> ClaimView:
    return ClaimView(
        id=kw.get("id", "clm_test_001"),
        state=state,
        content=kw.get("content", "The function returns true."),
        confidence=kw.get("confidence", 0.8),
        provenance=kw.get("provenance", "assumed"),
        evidence_links=kw.get("evidence_links", []),
        conflicting_claims=kw.get("conflicting_claims", []),
        stability=kw.get("stability", {}),
        created_at=kw.get("created_at", datetime.now(timezone.utc).isoformat()),
        raw=kw.get("raw", {}),
    )


def _make_execution(blocked: int = 0, pending: int = 0, running: int = 0, completed: int = 0) -> ExecutionView:
    def _actions(n: int, status: str) -> list[ExecutionActionView]:
        return [
            ExecutionActionView(id=f"act_{status}_{i}", description=f"action {i}", status=status, detail="")
            for i in range(n)
        ]
    return ExecutionView(
        pending=_actions(pending, "pending"),
        blocked=_actions(blocked, "blocked"),
        running=_actions(running, "running"),
        completed=_actions(completed, "completed"),
    )


# =============================================================================
# TestDeriveStatusPill
# =============================================================================


class TestDeriveStatusPill:
    """Tests for derive_status_pill."""

    def test_ok_when_empty(self) -> None:
        vm = _make_vm()
        assert derive_status_pill(vm) == "ok"

    def test_ok_with_elastic_regime(self) -> None:
        vm = _make_vm(regime=_make_regime("ELASTIC"))
        assert derive_status_pill(vm) == "ok"

    def test_blocked_unstable_regime(self) -> None:
        vm = _make_vm(regime=_make_regime("UNSTABLE"))
        assert derive_status_pill(vm) == "blocked"

    def test_blocked_ductile_regime(self) -> None:
        vm = _make_vm(regime=_make_regime("DUCTILE"))
        assert derive_status_pill(vm) == "blocked"

    def test_blocked_quarantine_drift(self) -> None:
        vm = _make_vm(stability=_make_stability("QUARANTINE"))
        assert derive_status_pill(vm) == "blocked"

    def test_blocked_warn_drift(self) -> None:
        vm = _make_vm(stability=_make_stability("WARN"))
        assert derive_status_pill(vm) == "blocked"

    def test_blocked_critical_violation(self) -> None:
        vm = _make_vm(violations=[_make_violation("critical")])
        assert derive_status_pill(vm) == "blocked"

    def test_blocked_execution_items(self) -> None:
        vm = _make_vm(execution=_make_execution(blocked=2))
        assert derive_status_pill(vm) == "blocked"

    def test_needs_attention_warm_regime(self) -> None:
        vm = _make_vm(regime=_make_regime("WARM"))
        assert derive_status_pill(vm) == "needs_attention"

    def test_needs_attention_watch_drift(self) -> None:
        vm = _make_vm(stability=_make_stability("WATCH"))
        assert derive_status_pill(vm) == "needs_attention"

    def test_needs_attention_medium_violation(self) -> None:
        vm = _make_vm(violations=[_make_violation("medium")])
        assert derive_status_pill(vm) == "needs_attention"

    def test_needs_attention_recent_rejection(self) -> None:
        vm = _make_vm(decisions=[_make_decision("rejected")])
        assert derive_status_pill(vm) == "needs_attention"

    def test_resolved_violation_not_blocked(self) -> None:
        vm = _make_vm(violations=[_make_violation("critical", resolution="resolved")])
        assert derive_status_pill(vm) == "ok"

    def test_blocked_takes_priority_over_attention(self) -> None:
        """Blocked wins when both blocked and attention signals present."""
        vm = _make_vm(
            regime=_make_regime("UNSTABLE"),
            decisions=[_make_decision("rejected")],
        )
        assert derive_status_pill(vm) == "blocked"


# =============================================================================
# TestDeriveOneSentence
# =============================================================================


class TestDeriveOneSentence:
    """Tests for derive_one_sentence."""

    def test_ok_sentence(self) -> None:
        vm = _make_vm()
        assert derive_one_sentence(vm) == "OK: consistent across last check."

    def test_blocked_regime(self) -> None:
        vm = _make_vm(regime=_make_regime("UNSTABLE"))
        result = derive_one_sentence(vm)
        assert result.startswith("Blocked:")
        assert "unstable" in result

    def test_blocked_drift(self) -> None:
        vm = _make_vm(stability=_make_stability("QUARANTINE"))
        result = derive_one_sentence(vm)
        assert result.startswith("Blocked:")
        assert "quarantine" in result

    def test_blocked_violations(self) -> None:
        vm = _make_vm(violations=[
            _make_violation("high"),
            _make_violation("high", id="vio_test_002"),
        ])
        result = derive_one_sentence(vm)
        assert result.startswith("Blocked:")
        assert "2 unresolved violations" in result

    def test_blocked_single_violation(self) -> None:
        vm = _make_vm(violations=[_make_violation("critical")])
        result = derive_one_sentence(vm)
        assert "1 unresolved violation" in result

    def test_needs_attention_warm(self) -> None:
        vm = _make_vm(regime=_make_regime("WARM"))
        result = derive_one_sentence(vm)
        assert result.startswith("Needs attention:")
        assert "warming" in result

    def test_needs_attention_rejection(self) -> None:
        vm = _make_vm(decisions=[_make_decision("rejected")])
        result = derive_one_sentence(vm)
        assert "rejection" in result

    def test_sentence_length_reasonable(self) -> None:
        vm = _make_vm()
        result = derive_one_sentence(vm)
        assert len(result) < 200


# =============================================================================
# TestDeriveSuggestedAction
# =============================================================================


class TestDeriveSuggestedAction:
    """Tests for derive_suggested_action."""

    def test_none_when_ok(self) -> None:
        vm = _make_vm()
        assert derive_suggested_action(vm) is None

    def test_grounding_suggestion(self) -> None:
        vm = _make_vm(violations=[_make_violation("high", rule_breached="grounding_failure")])
        result = derive_suggested_action(vm)
        assert result is not None
        assert "source" in result.lower() or "ungrounded" in result.lower()

    def test_premise_suggestion(self) -> None:
        vm = _make_vm(violations=[_make_violation("high", rule_breached="premise_quarantine")])
        result = derive_suggested_action(vm)
        assert result is not None
        assert "premise" in result.lower()

    def test_highest_severity_first(self) -> None:
        vm = _make_vm(violations=[
            _make_violation("low", id="vio_low", rule_breached="minor_issue"),
            _make_violation("critical", id="vio_crit", rule_breached="grounding_failure"),
        ])
        result = derive_suggested_action(vm)
        assert result is not None
        assert "source" in result.lower() or "ungrounded" in result.lower()

    def test_regime_suggestion(self) -> None:
        vm = _make_vm(regime=_make_regime("UNSTABLE"))
        result = derive_suggested_action(vm)
        assert result is not None
        assert "stabilize" in result.lower()

    def test_rejection_suggestion(self) -> None:
        vm = _make_vm(decisions=[_make_decision("rejected")])
        result = derive_suggested_action(vm)
        assert result is not None
        assert "rejected" in result.lower() or "evidence" in result.lower()


# =============================================================================
# TestDeriveLastEvent
# =============================================================================


class TestDeriveLastEvent:
    """Tests for derive_last_event."""

    def test_none_when_empty(self) -> None:
        vm = _make_vm()
        assert derive_last_event(vm) is None

    def test_returns_most_recent_decision(self) -> None:
        now = datetime.now(timezone.utc)
        old = (now - timedelta(hours=2)).isoformat()
        recent = (now - timedelta(minutes=5)).isoformat()
        vm = _make_vm(decisions=[
            _make_decision("accepted", id="dec_old", created_at=old),
            _make_decision("rejected", id="dec_new", created_at=recent),
        ])
        result = derive_last_event(vm)
        assert result is not None
        assert "summary" in result
        assert "when" in result
        assert "Rejected" in result["summary"]

    def test_returns_violation_event(self) -> None:
        vm = _make_vm(violations=[_make_violation("high")])
        result = derive_last_event(vm)
        assert result is not None
        assert "Blocked" in result["summary"]

    def test_when_is_relative(self) -> None:
        recent = (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat()
        vm = _make_vm(decisions=[_make_decision("accepted", created_at=recent)])
        result = derive_last_event(vm)
        assert result is not None
        assert "min ago" in result["when"]


# =============================================================================
# TestDeriveWhyFeed
# =============================================================================


class TestDeriveWhyFeed:
    """Tests for derive_why_feed."""

    def test_empty_when_no_items(self) -> None:
        vm = _make_vm()
        assert derive_why_feed(vm) == []

    def test_accepted_is_info(self) -> None:
        vm = _make_vm(decisions=[_make_decision("accepted")])
        feed = derive_why_feed(vm)
        assert len(feed) == 1
        assert feed[0]["severity"] == "info"
        assert feed[0]["type"] == "decision"

    def test_rejected_is_error(self) -> None:
        vm = _make_vm(decisions=[_make_decision("rejected")])
        feed = derive_why_feed(vm)
        assert len(feed) == 1
        assert feed[0]["severity"] == "error"

    def test_violation_severity_mapping(self) -> None:
        vm = _make_vm(violations=[_make_violation("critical")])
        feed = derive_why_feed(vm)
        assert len(feed) == 1
        assert feed[0]["severity"] == "error"
        assert feed[0]["type"] == "violation"

    def test_medium_violation_is_warning(self) -> None:
        vm = _make_vm(violations=[_make_violation("medium")])
        feed = derive_why_feed(vm)
        assert feed[0]["severity"] == "warning"

    def test_contradicted_claim_appears(self) -> None:
        vm = _make_vm(claims=[_make_claim("contradicted")])
        feed = derive_why_feed(vm)
        assert len(feed) == 1
        assert feed[0]["type"] == "claim"
        assert "Contradicted" in feed[0]["title"]

    def test_stale_claim_appears(self) -> None:
        vm = _make_vm(claims=[_make_claim("stale")])
        feed = derive_why_feed(vm)
        assert len(feed) == 1
        assert "Stale" in feed[0]["title"]

    def test_proposed_claim_excluded(self) -> None:
        """Only contradicted/stale claims appear in feed."""
        vm = _make_vm(claims=[_make_claim("proposed")])
        assert derive_why_feed(vm) == []

    def test_stabilized_claim_excluded(self) -> None:
        vm = _make_vm(claims=[_make_claim("stabilized")])
        assert derive_why_feed(vm) == []

    def test_limit_parameter(self) -> None:
        decisions = [_make_decision("accepted", id=f"dec_{i}") for i in range(10)]
        vm = _make_vm(decisions=decisions)
        feed = derive_why_feed(vm, limit=3)
        assert len(feed) == 3

    def test_severity_filter(self) -> None:
        vm = _make_vm(decisions=[
            _make_decision("accepted", id="dec_a"),
            _make_decision("rejected", id="dec_b"),
        ])
        feed = derive_why_feed(vm, severity_filter="error")
        assert len(feed) == 1
        assert feed[0]["severity"] == "error"

    def test_detail_contains_full_dict(self) -> None:
        vm = _make_vm(decisions=[_make_decision("accepted")])
        feed = derive_why_feed(vm)
        assert "detail" in feed[0]
        assert isinstance(feed[0]["detail"], dict)
        assert "id" in feed[0]["detail"]

    def test_sorted_by_created_at_desc(self) -> None:
        now = datetime.now(timezone.utc)
        old = (now - timedelta(hours=5)).isoformat()
        new = (now - timedelta(minutes=1)).isoformat()
        vm = _make_vm(decisions=[
            _make_decision("accepted", id="dec_old", created_at=old),
            _make_decision("rejected", id="dec_new", created_at=new),
        ])
        feed = derive_why_feed(vm)
        assert feed[0]["id"] == "dec_new"
        assert feed[1]["id"] == "dec_old"


# =============================================================================
# TestDeriveHistoryDays
# =============================================================================


class TestDeriveHistoryDays:
    """Tests for derive_history_days."""

    def test_empty_when_no_items(self) -> None:
        vm = _make_vm()
        assert derive_history_days(vm) == []

    def test_groups_by_date(self) -> None:
        now = datetime.now(timezone.utc)
        today = now.isoformat()
        yesterday = (now - timedelta(days=1)).isoformat()
        vm = _make_vm(decisions=[
            _make_decision("accepted", id="dec_today", created_at=today),
            _make_decision("rejected", id="dec_yest", created_at=yesterday),
        ])
        result = derive_history_days(vm)
        assert len(result) == 2
        # First day should be most recent
        assert result[0]["date"] > result[1]["date"]

    def test_counts_rejections(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        vm = _make_vm(decisions=[
            _make_decision("accepted", id="dec_1", created_at=now),
            _make_decision("rejected", id="dec_2", created_at=now),
            _make_decision("rejected", id="dec_3", created_at=now),
        ])
        result = derive_history_days(vm)
        assert len(result) == 1
        assert result[0]["items_count"] == 3
        assert result[0]["rejection_count"] == 2

    def test_days_limit(self) -> None:
        now = datetime.now(timezone.utc)
        decisions = []
        for i in range(10):
            d = (now - timedelta(days=i)).isoformat()
            decisions.append(_make_decision("accepted", id=f"dec_{i}", created_at=d))
        vm = _make_vm(decisions=decisions)
        result = derive_history_days(vm, days=3)
        assert len(result) == 3

    def test_outcomes_list(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        vm = _make_vm(decisions=[
            _make_decision("accepted", id="dec_a", created_at=now),
            _make_decision("rejected", id="dec_b", created_at=now),
        ])
        result = derive_history_days(vm)
        assert "accepted" in result[0]["outcomes"]
        assert "rejected" in result[0]["outcomes"]

    def test_items_in_day(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        vm = _make_vm(decisions=[_make_decision("accepted", created_at=now)])
        result = derive_history_days(vm)
        assert len(result[0]["items"]) == 1
        assert result[0]["items"][0]["type"] == "decision"


# =============================================================================
# TestRefereeVoiceHelpers
# =============================================================================


class TestRefereeVoiceHelpers:
    """Tests for _referee_voice_decision, _referee_voice_violation, _relative_time."""

    def test_decision_rejected_voice(self) -> None:
        dec = _make_decision("rejected", type="TESTS_PASS")
        title, summary = _referee_voice_decision(dec)
        assert title.startswith("Rejected:")

    def test_decision_accepted_voice(self) -> None:
        dec = _make_decision("accepted", type="FILE_EXISTS", rationale="File verified.")
        title, summary = _referee_voice_decision(dec)
        assert title.startswith("Accepted:")
        assert summary == "File verified."

    def test_decision_pending_voice(self) -> None:
        dec = _make_decision("pending")
        title, summary = _referee_voice_decision(dec)
        assert title.startswith("Pending:")

    def test_violation_voice(self) -> None:
        vio = _make_violation("high", detail="No evidence found.")
        title, summary = _referee_voice_violation(vio)
        assert title.startswith("Blocked:")
        assert summary == "No evidence found."

    def test_violation_resolved_voice(self) -> None:
        vio = _make_violation("high", resolution="resolved")
        title, summary = _referee_voice_violation(vio)
        assert title.startswith("Resolved:")

    def test_relative_time_just_now(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        assert _relative_time(now) == "just now"

    def test_relative_time_minutes(self) -> None:
        ts = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        assert "min ago" in _relative_time(ts)

    def test_relative_time_hours(self) -> None:
        ts = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        result = _relative_time(ts)
        assert "hour" in result

    def test_relative_time_yesterday(self) -> None:
        ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        assert _relative_time(ts) == "yesterday"

    def test_relative_time_days(self) -> None:
        ts = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        assert "days ago" in _relative_time(ts)

    def test_relative_time_invalid(self) -> None:
        assert _relative_time("not-a-date") == "unknown"
