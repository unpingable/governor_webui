# SPDX-License-Identifier: Apache-2.0
"""Tests for the /desk/* route group (U3-B) and the one-mutation-door (GS-3).

Three flavours:
- Route tests with an injected fake DaemonShellClient (no real daemon): list,
  resolve happy path, forged-action refusal, sessions/promotion, SSE smoke.
- Parity tripwire: the resolve route MUST go through DaemonShellClient and the
  desk module MUST NOT import governor.* (authority stays in the daemon).
- Live-daemon integration smoke, skipped by default.

Fake-client injection mirrors tests/test_parity.py's global-reset idiom.
"""

from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient

from gov_webui.daemon_client import DaemonAuthError, DecisionItem


# ============================================================================
# Fake shell client
# ============================================================================


class FakeShellClient:
    """Records calls; returns scripted results. No socket, no daemon."""

    def __init__(self, *, items=(), sessions=(), interventions=(),
                 promotion=None, diff=None, watch_updates=()):
        self._items = tuple(items)
        self._sessions = list(sessions)
        self._interventions = list(interventions)
        self._promotion = promotion
        self._diff = diff if diff is not None else {"diff": ""}
        self._watch_updates = list(watch_updates)
        # call recorders
        self.resolve_calls: list[tuple] = []
        self.intervention_resolve_calls: list[tuple] = []
        self.promotion_resolve_calls: list[tuple] = []
        self.list_call_count = 0
        # optional error injection
        self.list_error: Exception | None = None

    async def decisions_list(self):
        self.list_call_count += 1
        if self.list_error is not None:
            raise self.list_error
        return self._items

    async def decisions_resolve(self, decision_id, option_key, args=None):
        self.resolve_calls.append((decision_id, option_key, args))
        return {"resolved": True, "decision_id": decision_id, "option_key": option_key}

    async def session_list(self):
        return self._sessions

    async def intervention_list(self, session_id):
        return self._interventions

    async def intervention_resolve(self, session_id, tool_call_id, decision, reason=None):
        self.intervention_resolve_calls.append((session_id, tool_call_id, decision, reason))
        return {"resolved": True, "decision": decision}

    async def promotion_get(self, session_id):
        return self._promotion

    async def promotion_diff(self, session_id):
        return self._diff

    async def promotion_resolve(self, session_id, decision, reason=None):
        self.promotion_resolve_calls.append((session_id, decision, reason))
        return {"promoted": decision == "approve", "decision": decision}

    async def watch(self, *, interval_ms=1000, max_ticks=30, kinds=None):
        for u in self._watch_updates:
            yield u


def _item(decision_id="d-1", kind="intervention", options=None, **extra):
    d = {
        "decision_id": decision_id,
        "kind": kind,
        "session_ref": extra.get("session_ref", "s-1"),
        "created_at": "2026-07-05T10:00:00",
        "urgency": extra.get("urgency", "blocking"),
        "summary": extra.get("summary", "Approve write_file"),
        "options": options if options is not None else [
            {"key": "approve", "label": "Approve", "action": "approve"},
            {"key": "deny", "label": "Deny", "action": "deny"},
        ],
    }
    return DecisionItem.from_dict(d)


# ============================================================================
# Fixtures — inject a fake shell client into the desk module
# ============================================================================


@pytest.fixture
def desk_env(monkeypatch):
    """Import adapter (mounts the desk router) + desk_adapter, reset globals."""
    monkeypatch.setenv("BACKEND_TYPE", "ollama")
    monkeypatch.setenv("GOVERNOR_CONTEXT_ID", "test-desk")
    monkeypatch.setenv("GOVERNOR_MODE", "general")

    import importlib
    import gov_webui.desk_adapter as desk_mod
    import gov_webui.adapter as adapter_mod
    importlib.reload(desk_mod)
    importlib.reload(adapter_mod)
    desk_mod._shell_client = None
    yield adapter_mod, desk_mod
    desk_mod._shell_client = None


@pytest.fixture
def client(desk_env):
    adapter_mod, _ = desk_env
    return TestClient(adapter_mod.app)


def _inject(desk_mod, fake):
    desk_mod._shell_client = fake


# ============================================================================
# GET /desk/decisions
# ============================================================================


class TestListDecisions:
    def test_list_returns_typed_serialised_items(self, client, desk_env):
        _, desk_mod = desk_env
        _inject(desk_mod, FakeShellClient(items=[_item()]))
        r = client.get("/desk/decisions")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1
        it = data["items"][0]
        assert it["decision_id"] == "d-1"
        assert it["is_known_kind"] is True
        assert [o["action"] for o in it["options"]] == ["approve", "deny"]

    def test_list_empty(self, client, desk_env):
        _, desk_mod = desk_env
        _inject(desk_mod, FakeShellClient(items=[]))
        r = client.get("/desk/decisions")
        assert r.status_code == 200
        assert r.json() == {"items": [], "count": 0}

    def test_unknown_kind_preserved_and_flagged(self, client, desk_env):
        _, desk_mod = desk_env
        _inject(desk_mod, FakeShellClient(items=[_item(kind="future_kind")]))
        r = client.get("/desk/decisions")
        it = r.json()["items"][0]
        assert it["kind"] == "future_kind"
        assert it["is_known_kind"] is False

    def test_transport_error_is_502(self, client, desk_env):
        _, desk_mod = desk_env
        fake = FakeShellClient()
        fake.list_error = RuntimeError("connection refused")
        _inject(desk_mod, fake)
        r = client.get("/desk/decisions")
        assert r.status_code == 502

    def test_auth_error_is_401(self, client, desk_env):
        _, desk_mod = desk_env
        fake = FakeShellClient()
        fake.list_error = DaemonAuthError("not logged in")
        _inject(desk_mod, fake)
        r = client.get("/desk/decisions")
        assert r.status_code == 401


# ============================================================================
# POST /desk/decisions/{id}/resolve — the decision door (GS-3)
# ============================================================================


class TestResolveDecision:
    def test_happy_path_forwards_matched_key_and_args_verbatim(self, client, desk_env):
        _, desk_mod = desk_env
        fake = FakeShellClient(items=[_item()])
        _inject(desk_mod, fake)
        r = client.post("/desk/decisions/d-1/resolve",
                        json={"action": "approve", "args": {"reason": "looks good"}})
        assert r.status_code == 200
        assert r.json()["resolved"] is True
        # Exactly one forward, with the matched option KEY and verbatim args.
        assert fake.resolve_calls == [("d-1", "approve", {"reason": "looks good"})]

    def test_forged_action_refused_409_and_never_forwarded(self, client, desk_env):
        """CORE INVARIANT: an action the feed did not offer is refused, never forwarded."""
        _, desk_mod = desk_env
        fake = FakeShellClient(items=[_item()])  # offers approve/deny only
        _inject(desk_mod, fake)
        r = client.post("/desk/decisions/d-1/resolve",
                        json={"action": "rm_rf_everything", "args": {"scope": "*"}})
        assert r.status_code == 409
        assert r.json()["detail"] == "option_not_available"
        # The door never forwarded the forged action to the daemon.
        assert fake.resolve_calls == []
        # The route DID re-fetch the live feed to validate.
        assert fake.list_call_count == 1

    def test_forged_decision_id_refused_404_and_never_forwarded(self, client, desk_env):
        _, desk_mod = desk_env
        fake = FakeShellClient(items=[_item(decision_id="d-1")])
        _inject(desk_mod, fake)
        r = client.post("/desk/decisions/does-not-exist/resolve",
                        json={"action": "approve"})
        assert r.status_code == 404
        assert r.json()["detail"] == "decision_not_found"
        assert fake.resolve_calls == []

    def test_no_args_forwards_empty_dict(self, client, desk_env):
        _, desk_mod = desk_env
        fake = FakeShellClient(items=[_item()])
        _inject(desk_mod, fake)
        r = client.post("/desk/decisions/d-1/resolve", json={"action": "deny"})
        assert r.status_code == 200
        assert fake.resolve_calls == [("d-1", "deny", {})]

    def test_docket_action_matched_by_action_not_key(self, client, desk_env):
        """Options may have key != action; the door matches by action, forwards key."""
        _, desk_mod = desk_env
        opts = [
            {"key": "opt-sustain", "label": "Sustain", "action": "sustain"},
            {"key": "opt-dismiss", "label": "Dismiss", "action": "dismiss"},
        ]
        fake = FakeShellClient(items=[_item(decision_id="dk-1", kind="docket_case", options=opts)])
        _inject(desk_mod, fake)
        r = client.post("/desk/decisions/dk-1/resolve",
                        json={"action": "sustain", "args": {"reason": "precedent holds"}})
        assert r.status_code == 200
        # Forwarded the KEY (opt-sustain), not the action string.
        assert fake.resolve_calls == [("dk-1", "opt-sustain", {"reason": "precedent holds"})]

    def test_body_cannot_smuggle_option_key(self, client, desk_env):
        """The route accepts only {action, args}; an extra option_key is ignored,
        and the forwarded key is always the one derived from the live feed."""
        _, desk_mod = desk_env
        fake = FakeShellClient(items=[_item()])
        _inject(desk_mod, fake)
        r = client.post("/desk/decisions/d-1/resolve",
                        json={"action": "approve", "option_key": "deny", "args": {}})
        assert r.status_code == 200
        # Derived from action=approve → key "approve"; the smuggled key is ignored.
        assert fake.resolve_calls == [("d-1", "approve", {})]


# ============================================================================
# Sessions / interventions / promotion
# ============================================================================


class TestSessions:
    def test_list_sessions(self, client, desk_env):
        _, desk_mod = desk_env
        _inject(desk_mod, FakeShellClient(sessions=[{"session_id": "s-1", "status": "running"}]))
        r = client.get("/desk/sessions")
        assert r.status_code == 200
        assert r.json()["count"] == 1

    def test_list_interventions(self, client, desk_env):
        _, desk_mod = desk_env
        _inject(desk_mod, FakeShellClient(interventions=[{"tool_call_id": "tc-1"}]))
        r = client.get("/desk/sessions/s-1/interventions")
        assert r.status_code == 200
        assert r.json()["interventions"][0]["tool_call_id"] == "tc-1"

    def test_intervention_resolve_forwards_verbatim(self, client, desk_env):
        _, desk_mod = desk_env
        fake = FakeShellClient()
        _inject(desk_mod, fake)
        r = client.post("/desk/sessions/s-1/interventions/tc-1/resolve",
                        json={"decision": "approve", "reason": "safe"})
        assert r.status_code == 200
        assert fake.intervention_resolve_calls == [("s-1", "tc-1", "approve", "safe")]


class TestPromotion:
    def test_promotion_get_none(self, client, desk_env):
        _, desk_mod = desk_env
        _inject(desk_mod, FakeShellClient(promotion=None))
        r = client.get("/desk/sessions/s-1/promotion")
        assert r.status_code == 200
        assert r.json() == {"promotion": None}

    def test_promotion_diff(self, client, desk_env):
        _, desk_mod = desk_env
        _inject(desk_mod, FakeShellClient(diff={"diff": "+a\n-b"}))
        r = client.get("/desk/sessions/s-1/promotion/diff")
        assert r.status_code == 200
        assert r.json()["diff"] == "+a\n-b"

    def test_promotion_resolve_forwards_verbatim(self, client, desk_env):
        _, desk_mod = desk_env
        fake = FakeShellClient()
        _inject(desk_mod, fake)
        r = client.post("/desk/sessions/s-1/promotion/resolve",
                        json={"decision": "reject", "reason": "not yet"})
        assert r.status_code == 200
        assert fake.promotion_resolve_calls == [("s-1", "reject", "not yet")]


# ============================================================================
# SSE /desk/watch smoke
# ============================================================================


class TestWatchSSE:
    def test_watch_streams_update_frames(self, client, desk_env):
        _, desk_mod = desk_env
        updates = [{"items": [], "count": 0, "tick": 0}]
        _inject(desk_mod, FakeShellClient(watch_updates=updates))
        r = client.get("/desk/watch")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        # The single scripted update should appear as an SSE data frame.
        assert "data:" in r.text
        assert '"count": 0' in r.text


# ============================================================================
# Parity tripwire — authority stays in the daemon
# ============================================================================


class TestParityAuthorityInDaemon:
    """The resolve route MUST go through DaemonShellClient, and the desk module
    MUST NOT import governor.* — authority lives in the daemon, not the webui."""

    def test_resolve_goes_through_shell_client(self, client, desk_env):
        """Behavioural parity: the injected shell client's resolve is what fires."""
        _, desk_mod = desk_env
        fake = FakeShellClient(items=[_item()])
        _inject(desk_mod, fake)
        client.post("/desk/decisions/d-1/resolve", json={"action": "approve"})
        # The mutation went through the shell client, not some direct path.
        assert len(fake.resolve_calls) == 1

    def test_desk_module_does_not_import_governor(self):
        """Source-level tripwire: no `import governor` / `from governor` in the
        desk route module. If this fails, authority may have leaked out of the
        daemon into the webui process."""
        import gov_webui.desk_adapter as desk_mod
        src = pathlib.Path(desk_mod.__file__).read_text(encoding="utf-8")
        offending = [
            ln.strip()
            for ln in src.splitlines()
            if ln.strip().startswith("import governor")
            or ln.strip().startswith("from governor")
        ]
        assert offending == [], f"desk_adapter imports governor.*: {offending}"


# ============================================================================
# Live-daemon integration smoke (skipped by default)
# ============================================================================


@pytest.mark.skip(
    reason=(
        "Live-daemon smoke: requires a running governor daemon. Run manually "
        "with GOVERNOR_SOCKET/GOVERNOR_DIR set and the skip removed."
    )
)
class TestLiveDaemonDeskSmoke:
    """Real round-trip against a live daemon through the HTTP surface.

    Set GOVERNOR_SOCKET (or GOVERNOR_DIR) so desk_adapter resolves the socket,
    then hit the app with a TestClient. Mirrors the skip-by-default pattern in
    test_shell_client.TestLiveDaemonSmoke.
    """

    def test_live_decisions_list(self):
        import importlib
        import gov_webui.desk_adapter as desk_mod
        import gov_webui.adapter as adapter_mod
        importlib.reload(desk_mod)
        importlib.reload(adapter_mod)
        desk_mod._shell_client = None  # use the real, env-resolved socket
        c = TestClient(adapter_mod.app)
        r = c.get("/desk/decisions")
        assert r.status_code == 200
        assert "items" in r.json()
