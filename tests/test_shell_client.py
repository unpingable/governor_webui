# SPDX-License-Identifier: Apache-2.0
"""Tests for DaemonShellClient and the vendored shell-contract typed models.

Two flavours:
- Unit tests: in-process fake Unix-socket server (asyncio start_unix_server)
  serving scripted JSON-RPC responses without a real daemon.
- Live-daemon smoke: marked ``pytest.mark.skip`` by default — run manually
  with ``pytest tests/test_shell_client.py -m live_daemon`` when a real daemon
  socket is available.

Pattern mirrors the ag_shell_client/tests/test_client.py convention.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import tempfile
from typing import AsyncIterator

import pytest

from gov_webui.daemon_client import (
    AUTH_ERROR_CODE,
    KNOWN_DECISION_KINDS,
    KNOWN_URGENCIES,
    DaemonAuthError,
    DaemonShellClient,
    DecisionItem,
    DecisionOption,
    decisions_from_response,
)


# =============================================================================
# Helpers — framing (same as daemon / DaemonChatClient)
# =============================================================================


def _frame(msg: dict) -> bytes:
    body = json.dumps(msg).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode() + body


def _result(request_id: int, result: object) -> bytes:
    return _frame({"jsonrpc": "2.0", "id": request_id, "result": result})


def _error(request_id: int, code: int, message: str) -> bytes:
    return _frame({
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    })


def _notification(method: str, params: dict) -> bytes:
    return _frame({"jsonrpc": "2.0", "method": method, "params": params})


# =============================================================================
# Fake Unix-socket server
# =============================================================================


class _FakeServer:
    """In-process fake daemon: handles one connection, sends scripted frames."""

    def __init__(self, handler) -> None:
        self._handler = handler
        self._server: asyncio.AbstractServer | None = None
        self._sock_path: str = ""

    async def start(self, sock_path: str) -> None:
        self._sock_path = sock_path
        self._server = await asyncio.start_unix_server(self._handler, path=sock_path)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()


async def _run_with_fake(
    sock_path: str,
    scripted_handler,
    body,
):
    """Run ``body(client)`` against a fake server using ``scripted_handler``."""
    server = _FakeServer(scripted_handler)
    await server.start(sock_path)
    try:
        client = DaemonShellClient(sock_path)
        return await body(client)
    finally:
        await server.stop()


def _parse_request(data: bytes) -> dict:
    """Parse the first framed JSON-RPC message out of raw bytes."""
    # Header ends at \r\n\r\n
    header_end = data.index(b"\r\n\r\n")
    header = data[:header_end].decode()
    length = 0
    for line in header.split("\r\n"):
        if line.lower().startswith("content-length:"):
            length = int(line.split(":", 1)[1].strip())
    body_start = header_end + 4
    return json.loads(data[body_start:body_start + length])


# =============================================================================
# Typed model tests (no I/O)
# =============================================================================


class TestDecisionOption:
    def test_from_dict_basic(self):
        d = {"key": "approve", "label": "Approve", "action": "approve"}
        opt = DecisionOption.from_dict(d)
        assert opt.key == "approve"
        assert opt.label == "Approve"
        assert opt.action == "approve"
        assert opt.args_schema is None

    def test_from_dict_with_args_schema(self):
        d = {"key": "fix", "label": "Fix", "action": "fix",
             "args_schema": {"type": "object"}}
        opt = DecisionOption.from_dict(d)
        assert opt.args_schema == {"type": "object"}

    def test_from_dict_tolerates_missing_fields(self):
        opt = DecisionOption.from_dict({})
        assert opt.key == ""
        assert opt.action == ""

    def test_frozen(self):
        opt = DecisionOption.from_dict({"key": "k", "label": "L", "action": "a"})
        with pytest.raises((AttributeError, TypeError)):
            opt.key = "x"  # type: ignore[misc]


class TestDecisionItem:
    def _minimal(self) -> dict:
        return {
            "decision_id": "d-001",
            "kind": "intervention",
            "session_ref": "sess-1",
            "created_at": "2026-07-05T12:00:00",
            "urgency": "blocking",
            "summary": "Approve tool call",
        }

    def test_from_dict_basic(self):
        item = DecisionItem.from_dict(self._minimal())
        assert item.decision_id == "d-001"
        assert item.kind == "intervention"
        assert item.is_known_kind is True
        assert item.is_interrupt is True

    def test_from_dict_normal_urgency_not_interrupt(self):
        d = {**self._minimal(), "urgency": "normal"}
        item = DecisionItem.from_dict(d)
        assert item.is_interrupt is False

    def test_from_dict_unknown_kind_preserved(self):
        d = {**self._minimal(), "kind": "future_kind"}
        item = DecisionItem.from_dict(d)
        assert item.is_known_kind is False
        assert item.kind == "future_kind"  # preserved, not dropped

    def test_from_dict_unknown_urgency_is_interrupt(self):
        d = {**self._minimal(), "urgency": "super_urgent_unknown"}
        item = DecisionItem.from_dict(d)
        assert item.is_interrupt is True  # conservative: unknown → interrupt

    def test_from_dict_requires_decision_id_and_kind(self):
        with pytest.raises(ValueError):
            DecisionItem.from_dict({"kind": "intervention"})  # missing decision_id
        with pytest.raises(ValueError):
            DecisionItem.from_dict({"decision_id": "d-1"})  # missing kind

    def test_from_dict_options_parsed(self):
        d = {
            **self._minimal(),
            "options": [
                {"key": "approve", "label": "Approve", "action": "approve"},
                {"key": "deny", "label": "Deny", "action": "deny"},
            ],
        }
        item = DecisionItem.from_dict(d)
        assert len(item.options) == 2
        assert item.options[0].key == "approve"
        assert item.options[1].key == "deny"

    def test_from_dict_tolerates_malformed_detail(self):
        d = {**self._minimal(), "detail": "not-a-dict"}
        item = DecisionItem.from_dict(d)
        assert item.detail == {}  # safe-defaults

    def test_from_dict_tolerates_malformed_source(self):
        d = {**self._minimal(), "source": 42}
        item = DecisionItem.from_dict(d)
        assert item.source == {}

    def test_from_dict_ignores_unknown_fields(self):
        d = {**self._minimal(), "future_field": "ignored"}
        item = DecisionItem.from_dict(d)
        assert item.decision_id == "d-001"


class TestDecisionsFromResponse:
    def test_empty_items(self):
        assert decisions_from_response({}) == ()
        assert decisions_from_response({"items": []}) == ()

    def test_parses_items(self):
        result = {
            "items": [
                {
                    "decision_id": "d-1",
                    "kind": "violation",
                    "session_ref": None,
                    "created_at": "2026-07-05T00:00:00",
                    "urgency": "normal",
                    "summary": "Violation",
                }
            ]
        }
        items = decisions_from_response(result)
        assert len(items) == 1
        assert items[0].decision_id == "d-1"


class TestKnownVocabulary:
    def test_known_decision_kinds_are_present(self):
        for kind in ("intervention", "violation", "promotion", "docket_case"):
            assert kind in KNOWN_DECISION_KINDS

    def test_known_urgencies_are_present(self):
        for urg in ("blocking", "expiring", "normal", "info"):
            assert urg in KNOWN_URGENCIES


# =============================================================================
# DaemonShellClient unit tests — fake socket server
# =============================================================================


@pytest.fixture
def sock_path(tmp_path) -> str:
    return str(tmp_path / "test.sock")


class TestDecisionsList:
    """decisions_list() — operator.decisions.list round-trip."""

    def test_list_returns_typed_items(self, sock_path):
        async def handler(reader, writer):
            data = await reader.read(4096)
            req = _parse_request(data)
            assert req["method"] == "operator.decisions.list"
            writer.write(_result(req["id"], {
                "items": [
                    {
                        "decision_id": "d-42",
                        "kind": "intervention",
                        "session_ref": "s-1",
                        "created_at": "2026-07-05T10:00:00",
                        "urgency": "blocking",
                        "summary": "Approve write_file",
                        "options": [
                            {"key": "approve", "label": "Approve", "action": "approve"},
                            {"key": "deny", "label": "Deny", "action": "deny"},
                        ],
                    }
                ],
                "count": 1,
            }))
            await writer.drain()
            writer.close()

        async def go():
            return await _run_with_fake(sock_path, handler, lambda c: c.decisions_list())

        items = asyncio.run(go())
        assert len(items) == 1
        assert items[0].decision_id == "d-42"
        assert items[0].kind == "intervention"
        assert items[0].is_known_kind is True
        assert len(items[0].options) == 2

    def test_list_empty_feed(self, sock_path):
        async def handler(reader, writer):
            data = await reader.read(4096)
            req = _parse_request(data)
            writer.write(_result(req["id"], {"items": [], "count": 0}))
            await writer.drain()
            writer.close()

        async def go():
            return await _run_with_fake(sock_path, handler, lambda c: c.decisions_list())

        assert asyncio.run(go()) == ()

    def test_list_rpc_error_propagates(self, sock_path):
        async def handler(reader, writer):
            data = await reader.read(4096)
            req = _parse_request(data)
            writer.write(_error(req["id"], -32600, "invalid request"))
            await writer.drain()
            writer.close()

        async def go():
            return await _run_with_fake(sock_path, handler, lambda c: c.decisions_list())

        with pytest.raises(RuntimeError, match="RPC error"):
            asyncio.run(go())


class TestDecisionsResolve:
    """decisions_resolve() — operator.decisions.resolve; one-mutation-door."""

    def test_resolve_passes_args_verbatim(self, sock_path):
        """Core one-mutation-door invariant: args must arrive at the daemon unchanged."""
        received: dict = {}

        async def handler(reader, writer):
            data = await reader.read(8192)
            req = _parse_request(data)
            received.update(req)
            writer.write(_result(req["id"], {"resolved": True}))
            await writer.drain()
            writer.close()

        async def go():
            async def body(client):
                return await client.decisions_resolve(
                    "d-42",
                    "approve",
                    args={"reason": "looks good"},
                )
            return await _run_with_fake(sock_path, handler, body)

        result = asyncio.run(go())
        assert result == {"resolved": True}
        assert received["method"] == "operator.decisions.resolve"
        params = received["params"]
        assert params["decision_id"] == "d-42"
        assert params["option_key"] == "approve"
        # args forwarded verbatim — not synthesised, not rewritten
        assert params["args"] == {"reason": "looks good"}

    def test_resolve_no_args_sends_empty_dict(self, sock_path):
        """No-args call sends {} — daemon can distinguish from missing."""
        received: dict = {}

        async def handler(reader, writer):
            data = await reader.read(8192)
            req = _parse_request(data)
            received.update(req)
            writer.write(_result(req["id"], {"resolved": True}))
            await writer.drain()
            writer.close()

        async def go():
            async def body(client):
                return await client.decisions_resolve("d-1", "deny")
            return await _run_with_fake(sock_path, handler, body)

        asyncio.run(go())
        assert received["params"]["args"] == {}

    def test_resolve_error_decision_not_found(self, sock_path):
        async def handler(reader, writer):
            data = await reader.read(8192)
            req = _parse_request(data)
            # Daemon returns a closed-vocabulary error in the result, not an
            # RPC error code — per shell-contract §3.
            writer.write(_result(req["id"], {
                "resolved": False, "error": "decision_not_found"
            }))
            await writer.drain()
            writer.close()

        async def go():
            async def body(client):
                return await client.decisions_resolve("missing", "approve")
            return await _run_with_fake(sock_path, handler, body)

        result = asyncio.run(go())
        assert result["resolved"] is False
        assert result["error"] == "decision_not_found"


class TestSessionMethods:
    """runtime.session.list/create/launch/kill round-trips."""

    def test_session_list(self, sock_path):
        async def handler(reader, writer):
            data = await reader.read(4096)
            req = _parse_request(data)
            assert req["method"] == "runtime.session.list"
            writer.write(_result(req["id"], {
                "sessions": [{"session_id": "s-1", "status": "running"}]
            }))
            await writer.drain()
            writer.close()

        async def go():
            return await _run_with_fake(sock_path, handler, lambda c: c.session_list())

        sessions = asyncio.run(go())
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s-1"

    def test_session_kill_sends_session_id(self, sock_path):
        received: dict = {}

        async def handler(reader, writer):
            data = await reader.read(4096)
            req = _parse_request(data)
            received.update(req)
            writer.write(_result(req["id"], {"killed": True}))
            await writer.drain()
            writer.close()

        async def go():
            async def body(client):
                return await client.session_kill("sess-99")
            return await _run_with_fake(sock_path, handler, body)

        result = asyncio.run(go())
        assert result == {"killed": True}
        assert received["method"] == "runtime.session.kill"
        assert received["params"]["session_id"] == "sess-99"


class TestInterventionMethods:
    """runtime.intervention.list/resolve round-trips."""

    def test_intervention_list(self, sock_path):
        async def handler(reader, writer):
            data = await reader.read(4096)
            req = _parse_request(data)
            assert req["method"] == "runtime.intervention.list"
            assert req["params"]["session_id"] == "s-1"
            writer.write(_result(req["id"], {
                "interventions": [{"tool_call_id": "tc-1", "tool": "write_file"}]
            }))
            await writer.drain()
            writer.close()

        async def go():
            async def body(client):
                return await client.intervention_list("s-1")
            return await _run_with_fake(sock_path, handler, body)

        items = asyncio.run(go())
        assert len(items) == 1
        assert items[0]["tool_call_id"] == "tc-1"

    def test_intervention_resolve_approve(self, sock_path):
        received: dict = {}

        async def handler(reader, writer):
            data = await reader.read(4096)
            req = _parse_request(data)
            received.update(req)
            writer.write(_result(req["id"], {"resolved": True, "decision": "approve"}))
            await writer.drain()
            writer.close()

        async def go():
            async def body(client):
                return await client.intervention_resolve(
                    "s-1", "tc-1", "approve", reason="looks safe"
                )
            return await _run_with_fake(sock_path, handler, body)

        asyncio.run(go())
        params = received["params"]
        assert params["session_id"] == "s-1"
        assert params["tool_call_id"] == "tc-1"
        assert params["decision"] == "approve"
        assert params["reason"] == "looks safe"


class TestPromotionMethods:
    """runtime.promotion.get/diff/resolve round-trips."""

    def test_promotion_get_returns_none_when_absent(self, sock_path):
        async def handler(reader, writer):
            data = await reader.read(4096)
            req = _parse_request(data)
            writer.write(_result(req["id"], None))
            await writer.drain()
            writer.close()

        async def go():
            async def body(client):
                return await client.promotion_get("s-1")
            return await _run_with_fake(sock_path, handler, body)

        assert asyncio.run(go()) is None

    def test_promotion_resolve_approve(self, sock_path):
        received: dict = {}

        async def handler(reader, writer):
            data = await reader.read(4096)
            req = _parse_request(data)
            received.update(req)
            writer.write(_result(req["id"], {"promoted": True}))
            await writer.drain()
            writer.close()

        async def go():
            async def body(client):
                return await client.promotion_resolve("s-1", "approve", reason="LGTM")
            return await _run_with_fake(sock_path, handler, body)

        asyncio.run(go())
        params = received["params"]
        assert params["session_id"] == "s-1"
        assert params["decision"] == "approve"
        assert params["reason"] == "LGTM"


class TestWatch:
    """operator.watch streaming generator."""

    def test_watch_yields_update_notifications(self, sock_path):
        item_payload = {
            "decision_id": "d-1",
            "kind": "intervention",
            "session_ref": "s-1",
            "created_at": "2026-07-05T10:00:00",
            "urgency": "blocking",
            "summary": "Approve write_file",
        }

        async def handler(reader, writer):
            data = await reader.read(4096)
            req = _parse_request(data)
            assert req["method"] == "operator.watch"
            # Emit one update notification then the final result
            writer.write(_notification(
                "operator.watch.update",
                {"items": [item_payload], "count": 1, "tick": 0, "changed": True},
            ))
            await writer.drain()
            writer.write(_result(req["id"], {
                "ticks": 1, "updates_emitted": 1, "final_count": 1,
                "stopped_early": False,
            }))
            await writer.drain()
            writer.close()

        async def go():
            yielded = []
            async def body(client):
                async for payload in client.watch(max_ticks=1, interval_ms=200):
                    yielded.append(payload)
            await _run_with_fake(sock_path, handler, body)
            return yielded

        updates = asyncio.run(go())
        assert len(updates) == 1
        assert updates[0]["count"] == 1
        assert updates[0]["items"][0]["decision_id"] == "d-1"

    def test_watch_ignores_non_update_notifications(self, sock_path):
        async def handler(reader, writer):
            data = await reader.read(4096)
            req = _parse_request(data)
            # Emit a different notification method — should be ignored
            writer.write(_notification("some.other.event", {"x": 1}))
            await writer.drain()
            writer.write(_notification(
                "operator.watch.update",
                {"items": [], "count": 0, "tick": 0, "changed": True},
            ))
            await writer.drain()
            writer.write(_result(req["id"], {
                "ticks": 1, "updates_emitted": 1, "final_count": 0,
            }))
            await writer.drain()
            writer.close()

        async def go():
            yielded = []
            async def body(client):
                async for payload in client.watch(max_ticks=1):
                    yielded.append(payload)
            await _run_with_fake(sock_path, handler, body)
            return yielded

        updates = asyncio.run(go())
        # Only the operator.watch.update was yielded, not the other notification
        assert len(updates) == 1
        assert updates[0]["count"] == 0

    def test_watch_params_forwarded(self, sock_path):
        received: dict = {}

        async def handler(reader, writer):
            data = await reader.read(4096)
            req = _parse_request(data)
            received.update(req)
            writer.write(_result(req["id"], {"ticks": 5}))
            await writer.drain()
            writer.close()

        async def go():
            async def body(client):
                # Consume the generator to completion
                async for _ in client.watch(interval_ms=500, max_ticks=5,
                                             kinds=["intervention"]):
                    pass
            await _run_with_fake(sock_path, handler, body)

        asyncio.run(go())
        assert received["params"]["interval_ms"] == 500
        assert received["params"]["max_ticks"] == 5
        assert received["params"]["kinds"] == ["intervention"]


# =============================================================================
# Live-daemon smoke (skipped by default — requires a running daemon)
# =============================================================================


@pytest.mark.skip(
    reason=(
        "Live-daemon smoke: requires a running governor daemon. "
        "Run manually: pytest tests/test_shell_client.py::TestLiveDaemonSmoke "
        "--no-header -s"
    )
)
class TestLiveDaemonSmoke:
    """Real round-trip against a live governor daemon.

    Set GOVERNOR_SOCKET or GOVERNOR_DIR to point at the daemon's socket.
    Same skip-by-default pattern as agent_gov integration tests.
    """

    def _socket_path(self) -> str:
        sock = os.environ.get("GOVERNOR_SOCKET", "")
        if sock:
            return sock
        gov_dir_env = os.environ.get("GOVERNOR_DIR", "")
        if gov_dir_env:
            import hashlib
            gov = pathlib.Path(gov_dir_env).resolve()
            xdg = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
            digest = hashlib.sha256(str(gov).encode()).hexdigest()[:12]
            return str(pathlib.Path(xdg) / f"governor-{digest}.sock")
        raise RuntimeError(
            "Set GOVERNOR_SOCKET or GOVERNOR_DIR to point at a live daemon socket"
        )

    def test_live_decisions_list(self):
        async def go():
            client = DaemonShellClient(self._socket_path())
            return await client.decisions_list()

        items = asyncio.run(go())
        assert isinstance(items, tuple)
        # An empty feed is fine — the daemon responded without error.

    def test_live_session_list(self):
        async def go():
            client = DaemonShellClient(self._socket_path())
            return await client.session_list()

        sessions = asyncio.run(go())
        assert isinstance(sessions, list)

    def test_live_watch_one_tick(self):
        async def go():
            client = DaemonShellClient(self._socket_path())
            updates = []
            async for payload in client.watch(max_ticks=1, interval_ms=200):
                updates.append(payload)
            return updates

        updates = asyncio.run(go())
        # One or zero updates (depending on whether the feed is non-empty).
        assert isinstance(updates, list)
