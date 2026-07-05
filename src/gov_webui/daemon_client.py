# SPDX-License-Identifier: Apache-2.0
"""Thin JSON-RPC 2.0 client over Unix socket for the governor daemon.

Provides only the chat-path methods needed by the webui adapter.
Non-chat endpoints (sessions, governor/status, etc.) stay as direct imports.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

logger = logging.getLogger(__name__)

AUTH_ERROR_CODE = -32001


class DaemonAuthError(RuntimeError):
    """Raised when the daemon reports a backend authentication failure.

    This typically means the Claude Code CLI is logged out and the user
    needs to run `claude /login` to re-authenticate.
    """


# =============================================================================
# Content-Length framing (same protocol as daemon / Maude rpc.py)
# =============================================================================


async def _read_message(reader: asyncio.StreamReader) -> dict | None:
    """Read a Content-Length framed JSON-RPC message."""
    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        if not line:
            return None  # EOF
        decoded = line.decode("utf-8")
        if decoded in ("\r\n", "\n"):
            break
        if ":" in decoded:
            key, _, value = decoded.partition(":")
            headers[key.strip()] = value.strip()

    content_length_str = headers.get("Content-Length")
    if content_length_str is None:
        return None

    content_length = int(content_length_str)
    body = await reader.readexactly(content_length)
    return json.loads(body.decode("utf-8"))


async def _write_message(writer: asyncio.StreamWriter, msg: dict) -> None:
    """Write a Content-Length framed JSON-RPC message."""
    json_bytes = json.dumps(msg).encode("utf-8")
    header = f"Content-Length: {len(json_bytes)}\r\n\r\n".encode("utf-8")
    writer.write(header + json_bytes)
    await writer.drain()


# =============================================================================
# Socket path resolution
# =============================================================================


def default_socket_path(governor_dir: Path) -> Path:
    """Compute the default Unix socket path for a governor directory.

    Same algorithm as governor.daemon.default_socket_path.
    """
    xdg = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
    dir_hash = hashlib.sha256(str(governor_dir.resolve()).encode()).hexdigest()[:12]
    return Path(xdg) / f"governor-{dir_hash}.sock"


# =============================================================================
# DaemonChatClient — chat-path only
# =============================================================================


class DaemonChatClient:
    """JSON-RPC 2.0 client for daemon chat methods over Unix socket.

    Only wraps chat.send, chat.stream, and commit.pending — the methods
    needed to replace the webui's direct ChatBridge usage.
    """

    def __init__(self, socket_path: str | Path) -> None:
        self._socket_path = Path(socket_path)
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._request_id: int = 0

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    async def connect(self) -> None:
        """Open the Unix socket connection."""
        if self._writer is not None and not self._writer.is_closing():
            return  # Already connected
        self._reader, self._writer = await asyncio.open_unix_connection(
            str(self._socket_path)
        )

    async def close(self) -> None:
        """Close the connection."""
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _call(self, method: str, params: dict | None = None) -> Any:
        """Send a JSON-RPC request and return the result."""
        await self.connect()
        assert self._reader is not None and self._writer is not None

        request_id = self._next_id()
        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "id": request_id,
            "params": params or {},
        }
        await _write_message(self._writer, msg)

        # Read responses, skipping notifications until we get our response
        while True:
            resp = await _read_message(self._reader)
            if resp is None:
                raise ConnectionError("Connection closed by daemon")

            # Skip notifications (no id)
            if "id" not in resp:
                continue

            if resp.get("id") != request_id:
                continue

            if "error" in resp:
                err = resp["error"]
                code = err.get("code", 0)
                message = err.get("message", "unknown error")
                if code == AUTH_ERROR_CODE:
                    raise DaemonAuthError(message)
                raise RuntimeError(f"RPC error {code}: {message}")
            return resp.get("result")

    # ========================================================================
    # Chat methods
    # ========================================================================

    async def chat_send(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        context_id: str = "default",
    ) -> dict:
        """Non-streaming governed chat. Returns daemon result dict.

        Result shape: {content, model, usage, violations, footer, pending}
        """
        return await self._call(
            "chat.send",
            {"messages": messages, "model": model, "context_id": context_id},
        )

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        context_id: str = "default",
    ) -> AsyncIterator[tuple[str | None, dict | None]]:
        """Streaming governed chat via daemon.

        Yields (delta_content, None) for each chunk, then
        (None, final_result) when the stream completes.

        The final_result has the same shape as chat_send's return value.
        """
        await self.connect()
        assert self._reader is not None and self._writer is not None

        request_id = self._next_id()
        msg = {
            "jsonrpc": "2.0",
            "method": "chat.stream",
            "id": request_id,
            "params": {"messages": messages, "model": model, "context_id": context_id},
        }
        await _write_message(self._writer, msg)

        while True:
            resp = await _read_message(self._reader)
            if resp is None:
                raise ConnectionError("Connection closed by daemon")

            # Notification — yield delta content
            if "id" not in resp:
                if resp.get("method") == "chat.delta":
                    content = resp.get("params", {}).get("content", "")
                    if content:
                        yield (content, None)
                continue

            # Final response
            if resp.get("id") == request_id:
                if "error" in resp:
                    err = resp["error"]
                    code = err.get("code", 0)
                    message = err.get("message", "unknown error")
                    if code == AUTH_ERROR_CODE:
                        raise DaemonAuthError(message)
                    raise RuntimeError(f"RPC error {code}: {message}")
                yield (None, resp.get("result"))
                return

    async def commit_pending(self) -> dict | None:
        """Check for pre-existing pending violation."""
        return await self._call("commit.pending")

    async def chat_models(self) -> list[dict[str, str]]:
        """List available models from the backend."""
        result = await self._call("chat.models")
        return result.get("models", [])

    async def chat_backend(self) -> dict:
        """Get current backend info."""
        return await self._call("chat.backend")


# =============================================================================
# Vendored shell-contract typed models
#
# SOURCE: agent_gov/libs/ag_shell_client/src/ag_shell_client/models.py
# VERSION: ag_shell_client 0.1.0 (2026-07-05)
# DRIFT RISK: any schema change in DecisionItem/DecisionOption or the
#   KNOWN_DECISION_KINDS/KNOWN_URGENCIES sets must be back-ported here manually.
#   The canonical definitions live in libs/ag_shell_client; this file must not
#   diverge in serialisation semantics (from_dict tolerance, safe-defaults idiom).
#
# Reason for vendoring: ag-shell-client is an in-repo library under agent_gov
# and is not published to PyPI. gov-webui's pyproject.toml does not (and
# cannot portably) declare a path-dependency on a sibling repo. Vendoring the
# minimal model surface keeps the install self-contained.
# =============================================================================

from dataclasses import dataclass as _dataclass
from dataclasses import field as _field

# Closed decision-kind vocabulary (shell contract v0 §1).
KNOWN_DECISION_KINDS: frozenset[str] = frozenset(
    {
        "intervention",
        "violation",
        "promotion",
        "docket_case",
        "admissibility_question",
        "operator_question",
    }
)

KNOWN_URGENCIES: frozenset[str] = frozenset({"blocking", "expiring", "normal", "info"})


def _as_dict(value: Any) -> dict:
    """A mapping field off the wire, or {} for a non-mapping value.

    Safe-defaults: a malformed payload degrades to empty, never crashes.
    """
    return dict(value) if isinstance(value, dict) else {}


@_dataclass(frozen=True)
class DecisionOption:
    """One selectable option on a decision item."""

    key: str
    label: str
    action: str
    args_schema: dict | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionOption":
        return cls(
            key=str(d.get("key", "")),
            label=str(d.get("label", "")),
            action=str(d.get("action", "")),
            args_schema=d.get("args_schema"),
        )


@_dataclass(frozen=True)
class DecisionItem:
    """One item from the operator decision feed."""

    decision_id: str
    kind: str
    session_ref: str | None
    created_at: str
    urgency: str
    summary: str
    timeout_at: str | None = None
    detail: dict = _field(default_factory=dict)
    options: tuple[DecisionOption, ...] = ()
    receipt_refs: tuple[str, ...] = ()
    why_ref: str | None = None
    refs: tuple[Any, ...] = ()
    source: dict = _field(default_factory=dict)

    @property
    def is_known_kind(self) -> bool:
        """False for a kind outside the contract's closed set."""
        return self.kind in KNOWN_DECISION_KINDS

    @property
    def is_interrupt(self) -> bool:
        """Whether this item should interrupt (bell) vs. accumulate silently."""
        return self.urgency not in ("normal", "info")

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionItem":
        decision_id = d.get("decision_id")
        kind = d.get("kind")
        if not decision_id or not kind:
            raise ValueError(f"decision envelope missing decision_id/kind: {d!r}")
        return cls(
            decision_id=str(decision_id),
            kind=str(kind),
            session_ref=d.get("session_ref"),
            created_at=str(d.get("created_at", "")),
            urgency=str(d.get("urgency", "normal")),
            summary=str(d.get("summary", "")),
            timeout_at=d.get("timeout_at"),
            detail=_as_dict(d.get("detail")),
            options=tuple(
                DecisionOption.from_dict(o)
                for o in (d.get("options") or ())
                if isinstance(o, dict)
            ),
            receipt_refs=tuple(str(r) for r in d.get("receipt_refs") or ()),
            why_ref=d.get("why_ref"),
            refs=tuple(d.get("refs") or ()),
            source=_as_dict(d.get("source")),
        )


def decisions_from_response(result: dict) -> tuple[DecisionItem, ...]:
    """Parse an operator.decisions.list result into typed items."""
    return tuple(DecisionItem.from_dict(i) for i in result.get("items", ()))


# =============================================================================
# DaemonShellClient — shell-contract surface (U3-A)
# =============================================================================


class DaemonShellClient:
    """JSON-RPC 2.0 client for daemon shell-contract methods over Unix socket.

    Wraps the operator.decisions.* / operator.watch and runtime.* methods that
    make up the desk-mode lane.  Uses the same Content-Length framing as
    DaemonChatClient; each call opens a fresh connection so concurrent callers
    do not share a sequential socket (matches the one-in-flight-per-connection
    model documented in ag_shell_client).

    One-mutation-door discipline (GS-3): decisions_resolve passes decision_id,
    option_key, and args verbatim to the daemon — it does NOT synthesise args.
    The caller owns the args dict; the daemon routes it to the backing handler
    unchanged.

    Confirmed daemon methods (agent_gov 2.8.x, verified 2026-07-05):
        operator.decisions.list     read
        operator.decisions.resolve  mutating
        operator.watch              streaming
        runtime.session.list        read
        runtime.session.create      mutating
        runtime.session.launch      mutating
        runtime.session.kill        mutating
        runtime.intervention.list   read
        runtime.intervention.resolve mutating
        runtime.promotion.get       read
        runtime.promotion.diff      read
        runtime.promotion.resolve   mutating
    """

    def __init__(self, socket_path: str | Path) -> None:
        self._socket_path = Path(socket_path)
        self._request_id: int = 0

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _call(self, method: str, params: dict | None = None) -> Any:
        """Open a fresh connection, send one JSON-RPC request, return result."""
        reader, writer = await asyncio.open_unix_connection(str(self._socket_path))
        try:
            request_id = self._next_id()
            msg: dict = {
                "jsonrpc": "2.0",
                "method": method,
                "id": request_id,
                "params": params or {},
            }
            await _write_message(writer, msg)
            while True:
                resp = await _read_message(reader)
                if resp is None:
                    raise ConnectionError("Connection closed by daemon")
                if "id" not in resp:
                    continue  # skip notifications on a unary call
                if resp.get("id") != request_id:
                    continue
                if "error" in resp:
                    err = resp["error"]
                    code = err.get("code", 0)
                    message = err.get("message", "unknown error")
                    if code == AUTH_ERROR_CODE:
                        raise DaemonAuthError(message)
                    raise RuntimeError(f"RPC error {code}: {message}")
                return resp.get("result")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # operator.decisions.*
    # -------------------------------------------------------------------------

    async def decisions_list(self) -> tuple[DecisionItem, ...]:
        """Fetch the current operator decision feed.

        Returns typed DecisionItem tuples.  Unknown decision kinds are preserved
        (is_known_kind=False) — never dropped, never guessed.
        """
        result = await self._call("operator.decisions.list")
        if not isinstance(result, dict):
            return ()
        return decisions_from_response(result)

    async def decisions_resolve(
        self,
        decision_id: str,
        option_key: str,
        args: dict | None = None,
    ) -> dict:
        """Resolve a decision through the ONE mutation door (GS-3).

        Passes decision_id, option_key and args verbatim — no synthesised args.
        The daemon routes them to the backing handler; its receipt is the record.

        Args:
            decision_id: the DecisionItem.decision_id to resolve.
            option_key:  the DecisionOption.key of the chosen option.
            args:        optional args dict forwarded verbatim to the handler
                         (e.g. {"reason": "approved by operator"}).  The daemon
                         ignores keys the backing handler does not expect, so
                         passing extra keys is safe but wastes bytes.
        """
        return await self._call(
            "operator.decisions.resolve",
            {
                "decision_id": decision_id,
                "option_key": option_key,
                "args": args or {},
            },
        )

    async def watch(
        self,
        *,
        interval_ms: int = 1000,
        max_ticks: int = 30,
        kinds: list[str] | None = None,
    ) -> AsyncIterator[dict]:
        """Stream the operator decision feed as an async generator.

        Yields each operator.watch.update notification payload dict.  The
        daemon returns a summary when the bounded poll loop ends; the generator
        exits after the final result frame.  Re-subscribe to get the next batch.

        Uses a dedicated fresh connection (streaming holds the socket).

        Params match the daemon (all clamped server-side):
            interval_ms: poll cadence, clamped [200, 10000], default 1000.
            max_ticks:   clamped [1, 600], default 30.
            kinds:       optional list to filter by decision kind.
        """
        params: dict = {"interval_ms": interval_ms, "max_ticks": max_ticks}
        if kinds is not None:
            params["kinds"] = kinds

        reader, writer = await asyncio.open_unix_connection(str(self._socket_path))
        try:
            request_id = self._next_id()
            msg: dict = {
                "jsonrpc": "2.0",
                "method": "operator.watch",
                "id": request_id,
                "params": params,
            }
            await _write_message(writer, msg)
            while True:
                resp = await _read_message(reader)
                if resp is None:
                    return  # daemon closed connection — stream ends
                if "id" not in resp:
                    # Notification frame — yield the payload if it's a watch update
                    if resp.get("method") == "operator.watch.update":
                        yield resp.get("params", {})
                    continue
                # Final response — stream complete
                if resp.get("id") == request_id:
                    return
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # runtime.session.*
    # -------------------------------------------------------------------------

    async def session_list(self) -> list[dict]:
        """List all supervised sessions."""
        result = await self._call("runtime.session.list")
        if isinstance(result, dict):
            return result.get("sessions", [])
        return result if isinstance(result, list) else []

    async def session_create(self, params: dict) -> dict:
        """Create a new supervised session.

        Params forwarded verbatim (backend, cwd, task, mode, allow_dirty, etc.)
        — see daemon runtime.session.create for the full schema.
        """
        return await self._call("runtime.session.create", params)

    async def session_launch(self, params: dict) -> dict:
        """Launch a supervised session (create + attach in one call).

        Params forwarded verbatim.
        """
        return await self._call("runtime.session.launch", params)

    async def session_kill(self, session_id: str) -> dict:
        """Terminate a supervised session."""
        return await self._call("runtime.session.kill", {"session_id": session_id})

    # -------------------------------------------------------------------------
    # runtime.intervention.*
    # -------------------------------------------------------------------------

    async def intervention_list(self, session_id: str) -> list[dict]:
        """List pending tool-call approvals for a session."""
        result = await self._call(
            "runtime.intervention.list", {"session_id": session_id}
        )
        if isinstance(result, dict):
            return result.get("interventions", [])
        return result if isinstance(result, list) else []

    async def intervention_resolve(
        self,
        session_id: str,
        tool_call_id: str,
        decision: str,
        reason: str | None = None,
    ) -> dict:
        """Approve or deny a tool call.

        decision: "approve" | "deny"
        """
        params: dict = {
            "session_id": session_id,
            "tool_call_id": tool_call_id,
            "decision": decision,
        }
        if reason is not None:
            params["reason"] = reason
        return await self._call("runtime.intervention.resolve", params)

    # -------------------------------------------------------------------------
    # runtime.promotion.*
    # -------------------------------------------------------------------------

    async def promotion_get(self, session_id: str) -> dict | None:
        """Get the pending workspace promotion for a session (None if absent)."""
        return await self._call("runtime.promotion.get", {"session_id": session_id})

    async def promotion_diff(self, session_id: str) -> dict:
        """Get the unified diff of pending workspace changes."""
        return await self._call("runtime.promotion.diff", {"session_id": session_id})

    async def promotion_resolve(
        self,
        session_id: str,
        decision: str,
        reason: str | None = None,
    ) -> dict:
        """Accept or reject pending workspace changes.

        decision: "approve" | "reject"
        """
        params: dict = {"session_id": session_id, "decision": decision}
        if reason is not None:
            params["reason"] = reason
        return await self._call("runtime.promotion.resolve", params)
