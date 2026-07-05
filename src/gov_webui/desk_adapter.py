# SPDX-License-Identifier: Apache-2.0
"""Desk-mode /desk/* route group — the operator decision cockpit (U3-B).

Mounts an ``APIRouter`` (prefix ``/desk``) exposing the governed-shell
contract surface to the browser THROUGH the daemon, via ``DaemonShellClient``.

Authority discipline (why this is a separate module, and why it never imports
``governor.*``): the desk routes carry NO governance authority of their own.
Every decision the operator can take is native-backed by the daemon; these
routes are a thin, auditable transport. Keeping them out of ``adapter.py`` and
off any direct governor import keeps the authority boundary legible for the
adversarial pass. THREE routes mutate daemon state: ``resolve_decision``
(the unified decision door, live-feed re-validated), plus
``resolve_intervention`` and ``resolve_promotion`` — native-method
passthroughs at the same trust level as maude's equivalent commands (the
daemon independently validates each). All three forward through
``DaemonShellClient``. (Wording corrected per the 2026-07-05 adversarial
pass, finding F1 — a prior draft claimed a single door.)

The one-mutation-door invariant (governed-shell GS-3, shell-contract §3):
``POST /desk/decisions/{id}/resolve`` accepts only ``{action, args}``. The
requested ``action`` must be one the daemon *itself* currently lists for that
decision — the route re-fetches the LIVE feed and validates before forwarding.
A forged ``action`` (one the feed did not offer for this item) is refused 4xx
and NEVER forwarded. ``args`` is forwarded verbatim: no synthesis, no
privilege escalation, no refusal vocabulary invented here (door-level errors
reuse the daemon's own closed vocabulary: ``decision_not_found`` /
``option_not_available``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from gov_webui.daemon_client import (
    DaemonAuthError,
    DaemonShellClient,
    default_socket_path,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/desk", tags=["desk"])


# ============================================================================
# Shell client wiring (lazy, test-injectable)
# ============================================================================

# Module global so tests can inject a fake — mirrors adapter._daemon_client.
_shell_client: DaemonShellClient | None = None


def _resolve_socket_path() -> str:
    """Resolve the daemon socket path the same way the chat adapter does.

    Precedence: GOVERNOR_SOCKET > GOVERNOR_DIR > GOVERNOR_CONTEXTS_DIR +
    GOVERNOR_CONTEXT_ID. Identical algorithm to adapter._get_daemon_client so
    both clients hit the same daemon.
    """
    socket_path = os.environ.get("GOVERNOR_SOCKET", "")
    if socket_path:
        return socket_path
    gov_dir_env = os.environ.get("GOVERNOR_DIR", "")
    if gov_dir_env:
        gov_dir = Path(gov_dir_env)
    else:
        contexts_dir = os.environ.get("GOVERNOR_CONTEXTS_DIR", "")
        context_id = os.environ.get("GOVERNOR_CONTEXT_ID", "default")
        base = Path(contexts_dir) if contexts_dir else Path.home() / ".governor-contexts"
        gov_dir = base / context_id / ".governor"
    return str(default_socket_path(gov_dir))


def _get_shell_client() -> DaemonShellClient:
    """Get or create the desk-mode shell RPC client."""
    global _shell_client
    if _shell_client is None:
        _shell_client = DaemonShellClient(_resolve_socket_path())
    return _shell_client


def _reset_shell_client() -> None:
    """Drop the cached client (used by tests between reloads)."""
    global _shell_client
    _shell_client = None


# ============================================================================
# Request models
# ============================================================================


class ResolveRequest(BaseModel):
    """Body for the one mutation door. Only ``action`` + ``args`` — the
    ``decision_id`` is the path parameter, never a body field."""

    action: str
    args: dict = Field(default_factory=dict)


class InterventionResolveRequest(BaseModel):
    decision: str  # approve | deny — forwarded verbatim; daemon validates
    reason: str | None = None


class PromotionResolveRequest(BaseModel):
    decision: str  # approve | reject — forwarded verbatim; daemon validates
    reason: str | None = None


# ============================================================================
# Error translation
# ============================================================================


def _raise_transport(exc: Exception) -> None:
    """Map a shell-client transport error to an HTTP status.

    Auth failure → 401 (operator must re-login); anything else → 502 (the
    daemon is the upstream). HTTPExceptions raised by the routes themselves
    are re-raised untouched by callers before this is reached.
    """
    if isinstance(exc, DaemonAuthError):
        raise HTTPException(
            status_code=401,
            detail=(
                "Governor daemon backend is not authenticated. "
                "Run `claude /login` in a terminal to re-authenticate."
            ),
        )
    raise HTTPException(status_code=502, detail=f"Daemon error: {exc}")


# ============================================================================
# operator.decisions.* — the decision feed + the one mutation door
# ============================================================================


@router.get("/decisions")
async def list_decisions() -> dict[str, Any]:
    """The unified operator decision feed (operator.decisions.list).

    Returns typed decision items serialised for the browser. Unknown decision
    kinds are preserved with ``is_known_kind: false`` — never dropped, never
    guessed (the UI renders them raw + flagged).
    """
    client = _get_shell_client()
    try:
        items = await client.decisions_list()
    except Exception as exc:  # noqa: BLE001 — mapped to HTTP status
        _raise_transport(exc)
    return {
        "items": [
            {
                "decision_id": it.decision_id,
                "kind": it.kind,
                "is_known_kind": it.is_known_kind,
                "session_ref": it.session_ref,
                "created_at": it.created_at,
                "urgency": it.urgency,
                "is_interrupt": it.is_interrupt,
                "summary": it.summary,
                "timeout_at": it.timeout_at,
                "detail": it.detail,
                "options": [
                    {
                        "key": o.key,
                        "label": o.label,
                        "action": o.action,
                        "args_schema": o.args_schema,
                    }
                    for o in it.options
                ],
                "receipt_refs": list(it.receipt_refs),
                "why_ref": it.why_ref,
                "source": it.source,
            }
            for it in items
        ],
        "count": len(items),
    }


@router.post("/decisions/{decision_id}/resolve")
async def resolve_decision(decision_id: str, req: ResolveRequest) -> dict[str, Any]:
    """THE one mutation door (governed-shell GS-3, shell-contract §3).

    Contract: accept ``{action, args}``; the ``action`` MUST be one the daemon
    currently lists for THIS ``decision_id``. We re-fetch the live feed and
    validate before forwarding — a forged action/args pair for an item the
    feed did not offer is refused 4xx and never reaches the daemon. On a match,
    we forward the matched option's ``key`` and ``args`` verbatim through
    ``DaemonShellClient.decisions_resolve``; the daemon routes to the backing
    handler whose receipt IS the record. This route mints nothing, invents no
    refusal vocabulary, and cannot broaden ``args`` (verbatim forward).

    Defense-in-depth: the daemon re-validates option membership independently
    (daemon.py operator_decisions_resolve). This route refuses at the HTTP
    boundary so a forged action never crosses the socket.
    """
    client = _get_shell_client()

    # Re-fetch the LIVE feed — validation is against what the daemon offers
    # right now, NOT against any caller-supplied option metadata.
    try:
        items = await client.decisions_list()
    except Exception as exc:  # noqa: BLE001 — mapped to HTTP status
        _raise_transport(exc)

    item = next((it for it in items if it.decision_id == decision_id), None)
    if item is None:
        # Reuse the daemon's closed vocabulary — no new refusal string.
        raise HTTPException(status_code=404, detail="decision_not_found")

    # The requested action must match an option the daemon listed for this item.
    option = next((o for o in item.options if o.action == req.action), None)
    if option is None:
        raise HTTPException(status_code=409, detail="option_not_available")

    # Forward through the one door: matched option key + args verbatim.
    try:
        return await client.decisions_resolve(decision_id, option.key, req.args)
    except Exception as exc:  # noqa: BLE001 — mapped to HTTP status
        _raise_transport(exc)


@router.get("/watch")
async def watch_decisions(
    interval_ms: int = 1000,
    max_ticks: int = 30,
) -> StreamingResponse:
    """SSE stream wrapping the bounded ``operator.watch`` poll loop.

    Emits one ``data:`` frame per ``operator.watch.update`` notification, then
    ends when the daemon's bounded loop returns (re-subscribe for the next
    batch — the daemon clamps ``max_ticks``/``interval_ms`` server-side). On
    client disconnect the generator is cancelled and the underlying socket is
    closed by ``DaemonShellClient.watch``'s finally block.
    """
    client = _get_shell_client()

    async def _gen():
        try:
            async for payload in client.watch(
                interval_ms=interval_ms, max_ticks=max_ticks
            ):
                yield f"data: {json.dumps(payload)}\n\n"
        except asyncio.CancelledError:
            # Client disconnected — let the client.watch finally close the socket.
            raise
        except DaemonAuthError:
            yield (
                "data: "
                + json.dumps({"error": "auth_error", "type": "auth_error"})
                + "\n\n"
            )
        except Exception as exc:  # noqa: BLE001 — surfaced as an SSE error frame
            yield "data: " + json.dumps({"error": str(exc)}) + "\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


# ============================================================================
# runtime.session.* — sessions board
# ============================================================================


@router.get("/sessions")
async def list_sessions() -> dict[str, Any]:
    """List supervised sessions (runtime.session.list)."""
    client = _get_shell_client()
    try:
        sessions = await client.session_list()
    except Exception as exc:  # noqa: BLE001
        _raise_transport(exc)
    return {"sessions": sessions, "count": len(sessions)}


@router.get("/sessions/{session_id}/interventions")
async def list_interventions(session_id: str) -> dict[str, Any]:
    """List pending tool-call approvals for a session
    (runtime.intervention.list)."""
    client = _get_shell_client()
    try:
        interventions = await client.intervention_list(session_id)
    except Exception as exc:  # noqa: BLE001
        _raise_transport(exc)
    return {"interventions": interventions, "count": len(interventions)}


@router.post("/sessions/{session_id}/interventions/{tool_call_id}/resolve")
async def resolve_intervention(
    session_id: str,
    tool_call_id: str,
    req: InterventionResolveRequest,
) -> dict[str, Any]:
    """Approve or deny a pending tool call (runtime.intervention.resolve).

    ``decision`` + ``reason`` are forwarded verbatim; the daemon owns the
    approve/deny vocabulary and its receipt is the record.
    """
    client = _get_shell_client()
    try:
        return await client.intervention_resolve(
            session_id, tool_call_id, req.decision, reason=req.reason
        )
    except Exception as exc:  # noqa: BLE001
        _raise_transport(exc)


# ============================================================================
# runtime.promotion.* — promotion panel
# ============================================================================


@router.get("/sessions/{session_id}/promotion")
async def get_promotion(session_id: str) -> dict[str, Any]:
    """Pending workspace promotion for a session (runtime.promotion.get).

    Returns ``{promotion: null}`` when there is nothing pending.
    """
    client = _get_shell_client()
    try:
        promotion = await client.promotion_get(session_id)
    except Exception as exc:  # noqa: BLE001
        _raise_transport(exc)
    return {"promotion": promotion}


@router.get("/sessions/{session_id}/promotion/diff")
async def get_promotion_diff(session_id: str) -> dict[str, Any]:
    """Unified diff of pending workspace changes (runtime.promotion.diff)."""
    client = _get_shell_client()
    try:
        return await client.promotion_diff(session_id)
    except Exception as exc:  # noqa: BLE001
        _raise_transport(exc)


@router.post("/sessions/{session_id}/promotion/resolve")
async def resolve_promotion(
    session_id: str,
    req: PromotionResolveRequest,
) -> dict[str, Any]:
    """Accept or reject pending workspace changes (runtime.promotion.resolve).

    ``decision`` + ``reason`` forwarded verbatim; the daemon's receipt is the
    record.
    """
    client = _get_shell_client()
    try:
        return await client.promotion_resolve(
            session_id, req.decision, reason=req.reason
        )
    except Exception as exc:  # noqa: BLE001
        _raise_transport(exc)
