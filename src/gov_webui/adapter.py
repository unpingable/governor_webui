# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible API adapter with Governor integration.

Serves a self-contained chat + governor UI at the root URL (/), and exposes
an OpenAI-compatible API for external clients. Supports switchable backends
(Anthropic Claude, Ollama, Claude Code CLI, Codex CLI) with isolated governor
contexts per user/project.

Run with: uvicorn gov_webui.adapter:app --host 0.0.0.0 --port 8000

Primary UI:  http://localhost:8000  (combined chat + governor panel)
API info:    http://localhost:8000/api/info

Configuration via environment variables:
    BACKEND_TYPE        - "anthropic", "ollama", "claude-code", or "codex" (default: "ollama")
    ANTHROPIC_API_KEY   - Required when BACKEND_TYPE=anthropic
    OLLAMA_HOST         - Ollama URL (default: http://localhost:11434)
    CLAUDE_PATH         - Path to claude CLI (default: "claude") for claude-code backend
    CODEX_PATH          - Path to codex CLI (default: "codex") for codex backend
    GOVERNOR_CONTEXT_ID - Active context ID (default: "default")
    GOVERNOR_MODE       - Context mode: fiction/code/nonfiction/general (default: "general")
    GOVERNOR_CONTEXTS_DIR - Base dir for contexts (default: ~/.governor-contexts)
    GOVERNOR_AUTH_TOKEN - Bearer token for mutating endpoints (default: "" = no auth)
    GOVERNOR_BIND_HOST  - Host to bind to (default: "127.0.0.1")

The claude-code backend uses your Claude Max subscription instead of API credits.
The codex backend uses your ChatGPT subscription instead of API credits.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from governor.chat_bridge import (
    ChatBridge,
    create_backend,
)
from governor.violation_resolver import (
    ViolationResolver,
    format_violation_prompt,
)
from gov_webui.daemon_client import DaemonAuthError, DaemonChatClient, default_socket_path
from governor.context_manager import GovernorContextManager
from governor.session_store import ChatSession, SessionMessage, SessionStore
from governor.viewmodel import build_viewmodel, GovernorViewModel
from gov_webui.summaries import (
    derive_status_pill,
    derive_one_sentence,
    derive_suggested_action,
    derive_last_event,
    derive_why_feed,
    derive_history_days,
)

# ============================================================================
# Configuration from environment
# ============================================================================

BACKEND_TYPE = os.environ.get("BACKEND_TYPE", "ollama")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CLAUDE_PATH = os.environ.get("CLAUDE_PATH", "claude")  # Path to claude CLI for claude-code backend
CODEX_PATH = os.environ.get("CODEX_PATH", "codex")  # Path to codex CLI for codex backend
GOVERNOR_CONTEXT_ID = os.environ.get("GOVERNOR_CONTEXT_ID", "default")
GOVERNOR_MODE = os.environ.get("GOVERNOR_MODE", "general")
GOVERNOR_CONTEXTS_DIR = os.environ.get("GOVERNOR_CONTEXTS_DIR", "")
GOVERNOR_SHOW_OK_FOOTER = os.environ.get("GOVERNOR_SHOW_OK_FOOTER", "true").lower() in ("true", "1", "yes")

# Auth token — when set, mutating endpoints require Authorization: Bearer <token>
# When unset, all endpoints are open (dev mode).
GOVERNOR_AUTH_TOKEN = os.environ.get("GOVERNOR_AUTH_TOKEN", "")

# Host binding — default to loopback for safety; set to 0.0.0.0 explicitly if needed
GOVERNOR_BIND_HOST = os.environ.get("GOVERNOR_BIND_HOST", "127.0.0.1")

# Mutable backend type — starts from env, switchable at runtime via /v1/backends/switch
_current_backend_type: str = BACKEND_TYPE

# ============================================================================
# Application setup
# ============================================================================

app = FastAPI(
    title="Governor Chat Adapter",
    description="OpenAI-compatible API with switchable backends and Governor integration",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Auth middleware — opt-in via GOVERNOR_AUTH_TOKEN
# ============================================================================

# Methods that require auth when token is configured
_AUTH_METHODS = {"POST", "PUT", "DELETE", "PATCH"}

# Paths exempt from auth even for mutating methods (health probes, etc.)
_AUTH_EXEMPT_PATHS = {"/health", "/api/info", "/docs", "/openapi.json"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Enforce bearer token auth on mutating endpoints when GOVERNOR_AUTH_TOKEN is set."""
    if GOVERNOR_AUTH_TOKEN and request.method in _AUTH_METHODS:
        if request.url.path not in _AUTH_EXEMPT_PATHS:
            auth_header = request.headers.get("authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Authorization header required: Bearer <token>"},
                )
            provided = auth_header[7:]  # strip "Bearer "
            if provided != GOVERNOR_AUTH_TOKEN:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Invalid auth token"},
                )
    return await call_next(request)


# ============================================================================
# Pydantic Models (OpenAI API format)
# ============================================================================


class ChatMessage(BaseModel):
    role: str
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: list[str] | str | None = None
    max_tokens: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage | None = None
    receipt: dict | None = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "system"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class BackendSwitchRequest(BaseModel):
    backend_type: str


# ============================================================================
# Bridge setup (lazy init on first request)
# ============================================================================

_bridge: ChatBridge | None = None
_context_manager: GovernorContextManager | None = None
_session_store: SessionStore | None = None
_daemon_client: DaemonChatClient | None = None


def _get_daemon_client() -> DaemonChatClient:
    """Get or create the daemon RPC client for the chat path."""
    global _daemon_client
    if _daemon_client is None:
        socket_path = os.environ.get("GOVERNOR_SOCKET", "")
        if not socket_path:
            gov_dir_env = os.environ.get("GOVERNOR_DIR", "")
            if gov_dir_env:
                gov_dir = Path(gov_dir_env)
            else:
                base = Path(GOVERNOR_CONTEXTS_DIR) if GOVERNOR_CONTEXTS_DIR else Path.home() / ".governor-contexts"
                gov_dir = base / GOVERNOR_CONTEXT_ID / ".governor"
            socket_path = str(default_socket_path(gov_dir))
        _daemon_client = DaemonChatClient(socket_path)
    return _daemon_client


def _get_context_manager() -> GovernorContextManager:
    global _context_manager
    if _context_manager is None:
        base_dir = Path(GOVERNOR_CONTEXTS_DIR) if GOVERNOR_CONTEXTS_DIR else None
        _context_manager = GovernorContextManager(base_dir=base_dir)
    return _context_manager


def _get_bridge() -> ChatBridge:
    global _bridge
    if _bridge is None:
        kwargs: dict[str, Any] = {}
        if _current_backend_type == "anthropic":
            kwargs["api_key"] = ANTHROPIC_API_KEY
        elif _current_backend_type == "ollama":
            kwargs["host"] = OLLAMA_HOST
        elif _current_backend_type == "claude-code":
            kwargs["claude_path"] = CLAUDE_PATH
        elif _current_backend_type == "codex":
            kwargs["codex_path"] = CODEX_PATH
        backend = create_backend(_current_backend_type, **kwargs)
        _bridge = ChatBridge(
            backend=backend,
            context_manager=_get_context_manager(),
            show_ok_footer=GOVERNOR_SHOW_OK_FOOTER,
        )
    return _bridge


def _get_session_store() -> SessionStore:
    global _session_store
    if _session_store is None:
        cm = _get_context_manager()
        ctx = cm.get(GOVERNOR_CONTEXT_ID)
        if ctx is not None:
            sessions_dir = ctx.root / "sessions"
        else:
            # Context not created yet — compute path without writing to disk.
            # SessionStore.list_summaries() handles non-existent dir gracefully;
            # create() calls _ensure_dir() only when actually needed.
            sessions_dir = cm.base_dir / GOVERNOR_CONTEXT_ID / "sessions"
        _session_store = SessionStore(sessions_dir)
    return _session_store


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/v1/models")
async def list_models() -> ModelList:
    """List available models from the backend."""
    try:
        bridge = _get_bridge()
        models = await bridge.list_models()
        return ModelList(
            data=[
                ModelInfo(id=m["id"], owned_by=m.get("owned_by", "system"))
                for m in models
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Backend error: {e}")


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> ModelInfo:
    """Get info about a specific model."""
    return ModelInfo(id=model_id, owned_by=_current_backend_type)


# ============================================================================
# Backend Switching Endpoints
# ============================================================================


def _get_available_backends() -> list[dict[str, Any]]:
    """Return list of available backends with availability status."""
    backends = []
    # Ollama — check host reachable (best-effort: just note the host)
    backends.append({
        "type": "ollama",
        "available": True,  # Always structurally available
        "config_hint": f"OLLAMA_HOST={OLLAMA_HOST}",
    })
    # Anthropic — needs API key
    backends.append({
        "type": "anthropic",
        "available": bool(ANTHROPIC_API_KEY),
        "config_hint": "Set ANTHROPIC_API_KEY" if not ANTHROPIC_API_KEY else "API key configured",
    })
    # Claude Code — needs CLI binary
    claude_found = shutil.which(CLAUDE_PATH) is not None or Path(CLAUDE_PATH).is_file()
    backends.append({
        "type": "claude-code",
        "available": claude_found,
        "config_hint": f"CLAUDE_PATH={CLAUDE_PATH}" if claude_found else "claude CLI not found",
    })
    # Codex — needs CLI binary
    codex_found = shutil.which(CODEX_PATH) is not None or Path(CODEX_PATH).is_file()
    backends.append({
        "type": "codex",
        "available": codex_found,
        "config_hint": f"CODEX_PATH={CODEX_PATH}" if codex_found else "codex CLI not found",
    })
    return backends


@app.get("/v1/backends")
async def list_backends() -> dict[str, Any]:
    """List available backends and mark the active one."""
    backends = _get_available_backends()
    for b in backends:
        b["active"] = b["type"] == _current_backend_type
    # Check connection for active backend
    connected = False
    try:
        bridge = _get_bridge()
        await bridge.list_models()
        connected = True
    except Exception:
        pass
    return {
        "backends": backends,
        "active": _current_backend_type,
        "connected": connected,
    }


@app.post("/v1/backends/switch")
async def switch_backend(request: BackendSwitchRequest) -> dict[str, Any]:
    """Switch the active backend at runtime."""
    global _bridge, _current_backend_type

    valid_types = {"ollama", "anthropic", "claude-code", "codex"}
    if request.backend_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid backend type: {request.backend_type}. Valid: {sorted(valid_types)}",
        )

    # Build kwargs for the new backend
    kwargs: dict[str, Any] = {}
    if request.backend_type == "anthropic":
        kwargs["api_key"] = ANTHROPIC_API_KEY
    elif request.backend_type == "ollama":
        kwargs["host"] = OLLAMA_HOST
    elif request.backend_type == "claude-code":
        kwargs["claude_path"] = CLAUDE_PATH
    elif request.backend_type == "codex":
        kwargs["codex_path"] = CODEX_PATH

    try:
        new_backend = create_backend(request.backend_type, **kwargs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create backend: {e}")

    _current_backend_type = request.backend_type
    _bridge = ChatBridge(
        backend=new_backend,
        context_manager=_get_context_manager(),
        show_ok_footer=GOVERNOR_SHOW_OK_FOOTER,
    )

    # Check connection and list models
    connected = False
    models: list[dict[str, str]] = []
    try:
        models = await _bridge.list_models()
        connected = True
    except Exception:
        pass

    return {
        "success": True,
        "backend_type": _current_backend_type,
        "connected": connected,
        "models": [m["id"] for m in models] if models else [],
    }


_LENGTH_BANDS = {
    "short": (0, 800),
    "medium": (800, 2000),
    "long": (2000, 10000),
}

CONSTRAINTS_FMT_VER = 1
_turn_seq = 0


def _build_constraints_message() -> tuple[dict[str, str] | None, dict]:
    """Build a system message from the active project's config constraints.

    Returns (constraints_msg, contract_meta) where constraints_msg is
    {"role": "system", "content": ...} or None, and contract_meta has
    config_hash, config_hash_full, strict, mode for receipt building.
    Only applies in code or research modes.
    """
    mode = GOVERNOR_MODE
    _meta_base = {"mode": mode, "strict": False}
    if mode not in ("code", "research"):
        return None, _meta_base

    try:
        store = _get_project_store() if mode == "code" else _get_research_project_store()
        state = store.get_state()
        contract = state.get("contract", {})
    except Exception as e:
        return None, {**_meta_base, "receipt_error": f"store_read_failed: {e}"}

    config = contract.get("config")
    if config:
        # Build typed constraints block from config
        lines = []
        hash_tag = contract.get("config_hash", "")
        hash_full = contract.get("config_hash_full", "")
        lines.append(f"[CONSTRAINTS config_hash={hash_tag}]")

        if config.get("artifact_type"):
            lines.append(f"artifact_type: {config['artifact_type']}")

        length = config.get("length", "medium")
        lo, hi = _LENGTH_BANDS.get(length, (800, 2000))
        lines.append(f"length_band: {length}")
        lines.append(f"length_min_words: {lo}")
        lines.append(f"length_max_words: {hi}")

        if config.get("voice"):
            v = config["voice"]
            lines.append(f"voice: {', '.join(v) if isinstance(v, list) else v}")

        if config.get("citations"):
            lines.append(f"citations: {config['citations']}")

        if config.get("personal_material"):
            lines.append(f"personal_material: {config['personal_material']}")

        fmt = config.get("format", {})
        if isinstance(fmt, dict):
            for key in ("tables", "bullets", "headings"):
                if key in fmt:
                    lines.append(f"format_{key}: {str(fmt[key]).lower()}")

        if config.get("audience"):
            lines.append(f"audience: {config['audience']}")

        if config.get("bans"):
            bans = config["bans"]
            lines.append(f"bans: {'; '.join(bans) if isinstance(bans, list) else bans}")

        if config.get("structure"):
            s = config["structure"]
            lines.append(f"structure: {', '.join(s) if isinstance(s, list) else s}")

        strict = config.get("strict", False)
        lines.append(f"strict: {str(strict).lower()}")

        lines.append("[/CONSTRAINTS]")
        meta = {
            "config_hash": hash_tag or None,
            "config_hash_full": hash_full or None,
            "strict": strict,
            "mode": mode,
        }
        return {"role": "system", "content": "\n".join(lines)}, meta

    # Fallback: raw constraints list
    constraints = contract.get("constraints", [])
    if constraints:
        block = "[CONSTRAINTS]\n" + "\n".join(constraints) + "\n[/CONSTRAINTS]"
        meta = {
            "config_hash": contract.get("config_hash"),
            "config_hash_full": contract.get("config_hash_full"),
            "strict": False,
            "mode": mode,
        }
        return {"role": "system", "content": block}, meta

    return None, _meta_base


def _build_receipt(
    constraints_msg: dict[str, str] | None,
    contract_meta: dict,
    request_id: str,
    turn_id: str,
    turn_seq: int,
    resolved_model: str,
) -> dict:
    """Build a per-turn receipt proving constraints reached the model."""
    constraints_hash = None
    constraints_hash_full = None
    if constraints_msg:
        raw = constraints_msg["content"].encode("utf-8")
        full = hashlib.sha256(raw).hexdigest()
        constraints_hash = full[:16]
        constraints_hash_full = full

    receipt: dict = {
        "config_hash": contract_meta.get("config_hash"),
        "config_hash_full": contract_meta.get("config_hash_full"),
        "constraints_hash": constraints_hash,
        "constraints_hash_full": constraints_hash_full,
        "constraints_format_version": CONSTRAINTS_FMT_VER,
        "rendered_by": "adapter",
        "mode": contract_meta.get("mode", GOVERNOR_MODE),
        "strict": contract_meta.get("strict", False),
        "forced": False,  # chat path always injects; reserved for future preflight bypass
        "model": resolved_model,
        "request_id": request_id,
        "turn_id": turn_id,
        "turn_seq": turn_seq,
    }
    if contract_meta.get("receipt_error"):
        receipt["receipt_error"] = contract_meta["receipt_error"]
    return receipt


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint.

    Delegates to the governor daemon for the full governed pipeline:
    pending check → augment → generate → check → receipt.
    """
    global _turn_seq
    client = _get_daemon_client()
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Inject constraints after last system message
    constraints_msg, contract_meta = _build_constraints_message()
    if constraints_msg:
        insert_idx = 0
        for i, m in enumerate(messages):
            if m["role"] == "system":
                insert_idx = i + 1
        messages.insert(insert_idx, constraints_msg)

    if request.stream:
        return StreamingResponse(
            _stream_via_daemon(client, messages, request.model, GOVERNOR_CONTEXT_ID,
                               constraints_msg, contract_meta),
            media_type="text/event-stream",
        )

    try:
        result = await client.chat_send(
            messages=messages,
            model=request.model,
            context_id=GOVERNOR_CONTEXT_ID,
        )
    except DaemonAuthError as e:
        raise HTTPException(
            status_code=401,
            detail=(
                "Claude Code is not logged in. "
                "Run `claude /login` in a terminal to re-authenticate."
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Daemon error: {e}")

    # If daemon returned a pending violation, format as violation prompt
    if result.get("pending"):
        return _format_violation_pending_response(result, request.model)

    # Build content with optional governor footer
    content = result.get("content", "")
    footer = result.get("footer")
    if footer:
        content = f"{content}\n\n{footer}"

    # Build receipt with resolved model from daemon
    resolved_model = result.get("model", request.model)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    _turn_seq += 1
    turn_id = f"turn-{uuid.uuid4().hex[:12]}"
    receipt = _build_receipt(constraints_msg, contract_meta,
                             request_id, turn_id, _turn_seq, resolved_model)

    usage_data = result.get("usage") or {}
    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=resolved_model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        ),
        receipt=receipt,
    )


async def _stream_via_daemon(
    client: DaemonChatClient,
    messages: list[dict[str, str]],
    model: str,
    context_id: str,
    constraints_msg: dict[str, str] | None = None,
    contract_meta: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Stream response via daemon in OpenAI SSE format."""
    global _turn_seq
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    final_result: dict | None = None

    # Build receipt before streaming so first chunk carries it
    _turn_seq += 1
    turn_id = f"turn-{uuid.uuid4().hex[:12]}"
    receipt = _build_receipt(constraints_msg, contract_meta or {},
                             request_id, turn_id, _turn_seq, model)

    try:
        # Emit receipt as first chunk (before content)
        first_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
            "receipt": receipt,
        }
        yield f"data: {json.dumps(first_chunk)}\n\n"

        async for delta, final in client.chat_stream(
            messages=messages, model=model, context_id=context_id
        ):
            if delta:
                sse_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(sse_chunk)}\n\n"
            if final is not None:
                final_result = final

        # Update receipt with resolved model from daemon if available
        if final_result and final_result.get("model"):
            receipt["model"] = final_result["model"]

        # If daemon returned a pending violation, emit it as a final chunk
        if final_result and final_result.get("pending"):
            pending = final_result["pending"]
            violations = pending.get("violations", [])
            mode = pending.get("mode", "general")
            prompt = format_violation_prompt(violations, mode)
            sse_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"\n\n{prompt}"},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(sse_chunk)}\n\n"

        # If daemon returned a footer, emit it
        if final_result and final_result.get("footer"):
            sse_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"\n\n{final_result['footer']}"},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(sse_chunk)}\n\n"

        # Final done chunk — includes turn_id + receipt for code builder
        done_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "turn_id": turn_id,
            "receipt": receipt,
        }
        yield f"data: {json.dumps(done_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except DaemonAuthError:
        error_chunk = {
            "error": {
                "message": (
                    "Claude Code is not logged in. "
                    "Run `claude /login` in a terminal to re-authenticate."
                ),
                "type": "auth_error",
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    except Exception as e:
        error_chunk = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"


def _format_violation_pending_response(
    pending: Any,
    model: str,
) -> ChatCompletionResponse:
    """Format a pending violation as a ChatCompletionResponse.

    Accepts either a daemon result dict (with 'pending' key) or a
    PendingViolation object. The response content is the violation prompt
    asking user to choose an action.
    """
    if isinstance(pending, dict):
        # Daemon result dict — extract violations/mode from the pending sub-dict
        p = pending.get("pending", pending)
        violations = p.get("violations", [])
        mode = p.get("mode", "general")
    elif hasattr(pending, "prompt"):
        # ViolationPendingResponse from check_response_blocking
        prompt_text = pending.prompt
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=prompt_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
    else:
        # PendingViolation object from get_pending
        violations = pending.violations
        mode = pending.mode

    prompt_text = format_violation_prompt(violations, mode)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=prompt_text),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )


# ============================================================================
# Session Endpoints
# ============================================================================


class CreateSessionRequest(BaseModel):
    model: str = ""
    title: str = "New conversation"


class UpdateSessionRequest(BaseModel):
    title: str


class AppendMessageRequest(BaseModel):
    role: str
    content: str
    model: str | None = None
    usage: dict[str, int] | None = None


@app.get("/sessions/")
async def list_sessions() -> dict[str, Any]:
    """List all session summaries (no messages), sorted by most recent."""
    store = _get_session_store()
    return {"sessions": store.list_summaries()}


@app.post("/sessions/")
async def create_session(request: CreateSessionRequest) -> dict[str, Any]:
    """Create a new chat session."""
    store = _get_session_store()
    session = store.create(
        context_id=GOVERNOR_CONTEXT_ID,
        model=request.model,
        title=request.title,
    )
    return session.to_dict()


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    """Get a session with full message history."""
    store = _get_session_store()
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, Any]:
    """Delete a session."""
    store = _get_session_store()
    if not store.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True}


@app.patch("/sessions/{session_id}")
async def update_session(session_id: str, request: UpdateSessionRequest) -> dict[str, Any]:
    """Update a session's title."""
    store = _get_session_store()
    if not store.update_title(session_id, request.title):
        raise HTTPException(status_code=404, detail="Session not found")
    session = store.get(session_id)
    return session.to_dict() if session else {"success": True}


@app.post("/sessions/{session_id}/messages")
async def append_message(session_id: str, request: AppendMessageRequest) -> dict[str, Any]:
    """Append a message to a session (write-through target)."""
    store = _get_session_store()
    msg = SessionMessage.create(
        role=request.role,
        content=request.content,
        model=request.model,
        usage=request.usage,
    )
    if not store.append_message(session_id, msg):
        raise HTTPException(status_code=404, detail="Session not found")
    return msg.to_dict()


# ============================================================================
# Governor Endpoints
# ============================================================================


def _resolve_context() -> tuple[Any | None, str]:
    """Resolve the active governor context.

    Returns (context_or_None, context_id).
    """
    cm = _get_context_manager()
    ctx = cm.get(GOVERNOR_CONTEXT_ID)
    return ctx, GOVERNOR_CONTEXT_ID


def _build_vm_for_context(ctx: Any) -> GovernorViewModel:
    """Build a GovernorViewModel from a resolved context."""
    return build_viewmodel(ctx.governor_dir, ctx.root)


@app.get("/governor/contexts")
async def list_contexts() -> dict[str, Any]:
    """List all governor contexts."""
    cm = _get_context_manager()
    contexts = cm.list_contexts()
    return {
        "active_context_id": GOVERNOR_CONTEXT_ID,
        "contexts": [ctx.to_dict() for ctx in contexts],
    }


@app.get("/governor/status")
async def governor_status() -> dict[str, Any]:
    """Show governor state for the active context.

    Backward-compat fields preserved; adds 'viewmodel' key with v2 schema.
    """
    ctx, context_id = _resolve_context()

    if ctx is None:
        return {
            "context_id": context_id,
            "initialized": False,
            "mode": GOVERNOR_MODE,
        }

    gov_dir = ctx.governor_dir
    has_governor = gov_dir.exists()
    has_fiction = (ctx.root / ".fiction-gov").exists()

    # Count facts and decisions
    facts_count = 0
    decisions_count = 0
    if has_governor:
        facts_index = gov_dir / "facts" / "index.json"
        if facts_index.exists():
            try:
                facts_data = json.loads(facts_index.read_text())
                facts_count = len(facts_data) if isinstance(facts_data, list) else 0
            except (json.JSONDecodeError, OSError):
                pass
        decisions_index = gov_dir / "decisions" / "index.json"
        if decisions_index.exists():
            try:
                dec_data = json.loads(decisions_index.read_text())
                decisions_count = len(dec_data) if isinstance(dec_data, list) else 0
            except (json.JSONDecodeError, OSError):
                pass

    # Build ViewModel v2
    vm = _build_vm_for_context(ctx)

    result: dict[str, Any] = {
        "context_id": ctx.context_id,
        "initialized": True,
        "mode": ctx.mode,
        "created_at": ctx.created_at,
        "has_governor": has_governor,
        "has_fiction_governor": has_fiction,
        "facts_count": facts_count,
        "decisions_count": decisions_count,
        "metadata": ctx.metadata,
        "viewmodel": vm.to_dict(),
    }

    # Research mode: include ED summary
    if ctx.mode == "research":
        try:
            from governor.research_store import ResearchStore

            store = ResearchStore(ctx.governor_dir)
            ed = store.compute_ed()
            result["research_ed"] = ed
        except Exception:
            pass

    return result


@app.get("/governor/now")
async def governor_now() -> dict[str, Any]:
    """Now screen: glanceable status for the active context."""
    ctx, context_id = _resolve_context()

    if ctx is None:
        return {
            "context_id": context_id,
            "status": "ok",
            "sentence": "OK: no governor context initialized.",
            "last_event": None,
            "suggested_action": None,
            "regime": None,
            "mode": GOVERNOR_MODE,
        }

    vm = _build_vm_for_context(ctx)

    now_result: dict[str, Any] = {
        "context_id": context_id,
        "status": derive_status_pill(vm),
        "sentence": derive_one_sentence(vm),
        "last_event": derive_last_event(vm),
        "suggested_action": derive_suggested_action(vm),
        "regime": vm.regime.name if vm.regime else None,
        "mode": ctx.mode,
    }

    # Research mode: override sentence with ED summary
    if ctx.mode == "research":
        try:
            from governor.research_store import ResearchStore

            store = ResearchStore(ctx.governor_dir)
            ed = store.compute_ed()
            now_result["sentence"] = (
                f"Discipline: {ed['total']} | {ed['floating']} floating | "
                f"{ed['open_uncertain']} uncertain"
            )
            now_result["research_ed"] = ed
        except Exception:
            pass

    return now_result


@app.get("/governor/why")
async def governor_why(limit: int = 20, severity: str | None = None) -> dict[str, Any]:
    """Why screen: decision/violation/claim feed."""
    ctx, context_id = _resolve_context()

    if ctx is None:
        return {"context_id": context_id, "feed": [], "total": 0}

    vm = _build_vm_for_context(ctx)
    feed = derive_why_feed(vm, limit=limit, severity_filter=severity)

    return {
        "context_id": context_id,
        "feed": feed,
        "total": len(feed),
    }


@app.get("/governor/history")
async def governor_history(days: int = 7) -> dict[str, Any]:
    """History screen: events grouped by calendar day."""
    ctx, context_id = _resolve_context()

    if ctx is None:
        return {"context_id": context_id, "days": []}

    vm = _build_vm_for_context(ctx)
    grouped = derive_history_days(vm, days=days)

    return {
        "context_id": context_id,
        "days": grouped,
    }


@app.get("/governor/detail/{item_id}")
async def governor_detail(item_id: str) -> dict[str, Any]:
    """Drill-down by ID prefix (dec_, clm_, ev_, vio_)."""
    ctx, context_id = _resolve_context()

    if ctx is None:
        raise HTTPException(status_code=404, detail="No governor context initialized.")

    vm = _build_vm_for_context(ctx)

    # Search by prefix
    if item_id.startswith("dec_"):
        for d in vm.decisions:
            if d.id == item_id:
                return {"id": item_id, "type": "decision", "data": d.to_dict()}
    elif item_id.startswith("clm_"):
        for c in vm.claims:
            if c.id == item_id:
                return {"id": item_id, "type": "claim", "data": c.to_dict()}
    elif item_id.startswith("ev_"):
        for e in vm.evidence:
            if e.id == item_id:
                return {"id": item_id, "type": "evidence", "data": e.to_dict()}
    elif item_id.startswith("vio_"):
        for v in vm.violations:
            if v.id == item_id:
                return {"id": item_id, "type": "violation", "data": v.to_dict()}

    raise HTTPException(status_code=404, detail=f"Item not found: {item_id}")


# ============================================================================
# Mode-Specific Endpoints (Fiction / Code)
# ============================================================================


# ============================================================================
# Research Mode Pydantic Models
# ============================================================================


class ClaimRequest(BaseModel):
    content: str
    scope: str = ""


class AssumptionRequest(BaseModel):
    content: str


class UncertaintyRequest(BaseModel):
    content: str
    attached_to: str = ""


class LinkRequest(BaseModel):
    link_type: str
    from_id: str
    to_id: str
    subtype: str = ""


class StatusChangeRequest(BaseModel):
    status: str


# ============================================================================
# Research Store (lazy init)
# ============================================================================

_research_store: Any = None


def _get_research_store() -> Any:
    """Lazy-init research store for the active context."""
    global _research_store
    if _research_store is None:
        from governor.research_store import ResearchStore

        cm = _get_context_manager()
        ctx = cm.get_or_create(GOVERNOR_CONTEXT_ID, mode=GOVERNOR_MODE)
        _research_store = ResearchStore(ctx.governor_dir)
    return _research_store


# ============================================================================
# Code Project Store (lazy init)
# ============================================================================

_project_store: Any = None


def _get_project_store() -> Any:
    """Lazy-init project store for code builder."""
    global _project_store
    if _project_store is None:
        from gov_webui.project_store import CodeProjectStore

        cm = _get_context_manager()
        ctx = cm.get_or_create(GOVERNOR_CONTEXT_ID, mode=GOVERNOR_MODE)
        _project_store = CodeProjectStore(ctx.governor_dir)
    return _project_store


_research_project_store: Any = None


def _get_research_project_store() -> Any:
    """Lazy-init project store for research builder."""
    global _research_project_store
    if _research_project_store is None:
        from gov_webui.project_store import CodeProjectStore, RESEARCH_EXTENSIONS

        cm = _get_context_manager()
        ctx = cm.get_or_create(GOVERNOR_CONTEXT_ID, mode=GOVERNOR_MODE)
        _research_project_store = CodeProjectStore(
            ctx.governor_dir,
            subdir="research_project",
            allowed_extensions=RESEARCH_EXTENSIONS,
        )
    return _research_project_store


_artifact_store: Any = None


def _get_artifact_store() -> Any:
    """Lazy-init artifact store."""
    global _artifact_store
    if _artifact_store is None:
        from gov_webui.artifact_store import ArtifactStore

        cm = _get_context_manager()
        ctx = cm.get_or_create(GOVERNOR_CONTEXT_ID, mode=GOVERNOR_MODE)
        _artifact_store = ArtifactStore(ctx.governor_dir)
    return _artifact_store


class CaptureRequest(BaseModel):
    """Request to scan text for canon-worthy statements."""
    text: str
    message_id: str = ""


class CaptureAcceptRequest(BaseModel):
    """Request to promote a pending capture to canon."""
    name: str = ""           # Character/entity name (may override detected)
    description: str = ""    # Description text
    capture_type: str = ""   # character, world_rule, relationship, constraint


class CharacterRequest(BaseModel):
    """Request to add a character."""
    name: str
    description: str | None = None
    voice: str | None = None
    wont: str | None = None  # Things they wouldn't do


class WorldRuleRequest(BaseModel):
    """Request to add a world rule."""
    rule: str


class ForbiddenRequest(BaseModel):
    """Request to add a forbidden thing."""
    description: str
    patterns: list[str] = Field(default_factory=list)


class DecisionRequest(BaseModel):
    """Request to add a decision."""
    decision: str
    rationale: str | None = None


class ConstraintRequest(BaseModel):
    """Request to add a constraint."""
    constraint: str
    patterns: list[str] = Field(default_factory=list)


# -- Code Builder request models -------------------------------------------

class IntentUpdateRequest(BaseModel):
    text: str
    locked: bool = False
    expected_version: int | None = None


class ContractUpdateRequest(BaseModel):
    description: str = ""
    inputs: list[dict] = Field(default_factory=list)
    outputs: list[dict] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    transport: str = "stdio"
    expected_version: int | None = None
    config: dict | None = None
    config_hash: str | None = None
    config_hash_full: str | None = None


class PlanItemRequest(BaseModel):
    phase_idx: int
    text: str


class PlanItemStatusRequest(BaseModel):
    status: str
    expected_version: int | None = None


class PhaseRequest(BaseModel):
    name: str


class PhaseUpdateRequest(BaseModel):
    name: str | None = None
    locked: bool | None = None


class FileUpdateRequest(BaseModel):
    content: str
    turn_id: str | None = None


class RunRequest(BaseModel):
    filepath: str = "tool.py"
    stdin: str = ""
    timeout: int = 30
    force: bool = False


# -- Artifact Engine request models ----------------------------------------

class ArtifactCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str
    kind: str = "text"
    language: str = ""
    message_id: str | None = None
    source: str = "promote"


class ArtifactUpdateRequest(BaseModel):
    content: str
    title: str | None = None
    expected_current_version: int | None = Field(default=None, ge=1)
    source: str = "manual"
    message_id: str | None = None


@app.get("/governor/fiction/characters")
async def list_characters() -> dict[str, Any]:
    """List all characters for fiction mode."""
    ctx, _ = _resolve_context()
    if ctx is None:
        return {"characters": [], "message": "No governor context initialized."}

    from governor.continuity import AnchorType, create_registry

    registry = create_registry(ctx.governor_dir)
    anchors = registry.all()

    characters = []
    for a in anchors:
        if a.anchor_type == AnchorType.CANON and "char-" in a.id.lower():
            name = a.id.replace("char-", "").replace("-", " ").title()
            # Check for associated "wont" anchor
            wont_anchor = registry.get(f"{a.id}-wont")
            characters.append({
                "id": a.id,
                "name": name,
                "description": a.description,
                "wont": wont_anchor.description if wont_anchor else None,
            })

    return {"characters": characters}


@app.post("/governor/fiction/characters")
async def add_character(request: CharacterRequest) -> dict[str, Any]:
    """Add a character for fiction mode."""
    ctx, _ = _resolve_context()
    if ctx is None:
        raise HTTPException(status_code=400, detail="No governor context initialized.")

    from governor.continuity import Anchor, AnchorType, Severity, create_registry

    registry = create_registry(ctx.governor_dir)

    char_id = f"char-{request.name.lower().replace(' ', '-')}"

    # Build description
    desc_parts = []
    if request.description:
        desc_parts.append(f"Appearance: {request.description}")
    if request.voice:
        desc_parts.append(f"Voice: {request.voice}")
    description = "; ".join(desc_parts) if desc_parts else f"Character: {request.name}"

    # Create character anchor
    anchor = Anchor(
        id=char_id,
        anchor_type=AnchorType.CANON,
        description=description,
        severity=Severity.REJECT,
    )
    registry.register(anchor)

    # Create prohibition anchor if wont provided
    if request.wont:
        patterns = [p.strip() for p in request.wont.split(",")]
        wont_anchor = Anchor(
            id=f"{char_id}-wont",
            anchor_type=AnchorType.PROHIBITION,
            description=f"{request.name} wouldn't: {request.wont}",
            forbidden_patterns=patterns,
            severity=Severity.REJECT,
        )
        registry.register(wont_anchor)

    # Save
    registry.save(ctx.governor_dir / "continuity" / "anchors.json")

    return {
        "success": True,
        "message": f"{request.name} added. I'll remember.",
        "id": char_id,
    }


@app.delete("/governor/fiction/characters/{char_id}")
async def remove_character(char_id: str) -> dict[str, Any]:
    """Remove a character."""
    ctx, _ = _resolve_context()
    if ctx is None:
        raise HTTPException(status_code=400, detail="No governor context initialized.")

    from governor.continuity import create_registry

    registry = create_registry(ctx.governor_dir)
    anchor = registry.unregister(char_id)
    registry.unregister(f"{char_id}-wont")

    if anchor:
        registry.save(ctx.governor_dir / "continuity" / "anchors.json")
        return {"success": True, "message": "Character removed."}

    raise HTTPException(status_code=404, detail="Character not found.")


@app.get("/governor/fiction/world-rules")
async def list_world_rules() -> dict[str, Any]:
    """List all world rules for fiction mode."""
    ctx, _ = _resolve_context()
    if ctx is None:
        return {"rules": [], "message": "No governor context initialized."}

    from governor.continuity import AnchorType, create_registry

    registry = create_registry(ctx.governor_dir)
    anchors = registry.all()

    rules = []
    for a in anchors:
        if a.anchor_type == AnchorType.DEFINITION:
            rules.append({
                "id": a.id,
                "rule": a.description,
            })

    return {"rules": rules}


@app.post("/governor/fiction/world-rules")
async def add_world_rule(request: WorldRuleRequest) -> dict[str, Any]:
    """Add a world rule for fiction mode."""
    ctx, _ = _resolve_context()
    if ctx is None:
        raise HTTPException(status_code=400, detail="No governor context initialized.")

    from governor.continuity import Anchor, AnchorType, Severity, create_registry

    registry = create_registry(ctx.governor_dir)

    rule_id = f"world-{len([a for a in registry.all() if 'world-' in a.id]) + 1}"

    anchor = Anchor(
        id=rule_id,
        anchor_type=AnchorType.DEFINITION,
        description=request.rule,
        severity=Severity.REJECT,
    )
    registry.register(anchor)
    registry.save(ctx.governor_dir / "continuity" / "anchors.json")

    return {
        "success": True,
        "message": "Rule added. I'll keep it consistent.",
        "id": rule_id,
    }


@app.get("/governor/fiction/forbidden")
async def list_forbidden() -> dict[str, Any]:
    """List all forbidden things for fiction mode."""
    ctx, _ = _resolve_context()
    if ctx is None:
        return {"forbidden": [], "message": "No governor context initialized."}

    from governor.continuity import AnchorType, create_registry

    registry = create_registry(ctx.governor_dir)
    anchors = registry.all()

    forbidden = []
    for a in anchors:
        if a.anchor_type == AnchorType.PROHIBITION and not a.id.endswith("-wont"):
            forbidden.append({
                "id": a.id,
                "description": a.description,
                "patterns": a.forbidden_patterns,
            })

    return {"forbidden": forbidden}


@app.post("/governor/fiction/forbidden")
async def add_forbidden(request: ForbiddenRequest) -> dict[str, Any]:
    """Add a forbidden thing for fiction mode."""
    ctx, _ = _resolve_context()
    if ctx is None:
        raise HTTPException(status_code=400, detail="No governor context initialized.")

    from governor.continuity import Anchor, AnchorType, Severity, create_registry

    registry = create_registry(ctx.governor_dir)

    forbid_id = f"forbid-{len([a for a in registry.all() if 'forbid-' in a.id]) + 1}"

    anchor = Anchor(
        id=forbid_id,
        anchor_type=AnchorType.PROHIBITION,
        description=request.description,
        forbidden_patterns=request.patterns,
        severity=Severity.REJECT,
    )
    registry.register(anchor)
    registry.save(ctx.governor_dir / "continuity" / "anchors.json")

    return {
        "success": True,
        "message": "I'll watch for that.",
        "id": forbid_id,
    }


# =============================================================================
# Canon Capture (fiction mode — pending promotion pipeline)
# =============================================================================

# In-memory pending captures (per process; cleared on restart)
_pending_captures: dict[str, dict[str, Any]] = {}
_capture_counter: int = 0


@app.post("/governor/fiction/capture/scan")
async def capture_scan(request: CaptureRequest) -> dict[str, Any]:
    """Scan text for canon-worthy statements. Returns capture candidates."""
    global _capture_counter

    try:
        from fiction_governor.canon_capture import CanonCaptureClassifier
    except ImportError:
        return {"captures": [], "error": "Canon capture classifier not available."}

    classifier = CanonCaptureClassifier()
    items, receipt = classifier.scan(request.text)

    captures = []
    for item in items:
        _capture_counter += 1
        cap_id = f"cap-{_capture_counter}"
        cap = {
            "id": cap_id,
            "kind": item.kind if isinstance(item.kind, str) else item.kind.value,
            "confidence": round(item.confidence, 2),
            "subject": item.subject_guess or "",
            "statement": item.statement,
            "field": item.field_guess or "",
            "spans": [list(s) for s in item.evidence_spans],
            "message_id": request.message_id,
            "status": "pending",
        }
        if item.draft_payload:
            cap["draft"] = item.draft_payload
        _pending_captures[cap_id] = cap
        captures.append(cap)

    return {
        "captures": captures,
        "receipt": {
            "classifier_version": receipt.classifier_version,
            "content_hash": receipt.content_hash,
            "pattern_hits": receipt.pattern_hits,
        },
    }


@app.get("/governor/fiction/captures")
async def list_pending_captures() -> dict[str, Any]:
    """List all pending (unresolved) captures."""
    pending = [c for c in _pending_captures.values() if c["status"] == "pending"]
    return {"captures": pending, "count": len(pending)}


@app.post("/governor/fiction/capture/{capture_id}/accept")
async def accept_capture(capture_id: str, request: CaptureAcceptRequest) -> dict[str, Any]:
    """Promote a pending capture to canon (creates character or world rule anchor)."""
    if capture_id not in _pending_captures:
        raise HTTPException(status_code=404, detail="Capture not found.")

    cap = _pending_captures[capture_id]
    if cap["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"Capture already {cap['status']}.")

    ctx, _ = _resolve_context()
    if ctx is None:
        raise HTTPException(status_code=400, detail="No governor context initialized.")

    from governor.continuity import Anchor, AnchorType, Severity, create_registry

    registry = create_registry(ctx.governor_dir)

    kind = request.capture_type or cap.get("kind", "character")
    name = request.name or cap.get("subject", "")
    desc = request.description or cap.get("statement", "")

    if kind in ("character", "relationship"):
        char_id = f"char-{name.lower().replace(' ', '-')}" if name else f"char-cap-{capture_id}"
        anchor = Anchor(
            id=char_id,
            anchor_type=AnchorType.CANON,
            description=f"{name}: {desc}" if name else desc,
            severity=Severity.REJECT,
        )
        registry.register(anchor)
        registry.save(ctx.governor_dir / "continuity" / "anchors.json")
        cap["status"] = "accepted"
        cap["promoted_to"] = char_id
        return {"success": True, "message": f"Canon: {name or char_id}", "id": char_id}

    elif kind in ("world_rule", "constraint"):
        rule_count = len([a for a in registry.all() if "rule-" in a.id])
        rule_id = f"rule-{rule_count + 1}"

        if kind == "constraint":
            patterns = [p.strip() for p in desc.split(",") if p.strip()]
            anchor = Anchor(
                id=rule_id,
                anchor_type=AnchorType.PROHIBITION,
                description=desc,
                forbidden_patterns=patterns,
                severity=Severity.REJECT,
            )
        else:
            anchor = Anchor(
                id=rule_id,
                anchor_type=AnchorType.DEFINITION,
                description=desc,
                severity=Severity.WARN,
            )
        registry.register(anchor)
        registry.save(ctx.governor_dir / "continuity" / "anchors.json")
        cap["status"] = "accepted"
        cap["promoted_to"] = rule_id
        return {"success": True, "message": f"Canon: {desc[:40]}", "id": rule_id}

    raise HTTPException(status_code=400, detail=f"Unknown capture kind: {kind}")


@app.post("/governor/fiction/capture/{capture_id}/reject")
async def reject_capture(capture_id: str) -> dict[str, Any]:
    """Reject a pending capture."""
    if capture_id not in _pending_captures:
        raise HTTPException(status_code=404, detail="Capture not found.")

    cap = _pending_captures[capture_id]
    cap["status"] = "rejected"
    return {"success": True, "message": "Capture dismissed."}


@app.get("/governor/code/decisions")
async def list_decisions() -> dict[str, Any]:
    """List all decisions for code mode."""
    ctx, _ = _resolve_context()
    if ctx is None:
        return {"decisions": [], "message": "No governor context initialized."}

    from governor.ledgers import DecisionLedger

    try:
        ledger = DecisionLedger(ctx.governor_dir)
        decisions = list(ledger.all())

        return {
            "decisions": [
                {
                    "id": str(d.id),
                    "topic": d.topic,
                    "choice": d.choice,
                    "rationale": d.rationale,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                }
                for d in decisions
            ]
        }
    except Exception:
        return {"decisions": []}


@app.post("/governor/code/decisions")
async def add_decision(request: DecisionRequest) -> dict[str, Any]:
    """Add a decision for code mode."""
    ctx, _ = _resolve_context()
    if ctx is None:
        raise HTTPException(status_code=400, detail="No governor context initialized.")

    from governor.claims import decision as make_decision
    from governor.ledgers import DecisionLedger

    # Parse decision into topic/choice
    if ":" in request.decision:
        topic, choice = request.decision.split(":", 1)
    elif "," in request.decision:
        parts = request.decision.split(",", 1)
        topic = parts[0].strip()
        choice = parts[1].strip() if len(parts) > 1 else topic
    else:
        topic = "architecture"
        choice = request.decision

    topic = topic.strip()
    choice = choice.strip()

    ledger = DecisionLedger(ctx.governor_dir)
    claim = make_decision(topic, choice)
    decision = ledger.add(claim, rationale=request.rationale)

    return {
        "success": True,
        "message": "Decision recorded. I'll catch anything that contradicts it.",
        "id": str(decision.id),
    }


@app.get("/governor/code/constraints")
async def list_constraints() -> dict[str, Any]:
    """List all constraints for code mode."""
    ctx, _ = _resolve_context()
    if ctx is None:
        return {"constraints": [], "message": "No governor context initialized."}

    from governor.continuity import AnchorType, create_registry

    registry = create_registry(ctx.governor_dir)
    anchors = registry.all()

    constraints = []
    for a in anchors:
        if a.anchor_type == AnchorType.PROHIBITION and "constraint-" in a.id:
            constraints.append({
                "id": a.id,
                "description": a.description,
                "patterns": a.forbidden_patterns,
            })

    return {"constraints": constraints}


@app.post("/governor/code/constraints")
async def add_constraint(request: ConstraintRequest) -> dict[str, Any]:
    """Add a constraint for code mode."""
    ctx, _ = _resolve_context()
    if ctx is None:
        raise HTTPException(status_code=400, detail="No governor context initialized.")

    from governor.continuity import Anchor, AnchorType, Severity, create_registry

    registry = create_registry(ctx.governor_dir)

    con_id = f"constraint-{len([a for a in registry.all() if 'constraint-' in a.id]) + 1}"

    anchor = Anchor(
        id=con_id,
        anchor_type=AnchorType.PROHIBITION,
        description=request.constraint,
        forbidden_patterns=request.patterns,
        severity=Severity.REJECT,
    )
    registry.register(anchor)
    registry.save(ctx.governor_dir / "continuity" / "anchors.json")

    return {
        "success": True,
        "message": "Constraint added.",
        "id": con_id,
    }


# ============================================================================
# Code Interferometry Endpoints
# ============================================================================


@app.get("/governor/code/compare/last")
async def code_compare_last() -> dict[str, Any]:
    """Get the latest code divergence report, or null if none exists."""
    ctx, _ = _resolve_context()
    if ctx is None:
        return {"report": None}

    try:
        from governor.interferometry import InterferometryStore
        from governor.code_interferometry import compute_code_divergence
        from governor.continuity import create_registry

        store = InterferometryStore(ctx.governor_dir)
        irun = store.last()
        if irun is None:
            return {"report": None}

        try:
            registry = create_registry(ctx.governor_dir)
            anchors = registry.all()
        except Exception:
            anchors = []

        report = compute_code_divergence(irun, anchors)
        return {"report": report.to_dict()}
    except Exception:
        return {"report": None}


class CompareRequest(BaseModel):
    prompt: str
    backends: str  # comma-separated backend:model pairs


@app.post("/governor/code/compare")
async def code_compare_run(request: CompareRequest) -> dict[str, Any]:
    """Run code interferometry compare. Returns report + run_id."""
    ctx, _ = _resolve_context()
    if ctx is None:
        raise HTTPException(status_code=400, detail="No governor context initialized.")

    import asyncio
    from governor.interferometry import InterferometryStore, run_ensemble
    from governor.code_interferometry import compute_code_divergence
    from governor.continuity import create_registry

    # Parse backend configs
    backend_configs = []
    for pair in request.backends.split(","):
        pair = pair.strip()
        if ":" not in pair:
            raise HTTPException(status_code=400, detail=f"Invalid backend:model pair: {pair}")
        bt, model = pair.split(":", 1)
        config: dict[str, Any] = {"backend_type": bt, "model": model}
        if bt == "ollama":
            config["host"] = OLLAMA_HOST
        elif bt == "anthropic":
            config["api_key"] = ANTHROPIC_API_KEY
        backend_configs.append(config)

    irun = await run_ensemble(request.prompt, backend_configs)

    store = InterferometryStore(ctx.governor_dir)
    store.save(irun)

    try:
        registry = create_registry(ctx.governor_dir)
        anchors = registry.all()
    except Exception:
        anchors = []

    report = compute_code_divergence(irun, anchors)
    return {"run_id": irun.id, "report": report.to_dict()}


# ============================================================================
# Code Builder Endpoints
# ============================================================================


@app.get("/governor/code/project")
async def code_project_state() -> dict[str, Any]:
    """Full project state for sidebar polling."""
    try:
        store = _get_project_store()
        return store.get_state()
    except Exception:
        return {"version": 0, "intent": {"text": "", "locked": False},
                "contract": {}, "plan": {"phases": []}, "files": {}}


@app.put("/governor/code/project/intent")
async def code_update_intent(request: IntentUpdateRequest) -> dict[str, Any]:
    """Update intent text + lock."""
    from gov_webui.project_store import StaleVersionError

    store = _get_project_store()
    try:
        intent = store.update_intent(
            request.text, request.locked, request.expected_version
        )
        return {"success": True, "intent": intent.model_dump()}
    except StaleVersionError:
        raise HTTPException(409, "Stale version")


@app.put("/governor/code/project/contract")
async def code_update_contract(request: ContractUpdateRequest) -> dict[str, Any]:
    """Update contract fields."""
    from gov_webui.project_store import Contract, ContractField, StaleVersionError, compute_config_hash

    # Parse input/output dicts into ContractField objects
    inputs = [ContractField(**f) for f in request.inputs]
    outputs = [ContractField(**f) for f in request.outputs]

    # Server always recomputes hash — never trust client hashes
    config = request.config
    config_hash = None
    config_hash_full = None
    if config:
        # Guard against oversized configs (50KB soft cap)
        config_size = len(json.dumps(config))
        if config_size > 50_000:
            raise HTTPException(400, f"Config too large: {config_size} bytes (max 50000)")
        short, full = compute_config_hash(config)
        if request.config_hash and request.config_hash != short:
            raise HTTPException(400, f"Config hash mismatch: got {request.config_hash}, expected {short}")
        config_hash = short
        config_hash_full = full

    contract = Contract(
        description=request.description,
        inputs=inputs,
        outputs=outputs,
        constraints=request.constraints,
        transport=request.transport,
        config=config,
        config_hash=config_hash,
        config_hash_full=config_hash_full,
    )
    store = _get_project_store()
    try:
        result = store.update_contract(contract, request.expected_version)
        return {"success": True, "contract": result.model_dump()}
    except StaleVersionError:
        raise HTTPException(409, "Stale version")


@app.post("/governor/code/plan/item")
async def code_add_plan_item(request: PlanItemRequest) -> dict[str, Any]:
    """Add a plan item to a phase."""
    store = _get_project_store()
    try:
        item = store.add_plan_item(request.phase_idx, request.text)
        return {"success": True, "item": item.model_dump()}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.patch("/governor/code/plan/item/{item_id}")
async def code_update_plan_item(
    item_id: str, request: PlanItemStatusRequest
) -> dict[str, Any]:
    """Update plan item status with state machine validation."""
    from gov_webui.project_store import PlanItemStatus, StaleVersionError

    store = _get_project_store()
    try:
        status = PlanItemStatus(request.status)
    except ValueError:
        raise HTTPException(400, f"Invalid status: {request.status}")

    try:
        item = store.update_item_status(
            item_id, status, request.expected_version
        )
        return {"success": True, "item": item.model_dump()}
    except StaleVersionError:
        raise HTTPException(409, "Stale version")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/governor/code/plan/phase")
async def code_add_phase(request: PhaseRequest) -> dict[str, Any]:
    """Add a new phase to the plan."""
    store = _get_project_store()
    phase = store.add_phase(request.name)
    return {"success": True, "phase": phase.model_dump()}


@app.patch("/governor/code/plan/phase/{idx}")
async def code_update_phase(idx: int, request: PhaseUpdateRequest) -> dict[str, Any]:
    """Update phase name/lock."""
    store = _get_project_store()
    try:
        phase = store.update_phase(idx, request.name, request.locked)
        return {"success": True, "phase": phase.model_dump()}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/governor/code/files")
async def code_list_files() -> dict[str, Any]:
    """List files with versions + hashes."""
    store = _get_project_store()
    return {"files": store.list_files()}


@app.get("/governor/code/file-prev/{path:path}")
async def code_get_file_prev(path: str) -> dict[str, Any]:
    """Get previous version for client-side diff."""
    store = _get_project_store()
    try:
        content = store.get_file_prev(path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"filepath": path, "content": content}


@app.get("/governor/code/files/{path:path}")
async def code_get_file(path: str) -> dict[str, Any]:
    """Get file content."""
    store = _get_project_store()
    try:
        content = store.get_file_content(path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if content is None:
        raise HTTPException(404, f"File not found: {path}")
    return {"filepath": path, "content": content}


@app.put("/governor/code/files/{path:path}")
async def code_put_file(path: str, request: FileUpdateRequest) -> dict[str, Any]:
    """Accept file, returns version + hash."""
    store = _get_project_store()
    try:
        entry = store.put_file(path, request.content, request.turn_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "success": True,
        "filepath": path,
        "version": entry.version,
        "content_hash": entry.content_hash,
    }


@app.post("/governor/code/run")
async def code_run(request: RunRequest) -> dict[str, Any]:
    """Execute project files in tempdir, returns output."""
    import subprocess
    import sys
    import tempfile

    store = _get_project_store()
    state = store.get_state()
    files = state.get("files", {})

    if not files:
        raise HTTPException(400, "No files in project")

    if request.filepath not in files:
        raise HTTPException(404, f"Entrypoint not found: {request.filepath}")

    # Constraint pre-flight check
    preflight_violations: list[str] = []
    try:
        ctx, _ = _resolve_context()
        if ctx is not None:
            from governor.continuity import AnchorType, create_registry

            registry = create_registry(ctx.governor_dir)
            anchors = registry.all()
            prohibitions = [
                a for a in anchors
                if a.anchor_type == AnchorType.PROHIBITION
            ]

            # Check all project files against prohibition patterns
            for fpath in files:
                content = store.get_file_content(fpath)
                if content is None:
                    continue
                for anchor in prohibitions:
                    for pattern in anchor.forbidden_patterns:
                        if pattern.lower() in content.lower():
                            preflight_violations.append(
                                f"{fpath}: matches '{pattern}' "
                                f"({anchor.description})"
                            )
    except Exception:
        pass  # Pre-flight is best-effort

    if preflight_violations and not request.force:
        return {
            "success": False,
            "preflight_violations": preflight_violations,
            "preflight_hit": True,
            "forced": False,
        }

    # Execute in tempdir
    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="gov-code-run-")
        tmpdir_path = Path(tmpdir)

        # Write all accepted files into tempdir
        for fpath in files:
            content = store.get_file_content(fpath)
            if content is None:
                continue
            dest = tmpdir_path / fpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)

        entrypoint = request.filepath
        timeout = min(max(request.timeout, 1), 120)  # clamp 1-120s

        result = subprocess.run(
            [sys.executable, entrypoint],
            cwd=tmpdir,
            input=request.stdin or None,
            capture_output=True,
            timeout=timeout,
            text=True,
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        # Cap combined output at 100KB
        max_output = 100 * 1024
        combined_len = len(stdout) + len(stderr)
        if combined_len > max_output:
            # Truncate proportionally
            ratio = max_output / combined_len
            stdout = stdout[:int(len(stdout) * ratio)] + "\n…(truncated)"
            stderr = stderr[:int(len(stderr) * ratio)] + "\n…(truncated)"

        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "filepath": request.filepath,
            "preflight_hit": bool(preflight_violations),
            "forced": request.force and bool(preflight_violations),
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Process killed: exceeded {request.timeout}s timeout",
            "filepath": request.filepath,
            "preflight_hit": bool(preflight_violations),
            "forced": request.force and bool(preflight_violations),
        }
    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "filepath": request.filepath,
            "preflight_hit": bool(preflight_violations),
            "forced": request.force and bool(preflight_violations),
        }
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Research Mode Endpoints
# ============================================================================


@app.get("/governor/research/state")
async def research_state() -> dict[str, Any]:
    """Full research state + ED score."""
    store = _get_research_store()
    return store.get_state()


@app.post("/governor/research/claims")
async def add_research_claim(request: ClaimRequest) -> dict[str, Any]:
    """Add a research claim."""
    store = _get_research_store()
    claim = store.add_claim(content=request.content, scope=request.scope)
    return {"success": True, "claim": claim.to_dict()}


@app.delete("/governor/research/claims/{claim_id}")
async def delete_research_claim(claim_id: str) -> dict[str, Any]:
    """Delete a research claim."""
    store = _get_research_store()
    if not store.delete_claim(claim_id):
        raise HTTPException(status_code=404, detail="Claim not found")
    return {"success": True}


@app.patch("/governor/research/claims/{claim_id}/status")
async def change_claim_status(
    claim_id: str, request: StatusChangeRequest
) -> dict[str, Any]:
    """Change a claim's status."""
    from governor.research_store import ClaimStatus as RClaimStatus

    try:
        status = RClaimStatus(request.status)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status: {request.status}. Valid: {[s.value for s in RClaimStatus]}",
        )

    store = _get_research_store()
    try:
        claim = store.update_claim_status(claim_id, status)
    except KeyError:
        raise HTTPException(status_code=404, detail="Claim not found")
    return {"success": True, "claim": claim.to_dict()}


@app.post("/governor/research/assumptions")
async def add_research_assumption(request: AssumptionRequest) -> dict[str, Any]:
    """Add a research assumption."""
    store = _get_research_store()
    assumption = store.add_assumption(content=request.content)
    return {"success": True, "assumption": assumption.to_dict()}


@app.delete("/governor/research/assumptions/{assumption_id}")
async def delete_research_assumption(assumption_id: str) -> dict[str, Any]:
    """Delete a research assumption."""
    store = _get_research_store()
    if not store.delete_assumption(assumption_id):
        raise HTTPException(status_code=404, detail="Assumption not found")
    return {"success": True}


@app.patch("/governor/research/assumptions/{assumption_id}/status")
async def change_assumption_status(
    assumption_id: str, request: StatusChangeRequest
) -> dict[str, Any]:
    """Change an assumption's status."""
    from governor.research_store import AssumptionStatus

    try:
        status = AssumptionStatus(request.status)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status: {request.status}. Valid: {[s.value for s in AssumptionStatus]}",
        )

    store = _get_research_store()
    try:
        assumption = store.update_assumption_status(assumption_id, status)
    except KeyError:
        raise HTTPException(status_code=404, detail="Assumption not found")
    return {"success": True, "assumption": assumption.to_dict()}


@app.post("/governor/research/uncertainties")
async def add_research_uncertainty(request: UncertaintyRequest) -> dict[str, Any]:
    """Add a research uncertainty."""
    store = _get_research_store()
    uncertainty = store.add_uncertainty(
        content=request.content, attached_to=request.attached_to
    )
    return {"success": True, "uncertainty": uncertainty.to_dict()}


@app.delete("/governor/research/uncertainties/{uncertainty_id}")
async def delete_research_uncertainty(uncertainty_id: str) -> dict[str, Any]:
    """Delete a research uncertainty."""
    store = _get_research_store()
    if not store.delete_uncertainty(uncertainty_id):
        raise HTTPException(status_code=404, detail="Uncertainty not found")
    return {"success": True}


@app.patch("/governor/research/uncertainties/{uncertainty_id}/status")
async def change_uncertainty_status(
    uncertainty_id: str, request: StatusChangeRequest
) -> dict[str, Any]:
    """Change an uncertainty's status."""
    from governor.research_store import UncertaintyStatus

    try:
        status = UncertaintyStatus(request.status)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status: {request.status}. Valid: {[s.value for s in UncertaintyStatus]}",
        )

    store = _get_research_store()
    try:
        uncertainty = store.update_uncertainty_status(uncertainty_id, status)
    except KeyError:
        raise HTTPException(status_code=404, detail="Uncertainty not found")
    return {"success": True, "uncertainty": uncertainty.to_dict()}


@app.post("/governor/research/links")
async def add_research_link(request: LinkRequest) -> dict[str, Any]:
    """Add a typed link between research items."""
    from governor.research_store import LinkType

    try:
        link_type = LinkType(request.link_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid link type: {request.link_type}. Valid: {[t.value for t in LinkType]}",
        )

    store = _get_research_store()
    link = store.add_link(
        link_type=link_type,
        from_id=request.from_id,
        to_id=request.to_id,
        subtype=request.subtype,
    )
    return {"success": True, "link": link.to_dict()}


@app.delete("/governor/research/links/{link_id}")
async def delete_research_link(link_id: str) -> dict[str, Any]:
    """Delete a research link."""
    store = _get_research_store()
    if not store.remove_link(link_id):
        raise HTTPException(status_code=404, detail="Link not found")
    return {"success": True}


# =============================================================================
# Research Capture (pending promotion pipeline — claims, citations, source refs)
# =============================================================================

# In-memory pending research captures (per process; cleared on restart)
_pending_research_captures: dict[str, dict[str, Any]] = {}
_research_capture_counter: int = 0


@app.post("/governor/research/capture/scan")
async def research_capture_scan(request: CaptureRequest) -> dict[str, Any]:
    """Scan text for claims, citations, and structured source refs."""
    global _research_capture_counter

    try:
        from governor.capture import ResearchCaptureClassifier
    except ImportError:
        return {"captures": [], "error": "Research capture classifier not available."}

    classifier = ResearchCaptureClassifier()
    items, receipt = classifier.scan(request.text, message_id=request.message_id)

    captures = []
    for item in items:
        _research_capture_counter += 1
        cap_id = f"rcap-{_research_capture_counter}"
        cap = {
            "id": cap_id,
            "kind": item.kind if isinstance(item.kind, str) else item.kind.value,
            "confidence": round(item.confidence, 2),
            "subject": item.subject_guess or "",
            "statement": item.statement,
            "field": item.field_guess or "",
            "spans": [list(s) for s in item.evidence_spans],
            "message_id": request.message_id,
            "status": "pending",
        }
        if item.draft_payload:
            cap["draft"] = item.draft_payload
        _pending_research_captures[cap_id] = cap
        captures.append(cap)

    return {
        "captures": captures,
        "receipt": {
            "classifier_version": receipt.classifier_version,
            "content_hash": receipt.content_hash,
            "pattern_hits": receipt.pattern_hits,
        },
    }


@app.get("/governor/research/captures")
async def list_pending_research_captures() -> dict[str, Any]:
    """List all pending (unresolved) research captures."""
    pending = [c for c in _pending_research_captures.values() if c["status"] == "pending"]
    return {"captures": pending, "count": len(pending)}


@app.post("/governor/research/capture/{capture_id}/accept")
async def accept_research_capture(capture_id: str) -> dict[str, Any]:
    """Promote a pending research capture to the claim ledger."""
    if capture_id not in _pending_research_captures:
        raise HTTPException(status_code=404, detail="Capture not found.")

    cap = _pending_research_captures[capture_id]
    if cap["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"Capture already {cap['status']}.")

    store = _get_research_store()

    kind = cap.get("kind", "claim")
    statement = cap.get("statement", "")
    draft = cap.get("draft", {})
    source_ref = draft.get("source_ref", "")

    if kind in ("claim", "experiment"):
        claim = store.add_claim(
            content=statement,
            source_ref=source_ref,
            captured_from=capture_id,
        )
        cap["status"] = "accepted"
        cap["promoted_to"] = claim.id
        return {"success": True, "message": f"Claim: {statement[:40]}", "id": claim.id}

    elif kind == "citation":
        # Citations with source_ref → claim with ref provenance
        label = source_ref or statement
        claim = store.add_claim(
            content=label,
            source_ref=source_ref,
            captured_from=capture_id,
        )
        cap["status"] = "accepted"
        cap["promoted_to"] = claim.id
        return {"success": True, "message": f"Source: {label[:40]}", "id": claim.id}

    elif kind == "assumption":
        assumption = store.add_assumption(content=statement)
        cap["status"] = "accepted"
        cap["promoted_to"] = assumption.id
        return {"success": True, "message": f"Assumption: {statement[:40]}", "id": assumption.id}

    raise HTTPException(status_code=400, detail=f"Unknown capture kind: {kind}")


@app.post("/governor/research/capture/{capture_id}/reject")
async def reject_research_capture(capture_id: str) -> dict[str, Any]:
    """Reject a pending research capture."""
    if capture_id not in _pending_research_captures:
        raise HTTPException(status_code=404, detail="Capture not found.")

    cap = _pending_research_captures[capture_id]
    cap["status"] = "rejected"
    return {"success": True, "message": "Capture dismissed."}


@app.post("/governor/research/why")
async def research_why_overlay(request: Request) -> dict[str, Any]:
    """Per-turn Why overlay: what was injected vs what the assistant referenced.

    Expects: {"text": "assistant response text"}
    Returns: WhyOverlay dict with injected/referenced/floating/matched.
    """
    from governor.research_why import build_why_overlay

    body = await request.json()
    text = body.get("text", "")

    ctx, _ = _resolve_context()
    accepted_sources: list[str] = []
    accepted_claim_ids: list[str] = []

    if ctx is not None:
        try:
            from governor.research_store import ResearchStore
            store = ResearchStore(ctx.governor_dir)
            # Mirror the logic from _build_accepted_context
            active_claims = [
                c for c in store.claims.values()
                if c.status.value not in ("retracted", "superseded")
            ]
            active_claims.sort(key=lambda c: c.created_at, reverse=True)
            active_claims = active_claims[:20]

            seen_refs: set[str] = set()
            for claim in active_claims:
                if claim.source_ref and claim.source_ref not in seen_refs:
                    accepted_sources.append(claim.source_ref)
                    seen_refs.add(claim.source_ref)
            accepted_sources = accepted_sources[:25]
            accepted_claim_ids = [c.id for c in active_claims]
        except Exception:
            pass

    overlay = build_why_overlay(text, accepted_sources, accepted_claim_ids)
    return overlay.to_dict()


# ============================================================================
# Research Project Endpoints (structured workflow — parallel to code builder)
# ============================================================================


@app.get("/governor/research/project")
async def research_project_state() -> dict[str, Any]:
    """Full research project state for sidebar polling."""
    try:
        store = _get_research_project_store()
        return store.get_state()
    except Exception:
        return {"version": 0, "intent": {"text": "", "locked": False},
                "contract": {}, "plan": {"phases": []}, "files": {}}


@app.put("/governor/research/project/intent")
async def research_update_intent(request: IntentUpdateRequest) -> dict[str, Any]:
    """Update thesis / research question."""
    from gov_webui.project_store import StaleVersionError

    store = _get_research_project_store()
    try:
        intent = store.update_intent(
            request.text, request.locked, request.expected_version
        )
        return {"success": True, "intent": intent.model_dump()}
    except StaleVersionError:
        raise HTTPException(409, "Stale version")


@app.put("/governor/research/project/contract")
async def research_update_contract(request: ContractUpdateRequest) -> dict[str, Any]:
    """Update research scope / methodology."""
    from gov_webui.project_store import Contract, ContractField, StaleVersionError, compute_config_hash

    inputs = [ContractField(**f) for f in request.inputs]
    outputs = [ContractField(**f) for f in request.outputs]

    # Server always recomputes hash — never trust client hashes
    config = request.config
    config_hash = None
    config_hash_full = None
    if config:
        # Guard against oversized configs (50KB soft cap)
        config_size = len(json.dumps(config))
        if config_size > 50_000:
            raise HTTPException(400, f"Config too large: {config_size} bytes (max 50000)")
        short, full = compute_config_hash(config)
        if request.config_hash and request.config_hash != short:
            raise HTTPException(400, f"Config hash mismatch: got {request.config_hash}, expected {short}")
        config_hash = short
        config_hash_full = full

    contract = Contract(
        description=request.description,
        inputs=inputs,
        outputs=outputs,
        constraints=request.constraints,
        transport=request.transport,
        config=config,
        config_hash=config_hash,
        config_hash_full=config_hash_full,
    )
    store = _get_research_project_store()
    try:
        result = store.update_contract(contract, request.expected_version)
        return {"success": True, "contract": result.model_dump()}
    except StaleVersionError:
        raise HTTPException(409, "Stale version")


@app.post("/governor/research/project/plan/item")
async def research_add_plan_item(request: PlanItemRequest) -> dict[str, Any]:
    """Add a plan item to a research phase."""
    store = _get_research_project_store()
    try:
        item = store.add_plan_item(request.phase_idx, request.text)
        return {"success": True, "item": item.model_dump()}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.patch("/governor/research/project/plan/item/{item_id}")
async def research_update_plan_item(
    item_id: str, request: PlanItemStatusRequest
) -> dict[str, Any]:
    """Update research plan item status."""
    from gov_webui.project_store import PlanItemStatus, StaleVersionError

    store = _get_research_project_store()
    try:
        status = PlanItemStatus(request.status)
    except ValueError:
        raise HTTPException(400, f"Invalid status: {request.status}")

    try:
        item = store.update_item_status(
            item_id, status, request.expected_version
        )
        return {"success": True, "item": item.model_dump()}
    except StaleVersionError:
        raise HTTPException(409, "Stale version")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/governor/research/project/plan/phase")
async def research_add_phase(request: PhaseRequest) -> dict[str, Any]:
    """Add a new phase to the research plan."""
    store = _get_research_project_store()
    phase = store.add_phase(request.name)
    return {"success": True, "phase": phase.model_dump()}


@app.patch("/governor/research/project/plan/phase/{idx}")
async def research_update_phase(idx: int, request: PhaseUpdateRequest) -> dict[str, Any]:
    """Update research phase name/lock."""
    store = _get_research_project_store()
    try:
        phase = store.update_phase(idx, request.name, request.locked)
        return {"success": True, "phase": phase.model_dump()}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/governor/research/project/files")
async def research_list_files() -> dict[str, Any]:
    """List research drafts/notes with versions + hashes."""
    store = _get_research_project_store()
    return {"files": store.list_files()}


@app.get("/governor/research/project/file-prev/{path:path}")
async def research_get_file_prev(path: str) -> dict[str, Any]:
    """Get previous version of a draft for client-side diff."""
    store = _get_research_project_store()
    try:
        content = store.get_file_prev(path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"filepath": path, "content": content}


@app.get("/governor/research/project/files/{path:path}")
async def research_get_file(path: str) -> dict[str, Any]:
    """Get draft/note content."""
    store = _get_research_project_store()
    try:
        content = store.get_file_content(path)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if content is None:
        raise HTTPException(404, f"File not found: {path}")
    return {"filepath": path, "content": content}


@app.put("/governor/research/project/files/{path:path}")
async def research_put_file(path: str, request: FileUpdateRequest) -> dict[str, Any]:
    """Accept draft/note, returns version + hash."""
    store = _get_research_project_store()
    try:
        entry = store.put_file(path, request.content, request.turn_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "success": True,
        "filepath": path,
        "version": entry.version,
        "content_hash": entry.content_hash,
    }


@app.post("/governor/research/project/validate")
async def research_validate(request: RunRequest) -> dict[str, Any]:
    """Validate research drafts: citation checks, claim consistency.

    Uses the same request shape as /code/run for UI symmetry, but instead
    of executing code, runs text-based validation against registered claims.
    """
    store = _get_research_project_store()
    state = store.get_state()
    files = state.get("files", {})

    if not files:
        raise HTTPException(400, "No drafts in project")

    if request.filepath not in files:
        raise HTTPException(404, f"Draft not found: {request.filepath}")

    draft_content = store.get_file_content(request.filepath)
    if draft_content is None:
        raise HTTPException(404, f"Draft content missing: {request.filepath}")

    # -- Validation checks --------------------------------------------------
    findings: list[str] = []

    # 1. Cross-check against registered research claims
    try:
        rstore = _get_research_store()
        rstate = rstore.get_state()
        claims = rstate.get("claims", [])
        assumptions = rstate.get("assumptions", [])

        # Check for floating claims not referenced in draft
        for claim in claims:
            status = claim.get("status", "floating")
            content = claim.get("content", "")
            if status == "floating" and content:
                # Simple heuristic: first 40 chars as search key
                key = content[:40].lower()
                if key not in draft_content.lower():
                    findings.append(
                        f"Floating claim not referenced in draft: "
                        f"\"{content[:60]}{'...' if len(content) > 60 else ''}\""
                    )

        # Check for unresolved assumptions
        for assumption in assumptions:
            status = assumption.get("status", "proposed")
            content = assumption.get("content", "")
            if status == "proposed" and content:
                findings.append(
                    f"Unresolved assumption: "
                    f"\"{content[:60]}{'...' if len(content) > 60 else ''}\""
                )
    except Exception:
        pass  # Research store may not be initialized — that's fine

    # 2. Check for common citation patterns + config bans
    import re

    contract = state.get("contract", {})
    config = contract.get("config")

    # Build weasel patterns: default set + config bans
    weasel_patterns = [
        r"studies\s+show",
        r"research\s+suggests",
        r"it\s+is\s+widely\s+accepted",
        r"experts\s+agree",
        r"it\s+is\s+well\s+known",
        r"evidence\s+suggests",
    ]

    # Config bans feed the weasel validator as literal (case-insensitive) matches
    ban_literals: list[str] = []
    if config and config.get("bans"):
        for ban in config["bans"]:
            ban_stripped = ban.strip()
            if ban_stripped:
                ban_literals.append(ban_stripped.lower())

    lines = draft_content.split("\n")
    for i, line in enumerate(lines):
        line_lower = line.lower()
        # Check regex weasel patterns
        for pattern in weasel_patterns:
            if re.search(pattern, line_lower):
                # Check if line has a citation marker [N] or (Author, YYYY)
                if not re.search(r"\[\d+\]|\([A-Z]\w+,?\s*\d{4}\)", line):
                    findings.append(
                        f"Line {i + 1}: \"{pattern}\" without citation — "
                        f"\"{line.strip()[:60]}{'...' if len(line.strip()) > 60 else ''}\""
                    )
        # Check literal ban matches
        for ban in ban_literals:
            if ban in line_lower:
                findings.append(
                    f"Line {i + 1}: banned phrase \"{ban}\" — "
                    f"\"{line.strip()[:60]}{'...' if len(line.strip()) > 60 else ''}\""
                )

    # 3. Config-aware typed checks
    if config:
        # Table format check: config.format.tables == false
        fmt = config.get("format", {})
        if isinstance(fmt, dict) and fmt.get("tables") is False:
            # Detect markdown tables: header row + |---| separator row
            for i, line in enumerate(lines):
                if i + 1 < len(lines) and re.search(r"\|.*\|", line):
                    next_line = lines[i + 1]
                    if re.search(r"\|[\s-]+\|", next_line):
                        findings.append(
                            f"Line {i + 1}: markdown table found but tables are disabled"
                        )

        # Length band check
        length_band = config.get("length")
        if length_band and length_band in _LENGTH_BANDS:
            lo, hi = _LENGTH_BANDS[length_band]
            word_count = len(draft_content.split())
            if word_count < lo:
                findings.append(
                    f"Word count {word_count} below minimum {lo} for '{length_band}' band"
                )
            elif word_count > hi:
                findings.append(
                    f"Word count {word_count} above maximum {hi} for '{length_band}' band"
                )

        # Citations-required check (warn only)
        if config.get("citations") == "required":
            if not re.search(r"\[\d+\]|\([A-Z]\w+,?\s*\d{4}\)", draft_content):
                findings.append(
                    "Citations required but no citation markers found"
                )

    # 4. Check for constraint violations from research scope
    constraints = contract.get("constraints", [])
    draft_lower = draft_content.lower()
    constraint_hits: list[str] = []
    for constraint in constraints:
        c_lower = constraint.lower()
        if any(neg in c_lower for neg in ["no ", "avoid ", "don't ", "never "]):
            for neg in ["no ", "avoid ", "don't ", "never "]:
                if c_lower.startswith(neg):
                    term = c_lower[len(neg):].strip().rstrip(".")
                    if term and term in draft_lower:
                        constraint_hits.append(
                            f"Scope constraint may be violated: \"{constraint}\""
                        )
                    break

    findings.extend(constraint_hits)

    # 5. Strict mode: when strict=true, all findings are hard fails
    strict = config.get("strict", False) if config else False
    success = len(findings) == 0 if strict else (
        len([f for f in findings if "banned phrase" in f or "constraint" in f.lower()]) == 0
        if not strict and findings else len(findings) == 0
    )
    # Simplify: strict=true means any finding fails, strict=false means success unless findings
    success = len(findings) == 0

    return {
        "success": success,
        "returncode": 0 if success else 1,
        "stdout": f"Validated {request.filepath}: "
                  + (f"{len(findings)} finding(s)" if findings else "no issues found"),
        "stderr": "\n".join(findings) if findings else "",
        "filepath": request.filepath,
        "preflight_hit": False,
        "forced": False,
        "findings": findings,
    }


@app.get("/governor/corrections")
async def list_corrections(limit: int = 20) -> dict[str, Any]:
    """List past corrections/resolutions."""
    ctx, _ = _resolve_context()
    if ctx is None:
        return {"corrections": [], "message": "No governor context initialized."}

    resolver = ViolationResolver(
        governor_dir=ctx.governor_dir,
        mode=ctx.mode,
        context_id=ctx.context_id,
    )
    exceptions = resolver.list_exceptions()

    corrections = []
    for exc in exceptions[:limit]:
        corrections.append({
            "id": exc.id,
            "action": exc.action.value,
            "anchor_id": exc.anchor_id,
            "summary": exc.scope,
            "created_at": exc.created_at.isoformat() if exc.created_at else None,
        })

    return {"corrections": corrections}


# ============================================================================
# Sidecar UI
# ============================================================================

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/governor/ui", response_class=HTMLResponse)
async def governor_ui() -> HTMLResponse:
    """Serve the single-page Governor UI."""
    html_path = _STATIC_DIR / "governor.html"
    return HTMLResponse(
        content=html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


# ============================================================================
# Health / Root
# ============================================================================


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    backend_ok = False
    bridge = _get_bridge()

    try:
        await bridge.list_models()
        backend_ok = True
    except Exception:
        pass

    cm = _get_context_manager()
    ctx = cm.get(GOVERNOR_CONTEXT_ID)

    return {
        "status": "healthy" if backend_ok else "degraded",
        "backend": {
            "type": _current_backend_type,
            "connected": backend_ok,
        },
        "governor": {
            "context_id": GOVERNOR_CONTEXT_ID,
            "mode": GOVERNOR_MODE,
            "initialized": ctx is not None,
        },
    }


@app.get("/")
async def root() -> HTMLResponse:
    """Serve the combined chat + governor UI."""
    html_path = _STATIC_DIR / "index.html"
    return HTMLResponse(
        content=html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


# ============================================================================
# Export / Import
# ============================================================================


@app.get("/governor/export")
async def export_governor_state() -> dict[str, Any]:
    """Export all governor state as a single JSON object for portability."""
    ctx, _ = _resolve_context()
    if ctx is None:
        return {"mode": GOVERNOR_MODE, "anchors": [], "corrections": []}

    from governor.continuity import AnchorType, create_registry

    registry = create_registry(ctx.governor_dir)
    anchors = registry.all()

    # Serialize all anchors with full data
    anchor_list = []
    for a in anchors:
        entry: dict[str, Any] = {
            "id": a.id,
            "anchor_type": a.anchor_type.value if hasattr(a.anchor_type, "value") else str(a.anchor_type),
            "description": a.description,
            "severity": a.severity.value if hasattr(a.severity, "value") else str(a.severity),
        }
        if a.required_patterns:
            entry["required_patterns"] = a.required_patterns
        if a.forbidden_patterns:
            entry["forbidden_patterns"] = a.forbidden_patterns
        anchor_list.append(entry)

    # Corrections (exception log)
    corrections = []
    try:
        from governor.violation_resolver import ViolationResolver

        resolver = ViolationResolver(
            governor_dir=ctx.governor_dir,
            mode=ctx.mode,
            context_id=ctx.context_id,
        )
        for exc in resolver.list_exceptions():
            corrections.append({
                "action": exc.action.value if hasattr(exc.action, "value") else str(exc.action),
                "anchor_id": exc.anchor_id,
                "scope": exc.scope,
                "summary": getattr(exc, "summary", ""),
            })
    except Exception:
        pass

    result = {
        "version": 1,
        "mode": GOVERNOR_MODE,
        "exported_at": __import__("datetime").datetime.now().isoformat(),
        "anchors": anchor_list,
        "corrections": corrections,
    }

    # Research mode: include research store data
    if ctx.mode == "research":
        try:
            from governor.research_store import ResearchStore

            store = ResearchStore(ctx.governor_dir)
            result["research"] = store.export_data()
        except Exception:
            pass

    return result


@app.post("/governor/import")
async def import_governor_state(payload: dict[str, Any]) -> dict[str, Any]:
    """Import governor state from an exported JSON object."""
    ctx, _ = _resolve_context()
    if ctx is None:
        raise HTTPException(status_code=400, detail="No governor context initialized.")

    from governor.continuity import Anchor, AnchorType, Severity, create_registry

    registry = create_registry(ctx.governor_dir)

    anchors_data = payload.get("anchors", [])
    imported = 0
    skipped = 0

    type_map = {t.value: t for t in AnchorType}
    sev_map = {s.value: s for s in Severity}

    for entry in anchors_data:
        anchor_id = entry.get("id", "")
        if not anchor_id:
            skipped += 1
            continue

        # Skip if already exists
        if registry.get(anchor_id) is not None:
            skipped += 1
            continue

        anchor_type = type_map.get(entry.get("anchor_type", ""), AnchorType.CANON)
        severity = sev_map.get(entry.get("severity", ""), Severity.REJECT)

        anchor = Anchor(
            id=anchor_id,
            anchor_type=anchor_type,
            description=entry.get("description", ""),
            required_patterns=entry.get("required_patterns", []),
            forbidden_patterns=entry.get("forbidden_patterns", []),
            severity=severity,
        )
        registry.register(anchor)
        imported += 1

    if imported > 0:
        registry.save(ctx.governor_dir / "continuity" / "anchors.json")

    # Research mode: import research store data
    research_imported = 0
    if ctx.mode == "research" and "research" in payload:
        try:
            from governor.research_store import ResearchStore

            store = ResearchStore(ctx.governor_dir)
            research_imported = store.import_data(payload["research"])
        except Exception:
            pass

    total_imported = imported + research_imported

    return {
        "success": True,
        "imported": total_imported,
        "skipped": skipped,
        "message": f"Imported {total_imported} item(s), skipped {skipped} duplicate(s).",
    }


@app.get("/api/info")
async def api_info() -> dict[str, Any]:
    """JSON endpoint with API info and available endpoints."""
    return {
        "name": "Governor Chat Adapter",
        "version": "0.4.0",
        "backend": _current_backend_type,
        "openai_compatible": True,
        "governor_context": GOVERNOR_CONTEXT_ID,
        "governor_mode": GOVERNOR_MODE,
        "endpoints": {
            "ui": "/",
            "models": "/v1/models",
            "chat": "/v1/chat/completions",
            "backends": "/v1/backends",
            "backends_switch": "/v1/backends/switch",
            "health": "/health",
            "api_info": "/api/info",
            # Sessions
            "sessions_list": "/sessions/",
            "sessions_create": "/sessions/",
            "sessions_get": "/sessions/{id}",
            "sessions_delete": "/sessions/{id}",
            "sessions_update": "/sessions/{id}",
            "sessions_append_message": "/sessions/{id}/messages",
            # Governor
            "governor_contexts": "/governor/contexts",
            "governor_status": "/governor/status",
            "governor_now": "/governor/now",
            "governor_why": "/governor/why",
            "governor_history": "/governor/history",
            "governor_detail": "/governor/detail/{item_id}",
            "governor_corrections": "/governor/corrections",
            "governor_ui": "/governor/ui",
            # Fiction mode
            "fiction_characters": "/governor/fiction/characters",
            "fiction_world_rules": "/governor/fiction/world-rules",
            "fiction_forbidden": "/governor/fiction/forbidden",
            # Code mode
            "code_decisions": "/governor/code/decisions",
            "code_constraints": "/governor/code/constraints",
            # Research mode
            "research_state": "/governor/research/state",
            "research_claims": "/governor/research/claims",
            "research_assumptions": "/governor/research/assumptions",
            "research_uncertainties": "/governor/research/uncertainties",
            "research_links": "/governor/research/links",
            # Export/Import
            "governor_export": "/governor/export",
            "governor_import": "/governor/import",
            # V2 Dashboard
            "v2_runs": "/v2/runs",
            "v2_run_detail": "/v2/runs/{run_id}",
            "v2_run_events": "/v2/runs/{run_id}/events",
            "v2_run_claims": "/v2/runs/{run_id}/claims",
            "v2_run_violations": "/v2/runs/{run_id}/violations",
            "v2_run_report": "/v2/runs/{run_id}/report",
            "v2_run_cancel": "/v2/runs/{run_id}/cancel",
            "v2_runs_compare": "/v2/runs/compare",
            "v2_artifacts": "/v2/artifacts",
            "v2_artifact": "/v2/artifacts/{hash}",
            "v2_controls_schema": "/v2/controls/schema",
            "v2_controls_templates": "/v2/controls/templates",
            "v2_profiles": "/v2/profiles",
            "v2_anchors": "/v2/anchors",
            "v2_backends": "/v2/backends",
            "v2_dashboard_summary": "/v2/dashboard/summary",
            "v2_dashboard_regime": "/v2/dashboard/regime",
            "v2_demos": "/v2/demos",
            "v2_demo_playwright": "/v2/demos/{name}/playwright",
            "dashboard": "/dashboard",
            # V2 Intent Compiler
            "v2_intent_templates": "/v2/intent/templates",
            "v2_intent_schema": "/v2/intent/schema/{template_name}",
            "v2_intent_validate": "/v2/intent/validate",
            "v2_intent_compile": "/v2/intent/compile",
            "v2_intent_policy": "/v2/intent/policy",
        },
    }


# ============================================================================
# V2 Dashboard API — Run-centric governance dashboard
# ============================================================================

from governor.dashboard_ux import (
    DashboardStore,
    RunSummary,
    RunVerdict,
    StreamEvent,
    StreamEventType,
    CancelRequest,
    build_controls_schema,
    BUILTIN_TEMPLATES,
    BUILTIN_ACTIONS,
    generate_report,
    make_heartbeat,
)
from governor.instrument import InstrumentSystem, RunManifest, EventWriter

# Lazy-init singletons for v2 dashboard
_dashboard_store: DashboardStore | None = None
_instrument_system: InstrumentSystem | None = None


def _get_dashboard_store() -> DashboardStore:
    global _dashboard_store
    if _dashboard_store is None:
        cm = _get_context_manager()
        ctx = cm.get_or_create(GOVERNOR_CONTEXT_ID, mode=GOVERNOR_MODE)
        _dashboard_store = DashboardStore(ctx.governor_dir)
    return _dashboard_store


def _get_instrument_system() -> InstrumentSystem:
    global _instrument_system
    if _instrument_system is None:
        cm = _get_context_manager()
        ctx = cm.get_or_create(GOVERNOR_CONTEXT_ID, mode=GOVERNOR_MODE)
        _instrument_system = InstrumentSystem(ctx.governor_dir)
    return _instrument_system


# Pydantic models for v2 API

class CreateRunRequest(BaseModel):
    task: str
    profile: str = "established"
    backend: str = ""
    scope: list[str] = Field(default_factory=list)
    seed: int | None = None


class CancelRunResponse(BaseModel):
    run_id: str
    acknowledged_at: str


# Track active cancel requests
_cancel_requests: dict[str, CancelRequest] = {}


# ---- Runs ----

@app.post("/v2/runs")
async def v2_create_run(request: CreateRunRequest) -> dict[str, Any]:
    """Create a new instrumented run."""
    from governor.instrument import Actor, ActorKind, InstrumentProfile, RunInputs

    system = _get_instrument_system()
    store = _get_dashboard_store()

    # Map profile string to InstrumentProfile
    profile_map = {
        "greenfield": InstrumentProfile.GREENFIELD,
        "strict": InstrumentProfile.STRICT,
        "forensic": InstrumentProfile.FORENSIC,
    }
    profile = profile_map.get(request.profile, InstrumentProfile.GREENFIELD)

    manifest, writer = system.start_run(
        actor=Actor(ActorKind.HUMAN, "dashboard"),
        profile=profile,
        task_id=request.task,
    )

    # Record in dashboard store
    summary = RunSummary(
        run_id=manifest.run_id,
        created_at=manifest.created_at,
        model=request.backend or _current_backend_type,
        profile=request.profile,
        verdict=RunVerdict.PENDING,
        task=request.task,
    )
    store.record_run(summary)

    return {
        "run_id": manifest.run_id,
        "created_at": manifest.created_at,
        "profile": request.profile,
        "task": request.task,
    }


@app.get("/v2/runs")
async def v2_list_runs(
    profile: str = "",
    verdict: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    """List runs with optional filters."""
    store = _get_dashboard_store()
    runs = store.list_runs(profile=profile, verdict=verdict, limit=limit)
    return {"runs": [r.to_dict() for r in runs]}


@app.get("/v2/runs/{run_id}")
async def v2_get_run(run_id: str) -> dict[str, Any]:
    """Get run detail (manifest + summary)."""
    system = _get_instrument_system()
    store = _get_dashboard_store()

    manifest = system.run_store.load_manifest(run_id)
    summary = store.get_run(run_id)

    if manifest is None and summary is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    result: dict[str, Any] = {}
    if manifest:
        result["manifest"] = manifest.to_dict()
    if summary:
        result["summary"] = summary.to_dict()

    return result


@app.get("/v2/runs/{run_id}/events")
async def v2_run_events(run_id: str, stream: bool = False) -> Any:
    """Get events for a run. If stream=true, returns SSE."""
    system = _get_instrument_system()
    run_dir = system.instrument_dir / "runs" / run_id

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    writer = EventWriter(
        run_dir, system.artifact_store, system.config.artifact_size_threshold
    )

    if stream:
        async def event_stream():
            events = writer.read_events()
            for ev in events:
                se = StreamEvent(
                    event_type=StreamEventType.EVENT,
                    data=ev.to_dict(),
                )
                yield se.to_sse() + "\n"
            # End with heartbeat
            yield make_heartbeat().to_sse() + "\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    events = writer.read_events()
    return {"events": [e.to_dict() for e in events]}


@app.get("/v2/runs/{run_id}/claims")
async def v2_run_claims(run_id: str) -> dict[str, Any]:
    """Get claims for a run."""
    from governor.instrument import ClaimExtractor

    system = _get_instrument_system()
    run_dir = system.instrument_dir / "runs" / run_id

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    extractor = ClaimExtractor(run_dir)
    claims = extractor.read_claims()
    return {"claims": [c.to_dict() for c in claims]}


@app.get("/v2/runs/{run_id}/violations")
async def v2_run_violations(run_id: str) -> dict[str, Any]:
    """Get violations for a run (policy decisions with non-pass verdict)."""
    system = _get_instrument_system()
    run_dir = system.instrument_dir / "runs" / run_id

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    from governor.instrument import EventKind

    writer = EventWriter(
        run_dir, system.artifact_store, system.config.artifact_size_threshold
    )
    events = writer.read_events()

    violations = []
    for ev in events:
        if ev.kind == EventKind.POLICY_DECISION:
            verdict = ev.payload.get("verdict", "")
            if verdict and verdict != "pass":
                violations.append(ev.to_dict())

    return {"violations": violations}


@app.get("/v2/runs/{run_id}/report")
async def v2_run_report(run_id: str) -> dict[str, Any]:
    """Generate report for a run."""
    system = _get_instrument_system()
    store = _get_dashboard_store()

    manifest = system.run_store.load_manifest(run_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    from governor.instrument import ClaimExtractor

    run_dir = system.instrument_dir / "runs" / run_id
    writer = EventWriter(
        run_dir, system.artifact_store, system.config.artifact_size_threshold
    )
    events = writer.read_events()

    extractor = ClaimExtractor(run_dir)
    claims = extractor.read_claims()

    report = generate_report(
        run_id=run_id,
        manifest=manifest.to_dict(),
        events=[e.to_dict() for e in events],
        claims=[c.to_dict() for c in claims],
    )

    store.save_report(report)
    return report.to_dict()


@app.post("/v2/runs/{run_id}/cancel")
async def v2_cancel_run(run_id: str) -> dict[str, Any]:
    """Cancel an active run."""
    cancel = CancelRequest(run_id=run_id)
    cancel.acknowledge()
    _cancel_requests[run_id] = cancel

    return {
        "run_id": run_id,
        "acknowledged_at": cancel.acknowledged_at,
    }


@app.post("/v2/runs/compare")
async def v2_compare_runs() -> dict[str, Any]:
    """Placeholder for interferometry comparison."""
    return {"status": "not_implemented", "message": "Use /governor/code/compare instead."}


# ---- Artifacts ----

@app.get("/v2/artifacts/{artifact_hash}")
async def v2_get_artifact(artifact_hash: str) -> Any:
    """Retrieve a content-addressed artifact blob."""
    system = _get_instrument_system()
    data = system.artifact_store.retrieve(artifact_hash)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_hash}")

    from fastapi.responses import Response
    return Response(content=data, media_type="application/octet-stream")


@app.get("/v2/artifacts")
async def v2_list_artifacts(run_id: str = "") -> dict[str, Any]:
    """List artifacts for a run."""
    if not run_id:
        return {"artifacts": []}

    system = _get_instrument_system()
    run_dir = system.instrument_dir / "runs" / run_id
    if not run_dir.exists():
        return {"artifacts": []}

    writer = EventWriter(
        run_dir, system.artifact_store, system.config.artifact_size_threshold
    )
    receipts = writer.read_receipts()
    return {"artifacts": [r.to_dict() for r in receipts]}


# ---- Controls ----

@app.get("/v2/controls/schema")
async def v2_controls_schema() -> dict[str, Any]:
    """Return controls schema for the dashboard left panel."""
    return build_controls_schema()


@app.get("/v2/controls/templates")
async def v2_controls_templates() -> dict[str, Any]:
    """Return built-in run templates."""
    return {"templates": [t.to_dict() for t in BUILTIN_TEMPLATES]}


@app.get("/v2/profiles")
async def v2_list_profiles() -> dict[str, Any]:
    """Return available governance profiles."""
    from governor.profiles import BUILTIN_PROFILES

    return {"profiles": list(BUILTIN_PROFILES.keys())}


@app.get("/v2/anchors")
async def v2_list_anchors() -> dict[str, Any]:
    """Return active anchors (read-only)."""
    ctx, _ = _resolve_context()
    if ctx is None:
        return {"anchors": []}

    from governor.continuity import create_registry

    try:
        registry = create_registry(ctx.governor_dir)
        anchors = registry.all()
        return {
            "anchors": [
                {
                    "id": a.id,
                    "type": a.anchor_type.value if hasattr(a.anchor_type, "value") else str(a.anchor_type),
                    "description": a.description,
                    "severity": a.severity.value if hasattr(a.severity, "value") else str(a.severity),
                }
                for a in anchors
            ]
        }
    except Exception:
        return {"anchors": []}


@app.get("/v2/backends")
async def v2_list_backends() -> dict[str, Any]:
    """List available backends (delegates to v1)."""
    return await list_backends()


@app.post("/v2/backends/switch")
async def v2_switch_backend(request: BackendSwitchRequest) -> dict[str, Any]:
    """Switch backend (delegates to v1)."""
    return await switch_backend(request)


# ---- Dashboard ----

@app.get("/v2/dashboard/summary")
async def v2_dashboard_summary() -> dict[str, Any]:
    """Return aggregate dashboard statistics."""
    store = _get_dashboard_store()
    summary = store.dashboard_summary()
    return summary.to_dict()


@app.get("/v2/dashboard/regime")
async def v2_dashboard_regime() -> dict[str, Any]:
    """Return current regime state."""
    ctx, _ = _resolve_context()
    if ctx is None:
        return {"regime": None}

    vm = _build_vm_for_context(ctx)
    return {
        "regime": vm.regime.name if vm.regime else None,
        "session": vm.session.to_dict() if vm.session else {},
    }


# ============================================================================
# V2 Intent Compiler — Structured hypothesis-collapse for governance sessions
# ============================================================================


class IntentValidateRequest(BaseModel):
    schema_id: str
    values: dict[str, Any]
    escape_text: str | None = None


class IntentCompileRequest(BaseModel):
    schema_id: str
    values: dict[str, Any]
    escape_text: str | None = None
    template_name: str = "session_start"


@app.get("/v2/intent/templates")
async def v2_intent_templates() -> dict[str, Any]:
    """List available intent form templates."""
    from governor.intent_compiler import list_templates
    return {"templates": list_templates()}


@app.get("/v2/intent/schema/{template_name}")
async def v2_intent_schema(template_name: str) -> dict[str, Any]:
    """Build and return an IntentFormSchema for the current mode."""
    from governor.intent_compiler import build_form_schema

    ctx, _ = _resolve_context()
    mode = ctx.mode if ctx else GOVERNOR_MODE

    try:
        schema = build_form_schema(template_name, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return schema.to_dict()


@app.post("/v2/intent/validate")
async def v2_intent_validate(request: IntentValidateRequest) -> dict[str, Any]:
    """Validate a response against its schema."""
    from governor.intent_compiler import (
        IntentFormResponse,
        IntentFormSchema,
        build_form_schema,
        validate_response,
    )

    ctx, _ = _resolve_context()
    mode = ctx.mode if ctx else GOVERNOR_MODE

    response = IntentFormResponse(
        schema_id=request.schema_id,
        values=request.values,
        escape_text=request.escape_text,
    )

    # Try to rebuild the schema to validate against
    # The caller should have gotten schema from /v2/intent/schema/{template}
    # We need the template_name to rebuild — check all templates
    from governor.intent_compiler import BUILTIN_TEMPLATES
    schema = None
    for tname in BUILTIN_TEMPLATES:
        try:
            candidate = build_form_schema(tname, mode=mode)
            if candidate.schema_id == request.schema_id:
                schema = candidate
                break
        except ValueError:
            continue

    if schema is None:
        return {"valid": False, "errors": [f"Schema ID '{request.schema_id}' not found for mode '{mode}'"]}

    errors = validate_response(response, schema)
    return {"valid": len(errors) == 0, "errors": errors}


@app.post("/v2/intent/compile")
async def v2_intent_compile(request: IntentCompileRequest) -> dict[str, Any]:
    """Compile a form response into governance intent + constraints."""
    from governor.intent_compiler import (
        IntentFormResponse,
        build_form_schema,
        compile_intent,
    )

    ctx, _ = _resolve_context()
    mode = ctx.mode if ctx else GOVERNOR_MODE
    governor_dir = ctx.governor_dir if ctx else None

    try:
        schema = build_form_schema(request.template_name, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    response = IntentFormResponse(
        schema_id=request.schema_id,
        values=request.values,
        escape_text=request.escape_text,
    )

    result = compile_intent(response, schema, governor_dir=governor_dir)
    return result.to_dict()


@app.get("/v2/intent/policy")
async def v2_intent_policy() -> dict[str, Any]:
    """Return the current form policy for the active mode."""
    from governor.intent_compiler import get_form_policy

    ctx, _ = _resolve_context()
    mode = ctx.mode if ctx else GOVERNOR_MODE
    policy = get_form_policy(mode)
    return {"mode": mode, "policy": policy.value}


# ---- Dashboard UI ----

# ---- Demos ----

@app.get("/v2/demos")
async def v2_list_demos() -> dict[str, Any]:
    """List demo scenarios with freshness status."""
    from governor.webui_demo import DemoStore, BUILTIN_DEMOS

    cm = _get_context_manager()
    ctx = cm.get(GOVERNOR_CONTEXT_ID)
    gov_dir = ctx.governor_dir if ctx else Path(".governor")

    store = DemoStore(governor_dir=gov_dir)
    freshness = store.check_freshness()

    demos = []
    for demo in BUILTIN_DEMOS:
        fr = next((f for f in freshness if f["name"] == demo.name), None)
        demos.append({
            "name": demo.name,
            "description": demo.description,
            "surface": demo.surface.value,
            "tags": demo.tags,
            "step_count": len(demo.steps),
            "screenshot_count": len(demo.screenshot_paths),
            "status": fr["status"] if fr else "missing",
        })

    return {"demos": demos}


@app.get("/v2/demos/{name}/playwright")
async def v2_demo_playwright(name: str) -> dict[str, Any]:
    """Return generated Playwright spec text for a demo scenario."""
    from governor.webui_demo import BUILTIN_DEMOS, DemoStore, generate_playwright_spec

    # Search built-in demos first
    scenario = next((d for d in BUILTIN_DEMOS if d.name == name), None)

    # Fall back to custom scenarios in store
    if scenario is None:
        cm = _get_context_manager()
        ctx = cm.get(GOVERNOR_CONTEXT_ID)
        gov_dir = ctx.governor_dir if ctx else Path(".governor")
        store = DemoStore(governor_dir=gov_dir)
        for s in store.list_scenarios():
            if s.name == name:
                scenario = s
                break

    if scenario is None:
        raise HTTPException(status_code=404, detail=f"Demo scenario not found: {name}")

    spec_text = generate_playwright_spec(scenario)
    return {"name": name, "spec": spec_text}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_ui() -> HTMLResponse:
    """Serve the v2 governance dashboard."""
    html_path = _STATIC_DIR / "dashboard.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard HTML not found")
    return HTMLResponse(
        content=html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


# ============================================================================
# Artifact Engine — response helpers + routes
# ============================================================================


def _artifact_meta_to_dict(meta: Any, *, include_versions: bool = False) -> dict:
    """Convert ArtifactMeta to dict, optionally including version history."""
    d: dict[str, Any] = {
        "id": meta.id,
        "title": meta.title,
        "kind": meta.kind,
        "language": meta.language,
        "current_version": meta.current_version,
        "created_at": meta.created_at,
        "updated_at": meta.updated_at,
    }
    if include_versions:
        d["versions"] = [
            {
                "version": v.version,
                "created_at": v.created_at,
                "content_hash": v.content_hash,
                "source": v.source,
                "message_id": v.message_id,
            }
            for v in meta.versions
        ]
    return d


def _artifact_detail_response(
    *, meta: Any, content: str, index_version: int
) -> dict:
    return {
        "ok": True,
        "index_version": index_version,
        "artifact": _artifact_meta_to_dict(meta, include_versions=True),
        "content": content,
    }


def _artifact_list_response(
    *, summaries: list, index_version: int
) -> dict:
    return {
        "ok": True,
        "index_version": index_version,
        "artifacts": [
            {
                "id": s.id,
                "title": s.title,
                "kind": s.kind,
                "language": s.language,
                "current_version": s.current_version,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
            }
            for s in summaries
        ],
    }


def _artifact_error(
    *, status_code: int, code: str, message: str, details: dict | None = None
) -> JSONResponse:
    body: dict[str, Any] = {
        "ok": False,
        "error": {"code": code, "message": message},
    }
    if details:
        body["error"]["details"] = details
    return JSONResponse(content=body, status_code=status_code)


def _artifact_exception_response(exc: Exception) -> JSONResponse:
    """Map artifact store exceptions to structured JSON error responses."""
    from gov_webui.artifact_store import (
        ArtifactContentMissingError,
        ArtifactNotFoundError,
        ArtifactValidationError,
        ArtifactVersionNotFoundError,
        StaleArtifactVersionError,
    )

    if isinstance(exc, ArtifactNotFoundError):
        return _artifact_error(
            status_code=404,
            code="artifact_not_found",
            message=str(exc),
            details={"artifact_id": exc.artifact_id},
        )
    if isinstance(exc, ArtifactVersionNotFoundError):
        return _artifact_error(
            status_code=404,
            code="artifact_version_not_found",
            message=str(exc),
            details={"artifact_id": exc.artifact_id, "version": exc.version},
        )
    if isinstance(exc, StaleArtifactVersionError):
        return _artifact_error(
            status_code=409,
            code="stale_version",
            message=str(exc),
            details={
                "artifact_id": exc.artifact_id,
                "expected_current_version": exc.expected_current_version,
                "current_version": exc.current_version,
                "index_version": exc.index_version,
            },
        )
    if isinstance(exc, ArtifactValidationError):
        return _artifact_error(
            status_code=422,
            code="validation_error",
            message=str(exc),
        )
    if isinstance(exc, ArtifactContentMissingError):
        return _artifact_error(
            status_code=500,
            code="artifact_content_missing",
            message=str(exc),
            details={
                "artifact_id": exc.artifact_id,
                "version": exc.version,
                "path": exc.path,
            },
        )
    # Fallback for unexpected errors
    return _artifact_error(
        status_code=500, code="internal_error", message=str(exc)
    )


@app.get("/governor/artifacts")
async def artifacts_list() -> dict:
    """List all artifacts (metadata only)."""
    from gov_webui.artifact_store import ArtifactStoreError

    try:
        store = _get_artifact_store()
        summaries, idx_ver = store.list_all()
        return _artifact_list_response(summaries=summaries, index_version=idx_ver)
    except ArtifactStoreError as exc:
        return _artifact_exception_response(exc)


@app.post("/governor/artifacts", status_code=201)
async def artifacts_create(request: ArtifactCreateRequest) -> JSONResponse:
    """Create a new artifact."""
    from gov_webui.artifact_store import ArtifactStoreError

    try:
        store = _get_artifact_store()
        meta, content, idx_ver = store.create(
            title=request.title,
            content=request.content,
            kind=request.kind,
            language=request.language,
            message_id=request.message_id,
            source=request.source,
        )
        return JSONResponse(
            content=_artifact_detail_response(
                meta=meta, content=content, index_version=idx_ver
            ),
            status_code=201,
        )
    except ArtifactStoreError as exc:
        return _artifact_exception_response(exc)


@app.get("/governor/artifacts/state")
async def artifacts_state() -> dict:
    """Quick poll endpoint for artifact index version."""
    from gov_webui.artifact_store import ArtifactStoreError

    try:
        store = _get_artifact_store()
        state = store.get_state()
        return {
            "ok": True,
            "index_version": state["version"],
            "updated_at": state["updated_at"],
            "count": state["count"],
        }
    except ArtifactStoreError as exc:
        return _artifact_exception_response(exc)


@app.get("/governor/artifacts/{artifact_id}")
async def artifacts_get(artifact_id: str) -> dict:
    """Get artifact detail + latest content."""
    from gov_webui.artifact_store import ArtifactStoreError

    try:
        store = _get_artifact_store()
        meta, content, idx_ver = store.get(artifact_id)
        return _artifact_detail_response(
            meta=meta, content=content, index_version=idx_ver
        )
    except ArtifactStoreError as exc:
        return _artifact_exception_response(exc)


@app.put("/governor/artifacts/{artifact_id}")
async def artifacts_update(
    artifact_id: str, request: ArtifactUpdateRequest
) -> dict:
    """Update artifact content (creates new version)."""
    from gov_webui.artifact_store import ArtifactStoreError

    try:
        store = _get_artifact_store()
        meta, content, idx_ver = store.update(
            artifact_id,
            content=request.content,
            title=request.title,
            expected_current_version=request.expected_current_version,
            source=request.source,
            message_id=request.message_id,
        )
        return _artifact_detail_response(
            meta=meta, content=content, index_version=idx_ver
        )
    except ArtifactStoreError as exc:
        return _artifact_exception_response(exc)


@app.delete("/governor/artifacts/{artifact_id}")
async def artifacts_delete(artifact_id: str) -> dict:
    """Delete artifact from index."""
    from gov_webui.artifact_store import ArtifactStoreError

    try:
        store = _get_artifact_store()
        _, idx_ver = store.delete(artifact_id)
        return {
            "ok": True,
            "index_version": idx_ver,
            "deleted": {"artifact_id": artifact_id},
        }
    except ArtifactStoreError as exc:
        return _artifact_exception_response(exc)


@app.get("/governor/artifacts/{artifact_id}/version/{version}")
async def artifacts_get_version(artifact_id: str, version: int) -> dict:
    """Get content for a specific artifact version."""
    from gov_webui.artifact_store import ArtifactStoreError

    try:
        store = _get_artifact_store()
        content = store.get_version(artifact_id, version)
        return {
            "ok": True,
            "artifact_id": artifact_id,
            "version": version,
            "content": content,
        }
    except ArtifactStoreError as exc:
        return _artifact_exception_response(exc)


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """Run the adapter server."""
    import uvicorn

    uvicorn.run(
        "gov_webui.adapter:app",
        host=GOVERNOR_BIND_HOST,
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
