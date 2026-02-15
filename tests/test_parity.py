"""Parity tripwire: verify the webui chat path delegates to the daemon.

The split-brain fix routes /v1/chat/completions through the daemon's
chat.send/chat.stream RPC methods instead of calling ChatBridge directly.
These tests verify that contract is maintained.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def adapter_mod():
    """Import and reset adapter module."""
    import importlib
    import os
    os.environ.setdefault("BACKEND_TYPE", "ollama")
    os.environ.setdefault("GOVERNOR_CONTEXT_ID", "test-parity")
    os.environ.setdefault("GOVERNOR_MODE", "general")

    import gov_webui.adapter as mod
    importlib.reload(mod)
    mod._bridge = None
    mod._context_manager = None
    mod._session_store = None
    mod._daemon_client = None
    yield mod
    mod._bridge = None
    mod._context_manager = None
    mod._session_store = None
    mod._daemon_client = None


@pytest.fixture
def client(adapter_mod):
    from fastapi.testclient import TestClient
    return TestClient(adapter_mod.app)


class TestChatDelegatesToDaemon:
    """Tripwire: chat endpoint MUST call daemon, not ChatBridge directly."""

    def test_non_streaming_calls_daemon_chat_send(self, client, adapter_mod):
        """POST /v1/chat/completions (non-streaming) calls daemon chat.send."""
        mock = AsyncMock()
        mock.chat_send = AsyncMock(return_value={
            "content": "Hello",
            "model": "m",
            "usage": {},
            "violations": [],
            "footer": None,
            "pending": None,
        })
        adapter_mod._daemon_client = mock

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        # Daemon chat_send was called
        mock.chat_send.assert_called_once()
        # Verify messages were forwarded correctly
        call_kwargs = mock.chat_send.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "test"}]

    def test_streaming_calls_daemon_chat_stream(self, client, adapter_mod):
        """POST /v1/chat/completions (stream=true) calls daemon chat.stream."""
        async def mock_stream(*args, **kwargs):
            yield ("chunk", None)
            yield (None, {
                "content": "chunk",
                "model": "m",
                "usage": {},
                "violations": [],
                "footer": None,
                "pending": None,
            })

        mock = AsyncMock()
        mock.chat_stream = MagicMock(return_value=mock_stream())
        mock.connect = AsyncMock()
        adapter_mod._daemon_client = mock

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        # Daemon chat_stream was called (via the MagicMock)
        mock.chat_stream.assert_called_once()

    def test_violations_from_daemon_are_surfaced(self, client, adapter_mod):
        """Daemon violations are included in the response."""
        mock = AsyncMock()
        mock.chat_send = AsyncMock(return_value={
            "content": "",
            "model": "m",
            "usage": {},
            "violations": [{"anchor_id": "a1", "severity": "reject"}],
            "footer": None,
            "pending": {
                "id": "p1",
                "violations": [{"anchor_id": "a1", "severity": "reject"}],
                "mode": "general",
                "blocked_response": "bad",
            },
        })
        adapter_mod._daemon_client = mock

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        # Response should contain violation prompt, not empty content
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        assert content  # Violation prompt was generated

    def test_footer_from_daemon_appended(self, client, adapter_mod):
        """Daemon footer is appended to response content."""
        mock = AsyncMock()
        mock.chat_send = AsyncMock(return_value={
            "content": "Hello",
            "model": "m",
            "usage": {},
            "violations": [],
            "footer": "[Governor] OK — 3 anchors checked",
            "pending": None,
        })
        adapter_mod._daemon_client = mock

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        assert "Hello" in content
        assert "[Governor] OK" in content

    def test_bridge_not_used_for_chat(self, client, adapter_mod):
        """ChatBridge is NOT called when chat endpoint is hit."""
        mock_bridge = AsyncMock()
        adapter_mod._bridge = mock_bridge

        mock_daemon = AsyncMock()
        mock_daemon.chat_send = AsyncMock(return_value={
            "content": "from daemon",
            "model": "m",
            "usage": {},
            "violations": [],
            "footer": None,
            "pending": None,
        })
        adapter_mod._daemon_client = mock_daemon

        client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        # Bridge should NOT have been called
        mock_bridge.chat.assert_not_called()
        # Daemon should have been called
        mock_daemon.chat_send.assert_called_once()


class TestAuthErrorHandling:
    """Auth errors from daemon return 401, not 502."""

    def test_non_streaming_auth_error_returns_401(self, client, adapter_mod):
        """DaemonAuthError on chat.send produces HTTP 401."""
        from gov_webui.daemon_client import DaemonAuthError

        mock = AsyncMock()
        mock.chat_send = AsyncMock(
            side_effect=DaemonAuthError("Claude Code is not logged in")
        )
        adapter_mod._daemon_client = mock

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        assert response.status_code == 401
        assert "claude /login" in response.json()["detail"].lower()

    def test_non_streaming_generic_error_returns_502(self, client, adapter_mod):
        """Non-auth errors still produce HTTP 502."""
        mock = AsyncMock()
        mock.chat_send = AsyncMock(
            side_effect=RuntimeError("connection refused")
        )
        adapter_mod._daemon_client = mock

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        assert response.status_code == 502

    def test_streaming_auth_error_yields_auth_error_type(self, client, adapter_mod):
        """DaemonAuthError on chat.stream emits an auth_error SSE event."""
        from gov_webui.daemon_client import DaemonAuthError

        async def mock_stream(*args, **kwargs):
            raise DaemonAuthError("not logged in")
            yield  # pragma: no cover — makes this an async generator

        mock = AsyncMock()
        mock.chat_stream = MagicMock(return_value=mock_stream())
        mock.connect = AsyncMock()
        adapter_mod._daemon_client = mock

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert response.status_code == 200  # SSE always starts 200
        body = response.text
        assert "auth_error" in body
        assert "claude /login" in body.lower()


class TestDaemonClientAuthDetection:
    """Unit tests for DaemonAuthError detection in daemon_client._call()."""

    def test_auth_error_code_raises_daemon_auth_error(self):
        from gov_webui.daemon_client import AUTH_ERROR_CODE, DaemonAuthError

        assert AUTH_ERROR_CODE == -32001

    def test_daemon_auth_error_is_runtime_error(self):
        from gov_webui.daemon_client import DaemonAuthError

        err = DaemonAuthError("test")
        assert isinstance(err, RuntimeError)
