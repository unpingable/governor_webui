# SPDX-License-Identifier: Apache-2.0
"""Tests for the WebUI adapter (FastAPI endpoints)."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from governor.context_manager import GovernorContextManager


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_contexts_dir(tmp_path: Path) -> Path:
    """Temporary directory for governor contexts."""
    return tmp_path / "contexts"


@pytest.fixture
def mock_env(tmp_contexts_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set environment variables for adapter configuration."""
    monkeypatch.setenv("BACKEND_TYPE", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("GOVERNOR_CONTEXT_ID", "test-context")
    monkeypatch.setenv("GOVERNOR_MODE", "general")
    monkeypatch.setenv("GOVERNOR_CONTEXTS_DIR", str(tmp_contexts_dir))


@pytest.fixture
def reset_adapter_globals() -> None:
    """Reset module-level globals between tests."""
    import gov_webui.adapter as adapter_mod
    adapter_mod._bridge = None
    adapter_mod._context_manager = None
    adapter_mod._session_store = None
    adapter_mod._daemon_client = None
    adapter_mod._project_store = None
    adapter_mod._research_project_store = None
    adapter_mod._artifact_store = None
    adapter_mod._receipt_v1_bridge = None
    adapter_mod._pending_captures.clear()
    adapter_mod._capture_counter = 0
    adapter_mod._pending_research_captures.clear()
    adapter_mod._research_capture_counter = 0
    adapter_mod._turn_seq = 0
    yield
    adapter_mod._bridge = None
    adapter_mod._context_manager = None
    adapter_mod._session_store = None
    adapter_mod._daemon_client = None
    adapter_mod._project_store = None
    adapter_mod._research_project_store = None
    adapter_mod._artifact_store = None
    adapter_mod._receipt_v1_bridge = None
    adapter_mod._pending_captures.clear()
    adapter_mod._capture_counter = 0
    adapter_mod._pending_research_captures.clear()
    adapter_mod._research_capture_counter = 0
    adapter_mod._turn_seq = 0


@pytest.fixture
def app(mock_env, reset_adapter_globals):
    """Get the FastAPI app with test config."""
    # Re-import to pick up environment changes
    import importlib
    import gov_webui.adapter as adapter_mod
    importlib.reload(adapter_mod)
    return adapter_mod.app


@pytest.fixture
def client(app):
    """Create a test client."""
    from fastapi.testclient import TestClient
    return TestClient(app)


# ============================================================================
# TestRootEndpoint
# ============================================================================


class TestRootEndpoint:
    """Tests for GET / (serves combined UI) and GET /api/info."""

    def test_root_returns_html(self, client) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "Governor Chat" in response.text

    def test_root_contains_chat_and_governor(self, client) -> None:
        response = client.get("/")
        body = response.text
        assert "chat-panel" in body
        assert "governor-panel" in body
        assert "model-select" in body

    def test_api_info_returns_json(self, client) -> None:
        response = client.get("/api/info")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Governor Chat Adapter"
        assert data["openai_compatible"] is True

    def test_api_info_includes_version(self, client) -> None:
        response = client.get("/api/info")
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.4.0"

    def test_api_info_includes_endpoints(self, client) -> None:
        response = client.get("/api/info")
        data = response.json()
        assert "endpoints" in data
        assert "/v1/models" in data["endpoints"].values()
        assert "/v1/chat/completions" in data["endpoints"].values()


# ============================================================================
# TestHealthEndpoint
# ============================================================================


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_degraded_when_backend_down(self, client) -> None:
        """Health returns degraded when backend is unreachable."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        # Backend is not running so should be degraded
        assert data["status"] == "degraded"
        assert data["backend"]["connected"] is False

    def test_health_includes_governor_info(self, client) -> None:
        response = client.get("/health")
        data = response.json()
        assert "governor" in data
        assert "context_id" in data["governor"]
        assert "mode" in data["governor"]

    def test_health_includes_backend_type(self, client) -> None:
        response = client.get("/health")
        data = response.json()
        assert data["backend"]["type"] == "ollama"


# ============================================================================
# TestModelsEndpoint
# ============================================================================


class TestModelsEndpoint:
    """Tests for GET /v1/models."""

    def test_models_format(self, client) -> None:
        """Models endpoint returns correct format even on error."""
        # Backend is down, so this will raise 502
        response = client.get("/v1/models")
        # Could be 502 (backend down) or 200 (mocked)
        assert response.status_code in (200, 502)

    def test_get_model_by_id(self, client) -> None:
        response = client.get("/v1/models/test-model")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-model"
        assert data["object"] == "model"


# ============================================================================
# TestChatEndpoint
# ============================================================================


class TestChatEndpoint:
    """Tests for POST /v1/chat/completions (delegates to daemon)."""

    def _make_mock_daemon(self, content="Hello from test", model="test-model",
                          usage=None, violations=None, footer=None, pending=None):
        """Create a mock DaemonChatClient with chat_send returning given data."""
        mock = AsyncMock()
        mock.chat_send = AsyncMock(return_value={
            "content": content,
            "model": model,
            "usage": usage or {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "violations": violations or [],
            "footer": footer,
            "pending": pending,
        })
        mock.connect = AsyncMock()
        mock.close = AsyncMock()
        return mock

    def test_non_streaming_response_format(self, client) -> None:
        """Non-streaming response has correct OpenAI format."""
        import gov_webui.adapter as adapter_mod

        adapter_mod._daemon_client = self._make_mock_daemon()

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["content"] == "Hello from test"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["usage"]["total_tokens"] == 8

    def test_error_handling(self, client) -> None:
        """Daemon errors return 502."""
        import gov_webui.adapter as adapter_mod

        mock = AsyncMock()
        mock.chat_send = AsyncMock(side_effect=Exception("Connection refused"))
        adapter_mod._daemon_client = mock

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )
        assert response.status_code == 502

    def test_empty_messages(self, client) -> None:
        """Empty messages list is handled."""
        import gov_webui.adapter as adapter_mod

        adapter_mod._daemon_client = self._make_mock_daemon(content="OK", model="m")

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [],
                "stream": False,
            },
        )
        assert response.status_code == 200

    def test_model_passthrough(self, client) -> None:
        """Model name is passed through correctly."""
        import gov_webui.adapter as adapter_mod

        adapter_mod._daemon_client = self._make_mock_daemon(
            content="OK", model="custom-model"
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "custom-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        data = response.json()
        assert data["model"] == "custom-model"

    def test_daemon_called_with_messages(self, client) -> None:
        """Verify daemon receives the correct messages."""
        import gov_webui.adapter as adapter_mod

        mock = self._make_mock_daemon()
        adapter_mod._daemon_client = mock

        client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        # Verify chat_send was called (not ChatBridge.chat)
        mock.chat_send.assert_called_once()
        call_kwargs = mock.chat_send.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]
        assert call_kwargs["model"] == "test-model"

    def test_footer_appended_to_content(self, client) -> None:
        """Governor footer from daemon is appended to response content."""
        import gov_webui.adapter as adapter_mod

        adapter_mod._daemon_client = self._make_mock_daemon(
            content="Hello", footer="[Governor: OK]"
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        data = response.json()
        assert "[Governor: OK]" in data["choices"][0]["message"]["content"]

    def test_pending_violation_returns_prompt(self, client) -> None:
        """When daemon returns pending violation, format as violation prompt."""
        import gov_webui.adapter as adapter_mod

        adapter_mod._daemon_client = self._make_mock_daemon(
            content="",
            pending={
                "id": "p1",
                "violations": [{"anchor_id": "a1", "severity": "reject"}],
                "mode": "general",
                "blocked_response": "bad text",
            },
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Should contain violation prompt, not the original content
        content = data["choices"][0]["message"]["content"]
        assert content  # Not empty — violation prompt was generated


# ============================================================================
# TestGovernorEndpoints
# ============================================================================


class TestGovernorEndpoints:
    """Tests for governor-specific endpoints."""

    def test_contexts_list(self, client, tmp_contexts_dir) -> None:
        """GET /governor/contexts returns context list."""
        # Create a context manually
        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-ctx-1", mode="fiction")

        # Need to inject this context manager
        import gov_webui.adapter as adapter_mod
        adapter_mod._context_manager = cm

        response = client.get("/governor/contexts")
        assert response.status_code == 200
        data = response.json()
        assert "contexts" in data
        assert "active_context_id" in data
        assert len(data["contexts"]) == 1
        assert data["contexts"][0]["context_id"] == "test-ctx-1"

    def test_status_uninitialized(self, client) -> None:
        """GET /governor/status when context doesn't exist."""
        response = client.get("/governor/status")
        assert response.status_code == 200
        data = response.json()
        assert data["initialized"] is False

    def test_status_with_fiction_context(self, client, tmp_contexts_dir) -> None:
        """GET /governor/status with fiction context."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="fiction")
        adapter_mod._context_manager = cm

        response = client.get("/governor/status")
        assert response.status_code == 200
        data = response.json()
        assert data["initialized"] is True
        assert data["mode"] == "fiction"
        assert data["has_fiction_governor"] is True
        assert data["has_governor"] is True

    def test_status_with_code_context(self, client, tmp_contexts_dir) -> None:
        """GET /governor/status with code context."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="code")
        adapter_mod._context_manager = cm

        response = client.get("/governor/status")
        assert response.status_code == 200
        data = response.json()
        assert data["initialized"] is True
        assert data["mode"] == "code"
        assert data["has_fiction_governor"] is False


# ============================================================================
# TestBackendSelection
# ============================================================================


class TestBackendEndpoints:
    """Tests for GET /v1/backends and POST /v1/backends/switch."""

    def test_list_backends_returns_list(self, client) -> None:
        """GET /v1/backends returns a list of backend entries."""
        response = client.get("/v1/backends")
        assert response.status_code == 200
        data = response.json()
        assert "backends" in data
        assert isinstance(data["backends"], list)
        assert len(data["backends"]) >= 1

    def test_list_backends_has_ollama(self, client) -> None:
        """Ollama is always present in the backends list."""
        response = client.get("/v1/backends")
        types = [b["type"] for b in response.json()["backends"]]
        assert "ollama" in types

    def test_list_backends_marks_active(self, client) -> None:
        """Exactly one backend is marked active."""
        response = client.get("/v1/backends")
        data = response.json()
        active_list = [b for b in data["backends"] if b.get("active")]
        assert len(active_list) == 1
        assert data["active"] == active_list[0]["type"]

    def test_switch_invalid_type(self, client) -> None:
        """POST /v1/backends/switch with invalid type returns 400."""
        response = client.post(
            "/v1/backends/switch",
            json={"backend_type": "nonexistent"},
        )
        assert response.status_code == 400

    def test_switch_to_same_backend(self, client) -> None:
        """Switching to the same backend type succeeds."""
        response = client.post(
            "/v1/backends/switch",
            json={"backend_type": "ollama"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["backend_type"] == "ollama"

    def test_api_info_reflects_current_backend(self, client) -> None:
        """After switch, /api/info reflects the new backend type."""
        import gov_webui.adapter as adapter_mod
        adapter_mod._current_backend_type = "ollama"

        info = client.get("/api/info").json()
        assert info["backend"] == "ollama"


class TestBackendSelection:
    """Tests for backend type selection."""

    def test_default_is_ollama(self, monkeypatch, tmp_contexts_dir) -> None:
        """Default backend type is ollama."""
        monkeypatch.setenv("GOVERNOR_CONTEXTS_DIR", str(tmp_contexts_dir))
        monkeypatch.delenv("BACKEND_TYPE", raising=False)

        import importlib
        import gov_webui.adapter as adapter_mod
        adapter_mod._bridge = None
        adapter_mod._context_manager = None
        importlib.reload(adapter_mod)

        assert adapter_mod.BACKEND_TYPE == "ollama"

    def test_anthropic_from_env(self, monkeypatch, tmp_contexts_dir) -> None:
        """BACKEND_TYPE=anthropic is read from environment."""
        monkeypatch.setenv("BACKEND_TYPE", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("GOVERNOR_CONTEXTS_DIR", str(tmp_contexts_dir))

        import importlib
        import gov_webui.adapter as adapter_mod
        adapter_mod._bridge = None
        adapter_mod._context_manager = None
        importlib.reload(adapter_mod)

        assert adapter_mod.BACKEND_TYPE == "anthropic"

    def test_ollama_from_env(self, monkeypatch, tmp_contexts_dir) -> None:
        """BACKEND_TYPE=ollama is read from environment."""
        monkeypatch.setenv("BACKEND_TYPE", "ollama")
        monkeypatch.setenv("GOVERNOR_CONTEXTS_DIR", str(tmp_contexts_dir))

        import importlib
        import gov_webui.adapter as adapter_mod
        adapter_mod._bridge = None
        adapter_mod._context_manager = None
        importlib.reload(adapter_mod)

        assert adapter_mod.BACKEND_TYPE == "ollama"


# ============================================================================
# TestStreamingResponse
# ============================================================================


class TestStreamingResponse:
    """Tests for streaming chat responses (via daemon)."""

    def test_streaming_request(self, client) -> None:
        """Streaming request returns SSE format."""
        import gov_webui.adapter as adapter_mod

        async def mock_daemon_stream(*args, **kwargs):
            yield ("Hello ", None)
            yield ("world", None)
            yield (None, {
                "content": "Hello world",
                "model": "test-model",
                "usage": {},
                "violations": [],
                "footer": None,
                "pending": None,
            })

        mock = AsyncMock()
        mock.chat_stream = MagicMock(return_value=mock_daemon_stream())
        mock.connect = AsyncMock()
        adapter_mod._daemon_client = mock

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Parse SSE chunks
        content = response.text
        assert "data:" in content


# ============================================================================
# TestGovernorNow
# ============================================================================


class TestGovernorNow:
    """Tests for GET /governor/now."""

    def test_uninitialized_returns_ok(self, client) -> None:
        """Uninitialized context returns ok status."""
        response = client.get("/governor/now")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["sentence"].startswith("OK:")
        assert data["last_event"] is None
        assert data["suggested_action"] is None

    def test_with_empty_context(self, client, tmp_contexts_dir) -> None:
        """Empty initialized context returns ok."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="code")
        adapter_mod._context_manager = cm

        response = client.get("/governor/now")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["mode"] == "code"

    def test_includes_context_id(self, client) -> None:
        response = client.get("/governor/now")
        data = response.json()
        assert "context_id" in data

    def test_includes_regime(self, client, tmp_contexts_dir) -> None:
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="general")
        adapter_mod._context_manager = cm

        response = client.get("/governor/now")
        data = response.json()
        # regime is present (may be None or a string depending on state)
        assert "regime" in data

    def test_response_shape(self, client) -> None:
        """Response has all expected keys."""
        response = client.get("/governor/now")
        data = response.json()
        expected_keys = {"context_id", "status", "sentence", "last_event", "suggested_action", "regime", "mode"}
        assert expected_keys == set(data.keys())

    def test_status_is_valid_pill(self, client) -> None:
        response = client.get("/governor/now")
        data = response.json()
        assert data["status"] in ("ok", "needs_attention", "blocked")


# ============================================================================
# TestGovernorWhy
# ============================================================================


class TestGovernorWhy:
    """Tests for GET /governor/why."""

    def test_uninitialized_returns_empty(self, client) -> None:
        response = client.get("/governor/why")
        assert response.status_code == 200
        data = response.json()
        assert data["feed"] == []
        assert data["total"] == 0

    def test_with_empty_context(self, client, tmp_contexts_dir) -> None:
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="code")
        adapter_mod._context_manager = cm

        response = client.get("/governor/why")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["feed"], list)

    def test_limit_parameter(self, client) -> None:
        response = client.get("/governor/why?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data["feed"]) <= 5

    def test_severity_parameter(self, client) -> None:
        response = client.get("/governor/why?severity=error")
        assert response.status_code == 200

    def test_response_shape(self, client) -> None:
        response = client.get("/governor/why")
        data = response.json()
        expected_keys = {"context_id", "feed", "total"}
        assert expected_keys == set(data.keys())


# ============================================================================
# TestGovernorHistory
# ============================================================================


class TestGovernorHistory:
    """Tests for GET /governor/history."""

    def test_uninitialized_returns_empty(self, client) -> None:
        response = client.get("/governor/history")
        assert response.status_code == 200
        data = response.json()
        assert data["days"] == []

    def test_with_empty_context(self, client, tmp_contexts_dir) -> None:
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="code")
        adapter_mod._context_manager = cm

        response = client.get("/governor/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["days"], list)

    def test_days_parameter(self, client) -> None:
        response = client.get("/governor/history?days=3")
        assert response.status_code == 200

    def test_response_shape(self, client) -> None:
        response = client.get("/governor/history")
        data = response.json()
        expected_keys = {"context_id", "days"}
        assert expected_keys == set(data.keys())


# ============================================================================
# TestGovernorDetail
# ============================================================================


class TestGovernorDetail:
    """Tests for GET /governor/detail/{item_id}."""

    def test_404_when_uninitialized(self, client) -> None:
        response = client.get("/governor/detail/dec_test123")
        assert response.status_code == 404

    def test_404_unknown_id(self, client, tmp_contexts_dir) -> None:
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="code")
        adapter_mod._context_manager = cm

        response = client.get("/governor/detail/dec_nonexistent")
        assert response.status_code == 404

    def test_404_unknown_prefix(self, client, tmp_contexts_dir) -> None:
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="code")
        adapter_mod._context_manager = cm

        response = client.get("/governor/detail/xxx_unknown")
        assert response.status_code == 404

    def test_valid_prefixes_handled(self, client, tmp_contexts_dir) -> None:
        """All valid prefixes (dec_, clm_, ev_, vio_) are handled without 500."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="code")
        adapter_mod._context_manager = cm

        for prefix in ["dec_", "clm_", "ev_", "vio_"]:
            response = client.get(f"/governor/detail/{prefix}nonexistent")
            # Should be 404 (not found), not 500 (server error)
            assert response.status_code == 404

    def test_response_shape_on_404(self, client, tmp_contexts_dir) -> None:
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="code")
        adapter_mod._context_manager = cm

        response = client.get("/governor/detail/dec_missing")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data


# ============================================================================
# TestGovernorStatusV2
# ============================================================================


class TestGovernorStatusV2:
    """Tests for /governor/status with viewmodel integration."""

    def test_uninitialized_no_viewmodel(self, client) -> None:
        """Uninitialized context does not include viewmodel key."""
        response = client.get("/governor/status")
        data = response.json()
        assert data["initialized"] is False
        assert "viewmodel" not in data

    def test_initialized_includes_viewmodel(self, client, tmp_contexts_dir) -> None:
        """Initialized context includes viewmodel key."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="code")
        adapter_mod._context_manager = cm

        response = client.get("/governor/status")
        data = response.json()
        assert data["initialized"] is True
        assert "viewmodel" in data
        assert data["viewmodel"]["schema_version"] == "v2"

    def test_backward_compat_fields(self, client, tmp_contexts_dir) -> None:
        """Backward-compat fields still present alongside viewmodel."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="fiction")
        adapter_mod._context_manager = cm

        response = client.get("/governor/status")
        data = response.json()
        # Old fields still present
        assert "context_id" in data
        assert "initialized" in data
        assert "mode" in data
        assert "facts_count" in data
        assert "decisions_count" in data
        assert "metadata" in data
        # New field present
        assert "viewmodel" in data

    def test_viewmodel_has_sections(self, client, tmp_contexts_dir) -> None:
        """Viewmodel contains the 8 standard sections."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="general")
        adapter_mod._context_manager = cm

        response = client.get("/governor/status")
        vm = response.json()["viewmodel"]
        expected_sections = {"schema_version", "generated_at", "session", "regime",
                             "decisions", "claims", "evidence", "violations",
                             "execution", "stability"}
        assert expected_sections == set(vm.keys())


# ============================================================================
# TestRootEndpointV2
# ============================================================================


class TestRootEndpointV2:
    """Tests for new routes in api/info endpoint."""

    def test_includes_new_endpoints(self, client) -> None:
        response = client.get("/api/info")
        data = response.json()
        endpoints = data["endpoints"]
        assert "governor_now" in endpoints
        assert "governor_why" in endpoints
        assert "governor_history" in endpoints
        assert "governor_detail" in endpoints


# ============================================================================
# TestGovernorUI
# ============================================================================


class TestGovernorUI:
    """Tests for GET /governor/ui."""

    def test_ui_returns_html(self, client) -> None:
        """GET /governor/ui returns 200 with HTML content type."""
        response = client.get("/governor/ui")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_ui_contains_sections(self, client) -> None:
        """HTML body contains the key UI elements."""
        response = client.get("/governor/ui")
        body = response.text
        # Human-friendly UI has mode-specific panels and corrections log
        assert "Governor" in body
        assert "mode" in body.lower()
        assert "Corrections" in body

    def test_ui_in_api_info_endpoints(self, client) -> None:
        """API info endpoint lists /governor/ui."""
        response = client.get("/api/info")
        data = response.json()
        assert "governor_ui" in data["endpoints"]
        assert data["endpoints"]["governor_ui"] == "/governor/ui"


# ============================================================================
# TestGovernorFooterIntegration
# ============================================================================


class TestGovernorFooterIntegration:
    """End-to-end tests for governor footer in chat responses (via daemon)."""

    def test_non_streaming_governor_ok_footer(self, client) -> None:
        """Non-streaming response includes [Governor] OK when daemon returns footer."""
        import gov_webui.adapter as adapter_mod

        mock = AsyncMock()
        mock.chat_send = AsyncMock(return_value={
            "content": "Alice walked peacefully.",
            "model": "test-model",
            "usage": {},
            "violations": [],
            "footer": "[Governor] OK — 0 anchors checked",
            "pending": None,
        })
        adapter_mod._daemon_client = mock

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Write a scene"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        assert "[Governor] OK" in content

    def test_streaming_governor_footer_in_sse(self, client) -> None:
        """Streaming response includes governor feedback in SSE chunks."""
        import gov_webui.adapter as adapter_mod

        async def mock_stream(*args, **kwargs):
            yield ("She was the chosen one.", None)
            yield (None, {
                "content": "She was the chosen one.",
                "model": "test-model",
                "usage": {},
                "violations": [],
                "footer": "[Governor] OK",
                "pending": None,
            })

        mock = AsyncMock()
        mock.chat_stream = MagicMock(return_value=mock_stream())
        mock.connect = AsyncMock()
        adapter_mod._daemon_client = mock

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Write a scene"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Parse SSE data to find governor feedback
        full_text = response.text
        assert "[Governor]" in full_text


# ============================================================================
# TestSessionEndpoints
# ============================================================================


class TestSessionEndpoints:
    """Tests for session CRUD endpoints."""

    def test_list_sessions_empty(self, client, tmp_contexts_dir) -> None:
        """GET /sessions/ returns empty list initially."""
        import gov_webui.adapter as adapter_mod
        from governor.session_store import SessionStore

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="general")
        adapter_mod._context_manager = cm
        adapter_mod._session_store = SessionStore(tmp_contexts_dir / "test-context" / "sessions")

        response = client.get("/sessions/")
        assert response.status_code == 200
        data = response.json()
        assert data["sessions"] == []

    def test_create_session(self, client, tmp_contexts_dir) -> None:
        """POST /sessions/ creates a new session."""
        import gov_webui.adapter as adapter_mod
        from governor.session_store import SessionStore

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="general")
        adapter_mod._context_manager = cm
        adapter_mod._session_store = SessionStore(tmp_contexts_dir / "test-context" / "sessions")

        response = client.post("/sessions/", json={"model": "test-model", "title": "My Chat"})
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "My Chat"
        assert data["model"] == "test-model"
        assert "id" in data
        assert data["message_count"] == 0

    def test_get_session(self, client, tmp_contexts_dir) -> None:
        """GET /sessions/{id} returns session with messages."""
        import gov_webui.adapter as adapter_mod
        from governor.session_store import SessionStore

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="general")
        adapter_mod._context_manager = cm
        store = SessionStore(tmp_contexts_dir / "test-context" / "sessions")
        adapter_mod._session_store = store

        # Create via API
        create_resp = client.post("/sessions/", json={"title": "Test"})
        session_id = create_resp.json()["id"]

        response = client.get(f"/sessions/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == session_id
        assert "messages" in data

    def test_get_session_404(self, client, tmp_contexts_dir) -> None:
        """GET /sessions/{id} returns 404 for missing session."""
        import gov_webui.adapter as adapter_mod
        from governor.session_store import SessionStore

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="general")
        adapter_mod._context_manager = cm
        adapter_mod._session_store = SessionStore(tmp_contexts_dir / "test-context" / "sessions")

        response = client.get("/sessions/nonexistent")
        assert response.status_code == 404

    def test_delete_session(self, client, tmp_contexts_dir) -> None:
        """DELETE /sessions/{id} removes session."""
        import gov_webui.adapter as adapter_mod
        from governor.session_store import SessionStore

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="general")
        adapter_mod._context_manager = cm
        adapter_mod._session_store = SessionStore(tmp_contexts_dir / "test-context" / "sessions")

        create_resp = client.post("/sessions/", json={"title": "To Delete"})
        session_id = create_resp.json()["id"]

        response = client.delete(f"/sessions/{session_id}")
        assert response.status_code == 200
        assert response.json()["success"] is True

        # Confirm deletion
        get_resp = client.get(f"/sessions/{session_id}")
        assert get_resp.status_code == 404

    def test_delete_session_404(self, client, tmp_contexts_dir) -> None:
        """DELETE /sessions/{id} returns 404 for missing session."""
        import gov_webui.adapter as adapter_mod
        from governor.session_store import SessionStore

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="general")
        adapter_mod._context_manager = cm
        adapter_mod._session_store = SessionStore(tmp_contexts_dir / "test-context" / "sessions")

        response = client.delete("/sessions/nonexistent")
        assert response.status_code == 404

    def test_update_title(self, client, tmp_contexts_dir) -> None:
        """PATCH /sessions/{id} updates title."""
        import gov_webui.adapter as adapter_mod
        from governor.session_store import SessionStore

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="general")
        adapter_mod._context_manager = cm
        adapter_mod._session_store = SessionStore(tmp_contexts_dir / "test-context" / "sessions")

        create_resp = client.post("/sessions/", json={"title": "Old"})
        session_id = create_resp.json()["id"]

        response = client.patch(f"/sessions/{session_id}", json={"title": "New Title"})
        assert response.status_code == 200
        assert response.json()["title"] == "New Title"

    def test_append_message(self, client, tmp_contexts_dir) -> None:
        """POST /sessions/{id}/messages appends a message."""
        import gov_webui.adapter as adapter_mod
        from governor.session_store import SessionStore

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="general")
        adapter_mod._context_manager = cm
        adapter_mod._session_store = SessionStore(tmp_contexts_dir / "test-context" / "sessions")

        create_resp = client.post("/sessions/", json={"title": "Chat"})
        session_id = create_resp.json()["id"]

        response = client.post(f"/sessions/{session_id}/messages", json={
            "role": "user", "content": "Hello world"
        })
        assert response.status_code == 200
        msg_data = response.json()
        assert msg_data["role"] == "user"
        assert msg_data["content"] == "Hello world"

        # Verify message was stored
        get_resp = client.get(f"/sessions/{session_id}")
        assert get_resp.json()["message_count"] == 1

    def test_session_roundtrip(self, client, tmp_contexts_dir) -> None:
        """Full roundtrip: create, add messages, retrieve."""
        import gov_webui.adapter as adapter_mod
        from governor.session_store import SessionStore

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="general")
        adapter_mod._context_manager = cm
        adapter_mod._session_store = SessionStore(tmp_contexts_dir / "test-context" / "sessions")

        # Create
        create_resp = client.post("/sessions/", json={"title": "Roundtrip", "model": "m1"})
        session_id = create_resp.json()["id"]

        # Add messages
        client.post(f"/sessions/{session_id}/messages", json={
            "role": "user", "content": "First"
        })
        client.post(f"/sessions/{session_id}/messages", json={
            "role": "assistant", "content": "Response", "model": "m1"
        })
        client.post(f"/sessions/{session_id}/messages", json={
            "role": "user", "content": "Second"
        })

        # Retrieve
        get_resp = client.get(f"/sessions/{session_id}")
        data = get_resp.json()
        assert data["message_count"] == 3
        assert data["messages"][0]["content"] == "First"
        assert data["messages"][1]["content"] == "Response"
        assert data["messages"][2]["content"] == "Second"

    def test_api_info_includes_session_endpoints(self, client) -> None:
        """GET /api/info includes session endpoints."""
        response = client.get("/api/info")
        data = response.json()
        endpoints = data["endpoints"]
        assert "sessions_list" in endpoints
        assert "sessions_create" in endpoints
        assert "sessions_get" in endpoints
        assert "sessions_delete" in endpoints
        assert "sessions_update" in endpoints
        assert "sessions_append_message" in endpoints

    def test_list_sessions_sorted_by_recent(self, client, tmp_contexts_dir) -> None:
        """Sessions are listed most-recent first."""
        import time
        import gov_webui.adapter as adapter_mod
        from governor.session_store import SessionStore

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="general")
        adapter_mod._context_manager = cm
        adapter_mod._session_store = SessionStore(tmp_contexts_dir / "test-context" / "sessions")

        client.post("/sessions/", json={"title": "Older"})
        time.sleep(0.01)
        client.post("/sessions/", json={"title": "Newer"})

        response = client.get("/sessions/")
        sessions = response.json()["sessions"]
        assert len(sessions) == 2
        assert sessions[0]["title"] == "Newer"

    def test_append_message_to_missing_session(self, client, tmp_contexts_dir) -> None:
        """POST /sessions/{id}/messages returns 404 for missing session."""
        import gov_webui.adapter as adapter_mod
        from governor.session_store import SessionStore

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="general")
        adapter_mod._context_manager = cm
        adapter_mod._session_store = SessionStore(tmp_contexts_dir / "test-context" / "sessions")

        response = client.post("/sessions/nonexistent/messages", json={
            "role": "user", "content": "Hello"
        })
        assert response.status_code == 404

    def test_update_title_missing_session(self, client, tmp_contexts_dir) -> None:
        """PATCH /sessions/{id} returns 404 for missing session."""
        import gov_webui.adapter as adapter_mod
        from governor.session_store import SessionStore

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="general")
        adapter_mod._context_manager = cm
        adapter_mod._session_store = SessionStore(tmp_contexts_dir / "test-context" / "sessions")

        response = client.patch("/sessions/nonexistent", json={"title": "X"})
        assert response.status_code == 404

    def test_create_session_default_title(self, client, tmp_contexts_dir) -> None:
        """POST /sessions/ with no title gets default."""
        import gov_webui.adapter as adapter_mod
        from governor.session_store import SessionStore

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="general")
        adapter_mod._context_manager = cm
        adapter_mod._session_store = SessionStore(tmp_contexts_dir / "test-context" / "sessions")

        response = client.post("/sessions/", json={})
        assert response.status_code == 200
        assert response.json()["title"] == "New conversation"


# ============================================================================
# TestExportImport
# ============================================================================


class TestExportImport:
    """Tests for governor export/import endpoints."""

    def test_export_empty(self, client, tmp_contexts_dir) -> None:
        """GET /governor/export returns empty state when nothing configured."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="fiction")
        adapter_mod._context_manager = cm

        response = client.get("/governor/export")
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == 1
        assert data["anchors"] == []
        assert "exported_at" in data

    def test_export_with_anchors(self, client, tmp_contexts_dir) -> None:
        """GET /governor/export includes all registered anchors."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        ctx = cm.create("test-context", mode="fiction")
        adapter_mod._context_manager = cm

        # Add a character via the POST endpoint
        client.post("/governor/fiction/characters", json={
            "name": "Elena", "description": "Tall, green eyes", "voice": "Formal", "wont": "Show weakness"
        })

        response = client.get("/governor/export")
        data = response.json()
        assert len(data["anchors"]) >= 1
        ids = [a["id"] for a in data["anchors"]]
        assert "char-elena" in ids

    def test_import_empty(self, client, tmp_contexts_dir) -> None:
        """POST /governor/import with empty anchors imports nothing."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="fiction")
        adapter_mod._context_manager = cm

        response = client.post("/governor/import", json={"anchors": []})
        assert response.status_code == 200
        data = response.json()
        assert data["imported"] == 0

    def test_import_anchors(self, client, tmp_contexts_dir) -> None:
        """POST /governor/import creates anchors from exported data."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="fiction")
        adapter_mod._context_manager = cm

        payload = {
            "anchors": [
                {
                    "id": "char-bob",
                    "anchor_type": "canon",
                    "description": "Appearance: Tall; Voice: Gruff",
                    "severity": "reject",
                },
                {
                    "id": "world-1",
                    "anchor_type": "definition",
                    "description": "Magic requires spoken words",
                    "severity": "reject",
                },
            ]
        }
        response = client.post("/governor/import", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["imported"] == 2
        assert data["skipped"] == 0

        # Verify they show up in export
        export = client.get("/governor/export").json()
        ids = [a["id"] for a in export["anchors"]]
        assert "char-bob" in ids
        assert "world-1" in ids

    def test_import_skips_duplicates(self, client, tmp_contexts_dir) -> None:
        """POST /governor/import skips anchors that already exist."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="fiction")
        adapter_mod._context_manager = cm

        anchor = {
            "id": "char-dup",
            "anchor_type": "canon",
            "description": "Test",
            "severity": "reject",
        }
        # Import once
        client.post("/governor/import", json={"anchors": [anchor]})
        # Import again — should skip
        response = client.post("/governor/import", json={"anchors": [anchor]})
        data = response.json()
        assert data["imported"] == 0
        assert data["skipped"] == 1

    def test_export_import_roundtrip(self, client, tmp_contexts_dir) -> None:
        """Export then import into a fresh context produces the same anchors."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="fiction")
        adapter_mod._context_manager = cm

        # Add some state
        client.post("/governor/fiction/characters", json={"name": "Alice", "description": "Brave"})
        client.post("/governor/fiction/world-rules", json={"rule": "No flying"})
        client.post("/governor/fiction/forbidden", json={"description": "Time travel"})

        # Export
        exported = client.get("/governor/export").json()
        original_count = len(exported["anchors"])
        assert original_count >= 3

        # Create fresh context and import
        cm2 = GovernorContextManager(base_dir=tmp_contexts_dir / "fresh")
        cm2.create("test-context", mode="fiction")
        adapter_mod._context_manager = cm2

        response = client.post("/governor/import", json=exported)
        data = response.json()
        assert data["imported"] == original_count

        # Verify export matches
        re_exported = client.get("/governor/export").json()
        assert len(re_exported["anchors"]) == original_count

    def test_import_no_context(self, client, tmp_contexts_dir) -> None:
        """POST /governor/import returns 400 when no context exists."""
        import gov_webui.adapter as adapter_mod
        adapter_mod._context_manager = GovernorContextManager(base_dir=tmp_contexts_dir)

        response = client.post("/governor/import", json={"anchors": []})
        assert response.status_code == 400

    def test_api_info_includes_export_import(self, client, tmp_contexts_dir) -> None:
        """GET /api/info includes export/import endpoint URLs."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="general")
        adapter_mod._context_manager = cm

        response = client.get("/api/info")
        data = response.json()
        assert "governor_export" in data["endpoints"]
        assert "governor_import" in data["endpoints"]


# ============================================================================
# TestResearchEndpoints
# ============================================================================


class TestResearchEndpoints:
    """Tests for /governor/research/ endpoints."""

    @pytest.fixture(autouse=True)
    def setup_research(self, client, tmp_contexts_dir) -> None:
        """Create research mode context and reset research store."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="research")
        adapter_mod._context_manager = cm
        adapter_mod._research_store = None
        adapter_mod.GOVERNOR_MODE = "research"

    def test_state_empty(self, client) -> None:
        """GET /governor/research/state returns empty state."""
        response = client.get("/governor/research/state")
        assert response.status_code == 200
        data = response.json()
        assert data["claims"] == []
        assert data["assumptions"] == []
        assert data["ed"]["total"] == 0

    def test_add_claim(self, client) -> None:
        """POST /governor/research/claims creates a claim."""
        response = client.post(
            "/governor/research/claims",
            json={"content": "Temperature affects rate"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert data["claim"]["content"] == "Temperature affects rate"
        assert data["claim"]["status"] == "floating"

    def test_add_claim_with_scope(self, client) -> None:
        """POST /governor/research/claims with scope."""
        response = client.post(
            "/governor/research/claims",
            json={"content": "Test claim", "scope": "Chapter 3"},
        )
        data = response.json()
        assert data["claim"]["scope"] == "Chapter 3"

    def test_delete_claim(self, client) -> None:
        """DELETE /governor/research/claims/{id} removes claim."""
        resp = client.post("/governor/research/claims", json={"content": "To delete"})
        claim_id = resp.json()["claim"]["id"]

        del_resp = client.delete(f"/governor/research/claims/{claim_id}")
        assert del_resp.status_code == 200

        state = client.get("/governor/research/state").json()
        assert len(state["claims"]) == 0

    def test_delete_claim_not_found(self, client) -> None:
        response = client.delete("/governor/research/claims/C-NONEXISTENT")
        assert response.status_code == 404

    def test_change_claim_status(self, client) -> None:
        """PATCH /governor/research/claims/{id}/status changes status."""
        resp = client.post("/governor/research/claims", json={"content": "Test"})
        claim_id = resp.json()["claim"]["id"]

        patch_resp = client.patch(
            f"/governor/research/claims/{claim_id}/status",
            json={"status": "retracted"},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["claim"]["status"] == "retracted"

    def test_change_claim_status_invalid(self, client) -> None:
        resp = client.post("/governor/research/claims", json={"content": "Test"})
        claim_id = resp.json()["claim"]["id"]

        patch_resp = client.patch(
            f"/governor/research/claims/{claim_id}/status",
            json={"status": "bogus"},
        )
        assert patch_resp.status_code == 400

    def test_add_assumption(self, client) -> None:
        """POST /governor/research/assumptions creates an assumption."""
        response = client.post(
            "/governor/research/assumptions",
            json={"content": "Stable incentives"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["assumption"]["status"] == "proposed"

    def test_delete_assumption(self, client) -> None:
        resp = client.post("/governor/research/assumptions", json={"content": "Test"})
        a_id = resp.json()["assumption"]["id"]
        del_resp = client.delete(f"/governor/research/assumptions/{a_id}")
        assert del_resp.status_code == 200

    def test_change_assumption_status(self, client) -> None:
        resp = client.post("/governor/research/assumptions", json={"content": "Test"})
        a_id = resp.json()["assumption"]["id"]
        patch_resp = client.patch(
            f"/governor/research/assumptions/{a_id}/status",
            json={"status": "accepted"},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["assumption"]["status"] == "accepted"

    def test_add_uncertainty(self, client) -> None:
        """POST /governor/research/uncertainties creates an uncertainty."""
        resp = client.post("/governor/research/claims", json={"content": "Claim"})
        claim_id = resp.json()["claim"]["id"]

        response = client.post(
            "/governor/research/uncertainties",
            json={"content": "Sample size", "attached_to": claim_id},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["uncertainty"]["attached_to"] == claim_id

    def test_delete_uncertainty(self, client) -> None:
        resp = client.post("/governor/research/uncertainties", json={"content": "Test"})
        u_id = resp.json()["uncertainty"]["id"]
        del_resp = client.delete(f"/governor/research/uncertainties/{u_id}")
        assert del_resp.status_code == 200

    def test_change_uncertainty_status(self, client) -> None:
        resp = client.post("/governor/research/uncertainties", json={"content": "Test"})
        u_id = resp.json()["uncertainty"]["id"]
        patch_resp = client.patch(
            f"/governor/research/uncertainties/{u_id}/status",
            json={"status": "resolved"},
        )
        assert patch_resp.status_code == 200

    def test_add_link(self, client) -> None:
        """POST /governor/research/links creates a typed link."""
        c1 = client.post("/governor/research/claims", json={"content": "Evidence"}).json()["claim"]
        c2 = client.post("/governor/research/claims", json={"content": "Main"}).json()["claim"]

        response = client.post(
            "/governor/research/links",
            json={"link_type": "supports", "from_id": c1["id"], "to_id": c2["id"]},
        )
        assert response.status_code == 200
        assert response.json()["link"]["link_type"] == "supports"

    def test_add_link_invalid_type(self, client) -> None:
        response = client.post(
            "/governor/research/links",
            json={"link_type": "bogus", "from_id": "a", "to_id": "b"},
        )
        assert response.status_code == 400

    def test_delete_link(self, client) -> None:
        c1 = client.post("/governor/research/claims", json={"content": "A"}).json()["claim"]
        c2 = client.post("/governor/research/claims", json={"content": "B"}).json()["claim"]
        link = client.post(
            "/governor/research/links",
            json={"link_type": "supports", "from_id": c1["id"], "to_id": c2["id"]},
        ).json()["link"]
        del_resp = client.delete(f"/governor/research/links/{link['id']}")
        assert del_resp.status_code == 200

    def test_state_reflects_ed(self, client) -> None:
        """State endpoint reflects ED computation."""
        client.post("/governor/research/claims", json={"content": "Floating claim"})
        state = client.get("/governor/research/state").json()
        assert state["ed"]["floating"] == 1
        assert state["ed"]["total"] > 0

    def test_export_includes_research(self, client) -> None:
        """Export includes research data in research mode."""
        client.post("/governor/research/claims", json={"content": "Test claim"})
        exported = client.get("/governor/export").json()
        assert "research" in exported
        assert len(exported["research"]["claims"]) == 1

    def test_import_research_data(self, client) -> None:
        """Import restores research data."""
        client.post("/governor/research/claims", json={"content": "Original"})
        exported = client.get("/governor/export").json()

        # Add a new claim to the exported data
        exported["research"]["claims"].append({
            "id": "C-IMPORT1",
            "content": "Imported",
            "status": "floating",
            "scope": "",
            "created_at": "2024-01-01T00:00:00",
        })

        result = client.post("/governor/import", json=exported).json()
        assert result["imported"] >= 1


# ============================================================================
# TestAuthMiddleware
# ============================================================================


class TestAuthMiddleware:
    """Tests for bearer token auth on mutating endpoints."""

    @pytest.fixture
    def auth_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Set up environment with auth token."""
        contexts_dir = tmp_path / "contexts"
        monkeypatch.setenv("BACKEND_TYPE", "ollama")
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
        monkeypatch.setenv("GOVERNOR_CONTEXT_ID", "auth-test")
        monkeypatch.setenv("GOVERNOR_MODE", "general")
        monkeypatch.setenv("GOVERNOR_CONTEXTS_DIR", str(contexts_dir))
        monkeypatch.setenv("GOVERNOR_AUTH_TOKEN", "test-secret-token")

    @pytest.fixture
    def auth_app(self, auth_env):
        import importlib
        import gov_webui.adapter as adapter_mod
        adapter_mod._bridge = None
        adapter_mod._context_manager = None
        adapter_mod._session_store = None
        importlib.reload(adapter_mod)
        yield adapter_mod.app
        adapter_mod._bridge = None
        adapter_mod._context_manager = None
        adapter_mod._session_store = None

    @pytest.fixture
    def auth_client(self, auth_app):
        from fastapi.testclient import TestClient
        return TestClient(auth_app)

    def test_get_endpoints_open(self, auth_client) -> None:
        """GET requests don't require auth even with token configured."""
        resp = auth_client.get("/health")
        assert resp.status_code == 200

    def test_post_without_token_rejected(self, auth_client) -> None:
        """POST without Authorization header returns 401."""
        resp = auth_client.post("/sessions/", json={"title": "test"})
        assert resp.status_code == 401
        assert "Authorization" in resp.json()["detail"]

    def test_post_with_wrong_token_rejected(self, auth_client) -> None:
        """POST with wrong token returns 403."""
        resp = auth_client.post(
            "/sessions/",
            json={"title": "test"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 403

    def test_post_with_correct_token_allowed(self, auth_client) -> None:
        """POST with correct token succeeds."""
        resp = auth_client.post(
            "/sessions/",
            json={"title": "test"},
            headers={"Authorization": "Bearer test-secret-token"},
        )
        assert resp.status_code == 200

    def test_delete_requires_token(self, auth_client) -> None:
        """DELETE requests also require auth."""
        resp = auth_client.delete("/sessions/nonexistent")
        assert resp.status_code == 401

    def test_get_governor_now_open(self, auth_client) -> None:
        """GET /governor/now doesn't require auth."""
        resp = auth_client.get("/governor/now")
        assert resp.status_code == 200


# ============================================================================
# TestCaptureEndpoints
# ============================================================================


class TestCaptureEndpoints:
    """Tests for fiction canon capture pipeline endpoints."""

    def test_scan_returns_captures(self, client) -> None:
        """POST /governor/fiction/capture/scan returns capture candidates."""
        text = "Character: Elena is a tall warrior with silver hair"
        resp = client.post("/governor/fiction/capture/scan", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert "captures" in data
        assert "receipt" in data
        assert data["receipt"]["classifier_version"].startswith("fiction.canon@")
        assert len(data["receipt"]["content_hash"]) == 64  # SHA-256 hex

    def test_scan_assigns_sequential_ids(self, client) -> None:
        """Each capture gets a unique sequential ID."""
        resp = client.post("/governor/fiction/capture/scan", json={
            "text": "Character: Alice is brave. Character: Bob is quiet."
        })
        data = resp.json()
        if len(data["captures"]) >= 2:
            ids = [c["id"] for c in data["captures"]]
            assert ids[0] != ids[1]
            # IDs are sequential
            assert all(i.startswith("cap-") for i in ids)

    def test_scan_captures_start_pending(self, client) -> None:
        """All captures start with status 'pending'."""
        resp = client.post("/governor/fiction/capture/scan", json={
            "text": "Character: Elena is a warrior"
        })
        data = resp.json()
        for cap in data["captures"]:
            assert cap["status"] == "pending"

    def test_scan_empty_text_returns_empty(self, client) -> None:
        """Scanning empty text returns no captures."""
        resp = client.post("/governor/fiction/capture/scan", json={"text": ""})
        assert resp.status_code == 200
        assert resp.json()["captures"] == []

    def test_scan_preserves_message_id(self, client) -> None:
        """message_id from request is stored on captures."""
        resp = client.post("/governor/fiction/capture/scan", json={
            "text": "Character: Elena",
            "message_id": "msg-42",
        })
        data = resp.json()
        for cap in data["captures"]:
            assert cap["message_id"] == "msg-42"

    def test_list_pending_empty(self, client) -> None:
        """GET /governor/fiction/captures returns empty when nothing scanned."""
        resp = client.get("/governor/fiction/captures")
        assert resp.status_code == 200
        data = resp.json()
        assert data["captures"] == []
        assert data["count"] == 0

    def test_list_pending_after_scan(self, client) -> None:
        """Scanned captures appear in pending list."""
        client.post("/governor/fiction/capture/scan", json={
            "text": "Character: Elena is a warrior"
        })
        resp = client.get("/governor/fiction/captures")
        data = resp.json()
        assert data["count"] >= 1
        assert all(c["status"] == "pending" for c in data["captures"])

    def test_accept_capture_character(self, client, tmp_contexts_dir) -> None:
        """Accept a character capture promotes to canon anchor."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="fiction")
        adapter_mod._context_manager = cm

        # Scan to create a pending capture
        scan_resp = client.post("/governor/fiction/capture/scan", json={
            "text": "Character: Elena is a tall warrior"
        })
        captures = scan_resp.json()["captures"]
        if not captures:
            pytest.skip("No captures detected from test text")

        cap_id = captures[0]["id"]

        # Accept it
        resp = client.post(f"/governor/fiction/capture/{cap_id}/accept", json={
            "name": "Elena",
            "capture_type": "character",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "char-elena" in data["id"]

    def test_accept_capture_world_rule(self, client, tmp_contexts_dir) -> None:
        """Accept a world_rule capture creates a DEFINITION anchor."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="fiction")
        adapter_mod._context_manager = cm

        # Inject a pending capture directly
        adapter_mod._pending_captures["cap-99"] = {
            "id": "cap-99",
            "kind": "world_rule",
            "confidence": 0.9,
            "subject": "",
            "statement": "Magic requires spoken words",
            "status": "pending",
        }

        resp = client.post("/governor/fiction/capture/cap-99/accept", json={
            "capture_type": "world_rule",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["id"].startswith("rule-")

    def test_accept_capture_constraint(self, client, tmp_contexts_dir) -> None:
        """Accept a constraint capture creates a PROHIBITION anchor."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="fiction")
        adapter_mod._context_manager = cm

        adapter_mod._pending_captures["cap-100"] = {
            "id": "cap-100",
            "kind": "constraint",
            "confidence": 0.85,
            "subject": "",
            "statement": "No time travel allowed",
            "status": "pending",
        }

        resp = client.post("/governor/fiction/capture/cap-100/accept", json={
            "capture_type": "constraint",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["id"].startswith("rule-")

    def test_accept_nonexistent_capture_404(self, client) -> None:
        """Accepting a nonexistent capture returns 404."""
        resp = client.post("/governor/fiction/capture/cap-999/accept", json={})
        assert resp.status_code == 404

    def test_accept_already_accepted_400(self, client, tmp_contexts_dir) -> None:
        """Accepting an already-accepted capture returns 400."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="fiction")
        adapter_mod._context_manager = cm

        adapter_mod._pending_captures["cap-50"] = {
            "id": "cap-50",
            "kind": "character",
            "confidence": 0.9,
            "subject": "Bob",
            "statement": "Bob is tall",
            "status": "accepted",
            "promoted_to": "char-bob",
        }

        resp = client.post("/governor/fiction/capture/cap-50/accept", json={
            "name": "Bob",
        })
        assert resp.status_code == 400
        assert "already accepted" in resp.json()["detail"]

    def test_reject_capture(self, client) -> None:
        """Rejecting a capture sets status to 'rejected'."""
        import gov_webui.adapter as adapter_mod

        adapter_mod._pending_captures["cap-77"] = {
            "id": "cap-77",
            "kind": "character",
            "confidence": 0.5,
            "subject": "Nobody",
            "statement": "Nobody is important",
            "status": "pending",
        }

        resp = client.post("/governor/fiction/capture/cap-77/reject")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert adapter_mod._pending_captures["cap-77"]["status"] == "rejected"

    def test_reject_nonexistent_capture_404(self, client) -> None:
        """Rejecting a nonexistent capture returns 404."""
        resp = client.post("/governor/fiction/capture/cap-999/reject")
        assert resp.status_code == 404

    def test_rejected_not_in_pending_list(self, client) -> None:
        """Rejected captures don't appear in pending list."""
        import gov_webui.adapter as adapter_mod

        adapter_mod._pending_captures["cap-88"] = {
            "id": "cap-88",
            "kind": "character",
            "confidence": 0.5,
            "subject": "Ghost",
            "statement": "Ghost haunts the manor",
            "status": "pending",
        }

        # Reject it
        client.post("/governor/fiction/capture/cap-88/reject")

        # Should not appear in pending list
        resp = client.get("/governor/fiction/captures")
        data = resp.json()
        assert data["count"] == 0

    def test_accepted_not_in_pending_list(self, client, tmp_contexts_dir) -> None:
        """Accepted captures don't appear in pending list."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="fiction")
        adapter_mod._context_manager = cm

        adapter_mod._pending_captures["cap-60"] = {
            "id": "cap-60",
            "kind": "character",
            "confidence": 0.9,
            "subject": "Zara",
            "statement": "Zara is a thief",
            "status": "pending",
        }

        client.post("/governor/fiction/capture/cap-60/accept", json={
            "name": "Zara",
            "capture_type": "character",
        })

        resp = client.get("/governor/fiction/captures")
        assert resp.json()["count"] == 0

    def test_scan_with_rule_text(self, client) -> None:
        """Scanning text with rule patterns detects world_rule captures."""
        resp = client.post("/governor/fiction/capture/scan", json={
            "text": "Rule: Magic requires spoken words to function"
        })
        data = resp.json()
        assert len(data["captures"]) >= 1
        kinds = [c["kind"] for c in data["captures"]]
        assert "world_rule" in kinds


# ============================================================================
# TestResearchCaptureEndpoints
# ============================================================================


class TestResearchCaptureEndpoints:
    """Tests for research capture pipeline endpoints."""

    def test_scan_detects_claim(self, client) -> None:
        """POST /governor/research/capture/scan detects claim patterns."""
        resp = client.post("/governor/research/capture/scan", json={
            "text": "Claim: Higher temperatures increase reaction rates."
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["captures"]) >= 1
        assert data["receipt"]["classifier_version"].startswith("research@")

    def test_scan_detects_doi(self, client) -> None:
        """DOI references are extracted as citation captures."""
        resp = client.post("/governor/research/capture/scan", json={
            "text": "See doi:10.1038/nature12373 for the original paper."
        })
        data = resp.json()
        refs = [c for c in data["captures"] if c.get("draft", {}).get("ref_type") == "doi"]
        assert len(refs) >= 1
        assert "10.1038/nature12373" in refs[0]["statement"]

    def test_scan_detects_cve(self, client) -> None:
        """CVE references are extracted as citation captures."""
        resp = client.post("/governor/research/capture/scan", json={
            "text": "This is related to CVE-2021-44228 (Log4Shell)."
        })
        data = resp.json()
        refs = [c for c in data["captures"] if c.get("draft", {}).get("ref_type") == "cve"]
        assert len(refs) >= 1

    def test_scan_detects_pypi(self, client) -> None:
        """PyPI references are extracted as citation captures."""
        resp = client.post("/governor/research/capture/scan", json={
            "text": "Install with pip install requests for HTTP."
        })
        data = resp.json()
        refs = [c for c in data["captures"] if c.get("draft", {}).get("ref_type") == "pypi"]
        assert len(refs) >= 1

    def test_scan_ids_prefixed_rcap(self, client) -> None:
        """Research captures get rcap- prefix (distinguishes from fiction cap-)."""
        resp = client.post("/governor/research/capture/scan", json={
            "text": "Claim: X is true."
        })
        data = resp.json()
        if data["captures"]:
            assert data["captures"][0]["id"].startswith("rcap-")

    def test_list_pending_empty(self, client) -> None:
        """GET /governor/research/captures returns empty when nothing scanned."""
        resp = client.get("/governor/research/captures")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_list_pending_after_scan(self, client) -> None:
        """Scanned captures appear in pending list."""
        client.post("/governor/research/capture/scan", json={
            "text": "Claim: The system converges under load."
        })
        resp = client.get("/governor/research/captures")
        assert resp.json()["count"] >= 1

    def test_accept_claim_to_ledger(self, client, tmp_contexts_dir) -> None:
        """Accept a claim capture → promotes to ResearchStore.claims."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="research")
        adapter_mod._context_manager = cm
        adapter_mod._research_store = None  # force re-init

        # Inject pending
        adapter_mod._pending_research_captures["rcap-10"] = {
            "id": "rcap-10",
            "kind": "claim",
            "confidence": 0.85,
            "subject": "",
            "statement": "Higher temperatures increase reaction rates",
            "status": "pending",
            "draft": {"assertion": "Higher temperatures increase reaction rates"},
        }

        resp = client.post("/governor/research/capture/rcap-10/accept", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["id"].startswith("C-")

    def test_accept_citation_with_source_ref(self, client, tmp_contexts_dir) -> None:
        """Accept a citation with DOI → promotes to ResearchStore with source_ref."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="research")
        adapter_mod._context_manager = cm
        adapter_mod._research_store = None

        adapter_mod._pending_research_captures["rcap-20"] = {
            "id": "rcap-20",
            "kind": "citation",
            "confidence": 0.95,
            "subject": "",
            "statement": "10.1038/nature12373",
            "status": "pending",
            "draft": {"source_ref": "doi:10.1038/nature12373", "ref_type": "doi"},
        }

        resp = client.post("/governor/research/capture/rcap-20/accept", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_accept_assumption(self, client, tmp_contexts_dir) -> None:
        """Accept an assumption capture → promotes to ResearchStore.assumptions."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="research")
        adapter_mod._context_manager = cm
        adapter_mod._research_store = None

        adapter_mod._pending_research_captures["rcap-30"] = {
            "id": "rcap-30",
            "kind": "assumption",
            "confidence": 0.80,
            "subject": "",
            "statement": "Incentive structures remain stable over time",
            "status": "pending",
            "draft": {"assumption": "Incentive structures remain stable over time"},
        }

        resp = client.post("/governor/research/capture/rcap-30/accept", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["id"].startswith("A-")

    def test_reject_capture(self, client) -> None:
        """Rejecting sets status to 'rejected'."""
        import gov_webui.adapter as adapter_mod

        adapter_mod._pending_research_captures["rcap-40"] = {
            "id": "rcap-40",
            "kind": "claim",
            "confidence": 0.5,
            "subject": "",
            "statement": "Something uncertain",
            "status": "pending",
        }

        resp = client.post("/governor/research/capture/rcap-40/reject")
        assert resp.status_code == 200
        assert adapter_mod._pending_research_captures["rcap-40"]["status"] == "rejected"

    def test_accept_nonexistent_404(self, client) -> None:
        """Accepting nonexistent returns 404."""
        resp = client.post("/governor/research/capture/rcap-999/accept", json={})
        assert resp.status_code == 404

    def test_reject_nonexistent_404(self, client) -> None:
        """Rejecting nonexistent returns 404."""
        resp = client.post("/governor/research/capture/rcap-999/reject")
        assert resp.status_code == 404

    def test_rejected_not_in_pending(self, client) -> None:
        """Rejected captures don't appear in pending list."""
        import gov_webui.adapter as adapter_mod

        adapter_mod._pending_research_captures["rcap-50"] = {
            "id": "rcap-50",
            "kind": "claim",
            "confidence": 0.5,
            "subject": "",
            "statement": "Ghost claim",
            "status": "pending",
        }
        client.post("/governor/research/capture/rcap-50/reject")
        resp = client.get("/governor/research/captures")
        assert resp.json()["count"] == 0

    def test_scan_empty_text(self, client) -> None:
        """Empty text returns no captures."""
        resp = client.post("/governor/research/capture/scan", json={"text": ""})
        assert resp.status_code == 200
        assert resp.json()["captures"] == []


# ============================================================================
# Why Overlay
# ============================================================================


class TestWhyOverlay:
    """Tests for the per-turn Why overlay endpoint."""

    @pytest.fixture(autouse=True)
    def setup_research(self, client, tmp_contexts_dir) -> None:
        """Create research mode context for Why overlay tests."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="research")
        adapter_mod._context_manager = cm
        adapter_mod.GOVERNOR_MODE = "research"

    def test_empty_text(self, client) -> None:
        """Empty text returns clean overlay."""
        resp = client.post("/governor/research/why", json={"text": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert data["injected"]["source_count"] == 0
        assert data["injected"]["claim_count"] == 0
        assert data["referenced"]["sources"] == []
        assert data["floating"] == []
        assert data["matched"] == []

    def test_floating_ref_detected(self, client) -> None:
        """Source ref not in accepted list is flagged as floating."""
        resp = client.post("/governor/research/why", json={
            "text": "See doi:10.9999/ghost for more."
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["floating"]) == 1
        assert data["floating"][0]["ref_type"] == "doi"

    def test_candidate_source_detected(self, client) -> None:
        """CANDIDATE_SOURCE lines are extracted."""
        resp = client.post("/governor/research/why", json={
            "text": "I found a new paper.\nCANDIDATE_SOURCE: doi:10.1234/new"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["referenced"]["candidates"]) == 1
        assert "doi:10.1234/new" in data["referenced"]["candidates"]

    def test_with_accepted_sources(self, client, tmp_contexts_dir) -> None:
        """When store has accepted claims with source_refs, they appear in injected."""
        from governor.research_store import ResearchStore

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        ctx = cm.get("test-context")
        store = ResearchStore(ctx.governor_dir)
        store.add_claim("Test claim", source_ref="doi:10.1234/accepted")

        resp = client.post("/governor/research/why", json={
            "text": "Based on doi:10.1234/accepted, the result is clear."
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["injected"]["source_count"] == 1
        assert len(data["matched"]) == 1
        assert data["matched"][0]["identifier"] == "10.1234/accepted"
        assert len(data["floating"]) == 0

    def test_matched_vs_floating(self, client, tmp_contexts_dir) -> None:
        """Mix of matched and floating refs classified correctly."""
        from governor.research_store import ResearchStore

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        ctx = cm.get("test-context")
        store = ResearchStore(ctx.governor_dir)
        store.add_claim("Known source", source_ref="doi:10.1234/known")

        resp = client.post("/governor/research/why", json={
            "text": "See doi:10.1234/known and also doi:10.9999/ghost."
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["matched"]) == 1
        assert len(data["floating"]) == 1

    def test_overlay_structure(self, client) -> None:
        """WhyOverlay dict has expected top-level keys."""
        resp = client.post("/governor/research/why", json={"text": "hello"})
        data = resp.json()
        assert "injected" in data
        assert "referenced" in data
        assert "floating" in data
        assert "matched" in data
        assert "source_count" in data["injected"]
        assert "claim_count" in data["injected"]
        assert "sources" in data["referenced"]
        assert "candidates" in data["referenced"]


# ============================================================================
# TestUninitializedContext — regression tests for missing _context.json
# ============================================================================


class TestUninitializedContext:
    """Verify that mutation endpoints return actionable errors when the
    governor context metadata (_context.json) is missing.

    Root cause of the Feb 2026 fiction panel bug: entrypoint.sh created
    .governor/ but never wrote _context.json, so GovernorContextManager.get()
    returned None and all mode-specific CRUD silently failed.
    """

    def test_fiction_characters_post_uninitialized(self, client) -> None:
        """POST /governor/fiction/characters returns 400 with detail when no context."""
        resp = client.post("/governor/fiction/characters", json={
            "name": "Alice", "description": "Brave", "voice": "Dry", "wont": ""
        })
        assert resp.status_code == 400
        data = resp.json()
        assert "detail" in data
        assert "context" in data["detail"].lower()

    def test_fiction_characters_get_uninitialized(self, client) -> None:
        """GET /governor/fiction/characters returns empty list (not error) when no context."""
        resp = client.get("/governor/fiction/characters")
        assert resp.status_code == 200
        data = resp.json()
        assert data["characters"] == []

    def test_fiction_world_rules_post_uninitialized(self, client) -> None:
        """POST /governor/fiction/world-rules returns 400 when no context."""
        resp = client.post("/governor/fiction/world-rules", json={"rule": "No flying"})
        assert resp.status_code == 400
        assert "detail" in resp.json()

    def test_fiction_forbidden_post_uninitialized(self, client) -> None:
        """POST /governor/fiction/forbidden returns 400 when no context."""
        resp = client.post("/governor/fiction/forbidden", json={"description": "Time travel"})
        assert resp.status_code == 400
        assert "detail" in resp.json()

    def test_code_decisions_post_uninitialized(self, client) -> None:
        """POST /governor/code/decisions returns 400 when no context."""
        resp = client.post("/governor/code/decisions", json={
            "decision": "framework: react", "rationale": "popular"
        })
        assert resp.status_code == 400
        assert "detail" in resp.json()

    def test_code_constraints_post_uninitialized(self, client) -> None:
        """POST /governor/code/constraints returns 400 when no context."""
        resp = client.post("/governor/code/constraints", json={"constraint": "No eval"})
        assert resp.status_code == 400
        assert "detail" in resp.json()


class TestInitializedFictionCRUD:
    """Verify fiction CRUD works end-to-end when context is properly initialized."""

    def test_character_roundtrip(self, client, tmp_contexts_dir) -> None:
        """POST then GET /governor/fiction/characters returns the character."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="fiction")
        adapter_mod._context_manager = cm

        # Add character
        resp = client.post("/governor/fiction/characters", json={
            "name": "Elena", "description": "Tall", "voice": "Formal", "wont": "Show weakness"
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        # Read back
        resp = client.get("/governor/fiction/characters")
        assert resp.status_code == 200
        chars = resp.json()["characters"]
        assert len(chars) == 1
        assert chars[0]["name"] == "Elena"
        assert "Formal" in chars[0]["description"]
        assert chars[0]["wont"] is not None

    def test_world_rule_roundtrip(self, client, tmp_contexts_dir) -> None:
        """POST then GET /governor/fiction/world-rules returns the rule."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="fiction")
        adapter_mod._context_manager = cm

        resp = client.post("/governor/fiction/world-rules", json={"rule": "Magic costs blood"})
        assert resp.status_code == 200

        resp = client.get("/governor/fiction/world-rules")
        rules = resp.json()["rules"]
        assert len(rules) == 1
        assert rules[0]["rule"] == "Magic costs blood"

    def test_forbidden_roundtrip(self, client, tmp_contexts_dir) -> None:
        """POST then GET /governor/fiction/forbidden returns the item."""
        import gov_webui.adapter as adapter_mod

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="fiction")
        adapter_mod._context_manager = cm

        resp = client.post("/governor/fiction/forbidden", json={"description": "Time travel"})
        assert resp.status_code == 200

        resp = client.get("/governor/fiction/forbidden")
        forbidden = resp.json()["forbidden"]
        assert len(forbidden) == 1
        assert forbidden[0]["description"] == "Time travel"


# ============================================================================
# Code Builder Integration Tests
# ============================================================================


class TestCodeBuilderProject:
    """Integration tests for code builder project endpoints."""

    def _init_code_context(self, tmp_contexts_dir):
        """Helper: create a code-mode context and wire up the adapter."""
        import gov_webui.adapter as adapter_mod
        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="code")
        adapter_mod._context_manager = cm
        adapter_mod._project_store = None  # force re-init

    def test_get_project_empty(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        resp = client.get("/governor/code/project")
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"]["text"] == ""
        assert data["plan"]["phases"] == []
        assert data["files"] == {}

    def test_put_intent(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        resp = client.put("/governor/code/project/intent", json={
            "text": "Parse CSV files", "locked": False
        })
        assert resp.status_code == 200
        assert resp.json()["intent"]["text"] == "Parse CSV files"

        # Verify via GET
        state = client.get("/governor/code/project").json()
        assert state["intent"]["text"] == "Parse CSV files"

    def test_put_intent_lock(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        resp = client.put("/governor/code/project/intent", json={
            "text": "Locked intent", "locked": True
        })
        assert resp.json()["intent"]["locked"] is True

    def test_put_contract(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        resp = client.put("/governor/code/project/contract", json={
            "description": "CSV parser",
            "inputs": [{"name": "filepath", "type": "str"}],
            "outputs": [{"name": "rows", "type": "list"}],
            "constraints": ["No pandas"],
            "transport": "stdio"
        })
        assert resp.status_code == 200
        contract = resp.json()["contract"]
        assert contract["description"] == "CSV parser"
        assert len(contract["inputs"]) == 1
        assert contract["constraints"] == ["No pandas"]

    def test_stale_version_409(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        # Get initial version
        state = client.get("/governor/code/project").json()
        v = state["version"]

        # Update once (bumps version)
        client.put("/governor/code/project/intent", json={
            "text": "first", "locked": False
        })

        # Try with stale version
        resp = client.put("/governor/code/project/intent", json={
            "text": "second", "locked": False, "expected_version": v
        })
        assert resp.status_code == 409


class TestCodeBuilderPlan:
    """Integration tests for plan CRUD + state machine."""

    def _init_code_context(self, tmp_contexts_dir):
        import gov_webui.adapter as adapter_mod
        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="code")
        adapter_mod._context_manager = cm
        adapter_mod._project_store = None

    def test_add_phase_and_item(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        resp = client.post("/governor/code/plan/phase", json={"name": "Build"})
        assert resp.status_code == 200

        resp = client.post("/governor/code/plan/item", json={
            "phase_idx": 0, "text": "Write parser"
        })
        assert resp.status_code == 200
        assert resp.json()["item"]["id"] == "p0-0"

    def test_item_status_transitions(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        client.post("/governor/code/plan/phase", json={"name": "Phase 1"})
        client.post("/governor/code/plan/item", json={"phase_idx": 0, "text": "Task A"})

        # proposed -> accepted -> in_progress -> completed
        for status in ["accepted", "in_progress", "completed"]:
            resp = client.patch("/governor/code/plan/item/p0-0", json={"status": status})
            assert resp.status_code == 200
            assert resp.json()["item"]["status"] == status

    def test_invalid_transition_400(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        client.post("/governor/code/plan/phase", json={"name": "Phase 1"})
        client.post("/governor/code/plan/item", json={"phase_idx": 0, "text": "Task A"})

        # proposed -> completed (skip accepted)
        resp = client.patch("/governor/code/plan/item/p0-0", json={"status": "completed"})
        assert resp.status_code == 400

    def test_phase_gating(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        client.post("/governor/code/plan/phase", json={"name": "Phase 1"})
        client.post("/governor/code/plan/phase", json={"name": "Phase 2"})
        client.post("/governor/code/plan/item", json={"phase_idx": 0, "text": "P1 task"})
        client.post("/governor/code/plan/item", json={"phase_idx": 1, "text": "P2 task"})

        # Try to advance phase 2 item before phase 1 is complete
        resp = client.patch("/governor/code/plan/item/p1-0", json={"status": "accepted"})
        assert resp.status_code == 400
        assert "incomplete" in resp.json()["detail"]

    def test_update_phase(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        client.post("/governor/code/plan/phase", json={"name": "Old"})

        resp = client.patch("/governor/code/plan/phase/0", json={"name": "New"})
        assert resp.status_code == 200
        assert resp.json()["phase"]["name"] == "New"


class TestCodeBuilderFiles:
    """Integration tests for file accept + version + hash."""

    def _init_code_context(self, tmp_contexts_dir):
        import gov_webui.adapter as adapter_mod
        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="code")
        adapter_mod._context_manager = cm
        adapter_mod._project_store = None

    def test_put_and_get_file(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        resp = client.put("/governor/code/files/tool.py", json={
            "content": "print('hello')\n", "turn_id": "turn-abc123"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == 1
        assert data["content_hash"]

        resp = client.get("/governor/code/files/tool.py")
        assert resp.status_code == 200
        assert resp.json()["content"] == "print('hello')\n"

    def test_file_versioning(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        r1 = client.put("/governor/code/files/tool.py", json={"content": "v1"})
        assert r1.json()["version"] == 1
        r2 = client.put("/governor/code/files/tool.py", json={"content": "v2"})
        assert r2.json()["version"] == 2

    def test_file_prev(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        client.put("/governor/code/files/tool.py", json={"content": "v1"})
        client.put("/governor/code/files/tool.py", json={"content": "v2"})

        resp = client.get("/governor/code/file-prev/tool.py")
        assert resp.status_code == 200
        assert resp.json()["content"] == "v1"

    def test_list_files(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        client.put("/governor/code/files/tool.py", json={"content": "code"})
        client.put("/governor/code/files/test_tool.py", json={"content": "tests"})

        resp = client.get("/governor/code/files")
        files = resp.json()["files"]
        assert "tool.py" in files
        assert "test_tool.py" in files

    def test_path_safety_rejects_traversal(self, client, tmp_contexts_dir) -> None:
        """Path traversal is tested at the store layer; HTTP layer may normalize.
        Test the store directly for completeness."""
        self._init_code_context(tmp_contexts_dir)
        import gov_webui.adapter as adapter_mod
        store = adapter_mod._get_project_store()
        with pytest.raises(ValueError, match="traversal"):
            store.put_file("../escape.py", "nope")

    def test_path_safety_rejects_bad_extension(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        resp = client.put("/governor/code/files/script.sh", json={
            "content": "#!/bin/bash"
        })
        assert resp.status_code == 400

    def test_file_not_found(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        resp = client.get("/governor/code/files/nonexistent.py")
        assert resp.status_code == 404


class TestCodeBuilderRun:
    """Integration tests for the run endpoint."""

    def _init_code_context(self, tmp_contexts_dir):
        import gov_webui.adapter as adapter_mod
        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("test-context", mode="code")
        adapter_mod._context_manager = cm
        adapter_mod._project_store = None

    def test_run_success(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        client.put("/governor/code/files/tool.py", json={
            "content": "print('hello world')"
        })

        resp = client.post("/governor/code/run", json={"filepath": "tool.py"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["returncode"] == 0
        assert "hello world" in data["stdout"]

    def test_run_failure(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        client.put("/governor/code/files/tool.py", json={
            "content": "raise ValueError('boom')"
        })

        resp = client.post("/governor/code/run", json={"filepath": "tool.py"})
        data = resp.json()
        assert data["success"] is False
        assert data["returncode"] != 0
        assert "boom" in data["stderr"]

    def test_run_timeout(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        client.put("/governor/code/files/tool.py", json={
            "content": "import time; time.sleep(10)"
        })

        resp = client.post("/governor/code/run", json={
            "filepath": "tool.py", "timeout": 1
        })
        data = resp.json()
        assert data["success"] is False
        assert "timeout" in data["stderr"].lower()

    def test_run_no_files(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        resp = client.post("/governor/code/run", json={"filepath": "tool.py"})
        assert resp.status_code == 400

    def test_run_missing_entrypoint(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        client.put("/governor/code/files/other.py", json={"content": "pass"})

        resp = client.post("/governor/code/run", json={"filepath": "tool.py"})
        assert resp.status_code == 404

    def test_run_multifile(self, client, tmp_contexts_dir) -> None:
        """File A imports file B — both should be available in tempdir."""
        self._init_code_context(tmp_contexts_dir)
        client.put("/governor/code/files/helper.py", json={
            "content": "def greet():\n    return 'hi from helper'"
        })
        client.put("/governor/code/files/tool.py", json={
            "content": "from helper import greet\nprint(greet())"
        })

        resp = client.post("/governor/code/run", json={"filepath": "tool.py"})
        data = resp.json()
        assert data["success"] is True
        assert "hi from helper" in data["stdout"]

    def test_run_with_stdin(self, client, tmp_contexts_dir) -> None:
        self._init_code_context(tmp_contexts_dir)
        client.put("/governor/code/files/tool.py", json={
            "content": "import sys; print(sys.stdin.read().upper())"
        })

        resp = client.post("/governor/code/run", json={
            "filepath": "tool.py", "stdin": "hello"
        })
        data = resp.json()
        assert data["success"] is True
        assert "HELLO" in data["stdout"]


# ============================================================================
# Constraints injection tests
# ============================================================================


class TestConstraintsInjection:
    """Test _build_constraints_message() and chat injection."""

    def test_constraints_injected_when_config_present(self, tmp_contexts_dir) -> None:
        """When config is set on the contract, _build_constraints_message returns a system message."""
        import gov_webui.adapter as adapter_mod

        # Reset singletons
        adapter_mod._project_store = None
        adapter_mod._research_project_store = None
        adapter_mod.GOVERNOR_MODE = "code"
        adapter_mod.GOVERNOR_CONTEXTS_DIR = str(tmp_contexts_dir)
        adapter_mod.GOVERNOR_CONTEXT_ID = "constraints-test"

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("constraints-test", mode="code")
        adapter_mod._context_manager = cm

        from fastapi.testclient import TestClient
        client = TestClient(adapter_mod.app, raise_server_exceptions=False)

        # Set contract with config
        client.put("/governor/code/project/contract", json={
            "description": "test",
            "config": {
                "artifact_type": "tool",
                "length": "medium",
                "voice": ["dry", "wry"],
                "citations": "none",
                "bans": ["studies show"],
                "strict": False,
            },
        })

        msg, meta = adapter_mod._build_constraints_message()
        assert msg is not None
        assert msg["role"] == "system"
        assert "[CONSTRAINTS" in msg["content"]
        assert "artifact_type: tool" in msg["content"]
        assert "length_band: medium" in msg["content"]
        assert "voice: dry, wry" in msg["content"]
        assert "bans: studies show" in msg["content"]
        assert "[/CONSTRAINTS]" in msg["content"]
        assert meta["mode"] == "code"

        # Clean up
        adapter_mod._project_store = None
        adapter_mod._context_manager = None

    def test_constraints_inserted_after_system_messages(self, tmp_contexts_dir) -> None:
        """Constraints system message goes after existing system messages."""
        import gov_webui.adapter as adapter_mod

        adapter_mod._project_store = None
        adapter_mod.GOVERNOR_MODE = "code"
        adapter_mod.GOVERNOR_CONTEXTS_DIR = str(tmp_contexts_dir)
        adapter_mod.GOVERNOR_CONTEXT_ID = "pos-test"

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("pos-test", mode="code")
        adapter_mod._context_manager = cm

        from fastapi.testclient import TestClient
        client = TestClient(adapter_mod.app, raise_server_exceptions=False)

        client.put("/governor/code/project/contract", json={
            "description": "test",
            "config": {"artifact_type": "tool", "length": "short"},
        })

        # Simulate message injection
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        msg, meta = adapter_mod._build_constraints_message()
        assert msg is not None

        insert_idx = 0
        for i, m in enumerate(messages):
            if m["role"] == "system":
                insert_idx = i + 1
        messages.insert(insert_idx, msg)

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "system"
        assert "[CONSTRAINTS" in messages[1]["content"]
        assert messages[2]["role"] == "user"

        adapter_mod._project_store = None
        adapter_mod._context_manager = None

    def test_no_constraints_when_config_absent(self, tmp_contexts_dir) -> None:
        """No constraints message when contract has no config."""
        import gov_webui.adapter as adapter_mod

        adapter_mod._project_store = None
        adapter_mod.GOVERNOR_MODE = "code"
        adapter_mod.GOVERNOR_CONTEXTS_DIR = str(tmp_contexts_dir)
        adapter_mod.GOVERNOR_CONTEXT_ID = "no-config-test"

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("no-config-test", mode="code")
        adapter_mod._context_manager = cm

        from fastapi.testclient import TestClient
        client = TestClient(adapter_mod.app, raise_server_exceptions=False)

        # Set contract without config
        client.put("/governor/code/project/contract", json={
            "description": "test",
        })

        msg, meta = adapter_mod._build_constraints_message()
        assert msg is None  # No constraints, no config

        adapter_mod._project_store = None
        adapter_mod._context_manager = None

    def test_no_constraints_in_general_mode(self) -> None:
        """No injection in general mode."""
        import gov_webui.adapter as adapter_mod

        old_mode = adapter_mod.GOVERNOR_MODE
        adapter_mod.GOVERNOR_MODE = "general"
        msg, meta = adapter_mod._build_constraints_message()
        assert msg is None
        assert meta["mode"] == "general"
        adapter_mod.GOVERNOR_MODE = old_mode

    def test_raw_constraints_fallback(self, tmp_contexts_dir) -> None:
        """When no config but constraints list present, uses raw fallback."""
        import gov_webui.adapter as adapter_mod

        adapter_mod._project_store = None
        adapter_mod.GOVERNOR_MODE = "code"
        adapter_mod.GOVERNOR_CONTEXTS_DIR = str(tmp_contexts_dir)
        adapter_mod.GOVERNOR_CONTEXT_ID = "raw-test"

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("raw-test", mode="code")
        adapter_mod._context_manager = cm

        from fastapi.testclient import TestClient
        client = TestClient(adapter_mod.app, raise_server_exceptions=False)

        client.put("/governor/code/project/contract", json={
            "description": "test",
            "constraints": ["No pandas", "Handle UTF-8"],
        })

        msg, meta = adapter_mod._build_constraints_message()
        assert msg is not None
        assert msg["role"] == "system"
        assert "[CONSTRAINTS]" in msg["content"]
        assert "No pandas" in msg["content"]
        assert "Handle UTF-8" in msg["content"]
        assert meta["mode"] == "code"

        adapter_mod._project_store = None
        adapter_mod._context_manager = None


# ============================================================================
# Receipt tests
# ============================================================================


class TestReceipt:
    """Tests for per-turn receipt in SSE and non-streaming responses."""

    def _make_mock_daemon(self, content="Hello from test", model="test-model",
                          usage=None, footer=None, pending=None):
        mock = AsyncMock()
        mock.chat_send = AsyncMock(return_value={
            "content": content,
            "model": model,
            "usage": usage or {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "violations": [],
            "footer": footer,
            "pending": pending,
        })
        mock.connect = AsyncMock()
        mock.close = AsyncMock()
        return mock

    def test_receipt_in_streaming_first_and_done_chunks(self, client) -> None:
        """Receipt appears in both the first SSE chunk and the done chunk."""
        import gov_webui.adapter as adapter_mod

        async def mock_stream(*args, **kwargs):
            yield ("Hello ", None)
            yield ("world", None)
            yield (None, {
                "content": "Hello world",
                "model": "test-model",
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
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert response.status_code == 200

        # Parse SSE chunks to find receipts
        chunks_with_receipt = []
        for line in response.text.split("\n"):
            line = line.strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            parsed = json.loads(line[6:])
            if "receipt" in parsed:
                chunks_with_receipt.append(parsed)

        # Should have receipt in first chunk and done chunk
        assert len(chunks_with_receipt) >= 2
        first_receipt = chunks_with_receipt[0]["receipt"]
        done_receipt = chunks_with_receipt[-1]["receipt"]

        # Both receipts have required fields
        for r in (first_receipt, done_receipt):
            assert "constraints_hash" in r
            assert "config_hash" in r
            assert "mode" in r
            assert "model" in r
            assert "turn_id" in r
            assert "turn_seq" in r
            assert "request_id" in r
            assert r["constraints_format_version"] == 1
            assert r["rendered_by"] == "adapter"

    def test_receipt_constraints_hash_correctness(self, tmp_contexts_dir) -> None:
        """constraints_hash matches SHA-256[:16] of the constraints block content."""
        import hashlib
        import gov_webui.adapter as adapter_mod

        adapter_mod._project_store = None
        adapter_mod._research_project_store = None
        adapter_mod.GOVERNOR_MODE = "code"
        adapter_mod.GOVERNOR_CONTEXTS_DIR = str(tmp_contexts_dir)
        adapter_mod.GOVERNOR_CONTEXT_ID = "receipt-hash-test"

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("receipt-hash-test", mode="code")
        adapter_mod._context_manager = cm

        from fastapi.testclient import TestClient
        test_client = TestClient(adapter_mod.app, raise_server_exceptions=False)

        test_client.put("/governor/code/project/contract", json={
            "description": "test",
            "config": {
                "artifact_type": "tool",
                "length": "medium",
                "strict": False,
            },
        })

        msg, meta = adapter_mod._build_constraints_message()
        assert msg is not None

        # Compute expected hash
        expected_full = hashlib.sha256(msg["content"].encode("utf-8")).hexdigest()
        expected_short = expected_full[:16]

        # Build receipt and verify
        receipt = adapter_mod._build_receipt(
            msg, meta, "req-1", "turn-1", 1, "test-model"
        )
        assert receipt["constraints_hash"] == expected_short
        assert receipt["constraints_hash_full"] == expected_full

        adapter_mod._project_store = None
        adapter_mod._context_manager = None

    def test_receipt_no_constraints_in_general_mode(self, client) -> None:
        """In general mode, constraints_hash is None, mode is 'general'."""
        import gov_webui.adapter as adapter_mod

        adapter_mod._daemon_client = self._make_mock_daemon()
        adapter_mod.GOVERNOR_MODE = "general"

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        receipt = data.get("receipt")
        assert receipt is not None
        assert receipt["constraints_hash"] is None
        assert receipt["mode"] == "general"

    def test_receipt_strict_flag(self, tmp_contexts_dir) -> None:
        """When config.strict = True, receipt.strict is True."""
        import gov_webui.adapter as adapter_mod

        adapter_mod._project_store = None
        adapter_mod._research_project_store = None
        adapter_mod.GOVERNOR_MODE = "code"
        adapter_mod.GOVERNOR_CONTEXTS_DIR = str(tmp_contexts_dir)
        adapter_mod.GOVERNOR_CONTEXT_ID = "strict-test"

        cm = GovernorContextManager(base_dir=tmp_contexts_dir)
        cm.create("strict-test", mode="code")
        adapter_mod._context_manager = cm

        from fastapi.testclient import TestClient
        test_client = TestClient(adapter_mod.app, raise_server_exceptions=False)

        test_client.put("/governor/code/project/contract", json={
            "description": "test",
            "config": {
                "artifact_type": "tool",
                "length": "short",
                "strict": True,
            },
        })

        msg, meta = adapter_mod._build_constraints_message()
        assert msg is not None
        assert meta["strict"] is True

        receipt = adapter_mod._build_receipt(
            msg, meta, "req-1", "turn-1", 1, "test-model"
        )
        assert receipt["strict"] is True

        adapter_mod._project_store = None
        adapter_mod._context_manager = None

    def test_receipt_in_non_streaming_response(self, client) -> None:
        """Non-streaming response includes receipt field."""
        import gov_webui.adapter as adapter_mod

        adapter_mod._daemon_client = self._make_mock_daemon()

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "receipt" in data
        receipt = data["receipt"]
        assert receipt["model"] == "test-model"
        assert receipt["turn_seq"] >= 1
        assert receipt["turn_id"].startswith("turn-")
        assert receipt["request_id"].startswith("chatcmpl-")


# ============================================================================
# Artifact Engine Tests
# ============================================================================


class FakeArtifactStore:
    """In-memory artifact store for adapter endpoint tests."""

    def __init__(self):
        from gov_webui.artifact_store import (
            ArtifactMeta,
            ArtifactNotFoundError,
            ArtifactSummary,
            ArtifactValidationError,
            ArtifactVersion,
            ArtifactVersionNotFoundError,
            StaleArtifactVersionError,
        )
        self._NotFound = ArtifactNotFoundError
        self._VersionNotFound = ArtifactVersionNotFoundError
        self._Stale = StaleArtifactVersionError
        self._Validation = ArtifactValidationError
        self._Meta = ArtifactMeta
        self._Version = ArtifactVersion
        self._Summary = ArtifactSummary

        self._artifacts = {}   # id -> {meta, versions_content}
        self._index_version = 1

    def create(self, *, title, content, kind="text", language="",
               message_id=None, source="manual", source_turn_seq=None):
        if kind not in ("text", "markdown", "code"):
            raise self._Validation(f"Invalid kind '{kind}'")
        aid = f"fake{len(self._artifacts):04d}"
        now = "2026-02-21T00:00:00+00:00"
        ver = self._Version(
            version=1, created_at=now, content_hash="abcd1234abcd1234",
            source=source, message_id=message_id,
            source_turn_seq=source_turn_seq,
        )
        meta = self._Meta(
            id=aid, title=title, kind=kind, language=language,
            current_version=1, versions=[ver], created_at=now, updated_at=now,
        )
        self._artifacts[aid] = {"meta": meta, "content": {1: content}}
        self._index_version += 1
        return meta, content, self._index_version

    def update(self, artifact_id, *, content, title=None,
               expected_current_version=None, source="manual", message_id=None,
               source_turn_seq=None):
        if artifact_id not in self._artifacts:
            raise self._NotFound(artifact_id)
        entry = self._artifacts[artifact_id]
        meta = entry["meta"]
        if (expected_current_version is not None
                and expected_current_version != meta.current_version):
            raise self._Stale(
                artifact_id, expected_current_version,
                meta.current_version, self._index_version,
            )
        now = "2026-02-21T00:00:01+00:00"
        new_ver = meta.current_version + 1
        ver = self._Version(
            version=new_ver, created_at=now, content_hash="efgh5678efgh5678",
            source=source, message_id=message_id,
            source_turn_seq=source_turn_seq,
        )
        meta.current_version = new_ver
        meta.versions.append(ver)
        meta.updated_at = now
        if title is not None:
            meta.title = title
        entry["content"][new_ver] = content
        self._index_version += 1
        return meta, content, self._index_version

    def get(self, artifact_id):
        if artifact_id not in self._artifacts:
            raise self._NotFound(artifact_id)
        entry = self._artifacts[artifact_id]
        meta = entry["meta"]
        content = entry["content"][meta.current_version]
        return meta, content, self._index_version

    def get_version(self, artifact_id, version):
        if artifact_id not in self._artifacts:
            raise self._NotFound(artifact_id)
        entry = self._artifacts[artifact_id]
        if version not in entry["content"]:
            raise self._VersionNotFound(artifact_id, version)
        return entry["content"][version]

    def list_all(self):
        summaries = [
            self._Summary(
                id=m["meta"].id, title=m["meta"].title, kind=m["meta"].kind,
                language=m["meta"].language,
                current_version=m["meta"].current_version,
                created_at=m["meta"].created_at, updated_at=m["meta"].updated_at,
            )
            for m in self._artifacts.values()
        ]
        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries, self._index_version

    def delete(self, artifact_id):
        if artifact_id not in self._artifacts:
            raise self._NotFound(artifact_id)
        del self._artifacts[artifact_id]
        self._index_version += 1
        return True, self._index_version

    def get_state(self):
        return {
            "version": self._index_version,
            "updated_at": "2026-02-21T00:00:00+00:00",
            "count": len(self._artifacts),
        }

    def exists(self, artifact_id):
        return artifact_id in self._artifacts


@pytest.fixture
def fake_artifact_store(app, monkeypatch):
    """Inject FakeArtifactStore into the adapter module."""
    import gov_webui.adapter as adapter_mod
    fake = FakeArtifactStore()
    monkeypatch.setattr(adapter_mod, "_artifact_store", fake)
    return fake


@pytest.fixture
def art_client(client, fake_artifact_store):
    """Test client with fake artifact store injected."""
    return client


def test_artifacts_list_returns_metadata_only(art_client, fake_artifact_store):
    fake_artifact_store.create(title="Note", content="hello", kind="text")
    resp = art_client.get("/governor/artifacts")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert len(data["artifacts"]) == 1
    assert data["artifacts"][0]["title"] == "Note"
    assert "content" not in data["artifacts"][0]
    assert "versions" not in data["artifacts"][0]


def test_artifacts_create_returns_detail_and_content(art_client):
    resp = art_client.post("/governor/artifacts", json={
        "title": "My Draft", "content": "Draft body", "kind": "markdown",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["ok"] is True
    assert data["artifact"]["title"] == "My Draft"
    assert data["artifact"]["kind"] == "markdown"
    assert data["content"] == "Draft body"
    assert "versions" in data["artifact"]
    assert data["index_version"] >= 2


def test_artifacts_create_validation_error_shape(art_client):
    resp = art_client.post("/governor/artifacts", json={
        "title": "Bad", "content": "x", "kind": "spreadsheet",
    })
    assert resp.status_code == 422
    data = resp.json()
    assert data["ok"] is False
    assert data["error"]["code"] == "validation_error"


def test_artifacts_get_returns_detail(art_client, fake_artifact_store):
    meta, _, _ = fake_artifact_store.create(title="Get Me", content="body", kind="text")
    resp = art_client.get(f"/governor/artifacts/{meta.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["artifact"]["title"] == "Get Me"
    assert data["content"] == "body"


def test_artifacts_get_not_found(art_client):
    resp = art_client.get("/governor/artifacts/doesnotexist")
    assert resp.status_code == 404
    data = resp.json()
    assert data["ok"] is False
    assert data["error"]["code"] == "artifact_not_found"


def test_artifacts_update_bumps_version(art_client, fake_artifact_store):
    meta, _, _ = fake_artifact_store.create(title="Upd", content="v1", kind="text")
    resp = art_client.put(f"/governor/artifacts/{meta.id}", json={
        "content": "v2 content", "expected_current_version": 1,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["artifact"]["current_version"] == 2
    assert data["content"] == "v2 content"


def test_artifacts_update_stale_returns_409(art_client, fake_artifact_store):
    meta, _, _ = fake_artifact_store.create(title="Stale", content="v1", kind="text")
    fake_artifact_store.update(meta.id, content="v2", expected_current_version=1)

    resp = art_client.put(f"/governor/artifacts/{meta.id}", json={
        "content": "v3", "expected_current_version": 1,
    })
    assert resp.status_code == 409
    data = resp.json()
    assert data["ok"] is False
    assert data["error"]["code"] == "stale_version"
    details = data["error"]["details"]
    assert details["artifact_id"] == meta.id
    assert details["expected_current_version"] == 1
    assert details["current_version"] == 2
    assert "index_version" in details


def test_artifacts_delete_returns_deleted_payload(art_client, fake_artifact_store):
    meta, _, _ = fake_artifact_store.create(title="Del", content="bye", kind="text")
    resp = art_client.delete(f"/governor/artifacts/{meta.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["deleted"]["artifact_id"] == meta.id


def test_artifacts_get_specific_version(art_client, fake_artifact_store):
    meta, _, _ = fake_artifact_store.create(title="Ver", content="v1", kind="text")
    fake_artifact_store.update(meta.id, content="v2", expected_current_version=1)

    resp = art_client.get(f"/governor/artifacts/{meta.id}/version/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["content"] == "v1"
    assert data["version"] == 1


def test_artifacts_get_specific_version_not_found(art_client, fake_artifact_store):
    meta, _, _ = fake_artifact_store.create(title="Ver", content="v1", kind="text")
    resp = art_client.get(f"/governor/artifacts/{meta.id}/version/99")
    assert resp.status_code == 404
    data = resp.json()
    assert data["ok"] is False
    assert data["error"]["code"] == "artifact_version_not_found"


def test_artifacts_state_endpoint(art_client, fake_artifact_store):
    fake_artifact_store.create(title="A", content="a", kind="text")
    resp = art_client.get("/governor/artifacts/state")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["count"] == 1
    assert "index_version" in data
    assert "updated_at" in data


# ============================================================================
# Artifact Phase 2 — source_turn_seq + promote-as-revision
# ============================================================================


def test_artifact_create_with_source_turn_seq(art_client):
    """POST with source_turn_seq → response version includes it."""
    resp = art_client.post("/governor/artifacts", json={
        "title": "Turn Test", "content": "body", "kind": "text",
        "source": "promote", "source_turn_seq": 3,
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["ok"] is True
    v = data["artifact"]["versions"][0]
    assert v["source_turn_seq"] == 3


def test_artifact_create_without_source_turn_seq(art_client):
    """POST without source_turn_seq → field is null in response."""
    resp = art_client.post("/governor/artifacts", json={
        "title": "No Turn", "content": "body", "kind": "text",
    })
    assert resp.status_code == 201
    data = resp.json()
    v = data["artifact"]["versions"][0]
    assert v["source_turn_seq"] is None


def test_artifact_update_with_source_turn_seq(art_client, fake_artifact_store):
    """PUT with source_turn_seq → new version has it."""
    meta, _, _ = fake_artifact_store.create(title="Upd", content="v1", kind="text")
    resp = art_client.put(f"/governor/artifacts/{meta.id}", json={
        "content": "v2 content", "expected_current_version": 1,
        "source": "promote", "source_turn_seq": 7,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    # Latest version (last in list) should have source_turn_seq
    latest_v = data["artifact"]["versions"][-1]
    assert latest_v["source_turn_seq"] == 7


def test_artifact_update_preserves_concurrency_with_turn_seq(art_client, fake_artifact_store):
    """PUT with expected_current_version mismatch → 409, even when source_turn_seq present."""
    meta, _, _ = fake_artifact_store.create(title="Conc", content="v1", kind="text")
    fake_artifact_store.update(meta.id, content="v2", expected_current_version=1)

    resp = art_client.put(f"/governor/artifacts/{meta.id}", json={
        "content": "v3", "expected_current_version": 1,
        "source": "promote", "source_turn_seq": 5,
    })
    assert resp.status_code == 409
    data = resp.json()
    assert data["ok"] is False
    assert data["error"]["code"] == "stale_version"


def test_artifact_promote_as_revision_flow(art_client, fake_artifact_store):
    """Full promote-as-revision: create, then PUT with source=promote + message_id + source_turn_seq."""
    meta, _, _ = fake_artifact_store.create(
        title="Original", content="v1", kind="markdown",
    )
    resp = art_client.put(f"/governor/artifacts/{meta.id}", json={
        "content": "revised content",
        "expected_current_version": 1,
        "source": "promote",
        "message_id": "msg-xyz-789",
        "source_turn_seq": 12,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["artifact"]["current_version"] == 2
    v2 = data["artifact"]["versions"][-1]
    assert v2["source"] == "promote"
    assert v2["message_id"] == "msg-xyz-789"
    assert v2["source_turn_seq"] == 12
    assert data["content"] == "revised content"


def test_artifact_version_history_includes_turn_seq(art_client, fake_artifact_store):
    """GET /{id}/version/{v} endpoint — version metadata accessible via parent GET."""
    meta, _, _ = fake_artifact_store.create(
        title="Hist", content="v1", kind="text", source_turn_seq=1,
    )
    fake_artifact_store.update(
        meta.id, content="v2", expected_current_version=1,
        source="promote", source_turn_seq=5,
    )
    # Fetch full artifact to see version list
    resp = art_client.get(f"/governor/artifacts/{meta.id}")
    assert resp.status_code == 200
    data = resp.json()
    versions = data["artifact"]["versions"]
    assert versions[0]["source_turn_seq"] == 1
    assert versions[1]["source_turn_seq"] == 5


def test_artifact_list_unchanged(art_client, fake_artifact_store):
    """GET /governor/artifacts response shape unchanged — summaries don't include version details."""
    fake_artifact_store.create(title="A", content="a", kind="text")
    fake_artifact_store.create(title="B", content="b", kind="markdown")
    resp = art_client.get("/governor/artifacts")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert len(data["artifacts"]) == 2
    for art in data["artifacts"]:
        assert "title" in art
        assert "kind" in art
        assert "versions" not in art
        assert "content" not in art


def test_artifact_state_returns_count(art_client, fake_artifact_store):
    """GET /governor/artifacts/state returns count for auto-hydrate badge."""
    assert art_client.get("/governor/artifacts/state").json()["count"] == 0
    fake_artifact_store.create(title="X", content="x", kind="text")
    data = art_client.get("/governor/artifacts/state").json()
    assert data["ok"] is True
    assert data["count"] == 1
    fake_artifact_store.create(title="Y", content="y", kind="text")
    assert art_client.get("/governor/artifacts/state").json()["count"] == 2


# ============================================================================
# Style Policy Integration Tests
# ============================================================================


def test_artifact_create_fiction_applies_fix(art_client, fake_artifact_store):
    """In fiction mode (fix), content is normalized and style_status shows applied."""
    import gov_webui.adapter as adapter_mod

    old_mode = adapter_mod.GOVERNOR_MODE
    adapter_mod.GOVERNOR_MODE = "fiction"
    try:
        resp = art_client.post("/governor/artifacts", json={
            "title": "Draft", "content": "Hello -- world...", "kind": "text",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["ok"] is True
        # Content should be normalized
        assert data["content"] == "Hello \u2014 world\u2026"
        # style_status present
        ss = data["style_status"]
        assert ss["profile"] == "fiction_typography_v1"
        assert ss["action"] == "fix"
        assert ss["corrections_applied"] is True
        assert ss["correction_count"] == 2
        assert len(data["style_corrections"]) == 2
    finally:
        adapter_mod.GOVERNOR_MODE = old_mode


def test_artifact_create_research_warns_only(art_client, fake_artifact_store):
    """In research mode (warn), content is unchanged but corrections reported."""
    import gov_webui.adapter as adapter_mod

    old_mode = adapter_mod.GOVERNOR_MODE
    adapter_mod.GOVERNOR_MODE = "research"
    try:
        resp = art_client.post("/governor/artifacts", json={
            "title": "Paper", "content": "Hello -- world", "kind": "text",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["ok"] is True
        # Content unchanged in warn mode
        assert data["content"] == "Hello -- world"
        ss = data["style_status"]
        assert ss["action"] == "warn"
        assert ss["corrections_applied"] is False
        assert ss["correction_count"] == 1
    finally:
        adapter_mod.GOVERNOR_MODE = old_mode


def test_artifact_create_code_no_style(art_client, fake_artifact_store):
    """In code mode, no style policy applies."""
    import gov_webui.adapter as adapter_mod

    old_mode = adapter_mod.GOVERNOR_MODE
    adapter_mod.GOVERNOR_MODE = "code"
    try:
        resp = art_client.post("/governor/artifacts", json={
            "title": "Script", "content": "x -- y", "kind": "code",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["ok"] is True
        assert data["content"] == "x -- y"
        assert "style_status" not in data
        assert "style_corrections" not in data
    finally:
        adapter_mod.GOVERNOR_MODE = old_mode


def test_effective_config_includes_style():
    """Fiction mode effective config includes style_policy section."""
    import importlib
    import gov_webui.adapter as adapter_mod

    old_mode = adapter_mod.GOVERNOR_MODE
    adapter_mod.GOVERNOR_MODE = "fiction"
    try:
        cfg = adapter_mod.resolve_effective_config()
        assert "style_policy" in cfg
        sp = cfg["style_policy"]
        assert sp["profile"] == "fiction_typography_v1"
        assert sp["action"] == "fix"
        assert "fiction_typography_v1" in sp["available_profiles"]
        assert "research_typography_v1" in sp["available_profiles"]
    finally:
        adapter_mod.GOVERNOR_MODE = old_mode


def test_effective_config_code_no_style():
    """Code mode effective config has no style_policy section."""
    import gov_webui.adapter as adapter_mod

    old_mode = adapter_mod.GOVERNOR_MODE
    adapter_mod.GOVERNOR_MODE = "code"
    try:
        cfg = adapter_mod.resolve_effective_config()
        assert "style_policy" not in cfg
    finally:
        adapter_mod.GOVERNOR_MODE = old_mode


# ============================================================================
# Receipt V1 Export / Verify Tests
# ============================================================================


class FakeReceiptV1Bridge:
    """Stub that captures emit calls without writing files."""
    def __init__(self):
        self.enabled = True
        self.calls = []

    def emit(self, ctx):
        self.calls.append(ctx)
        return "fake-receipt-id"


@pytest.fixture
def fake_receipt_bridge(app, monkeypatch):
    import gov_webui.adapter as adapter_mod
    fake = FakeReceiptV1Bridge()
    monkeypatch.setattr(adapter_mod, "_receipt_v1_bridge", fake)
    return fake


def _build_test_receipt_dicts(count: int) -> list[dict]:
    """Build N chained receipt_v1 dicts for testing."""
    from receipt_v1 import ReceiptBuilder, ReceiptChain
    from receipt_v1.types import Action, Actor, ExecutionStatus, Provenance

    chain = ReceiptChain()
    dicts = []
    for i in range(count):
        r = (
            ReceiptBuilder()
            .actor(Actor(agent_id="webui", session_id="test-sess"))
            .tool("gov.chat_completion", {"turn_seq": i + 1})
            .decision(Action.ALLOW, "gov.passthrough", reason_human="test turn")
            .execution(ExecutionStatus.SUCCESS)
            .provenance(Provenance(
                deployment_id="test",
                instance_id="test-inst",
                governor_version="0.1.0",
            ))
            .build(chain.next())
        )
        chain.append(r)
        dicts.append(r.to_dict())
    return dicts


def test_receipt_v1_bridge_emits_on_turn(client, fake_receipt_bridge):
    """Bridge emits on non-streaming chat turn."""
    import gov_webui.adapter as adapter_mod

    mock = AsyncMock()
    mock.chat_send = AsyncMock(return_value={
        "content": "Hello",
        "model": "test-model",
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "violations": [],
        "footer": None,
        "pending": None,
    })
    mock.connect = AsyncMock()
    mock.close = AsyncMock()
    adapter_mod._daemon_client = mock

    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "Hi"}], "stream": False},
    )
    assert response.status_code == 200
    assert len(fake_receipt_bridge.calls) == 1
    ctx = fake_receipt_bridge.calls[0]
    assert ctx.agent_id == "webui"
    assert ctx.gate == "chat_completion"


def test_receipt_export_returns_jsonl(client, monkeypatch):
    """Export endpoint returns JSONL with correct content-type."""
    import gov_webui.adapter as adapter_mod

    dicts = _build_test_receipt_dicts(2)
    monkeypatch.setattr(adapter_mod, "_load_receipt_v1_dicts", lambda: dicts)

    resp = client.get("/governor/receipts/export")
    assert resp.status_code == 200
    assert "application/x-ndjson" in resp.headers["content-type"]
    lines = [l for l in resp.text.strip().split("\n") if l.strip()]
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)
        assert "receipt_id" in parsed


def test_receipt_verify_valid_chain(client, monkeypatch):
    """Verify reports valid for a properly chained set of receipts."""
    import gov_webui.adapter as adapter_mod

    dicts = _build_test_receipt_dicts(3)
    monkeypatch.setattr(adapter_mod, "_load_receipt_v1_dicts", lambda: dicts)

    resp = client.post("/governor/receipts/verify")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    report = data["report"]
    assert report["valid"] is True
    assert report["receipt_count"] == 3
    assert report["error_count"] == 0
    assert report["scheme"] == "receipt_v1"
    assert report["receipt_version"] == "1.0"


def test_receipt_verify_upload_detects_tampered_hash(client):
    """Verify-upload detects a tampered receipt_hash."""
    dicts = _build_test_receipt_dicts(1)
    dicts[0]["receipt_hash"] = "a" * 64  # tamper

    body = json.dumps(dicts[0]) + "\n"
    resp = client.post("/governor/receipts/verify-upload", content=body)
    assert resp.status_code == 200
    data = resp.json()
    report = data["report"]
    assert report["valid"] is False
    hash_findings = [f for f in report["findings"] if f["code"] == "hash_mismatch"]
    assert len(hash_findings) >= 1


def test_receipt_verify_upload_detects_chain_break(client):
    """Verify-upload detects a chain break (wrong parent hash)."""
    dicts = _build_test_receipt_dicts(2)
    dicts[1]["chain"]["parent_receipt_hash"] = "b" * 64  # break chain

    body = "\n".join(json.dumps(d) for d in dicts) + "\n"
    resp = client.post("/governor/receipts/verify-upload", content=body)
    assert resp.status_code == 200
    data = resp.json()
    report = data["report"]
    assert report["valid"] is False
    chain_findings = [f for f in report["findings"] if f["code"] == "chain_break"]
    assert len(chain_findings) >= 1


def test_receipt_export_empty_returns_404(client, monkeypatch):
    """Export returns 404 when no receipts exist."""
    import gov_webui.adapter as adapter_mod

    monkeypatch.setattr(adapter_mod, "_load_receipt_v1_dicts", lambda: [])

    resp = client.get("/governor/receipts/export")
    assert resp.status_code == 404
    data = resp.json()
    assert data["ok"] is False
    assert data["error"]["code"] == "no_receipts"


def test_receipt_verify_upload_invalid_jsonl_returns_400(client):
    """Verify-upload returns 400 with line number for bad JSON."""
    body = '{"valid": true}\nnot valid json\n'
    resp = client.post("/governor/receipts/verify-upload", content=body)
    assert resp.status_code == 400
    data = resp.json()
    assert data["ok"] is False
    assert data["error"]["code"] == "invalid_jsonl"
    assert "line 2" in data["error"]["message"]


# ============================================================================
# Effective Config (C2) Tests
# ============================================================================


def _setup_code_mode(adapter_mod, tmp_contexts_dir, context_id="ec-test"):
    """Helper: configure adapter for code mode with a fresh context."""
    adapter_mod._project_store = None
    adapter_mod._research_project_store = None
    adapter_mod.GOVERNOR_MODE = "code"
    adapter_mod.GOVERNOR_CONTEXTS_DIR = str(tmp_contexts_dir)
    adapter_mod.GOVERNOR_CONTEXT_ID = context_id

    cm = GovernorContextManager(base_dir=tmp_contexts_dir)
    cm.create(context_id, mode="code")
    adapter_mod._context_manager = cm

    from fastapi.testclient import TestClient
    return TestClient(adapter_mod.app, raise_server_exceptions=False)


def test_effective_config_defaults_only(tmp_contexts_dir):
    """Code mode, no contract config → all sources are 'default'."""
    import gov_webui.adapter as adapter_mod
    test_client = _setup_code_mode(adapter_mod, tmp_contexts_dir, "ec-defaults")

    # Set contract without config
    test_client.put("/governor/code/project/contract", json={"description": "test"})

    resp = test_client.get("/governor/config/effective")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["has_config"] is True
    assert data["has_session_overrides"] is False
    assert data["scope"] == "session_current"

    # All fields should be source=default
    for f in data["fields"]:
        assert f["source"] == "default", f"field {f['key']} should be default"
        assert f["clamped"] is False

    # Field order matches _CONFIG_DEFAULTS key order
    expected_keys = list(adapter_mod._CONFIG_DEFAULTS["code"].keys())
    actual_keys = [f["key"] for f in data["fields"]]
    assert actual_keys == expected_keys

    adapter_mod._project_store = None
    adapter_mod._context_manager = None


def test_effective_config_session_override(tmp_contexts_dir):
    """Session override: length='long' → source='session', rest stays 'default'."""
    import gov_webui.adapter as adapter_mod
    test_client = _setup_code_mode(adapter_mod, tmp_contexts_dir, "ec-session")

    test_client.put("/governor/code/project/contract", json={
        "description": "test",
        "config": {"length": "long"},
    })

    resp = test_client.get("/governor/config/effective")
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_session_overrides"] is True

    sources = data["sources"]
    assert sources["length"] == "session"
    assert data["effective"]["length"] == "long"

    # Other fields still default
    for key, src in sources.items():
        if key != "length":
            assert src == "default", f"field {key} should be default"

    adapter_mod._project_store = None
    adapter_mod._context_manager = None


def test_effective_config_system_clamp(tmp_contexts_dir, monkeypatch):
    """System constraint clamps a field: strict forced True despite session False."""
    import gov_webui.adapter as adapter_mod
    test_client = _setup_code_mode(adapter_mod, tmp_contexts_dir, "ec-clamp")

    # Monkeypatch system constraints to force strict=True
    monkeypatch.setitem(adapter_mod._SYSTEM_CONSTRAINTS["code"], "strict", True)

    test_client.put("/governor/code/project/contract", json={
        "description": "test",
        "config": {"strict": False},
    })

    resp = test_client.get("/governor/config/effective")
    assert resp.status_code == 200
    data = resp.json()
    assert data["effective"]["strict"] is True
    assert data["sources"]["strict"] == "system"

    # Find the strict field
    strict_field = [f for f in data["fields"] if f["key"] == "strict"][0]
    assert strict_field["clamped"] is True
    assert strict_field["value"] is True

    # Diagnostics should show the clamp
    clamped = data["diagnostics"]["clamped_fields"]
    assert len(clamped) == 1
    assert clamped[0]["key"] == "strict"
    assert clamped[0]["requested"] is False
    assert clamped[0]["effective"] is True

    adapter_mod._project_store = None
    adapter_mod._context_manager = None


def test_effective_config_unknown_key(tmp_contexts_dir):
    """Unknown keys in session config appear in diagnostics, not in fields/effective."""
    import gov_webui.adapter as adapter_mod
    test_client = _setup_code_mode(adapter_mod, tmp_contexts_dir, "ec-unknown")

    test_client.put("/governor/code/project/contract", json={
        "description": "test",
        "config": {"made_up_key": "x", "length": "short"},
    })

    resp = test_client.get("/governor/config/effective")
    assert resp.status_code == 200
    data = resp.json()

    # Unknown key NOT in fields or effective
    field_keys = [f["key"] for f in data["fields"]]
    assert "made_up_key" not in field_keys
    assert "made_up_key" not in data["effective"]

    # But IS in diagnostics
    assert "made_up_key" in data["diagnostics"]["unknown_keys"]

    adapter_mod._project_store = None
    adapter_mod._context_manager = None


def test_effective_config_hash_matches_receipt(tmp_contexts_dir):
    """contract_config_hash and constraints_hash match receipt values."""
    import hashlib
    import gov_webui.adapter as adapter_mod
    test_client = _setup_code_mode(adapter_mod, tmp_contexts_dir, "ec-hash")

    test_client.put("/governor/code/project/contract", json={
        "description": "test",
        "config": {"artifact_type": "tool", "length": "medium", "strict": False},
    })

    resp = test_client.get("/governor/config/effective")
    assert resp.status_code == 200
    data = resp.json()

    # Build receipt via the same path
    msg, meta = adapter_mod._build_constraints_message()
    assert msg is not None

    # contract_config_hash should match receipt's config_hash
    assert data["contract_config_hash"] == meta["config_hash"]
    assert data["contract_config_hash_full"] == meta["config_hash_full"]

    # constraints_hash should match SHA-256 of rendered block
    expected_full = hashlib.sha256(msg["content"].encode("utf-8")).hexdigest()
    assert data["constraints_hash"] == expected_full[:16]
    assert data["constraints_hash_full"] == expected_full

    adapter_mod._project_store = None
    adapter_mod._context_manager = None


def test_effective_config_general_mode(client):
    """General mode → has_config=False, fields=[], hashes=None."""
    import gov_webui.adapter as adapter_mod
    adapter_mod.GOVERNOR_MODE = "general"

    resp = client.get("/governor/config/effective")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["has_config"] is False
    assert data["fields"] == []
    assert data["contract_config_hash"] is None
    assert data["constraints_hash"] is None
    assert data["scope"] == "session_current"


def test_effective_config_endpoint_status_codes(tmp_contexts_dir):
    """200 on success; store failure → 500."""
    import gov_webui.adapter as adapter_mod
    test_client = _setup_code_mode(adapter_mod, tmp_contexts_dir, "ec-status")

    # Success case
    test_client.put("/governor/code/project/contract", json={"description": "test"})
    resp = test_client.get("/governor/config/effective")
    assert resp.status_code == 200

    # Simulate store failure by breaking the project store
    original = adapter_mod._get_project_store
    adapter_mod._get_project_store = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    adapter_mod._project_store = None

    resp = test_client.get("/governor/config/effective")
    assert resp.status_code == 500
    data = resp.json()
    assert data["ok"] is False
    assert data["error"]["code"] == "store_read_failed"

    # Restore
    adapter_mod._get_project_store = original
    adapter_mod._project_store = None
    adapter_mod._context_manager = None


def test_effective_config_field_order_stable(tmp_contexts_dir):
    """Field order matches _CONFIG_DEFAULTS key order across multiple calls."""
    import gov_webui.adapter as adapter_mod
    test_client = _setup_code_mode(adapter_mod, tmp_contexts_dir, "ec-order")

    test_client.put("/governor/code/project/contract", json={
        "description": "test",
        "config": {"strict": True, "length": "long"},
    })

    expected_keys = list(adapter_mod._CONFIG_DEFAULTS["code"].keys())

    for _ in range(3):
        resp = test_client.get("/governor/config/effective")
        assert resp.status_code == 200
        actual_keys = [f["key"] for f in resp.json()["fields"]]
        assert actual_keys == expected_keys

    adapter_mod._project_store = None
    adapter_mod._context_manager = None
