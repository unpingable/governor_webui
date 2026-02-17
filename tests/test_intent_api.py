# SPDX-License-Identifier: Apache-2.0
"""Tests for v2 Intent Compiler API routes (/v2/intent/*)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

# Set env vars before importing the app
os.environ.setdefault("BACKEND_TYPE", "ollama")
os.environ.setdefault("GOVERNOR_MODE", "general")


@pytest.fixture(autouse=True)
def _reset_adapter_singletons(tmp_path):
    """Reset lazy-init singletons between tests."""
    import gov_webui.adapter as adapter

    adapter._bridge = None
    adapter._context_manager = None
    adapter._session_store = None
    adapter._dashboard_store = None
    adapter._instrument_system = None
    adapter._cancel_requests = {}
    adapter._research_store = None

    adapter.GOVERNOR_CONTEXTS_DIR = str(tmp_path / "contexts")
    adapter.GOVERNOR_CONTEXT_ID = "test"
    adapter.GOVERNOR_MODE = "general"
    adapter.GOVERNOR_AUTH_TOKEN = ""

    yield

    adapter._bridge = None
    adapter._context_manager = None
    adapter._session_store = None
    adapter._dashboard_store = None
    adapter._instrument_system = None
    adapter._cancel_requests = {}
    adapter._research_store = None


@pytest.fixture
def client():
    from starlette.testclient import TestClient
    from gov_webui.adapter import app
    return TestClient(app, raise_server_exceptions=False)


# =============================================================================
# Templates
# =============================================================================


class TestIntentTemplates:
    def test_list_templates(self, client):
        res = client.get("/v2/intent/templates")
        assert res.status_code == 200
        data = res.json()
        assert "templates" in data
        assert len(data["templates"]) == 3

    def test_template_names(self, client):
        res = client.get("/v2/intent/templates")
        names = [t["name"] for t in res.json()["templates"]]
        assert "session_start" in names
        assert "task_scope" in names
        assert "verification_config" in names

    def test_templates_have_descriptions(self, client):
        res = client.get("/v2/intent/templates")
        for t in res.json()["templates"]:
            assert "description" in t
            assert len(t["description"]) > 0


# =============================================================================
# Schema
# =============================================================================


class TestIntentSchema:
    def test_session_start_schema(self, client):
        res = client.get("/v2/intent/schema/session_start")
        assert res.status_code == 200
        data = res.json()
        assert data["template_name"] == "session_start"
        assert "schema_id" in data
        assert "fields" in data
        assert len(data["fields"]) == 4

    def test_task_scope_schema(self, client):
        res = client.get("/v2/intent/schema/task_scope")
        assert res.status_code == 200
        data = res.json()
        assert data["template_name"] == "task_scope"
        assert len(data["fields"]) == 5

    def test_verification_config_schema(self, client):
        res = client.get("/v2/intent/schema/verification_config")
        assert res.status_code == 200
        data = res.json()
        assert data["template_name"] == "verification_config"
        assert len(data["fields"]) == 4

    def test_unknown_template_404_in_template_only(self, client):
        """TEMPLATE_ONLY mode (general) rejects unknown templates."""
        res = client.get("/v2/intent/schema/nonexistent_template")
        assert res.status_code == 404

    def test_schema_has_branches(self, client):
        res = client.get("/v2/intent/schema/session_start")
        data = res.json()
        assert "branches" in data
        assert len(data["branches"]) >= 2

    def test_schema_has_policy(self, client):
        res = client.get("/v2/intent/schema/session_start")
        data = res.json()
        assert "policy" in data
        assert data["policy"] in ("template_only", "validated_custom", "custom_ok")

    def test_schema_mode_matches_governor_mode(self, client):
        import gov_webui.adapter as adapter
        adapter.GOVERNOR_MODE = "fiction"
        res = client.get("/v2/intent/schema/session_start")
        data = res.json()
        assert data["mode"] == "fiction"
        assert data["policy"] == "custom_ok"


# =============================================================================
# Validate
# =============================================================================


class TestIntentValidate:
    def _get_schema_id(self, client, template: str = "session_start") -> str:
        res = client.get(f"/v2/intent/schema/{template}")
        return res.json()["schema_id"]

    def test_valid_response(self, client):
        schema_id = self._get_schema_id(client)
        res = client.post("/v2/intent/validate", json={
            "schema_id": schema_id,
            "values": {"profile": "strict", "mode": "general"},
        })
        assert res.status_code == 200
        data = res.json()
        assert data["valid"] is True
        assert data["errors"] == []

    def test_invalid_value(self, client):
        schema_id = self._get_schema_id(client)
        res = client.post("/v2/intent/validate", json={
            "schema_id": schema_id,
            "values": {"profile": "nonexistent", "mode": "general"},
        })
        assert res.status_code == 200
        data = res.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    def test_unknown_schema_id(self, client):
        res = client.post("/v2/intent/validate", json={
            "schema_id": "bogus_id",
            "values": {"profile": "strict"},
        })
        assert res.status_code == 200
        data = res.json()
        assert len(data["errors"]) > 0
        assert "not found" in data["errors"][0].lower()

    def test_missing_required_field(self, client):
        schema_id = self._get_schema_id(client)
        res = client.post("/v2/intent/validate", json={
            "schema_id": schema_id,
            "values": {},
        })
        assert res.status_code == 200
        data = res.json()
        assert data["valid"] is False


# =============================================================================
# Compile
# =============================================================================


class TestIntentCompile:
    def _get_schema_id(self, client, template: str = "session_start") -> str:
        res = client.get(f"/v2/intent/schema/{template}")
        return res.json()["schema_id"]

    def test_compile_session_start(self, client):
        schema_id = self._get_schema_id(client)
        res = client.post("/v2/intent/compile", json={
            "schema_id": schema_id,
            "values": {"profile": "strict", "mode": "general"},
            "template_name": "session_start",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["intent_profile"] == "strict"
        assert "receipt_hash" in data
        assert len(data["receipt_hash"]) == 64

    def test_compile_with_scope(self, client):
        schema_id = self._get_schema_id(client)
        res = client.post("/v2/intent/compile", json={
            "schema_id": schema_id,
            "values": {"profile": "strict", "mode": "general", "scope": "src/**,tests/**"},
            "template_name": "session_start",
        })
        data = res.json()
        assert data["intent_scope"] == ["src/**", "tests/**"]

    def test_compile_with_escape_text(self, client):
        schema_id = self._get_schema_id(client)
        res = client.post("/v2/intent/compile", json={
            "schema_id": schema_id,
            "values": {"profile": "strict", "mode": "general"},
            "escape_text": "allow exception for testing",
            "template_name": "session_start",
        })
        data = res.json()
        assert data["escape_classification"] == "waiver_candidate"

    def test_compile_unknown_template_400(self, client):
        res = client.post("/v2/intent/compile", json={
            "schema_id": "x",
            "values": {},
            "template_name": "nonexistent",
        })
        assert res.status_code == 400

    def test_compile_task_scope(self, client):
        schema_id = self._get_schema_id(client, "task_scope")
        res = client.post("/v2/intent/compile", json={
            "schema_id": schema_id,
            "values": {"task": "Refactor auth", "verification": "full"},
            "template_name": "task_scope",
        })
        assert res.status_code == 200
        data = res.json()
        assert "intent_profile" in data

    def test_compile_verification_config(self, client):
        schema_id = self._get_schema_id(client, "verification_config")
        res = client.post("/v2/intent/compile", json={
            "schema_id": schema_id,
            "values": {
                "evidence_level": "full",
                "security_scan": True,
                "continuity_check": True,
                "max_violations": 3,
            },
            "template_name": "verification_config",
        })
        assert res.status_code == 200


# =============================================================================
# Policy
# =============================================================================


class TestIntentPolicy:
    def test_policy_general_mode(self, client):
        res = client.get("/v2/intent/policy")
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "general"
        assert data["policy"] == "template_only"

    def test_policy_fiction_mode(self, client):
        import gov_webui.adapter as adapter
        adapter.GOVERNOR_MODE = "fiction"
        res = client.get("/v2/intent/policy")
        data = res.json()
        assert data["mode"] == "fiction"
        assert data["policy"] == "custom_ok"

    def test_policy_code_mode(self, client):
        import gov_webui.adapter as adapter
        adapter.GOVERNOR_MODE = "code"
        res = client.get("/v2/intent/policy")
        data = res.json()
        assert data["policy"] == "template_only"

    def test_policy_research_mode(self, client):
        import gov_webui.adapter as adapter
        adapter.GOVERNOR_MODE = "research"
        res = client.get("/v2/intent/policy")
        data = res.json()
        assert data["policy"] == "validated_custom"


# =============================================================================
# API Info
# =============================================================================


class TestApiInfoIntent:
    def test_api_info_includes_intent_endpoints(self, client):
        res = client.get("/api/info")
        assert res.status_code == 200
        endpoints = res.json()["endpoints"]
        assert "v2_intent_templates" in endpoints
        assert "v2_intent_schema" in endpoints
        assert "v2_intent_compile" in endpoints
        assert "v2_intent_policy" in endpoints
