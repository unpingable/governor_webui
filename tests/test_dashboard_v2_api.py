"""Tests for v2 Dashboard API routes in webui/adapter.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# Set env vars before importing the app
os.environ.setdefault("BACKEND_TYPE", "ollama")
os.environ.setdefault("GOVERNOR_MODE", "general")


@pytest.fixture(autouse=True)
def _reset_adapter_singletons(tmp_path):
    """Reset lazy-init singletons between tests."""
    import gov_webui.adapter as adapter

    # Point to a temp context dir so tests are isolated
    adapter._bridge = None
    adapter._context_manager = None
    adapter._session_store = None
    adapter._dashboard_store = None
    adapter._instrument_system = None
    adapter._cancel_requests = {}
    adapter._research_store = None

    # Override context dir to tmp
    adapter.GOVERNOR_CONTEXTS_DIR = str(tmp_path / "contexts")
    adapter.GOVERNOR_CONTEXT_ID = "test"
    adapter.GOVERNOR_MODE = "general"
    adapter.GOVERNOR_AUTH_TOKEN = ""  # Clear auth token from prior test reloads

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
    """Synchronous test client for FastAPI."""
    from starlette.testclient import TestClient
    from gov_webui.adapter import app

    return TestClient(app, raise_server_exceptions=False)


# =============================================================================
# Controls & Schema
# =============================================================================


class TestControlsSchema:
    def test_schema_returns_json_schema(self, client):
        res = client.get("/v2/controls/schema")
        assert res.status_code == 200
        data = res.json()
        assert data["type"] == "object"
        assert "$schema" in data
        assert "properties" in data
        props = data["properties"]
        assert "profile" in props
        assert "task" in props
        assert "backend" in props

    def test_schema_has_render_hints(self, client):
        res = client.get("/v2/controls/schema")
        data = res.json()
        profile = data["properties"]["profile"]
        assert profile["x-render"] == "dropdown"
        task = data["properties"]["task"]
        assert task["x-render"] == "textarea"

    def test_schema_has_actions(self, client):
        res = client.get("/v2/controls/schema")
        data = res.json()
        assert "x-actions" in data
        assert "start" in data["x-actions"]
        assert "cancel" in data["x-actions"]


class TestControlsTemplates:
    def test_returns_builtin_templates(self, client):
        res = client.get("/v2/controls/templates")
        assert res.status_code == 200
        data = res.json()
        assert "templates" in data
        assert len(data["templates"]) == 3  # smoke_test, security_scan, interferometry

    def test_template_structure(self, client):
        res = client.get("/v2/controls/templates")
        templates = res.json()["templates"]
        tmpl = templates[0]
        assert "name" in tmpl
        assert "description" in tmpl
        assert "manifest_defaults" in tmpl
        assert "example_task" in tmpl


# =============================================================================
# Runs
# =============================================================================


class TestRunsList:
    def test_empty_initially(self, client):
        res = client.get("/v2/runs")
        assert res.status_code == 200
        data = res.json()
        assert data["runs"] == []


class TestCreateRun:
    def test_creates_run(self, client):
        res = client.post("/v2/runs", json={"task": "Run pytest", "profile": "established"})
        assert res.status_code == 200
        data = res.json()
        assert "run_id" in data
        assert data["task"] == "Run pytest"
        assert data["profile"] == "established"

    def test_run_appears_in_list(self, client):
        client.post("/v2/runs", json={"task": "test-task", "profile": "strict"})
        res = client.get("/v2/runs")
        runs = res.json()["runs"]
        assert len(runs) == 1
        assert runs[0]["task"] == "test-task"
        assert runs[0]["verdict"] == "pending"

    def test_run_detail(self, client):
        create_res = client.post("/v2/runs", json={"task": "detail-test"})
        run_id = create_res.json()["run_id"]

        res = client.get(f"/v2/runs/{run_id}")
        assert res.status_code == 200
        data = res.json()
        assert "manifest" in data or "summary" in data

    def test_run_not_found(self, client):
        res = client.get("/v2/runs/nonexistent")
        assert res.status_code == 404


class TestRunClaims:
    def test_claims_empty_initially(self, client):
        create_res = client.post("/v2/runs", json={"task": "claims-test"})
        run_id = create_res.json()["run_id"]

        res = client.get(f"/v2/runs/{run_id}/claims")
        assert res.status_code == 200
        assert res.json()["claims"] == []

    def test_claims_not_found(self, client):
        res = client.get("/v2/runs/nonexistent/claims")
        assert res.status_code == 404


class TestRunViolations:
    def test_violations_empty(self, client):
        create_res = client.post("/v2/runs", json={"task": "violations-test"})
        run_id = create_res.json()["run_id"]

        res = client.get(f"/v2/runs/{run_id}/violations")
        assert res.status_code == 200
        assert res.json()["violations"] == []


class TestRunReport:
    def test_generates_report(self, client):
        create_res = client.post("/v2/runs", json={"task": "report-test"})
        run_id = create_res.json()["run_id"]

        res = client.get(f"/v2/runs/{run_id}/report")
        assert res.status_code == 200
        data = res.json()
        assert data["run_id"] == run_id
        assert "verdict" in data
        assert "sections" in data
        assert "manifest_hash" in data

    def test_report_not_found(self, client):
        res = client.get("/v2/runs/nonexistent/report")
        assert res.status_code == 404


class TestRunCancel:
    def test_cancel_acknowledges(self, client):
        create_res = client.post("/v2/runs", json={"task": "cancel-test"})
        run_id = create_res.json()["run_id"]

        res = client.post(f"/v2/runs/{run_id}/cancel")
        assert res.status_code == 200
        data = res.json()
        assert data["run_id"] == run_id
        assert "acknowledged_at" in data


class TestRunEvents:
    def test_events_empty(self, client):
        create_res = client.post("/v2/runs", json={"task": "events-test"})
        run_id = create_res.json()["run_id"]

        res = client.get(f"/v2/runs/{run_id}/events")
        assert res.status_code == 200
        assert res.json()["events"] == []

    def test_events_not_found(self, client):
        res = client.get("/v2/runs/nonexistent/events")
        assert res.status_code == 404


class TestRunsCompare:
    def test_compare_placeholder(self, client):
        res = client.post("/v2/runs/compare")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "not_implemented"


# =============================================================================
# Artifacts
# =============================================================================


class TestArtifacts:
    def test_artifact_not_found(self, client):
        res = client.get("/v2/artifacts/deadbeef1234567890abcdef")
        assert res.status_code == 404

    def test_list_artifacts_empty(self, client):
        res = client.get("/v2/artifacts")
        assert res.status_code == 200
        assert res.json()["artifacts"] == []

    def test_list_artifacts_for_run(self, client):
        create_res = client.post("/v2/runs", json={"task": "artifact-test"})
        run_id = create_res.json()["run_id"]

        res = client.get(f"/v2/artifacts?run_id={run_id}")
        assert res.status_code == 200
        assert isinstance(res.json()["artifacts"], list)


# =============================================================================
# Dashboard Summary & Regime
# =============================================================================


class TestDashboardSummary:
    def test_summary_empty(self, client):
        res = client.get("/v2/dashboard/summary")
        assert res.status_code == 200
        data = res.json()
        assert data["total_runs"] == 0
        assert data["pass_rate"] == 0.0

    def test_summary_after_run(self, client):
        client.post("/v2/runs", json={"task": "summary-test"})
        res = client.get("/v2/dashboard/summary")
        data = res.json()
        assert data["total_runs"] == 1


class TestDashboardRegime:
    def test_regime_returns(self, client):
        res = client.get("/v2/dashboard/regime")
        assert res.status_code == 200
        data = res.json()
        # regime may be None if no context initialized — that's fine
        assert "regime" in data


# =============================================================================
# Profiles & Anchors
# =============================================================================


class TestProfiles:
    def test_list_profiles(self, client):
        res = client.get("/v2/profiles")
        assert res.status_code == 200
        data = res.json()
        assert "profiles" in data
        assert isinstance(data["profiles"], list)
        assert len(data["profiles"]) > 0
        assert "strict" in data["profiles"]


class TestAnchors:
    def test_list_anchors(self, client):
        res = client.get("/v2/anchors")
        assert res.status_code == 200
        data = res.json()
        assert "anchors" in data
        assert isinstance(data["anchors"], list)


# =============================================================================
# Backends (v2 delegates to v1)
# =============================================================================


class TestV2Backends:
    def test_list_backends(self, client):
        res = client.get("/v2/backends")
        assert res.status_code == 200
        data = res.json()
        assert "backends" in data


# =============================================================================
# Dashboard UI
# =============================================================================


class TestDashboardUI:
    def test_dashboard_serves_html(self, client):
        res = client.get("/dashboard")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "Governor Dashboard" in res.text


# =============================================================================
# API Info includes v2 endpoints
# =============================================================================


class TestAPIInfo:
    def test_api_info_has_v2(self, client):
        res = client.get("/api/info")
        assert res.status_code == 200
        endpoints = res.json()["endpoints"]
        assert "v2_runs" in endpoints
        assert "v2_controls_schema" in endpoints
        assert "dashboard" in endpoints

    def test_api_info_has_demos(self, client):
        res = client.get("/api/info")
        assert res.status_code == 200
        endpoints = res.json()["endpoints"]
        assert "v2_demos" in endpoints
        assert "v2_demo_playwright" in endpoints


# =============================================================================
# Demo Endpoints
# =============================================================================


class TestDemosList:
    def test_lists_builtin_demos(self, client):
        res = client.get("/v2/demos")
        assert res.status_code == 200
        data = res.json()
        assert "demos" in data
        demos = data["demos"]
        assert len(demos) == 4
        names = [d["name"] for d in demos]
        assert "fiction_violation_flow" in names
        assert "code_governance_flow" in names
        assert "interferometry_flow" in names
        assert "dashboard_run_flow" in names

    def test_demo_structure(self, client):
        res = client.get("/v2/demos")
        demos = res.json()["demos"]
        d = demos[0]
        assert "name" in d
        assert "description" in d
        assert "surface" in d
        assert "tags" in d
        assert "step_count" in d
        assert "screenshot_count" in d
        assert "status" in d

    def test_all_demos_have_status(self, client):
        res = client.get("/v2/demos")
        demos = res.json()["demos"]
        for d in demos:
            assert d["status"] in ("fresh", "stale", "missing", "error")

    def test_dashboard_demo_present(self, client):
        res = client.get("/v2/demos")
        demos = res.json()["demos"]
        dash = next(d for d in demos if d["name"] == "dashboard_run_flow")
        assert dash["surface"] == "webui"
        assert "dashboard" in dash["tags"]
        assert dash["step_count"] == 8
        assert dash["screenshot_count"] == 2


class TestDemoPlaywright:
    def test_generates_spec_for_builtin(self, client):
        res = client.get("/v2/demos/fiction_violation_flow/playwright")
        assert res.status_code == 200
        data = res.json()
        assert data["name"] == "fiction_violation_flow"
        assert "spec" in data
        spec = data["spec"]
        assert "import { test, expect }" in spec
        assert "fiction_violation_flow" in spec
        assert "page.goto" in spec

    def test_generates_spec_for_dashboard_demo(self, client):
        res = client.get("/v2/demos/dashboard_run_flow/playwright")
        assert res.status_code == 200
        spec = res.json()["spec"]
        assert "dashboard_run_flow" in spec
        assert "/dashboard" in spec
        assert "page.fill" in spec

    def test_not_found_for_unknown(self, client):
        res = client.get("/v2/demos/nonexistent_demo/playwright")
        assert res.status_code == 404

    def test_all_builtins_generate_specs(self, client):
        """Every built-in demo should produce a valid Playwright spec."""
        list_res = client.get("/v2/demos")
        demos = list_res.json()["demos"]
        for d in demos:
            res = client.get(f"/v2/demos/{d['name']}/playwright")
            assert res.status_code == 200
            spec = res.json()["spec"]
            assert "test(" in spec
