# SPDX-License-Identifier: MIT
"""CI smoke test: API-walk the complete code builder loop.

Exercises the full mechanically binding loop:
  intent → contract → plan → accept file → run → verify output → state persisted

This is the "does it still boot and close the loop" gate for every PR.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("BACKEND_TYPE", "ollama")
os.environ.setdefault("GOVERNOR_MODE", "code")

from governor.context_manager import GovernorContextManager


@pytest.fixture()
def code_client(tmp_path: Path):
    """TestClient with a fresh code-mode governor context."""
    import gov_webui.adapter as adapter_mod

    # Reset all singletons
    adapter_mod._bridge = None
    adapter_mod._context_manager = None
    adapter_mod._session_store = None
    adapter_mod._daemon_client = None
    adapter_mod._project_store = None
    adapter_mod._research_store = None
    adapter_mod._research_project_store = None
    adapter_mod.GOVERNOR_CONTEXTS_DIR = str(tmp_path / "contexts")
    adapter_mod.GOVERNOR_CONTEXT_ID = "smoke-test"
    adapter_mod.GOVERNOR_MODE = "code"
    adapter_mod.GOVERNOR_AUTH_TOKEN = ""

    # Create the context
    cm = GovernorContextManager(base_dir=tmp_path / "contexts")
    cm.create("smoke-test", mode="code")
    adapter_mod._context_manager = cm

    from fastapi.testclient import TestClient
    client = TestClient(adapter_mod.app, raise_server_exceptions=False)
    yield client

    # Teardown
    adapter_mod._bridge = None
    adapter_mod._context_manager = None
    adapter_mod._session_store = None
    adapter_mod._daemon_client = None
    adapter_mod._project_store = None
    adapter_mod._research_store = None
    adapter_mod._research_project_store = None


class TestCodeBuilderSmokeLoop:
    """Full loop: intent → contract → plan → file → run → verify."""

    def test_complete_loop(self, code_client) -> None:
        c = code_client

        # 1. Get initial state
        r = c.get("/governor/code/project")
        assert r.status_code == 200
        state = r.json()
        v = state["version"]
        assert state["intent"]["text"] == ""
        assert state["files"] == {}

        # 2. Set intent (with expected_version)
        r = c.put("/governor/code/project/intent", json={
            "text": "Parse CSV files and output JSON",
            "locked": False,
            "expected_version": v,
        })
        assert r.status_code == 200
        v += 1  # version bumped

        # 3. Lock intent
        r = c.put("/governor/code/project/intent", json={
            "text": "Parse CSV files and output JSON",
            "locked": True,
            "expected_version": v,
        })
        assert r.status_code == 200
        assert r.json()["intent"]["locked"] is True
        v += 1

        # 4. Set contract
        r = c.put("/governor/code/project/contract", json={
            "description": "CSV to JSON converter",
            "inputs": [{"name": "filepath", "type": "str"}],
            "outputs": [{"name": "json_data", "type": "list[dict]"}],
            "constraints": ["No pandas", "Handle UTF-8"],
            "transport": "stdio",
            "expected_version": v,
        })
        assert r.status_code == 200
        contract = r.json()["contract"]
        assert contract["description"] == "CSV to JSON converter"
        assert len(contract["inputs"]) == 1
        assert contract["constraints"] == ["No pandas", "Handle UTF-8"]
        v += 1

        # 5. Create plan phases and items
        r = c.post("/governor/code/plan/phase", json={"name": "Implementation"})
        assert r.status_code == 200
        v += 1

        r = c.post("/governor/code/plan/phase", json={"name": "Testing"})
        assert r.status_code == 200
        v += 1

        r = c.post("/governor/code/plan/item", json={
            "phase_idx": 0, "text": "Write parser function"
        })
        assert r.status_code == 200
        assert r.json()["item"]["id"] == "p0-0"
        v += 1

        r = c.post("/governor/code/plan/item", json={
            "phase_idx": 1, "text": "Write unit tests"
        })
        assert r.status_code == 200
        v += 1

        # 6. Walk plan item through state machine
        # proposed -> accepted
        r = c.patch("/governor/code/plan/item/p0-0", json={
            "status": "accepted", "expected_version": v,
        })
        assert r.status_code == 200
        v += 1

        # accepted -> in_progress
        r = c.patch("/governor/code/plan/item/p0-0", json={
            "status": "in_progress", "expected_version": v,
        })
        assert r.status_code == 200
        v += 1

        # 7. Accept a file (simulating user clicking Accept on a code block)
        r = c.put("/governor/code/files/tool.py", json={
            "content": 'import sys\nimport json\nimport csv\n\ndef parse_csv(filepath):\n    with open(filepath) as f:\n        reader = csv.DictReader(f)\n        return list(reader)\n\nif __name__ == "__main__":\n    # Read filepath from stdin\n    filepath = sys.stdin.read().strip()\n    if filepath:\n        data = parse_csv(filepath)\n        print(json.dumps(data, indent=2))\n    else:\n        print("hello world")\n',
            "turn_id": "turn-smoke001",
        })
        assert r.status_code == 200
        file_result = r.json()
        assert file_result["version"] == 1
        assert file_result["content_hash"]  # non-empty
        v += 1

        # 8. Verify file round-trips
        r = c.get("/governor/code/files/tool.py")
        assert r.status_code == 200
        assert "hello world" in r.json()["content"]

        # 9. List files
        r = c.get("/governor/code/files")
        assert "tool.py" in r.json()["files"]

        # 10. Run the code
        r = c.post("/governor/code/run", json={
            "filepath": "tool.py",
            "stdin": "",
            "timeout": 10,
        })
        assert r.status_code == 200
        run_result = r.json()
        assert run_result["success"] is True
        assert run_result["returncode"] == 0
        assert "hello world" in run_result["stdout"]
        assert run_result["preflight_hit"] is False
        assert run_result["forced"] is False

        # 11. Complete the plan item
        r = c.patch("/governor/code/plan/item/p0-0", json={
            "status": "completed", "expected_version": v,
        })
        assert r.status_code == 200
        assert r.json()["item"]["status"] == "completed"
        v += 1

        # 12. Verify phase gating: Phase 2 item should now be accessible
        r = c.patch("/governor/code/plan/item/p1-0", json={
            "status": "accepted", "expected_version": v,
        })
        assert r.status_code == 200
        v += 1

        # 13. Accept a second version of the file
        r = c.put("/governor/code/files/tool.py", json={
            "content": 'print("v2 - improved")\n',
            "turn_id": "turn-smoke002",
        })
        assert r.status_code == 200
        assert r.json()["version"] == 2
        v += 1

        # 14. Verify .prev exists
        r = c.get("/governor/code/file-prev/tool.py")
        assert r.status_code == 200
        assert "hello world" in r.json()["content"]

        # 15. Final state check — everything persisted
        r = c.get("/governor/code/project")
        assert r.status_code == 200
        final = r.json()
        assert final["intent"]["text"] == "Parse CSV files and output JSON"
        assert final["intent"]["locked"] is True
        assert final["contract"]["description"] == "CSV to JSON converter"
        assert final["files"]["tool.py"]["version"] == 2
        assert final["files"]["tool.py"]["accepted_turn_id"] == "turn-smoke002"
        assert final["plan"]["phases"][0]["items"][0]["status"] == "completed"
        assert final["plan"]["phases"][1]["items"][0]["status"] == "accepted"
        assert final["version"] == v

    def test_stale_version_detected(self, code_client) -> None:
        """Verify that concurrent mutation from another tab would be caught."""
        c = code_client

        state = c.get("/governor/code/project").json()
        v = state["version"]

        # First mutation succeeds
        r = c.put("/governor/code/project/intent", json={
            "text": "first", "locked": False, "expected_version": v,
        })
        assert r.status_code == 200

        # Second mutation with the SAME (now stale) version fails
        r = c.put("/governor/code/project/intent", json={
            "text": "second", "locked": False, "expected_version": v,
        })
        assert r.status_code == 409

    def test_multifile_run(self, code_client) -> None:
        """File A imports file B — both available in tempdir."""
        c = code_client

        c.put("/governor/code/files/helper.py", json={
            "content": "def greet(name):\n    return f'Hello, {name}!'\n"
        })
        c.put("/governor/code/files/tool.py", json={
            "content": "from helper import greet\nprint(greet('world'))\n"
        })

        r = c.post("/governor/code/run", json={"filepath": "tool.py"})
        data = r.json()
        assert data["success"] is True
        assert "Hello, world!" in data["stdout"]

    def test_run_failure_captured(self, code_client) -> None:
        """Run that raises an exception returns returncode != 0."""
        c = code_client

        c.put("/governor/code/files/tool.py", json={
            "content": "raise RuntimeError('intentional')"
        })

        r = c.post("/governor/code/run", json={"filepath": "tool.py"})
        data = r.json()
        assert data["success"] is False
        assert data["returncode"] != 0
        assert "intentional" in data["stderr"]

    def test_invalid_transition_rejected(self, code_client) -> None:
        """State machine rejects invalid transitions."""
        c = code_client

        c.post("/governor/code/plan/phase", json={"name": "Phase 1"})
        c.post("/governor/code/plan/item", json={"phase_idx": 0, "text": "Task"})

        # proposed -> completed (skipping accepted)
        r = c.patch("/governor/code/plan/item/p0-0", json={"status": "completed"})
        assert r.status_code == 400
        assert "Invalid transition" in r.json()["detail"]
