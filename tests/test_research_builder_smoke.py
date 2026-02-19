# SPDX-License-Identifier: MIT
"""CI smoke test: API-walk the complete research builder loop.

Exercises the structured research workflow:
  thesis → scope → plan → accept draft → validate → state persisted

Mirrors test_code_builder_smoke.py but with research-mode labels,
RESEARCH_EXTENSIONS, and text validation instead of code execution.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("BACKEND_TYPE", "ollama")
os.environ.setdefault("GOVERNOR_MODE", "research")

from governor.context_manager import GovernorContextManager


@pytest.fixture()
def research_client(tmp_path: Path):
    """TestClient with a fresh research-mode governor context."""
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
    adapter_mod.GOVERNOR_CONTEXT_ID = "research-smoke"
    adapter_mod.GOVERNOR_MODE = "research"
    adapter_mod.GOVERNOR_AUTH_TOKEN = ""

    # Create the context
    cm = GovernorContextManager(base_dir=tmp_path / "contexts")
    cm.create("research-smoke", mode="research")
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


class TestResearchBuilderSmokeLoop:
    """Full loop: thesis → scope → plan → draft → validate → verify."""

    def test_complete_loop(self, research_client) -> None:
        c = research_client

        # 1. Get initial state
        r = c.get("/governor/research/project")
        assert r.status_code == 200
        state = r.json()
        v = state["version"]
        assert state["intent"]["text"] == ""
        assert state["files"] == {}

        # 2. Set thesis (intent)
        r = c.put("/governor/research/project/intent", json={
            "text": "How does cognitive load affect code review quality?",
            "locked": False,
            "expected_version": v,
        })
        assert r.status_code == 200
        v += 1

        # 3. Lock thesis
        r = c.put("/governor/research/project/intent", json={
            "text": "How does cognitive load affect code review quality?",
            "locked": True,
            "expected_version": v,
        })
        assert r.status_code == 200
        assert r.json()["intent"]["locked"] is True
        v += 1

        # 4. Set scope (contract) with wizard config
        r = c.put("/governor/research/project/contract", json={
            "description": "Literature review of cognitive load in code review",
            "inputs": [{"name": "search_terms", "type": "str"}],
            "outputs": [{"name": "summary", "type": "str"}],
            "constraints": ["No anecdotal evidence", "Avoid speculation"],
            "transport": "stdio",
            "expected_version": v,
            "config": {
                "artifact_type": "lit_review",
                "length": "short",
                "voice": ["academic"],
                "citations": "light",
                "bans": [],
                "strict": False,
            },
        })
        assert r.status_code == 200
        scope = r.json()["contract"]
        assert scope["description"] == "Literature review of cognitive load in code review"
        assert scope["constraints"] == ["No anecdotal evidence", "Avoid speculation"]
        assert scope["config"]["artifact_type"] == "lit_review"
        assert scope["config_hash"] is not None
        assert len(scope["config_hash"]) == 16
        assert scope["config_hash_full"] is not None
        assert scope["config_hash_full"].startswith(scope["config_hash"])
        v += 1

        # 5. Create plan phases and items
        r = c.post("/governor/research/project/plan/phase", json={"name": "Literature Review"})
        assert r.status_code == 200
        v += 1

        r = c.post("/governor/research/project/plan/phase", json={"name": "Synthesis"})
        assert r.status_code == 200
        v += 1

        r = c.post("/governor/research/project/plan/item", json={
            "phase_idx": 0, "text": "Collect primary sources"
        })
        assert r.status_code == 200
        assert r.json()["item"]["id"] == "p0-0"
        v += 1

        r = c.post("/governor/research/project/plan/item", json={
            "phase_idx": 1, "text": "Draft synthesis section"
        })
        assert r.status_code == 200
        v += 1

        # 6. Walk plan item through state machine
        r = c.patch("/governor/research/project/plan/item/p0-0", json={
            "status": "accepted", "expected_version": v,
        })
        assert r.status_code == 200
        v += 1

        r = c.patch("/governor/research/project/plan/item/p0-0", json={
            "status": "in_progress", "expected_version": v,
        })
        assert r.status_code == 200
        v += 1

        # 7. Accept a draft (research uses .md files)
        r = c.put("/governor/research/project/files/notes.md", json={
            "content": "# Literature Review\n\nSmith (2023) found that cognitive load "
                       "increases with code complexity [1].\n\n## References\n[1] Smith, "
                       "J. (2023). Cognitive Load in Code Review.\n",
            "turn_id": "turn-rsmoke001",
        })
        assert r.status_code == 200
        file_result = r.json()
        assert file_result["version"] == 1
        assert file_result["content_hash"]
        v += 1

        # 8. Verify file round-trips
        r = c.get("/governor/research/project/files/notes.md")
        assert r.status_code == 200
        assert "cognitive load" in r.json()["content"]

        # 9. List files
        r = c.get("/governor/research/project/files")
        assert "notes.md" in r.json()["files"]

        # 10. Validate the draft (no findings expected)
        r = c.post("/governor/research/project/validate", json={
            "filepath": "notes.md",
        })
        assert r.status_code == 200
        val = r.json()
        assert val["success"] is True
        assert val["returncode"] == 0
        assert "no issues" in val["stdout"]
        assert val["preflight_hit"] is False

        # 11. Complete the plan item
        r = c.patch("/governor/research/project/plan/item/p0-0", json={
            "status": "completed", "expected_version": v,
        })
        assert r.status_code == 200
        assert r.json()["item"]["status"] == "completed"
        v += 1

        # 12. Phase gating: Phase 2 item now accessible
        r = c.patch("/governor/research/project/plan/item/p1-0", json={
            "status": "accepted", "expected_version": v,
        })
        assert r.status_code == 200
        v += 1

        # 13. Accept a second version of the draft
        r = c.put("/governor/research/project/files/notes.md", json={
            "content": "# Literature Review v2\n\nExpanded with additional sources.\n",
            "turn_id": "turn-rsmoke002",
        })
        assert r.status_code == 200
        assert r.json()["version"] == 2
        v += 1

        # 14. Verify .prev exists
        r = c.get("/governor/research/project/file-prev/notes.md")
        assert r.status_code == 200
        assert "cognitive load" in r.json()["content"]

        # 15. Final state check
        r = c.get("/governor/research/project")
        assert r.status_code == 200
        final = r.json()
        assert final["intent"]["text"] == "How does cognitive load affect code review quality?"
        assert final["intent"]["locked"] is True
        assert final["contract"]["description"] == "Literature review of cognitive load in code review"
        assert final["files"]["notes.md"]["version"] == 2
        assert final["files"]["notes.md"]["accepted_turn_id"] == "turn-rsmoke002"
        assert final["plan"]["phases"][0]["items"][0]["status"] == "completed"
        assert final["plan"]["phases"][1]["items"][0]["status"] == "accepted"
        assert final["version"] == v

    def test_stale_version_detected(self, research_client) -> None:
        """Concurrent mutation from another tab is caught."""
        c = research_client

        state = c.get("/governor/research/project").json()
        v = state["version"]

        r = c.put("/governor/research/project/intent", json={
            "text": "first", "locked": False, "expected_version": v,
        })
        assert r.status_code == 200

        r = c.put("/governor/research/project/intent", json={
            "text": "second", "locked": False, "expected_version": v,
        })
        assert r.status_code == 409

    def test_research_extensions_accepted(self, research_client) -> None:
        """Research store accepts .md, .csv, .bib files (not just .py)."""
        c = research_client

        for fname, content in [
            ("draft.md", "# Draft\nSome research content.\n"),
            ("data.csv", "name,value\nfoo,1\nbar,2\n"),
            ("refs.bib", "@article{smith2023, author={Smith}}\n"),
            ("notes.txt", "Field notes from observation.\n"),
        ]:
            r = c.put(f"/governor/research/project/files/{fname}", json={
                "content": content,
            })
            assert r.status_code == 200, f"Failed to accept {fname}: {r.json()}"
            assert r.json()["version"] == 1

        r = c.get("/governor/research/project/files")
        files = r.json()["files"]
        assert "draft.md" in files
        assert "data.csv" in files
        assert "refs.bib" in files
        assert "notes.txt" in files

    def test_research_rejects_code_extensions(self, research_client) -> None:
        """Research store rejects .py and .cfg files."""
        c = research_client

        r = c.put("/governor/research/project/files/script.py", json={
            "content": "print('hello')",
        })
        assert r.status_code == 400
        assert "not allowed" in r.json()["detail"]

    def test_validate_catches_weasel_words(self, research_client) -> None:
        """Validator detects uncited weasel phrases."""
        c = research_client

        c.put("/governor/research/project/files/draft.md", json={
            "content": "# Analysis\n\nStudies show that X is important.\n"
                       "Research suggests Y may be relevant.\n"
                       "Smith (2023) confirmed Z [1].\n",
        })

        r = c.post("/governor/research/project/validate", json={
            "filepath": "draft.md",
        })
        assert r.status_code == 200
        val = r.json()
        assert val["success"] is False
        assert val["returncode"] == 1
        assert len(val["findings"]) >= 2  # both weasel phrases caught
        # The properly cited line should NOT trigger
        findings_text = "\n".join(val["findings"])
        assert "studies show" in findings_text.lower()
        assert "research suggests" in findings_text.lower()

    def test_validate_catches_constraint_violation(self, research_client) -> None:
        """Validator checks scope constraints."""
        c = research_client

        state = c.get("/governor/research/project").json()
        v = state["version"]

        # Set scope with constraint "Avoid speculation"
        c.put("/governor/research/project/contract", json={
            "description": "Focused review",
            "inputs": [],
            "outputs": [],
            "constraints": ["Avoid speculation"],
            "transport": "stdio",
            "expected_version": v,
        })

        # Accept draft containing the forbidden term
        c.put("/governor/research/project/files/draft.md", json={
            "content": "# Draft\n\nThis section involves some speculation about causes.\n",
        })

        r = c.post("/governor/research/project/validate", json={
            "filepath": "draft.md",
        })
        assert r.status_code == 200
        val = r.json()
        assert val["success"] is False
        findings_text = "\n".join(val["findings"])
        assert "constraint" in findings_text.lower() or "speculation" in findings_text.lower()

    def test_validate_no_drafts_400(self, research_client) -> None:
        """Validate returns 400 when no drafts exist."""
        c = research_client

        r = c.post("/governor/research/project/validate", json={
            "filepath": "nonexistent.md",
        })
        assert r.status_code == 400

    def test_validate_missing_draft_404(self, research_client) -> None:
        """Validate returns 404 for wrong filename when other drafts exist."""
        c = research_client

        c.put("/governor/research/project/files/draft.md", json={
            "content": "Some content.\n",
        })

        r = c.post("/governor/research/project/validate", json={
            "filepath": "other.md",
        })
        assert r.status_code == 404

    def test_invalid_transition_rejected(self, research_client) -> None:
        """State machine rejects invalid transitions."""
        c = research_client

        c.post("/governor/research/project/plan/phase", json={"name": "Phase 1"})
        c.post("/governor/research/project/plan/item", json={
            "phase_idx": 0, "text": "Research task"
        })

        # proposed -> completed (skipping accepted)
        r = c.patch("/governor/research/project/plan/item/p0-0", json={
            "status": "completed",
        })
        assert r.status_code == 400
        assert "Invalid transition" in r.json()["detail"]

    def test_phase_gating_blocks_early_advance(self, research_client) -> None:
        """Items in phase 2 can't advance while phase 1 is incomplete."""
        c = research_client

        c.post("/governor/research/project/plan/phase", json={"name": "Phase 1"})
        c.post("/governor/research/project/plan/phase", json={"name": "Phase 2"})
        c.post("/governor/research/project/plan/item", json={
            "phase_idx": 0, "text": "Task A"
        })
        c.post("/governor/research/project/plan/item", json={
            "phase_idx": 1, "text": "Task B"
        })

        # Try to accept phase 2 item while phase 1 has pending items
        r = c.patch("/governor/research/project/plan/item/p1-0", json={
            "status": "accepted",
        })
        assert r.status_code == 400
        assert "incomplete" in r.json()["detail"].lower()

    def test_bans_feed_validator(self, research_client) -> None:
        """Config bans are caught by the validator as literal matches."""
        c = research_client

        state = c.get("/governor/research/project").json()
        v = state["version"]

        # Set scope with config bans
        c.put("/governor/research/project/contract", json={
            "description": "Test",
            "inputs": [],
            "outputs": [],
            "constraints": [],
            "transport": "stdio",
            "expected_version": v,
            "config": {
                "artifact_type": "essay",
                "length": "medium",
                "bans": ["inspirational closer", "in conclusion"],
                "strict": False,
            },
        })

        # Accept draft containing banned phrase
        c.put("/governor/research/project/files/draft.md", json={
            "content": "# Analysis\n\nThis is a solid analysis.\n\n"
                       "In conclusion, everything worked out.\n",
        })

        r = c.post("/governor/research/project/validate", json={
            "filepath": "draft.md",
        })
        assert r.status_code == 200
        val = r.json()
        assert val["success"] is False
        findings_text = "\n".join(val["findings"])
        assert "in conclusion" in findings_text.lower()

    def test_config_hash_mismatch_rejected(self, research_client) -> None:
        """Server rejects mismatched client config hashes."""
        c = research_client

        state = c.get("/governor/research/project").json()
        v = state["version"]

        r = c.put("/governor/research/project/contract", json={
            "description": "Test",
            "inputs": [],
            "outputs": [],
            "constraints": [],
            "transport": "stdio",
            "expected_version": v,
            "config": {"artifact_type": "essay", "length": "short"},
            "config_hash": "deadbeefdeadbeef",  # wrong hash
        })
        assert r.status_code == 400
        assert "mismatch" in r.json()["detail"].lower()
