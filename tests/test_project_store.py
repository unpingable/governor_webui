# SPDX-License-Identifier: MIT
"""Unit tests for CodeProjectStore — data model, persistence, state machine."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from gov_webui.project_store import (
    ALLOWED_EXTENSIONS,
    CodeProjectStore,
    Contract,
    ContractField,
    PlanItemStatus,
    StaleVersionError,
    _validate_path,
)


@pytest.fixture()
def store(tmp_path: Path) -> CodeProjectStore:
    """Fresh store rooted in a temp dir."""
    return CodeProjectStore(tmp_path / ".governor")


# ── Persistence ───────────────────────────────────────────────────────────

class TestPersistence:
    def test_fresh_store_creates_state(self, store: CodeProjectStore) -> None:
        state = store.get_state()
        assert state["version"] >= 1
        assert state["intent"]["text"] == ""
        assert state["plan"]["phases"] == []
        assert state["files"] == {}

    def test_save_and_reload(self, tmp_path: Path) -> None:
        gov_dir = tmp_path / ".governor"
        s1 = CodeProjectStore(gov_dir)
        s1.update_intent("build a CLI tool", locked=False)

        # Reload from disk
        s2 = CodeProjectStore(gov_dir)
        assert s2.get_state()["intent"]["text"] == "build a CLI tool"

    def test_atomic_save_produces_json(self, store: CodeProjectStore) -> None:
        store.update_intent("test", locked=False)
        project_path = store._project_path
        assert project_path.exists()
        data = json.loads(project_path.read_text())
        assert data["intent"]["text"] == "test"

    def test_version_bumps_on_save(self, store: CodeProjectStore) -> None:
        v0 = store.get_state()["version"]
        store.update_intent("a", locked=False)
        v1 = store.get_state()["version"]
        store.update_intent("b", locked=False)
        v2 = store.get_state()["version"]
        assert v1 == v0 + 1
        assert v2 == v1 + 1

    def test_corrupt_json_starts_fresh(self, tmp_path: Path) -> None:
        gov_dir = tmp_path / ".governor"
        code_dir = gov_dir / "code"
        code_dir.mkdir(parents=True)
        (code_dir / "project.json").write_text("{invalid json!!")

        store = CodeProjectStore(gov_dir)
        state = store.get_state()
        assert state["intent"]["text"] == ""


# ── Optimistic concurrency ────────────────────────────────────────────────

class TestOptimisticConcurrency:
    def test_stale_version_raises(self, store: CodeProjectStore) -> None:
        store.update_intent("v1", locked=False)
        current_version = store.get_state()["version"]

        with pytest.raises(StaleVersionError):
            store.update_intent("v2", locked=False, expected_version=current_version - 1)

    def test_correct_version_succeeds(self, store: CodeProjectStore) -> None:
        store.update_intent("v1", locked=False)
        current_version = store.get_state()["version"]
        store.update_intent("v2", locked=False, expected_version=current_version)
        assert store.get_state()["intent"]["text"] == "v2"


# ── Intent ────────────────────────────────────────────────────────────────

class TestIntent:
    def test_update_intent(self, store: CodeProjectStore) -> None:
        intent = store.update_intent("parse CSV files", locked=False)
        assert intent.text == "parse CSV files"
        assert intent.locked is False

    def test_lock_intent(self, store: CodeProjectStore) -> None:
        intent = store.update_intent("parse CSV", locked=True)
        assert intent.locked is True


# ── Contract ──────────────────────────────────────────────────────────────

class TestContract:
    def test_update_contract(self, store: CodeProjectStore) -> None:
        contract = Contract(
            description="CSV parser",
            inputs=[ContractField(name="filepath", type="str")],
            outputs=[ContractField(name="rows", type="list[dict]")],
            constraints=["No pandas"],
            transport="stdio",
        )
        result = store.update_contract(contract)
        assert result.description == "CSV parser"
        assert len(result.inputs) == 1
        assert result.inputs[0].name == "filepath"
        assert result.constraints == ["No pandas"]


# ── Plan state machine ────────────────────────────────────────────────────

class TestPlanStateMachine:
    def test_add_phase_and_item(self, store: CodeProjectStore) -> None:
        store.add_phase("Implementation")
        item = store.add_plan_item(0, "Write parser")
        assert item.id == "p0-0"
        assert item.status == PlanItemStatus.proposed

    def test_valid_transitions(self, store: CodeProjectStore) -> None:
        store.add_phase("Phase 1")
        store.add_plan_item(0, "Task A")

        item = store.update_item_status("p0-0", PlanItemStatus.accepted)
        assert item.status == PlanItemStatus.accepted

        item = store.update_item_status("p0-0", PlanItemStatus.in_progress)
        assert item.status == PlanItemStatus.in_progress

        item = store.update_item_status("p0-0", PlanItemStatus.completed)
        assert item.status == PlanItemStatus.completed

    def test_invalid_transition_raises(self, store: CodeProjectStore) -> None:
        store.add_phase("Phase 1")
        store.add_plan_item(0, "Task A")

        with pytest.raises(ValueError, match="Invalid transition"):
            store.update_item_status("p0-0", PlanItemStatus.completed)

    def test_reject_always_allowed_from_non_terminal(self, store: CodeProjectStore) -> None:
        store.add_phase("Phase 1")
        store.add_plan_item(0, "Task A")
        store.update_item_status("p0-0", PlanItemStatus.accepted)

        item = store.update_item_status("p0-0", PlanItemStatus.rejected)
        assert item.status == PlanItemStatus.rejected

    def test_reject_from_proposed(self, store: CodeProjectStore) -> None:
        store.add_phase("Phase 1")
        store.add_plan_item(0, "Task A")
        item = store.update_item_status("p0-0", PlanItemStatus.rejected)
        assert item.status == PlanItemStatus.rejected

    def test_cannot_reject_completed(self, store: CodeProjectStore) -> None:
        store.add_phase("Phase 1")
        store.add_plan_item(0, "Task A")
        store.update_item_status("p0-0", PlanItemStatus.accepted)
        store.update_item_status("p0-0", PlanItemStatus.in_progress)
        store.update_item_status("p0-0", PlanItemStatus.completed)

        with pytest.raises(ValueError, match="terminal state"):
            store.update_item_status("p0-0", PlanItemStatus.rejected)

    def test_cannot_transition_from_rejected(self, store: CodeProjectStore) -> None:
        store.add_phase("Phase 1")
        store.add_plan_item(0, "Task A")
        store.update_item_status("p0-0", PlanItemStatus.rejected)

        with pytest.raises(ValueError, match="terminal state"):
            store.update_item_status("p0-0", PlanItemStatus.rejected)

    def test_item_not_found(self, store: CodeProjectStore) -> None:
        store.add_phase("Phase 1")
        with pytest.raises(ValueError, match="not found"):
            store.update_item_status("p99-0", PlanItemStatus.accepted)


# ── Phase gating ──────────────────────────────────────────────────────────

class TestPhaseGating:
    def test_locked_phase_blocks_non_reject(self, store: CodeProjectStore) -> None:
        store.add_phase("Locked Phase")
        store.update_phase(0, locked=True)
        store.add_plan_item(0, "Task A")

        with pytest.raises(ValueError, match="locked"):
            store.update_item_status("p0-0", PlanItemStatus.accepted)

    def test_locked_phase_allows_reject(self, store: CodeProjectStore) -> None:
        store.add_phase("Locked Phase")
        store.update_phase(0, locked=True)
        store.add_plan_item(0, "Task A")

        item = store.update_item_status("p0-0", PlanItemStatus.rejected)
        assert item.status == PlanItemStatus.rejected

    def test_previous_phase_incomplete_blocks(self, store: CodeProjectStore) -> None:
        store.add_phase("Phase 1")
        store.add_phase("Phase 2")
        store.add_plan_item(0, "Phase 1 task")
        store.add_plan_item(1, "Phase 2 task")

        with pytest.raises(ValueError, match="incomplete"):
            store.update_item_status("p1-0", PlanItemStatus.accepted)

    def test_previous_phase_complete_unblocks(self, store: CodeProjectStore) -> None:
        store.add_phase("Phase 1")
        store.add_phase("Phase 2")
        store.add_plan_item(0, "Phase 1 task")
        store.add_plan_item(1, "Phase 2 task")

        # Complete phase 1
        store.update_item_status("p0-0", PlanItemStatus.accepted)
        store.update_item_status("p0-0", PlanItemStatus.in_progress)
        store.update_item_status("p0-0", PlanItemStatus.completed)

        # Now phase 2 should be unblocked
        item = store.update_item_status("p1-0", PlanItemStatus.accepted)
        assert item.status == PlanItemStatus.accepted

    def test_previous_phase_with_rejected_items_is_complete(
        self, store: CodeProjectStore
    ) -> None:
        """Rejected items count as terminal — don't block next phase."""
        store.add_phase("Phase 1")
        store.add_phase("Phase 2")
        store.add_plan_item(0, "Task A")
        store.add_plan_item(0, "Task B")
        store.add_plan_item(1, "Phase 2 task")

        # Complete one, reject the other
        store.update_item_status("p0-0", PlanItemStatus.accepted)
        store.update_item_status("p0-0", PlanItemStatus.in_progress)
        store.update_item_status("p0-0", PlanItemStatus.completed)
        store.update_item_status("p0-1", PlanItemStatus.rejected)

        # Phase 2 should be accessible
        item = store.update_item_status("p1-0", PlanItemStatus.accepted)
        assert item.status == PlanItemStatus.accepted

    def test_update_phase_name(self, store: CodeProjectStore) -> None:
        store.add_phase("Old Name")
        phase = store.update_phase(0, name="New Name")
        assert phase.name == "New Name"

    def test_update_phase_out_of_range(self, store: CodeProjectStore) -> None:
        with pytest.raises(ValueError, match="out of range"):
            store.update_phase(0, name="Nope")


# ── File operations ───────────────────────────────────────────────────────

class TestFileOps:
    def test_put_and_get_file(self, store: CodeProjectStore) -> None:
        entry = store.put_file("tool.py", "print('hello')\n")
        assert entry.version == 1
        assert len(entry.content_hash) == 16

        content = store.get_file_content("tool.py")
        assert content == "print('hello')\n"

    def test_put_file_creates_prev(self, store: CodeProjectStore) -> None:
        store.put_file("tool.py", "v1")
        store.put_file("tool.py", "v2")

        assert store.get_file_content("tool.py") == "v2"
        assert store.get_file_prev("tool.py") == "v1"

    def test_file_version_increments(self, store: CodeProjectStore) -> None:
        e1 = store.put_file("tool.py", "v1")
        assert e1.version == 1
        e2 = store.put_file("tool.py", "v2")
        assert e2.version == 2
        e3 = store.put_file("tool.py", "v3")
        assert e3.version == 3

    def test_put_file_with_turn_id(self, store: CodeProjectStore) -> None:
        entry = store.put_file("tool.py", "code", turn_id="turn-abc123")
        assert entry.accepted_turn_id == "turn-abc123"

    def test_nested_path(self, store: CodeProjectStore) -> None:
        entry = store.put_file("lib/utils.py", "# utils")
        assert entry.version == 1
        assert store.get_file_content("lib/utils.py") == "# utils"

    def test_get_nonexistent_file(self, store: CodeProjectStore) -> None:
        assert store.get_file_content("nope.py") is None

    def test_get_nonexistent_prev(self, store: CodeProjectStore) -> None:
        store.put_file("tool.py", "v1")
        assert store.get_file_prev("tool.py") is None  # only one version

    def test_list_files(self, store: CodeProjectStore) -> None:
        store.put_file("tool.py", "code")
        store.put_file("test_tool.py", "tests")
        files = store.list_files()
        assert "tool.py" in files
        assert "test_tool.py" in files
        assert files["tool.py"]["version"] == 1

    def test_content_hash_changes(self, store: CodeProjectStore) -> None:
        e1 = store.put_file("tool.py", "version 1")
        e2 = store.put_file("tool.py", "version 2")
        assert e1.content_hash != e2.content_hash


# ── Path sanitization ────────────────────────────────────────────────────

class TestPathSafety:
    def test_reject_absolute_path(self, store: CodeProjectStore) -> None:
        with pytest.raises(ValueError, match="Absolute"):
            store.put_file("/etc/passwd", "nope")

    def test_reject_traversal(self, store: CodeProjectStore) -> None:
        with pytest.raises(ValueError, match="traversal"):
            store.put_file("../../../etc/passwd", "nope")

    def test_reject_disallowed_extension(self, store: CodeProjectStore) -> None:
        with pytest.raises(ValueError, match="not allowed"):
            store.put_file("script.sh", "#!/bin/bash")

    def test_reject_no_extension(self, store: CodeProjectStore) -> None:
        with pytest.raises(ValueError, match="not allowed"):
            store.put_file("Makefile", "all:")

    def test_allowed_extensions(self, store: CodeProjectStore) -> None:
        for ext in ALLOWED_EXTENSIONS:
            entry = store.put_file(f"test{ext}", "content")
            assert entry.version == 1

    def test_validate_path_standalone(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        resolved = _validate_path("tool.py", workspace)
        assert resolved == (workspace / "tool.py").resolve()

        with pytest.raises(ValueError):
            _validate_path("../escape.py", workspace)


# ── Thread safety ─────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_mutations_no_corruption(
        self, store: CodeProjectStore
    ) -> None:
        """Hammer the store from multiple threads; verify no data corruption."""
        store.add_phase("Phase 1")
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                store.add_plan_item(0, f"Item {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        state = store.get_state()
        items = state["plan"]["phases"][0]["items"]
        assert len(items) == 20

    def test_concurrent_file_writes(self, store: CodeProjectStore) -> None:
        errors: list[Exception] = []

        def writer(i: int) -> None:
            try:
                store.put_file(f"file_{i}.py", f"content {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(store.list_files()) == 10
