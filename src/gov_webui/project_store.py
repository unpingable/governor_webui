# SPDX-License-Identifier: MIT
"""Code Builder — project store with workspace-backed file storage.

Manages intent/contract/plan + workspace files for the structured
code-building workflow.  Thread-safe, atomic saves, optimistic concurrency.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StaleVersionError(Exception):
    """Raised when expected_version doesn't match current state version."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PlanItemStatus(str, Enum):
    proposed = "proposed"
    accepted = "accepted"
    in_progress = "in_progress"
    completed = "completed"
    rejected = "rejected"


VALID_TRANSITIONS: dict[PlanItemStatus, set[PlanItemStatus]] = {
    PlanItemStatus.proposed:    {PlanItemStatus.accepted, PlanItemStatus.rejected},
    PlanItemStatus.accepted:    {PlanItemStatus.in_progress, PlanItemStatus.rejected},
    PlanItemStatus.in_progress: {PlanItemStatus.completed, PlanItemStatus.rejected},
    PlanItemStatus.completed:   set(),
    PlanItemStatus.rejected:    set(),
}

TERMINAL_STATUSES = {PlanItemStatus.completed, PlanItemStatus.rejected}

# Phase 0 file extension allowlists
CODE_EXTENSIONS = {".py", ".txt", ".json", ".toml", ".cfg"}
RESEARCH_EXTENSIONS = {".md", ".txt", ".bib", ".json", ".toml", ".csv"}
ALLOWED_EXTENSIONS = CODE_EXTENSIONS  # default for backwards compat


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class Intent(BaseModel):
    text: str = ""
    locked: bool = False


class ContractField(BaseModel):
    name: str
    type: str = "str"
    required: bool = True
    default: Any = None
    description: str = ""


class Contract(BaseModel):
    description: str = ""
    inputs: list[ContractField] = Field(default_factory=list)
    outputs: list[ContractField] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    transport: str = "stdio"


class PlanItem(BaseModel):
    id: str
    text: str
    status: PlanItemStatus = PlanItemStatus.proposed
    turn_id: str | None = None


class Phase(BaseModel):
    name: str
    locked: bool = False
    items: list[PlanItem] = Field(default_factory=list)


class Plan(BaseModel):
    phases: list[Phase] = Field(default_factory=list)


class FileEntry(BaseModel):
    version: int = 1
    accepted_at: str = ""
    accepted_turn_id: str | None = None
    content_hash: str = ""


class ProjectState(BaseModel):
    version: int = 1
    intent: Intent = Field(default_factory=Intent)
    contract: Contract = Field(default_factory=Contract)
    plan: Plan = Field(default_factory=Plan)
    files: dict[str, FileEntry] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def _validate_path(
    path: str,
    workspace_root: Path,
    allowed_extensions: set[str] | None = None,
) -> Path:
    """Validate and resolve a workspace-relative path.

    Raises ValueError on:
    - absolute paths
    - '..' traversal
    - extensions not in allowed set
    - resolved path outside workspace_root
    """
    if allowed_extensions is None:
        allowed_extensions = ALLOWED_EXTENSIONS

    if not path:
        raise ValueError("Empty path")
    if path.startswith("/"):
        raise ValueError(f"Absolute paths not allowed: {path}")
    if ".." in Path(path).parts:
        raise ValueError(f"Path traversal not allowed: {path}")

    ext = Path(path).suffix.lower()
    if ext not in allowed_extensions:
        raise ValueError(
            f"Extension '{ext}' not allowed. "
            f"Allowed: {sorted(allowed_extensions)}"
        )

    resolved = (workspace_root / path).resolve()
    if not resolved.is_relative_to(workspace_root.resolve()):
        raise ValueError(f"Path escapes workspace: {path}")

    return resolved


# ---------------------------------------------------------------------------
# CodeProjectStore
# ---------------------------------------------------------------------------

class CodeProjectStore:
    """Manages project state + workspace files for the code/research builder.

    Args:
        governor_dir: Path to the .governor directory.
        subdir: Subdirectory name under governor_dir (default "code").
        allowed_extensions: Set of allowed file extensions (default CODE_EXTENSIONS).
    """

    def __init__(
        self,
        governor_dir: Path,
        subdir: str = "code",
        allowed_extensions: set[str] | None = None,
    ) -> None:
        self._dir = governor_dir / subdir
        self._project_path = self._dir / "project.json"
        self._workspace = self._dir / "workspace"
        self._allowed_extensions = allowed_extensions or CODE_EXTENSIONS
        self._lock = threading.Lock()
        self._state: ProjectState = self._load()

    # -- Persistence --------------------------------------------------------

    def _load(self) -> ProjectState:
        """Load state from project.json, or create fresh."""
        if self._project_path.exists():
            try:
                data = json.loads(self._project_path.read_text())
                return ProjectState(**data)
            except (json.JSONDecodeError, Exception):
                # Corrupt file — start fresh
                pass
        now = datetime.now(timezone.utc).isoformat()
        return ProjectState(version=1, created_at=now, updated_at=now)

    def _save(self) -> None:
        """Atomic save: write tmp then os.replace."""
        self._state.version += 1
        self._state.updated_at = datetime.now(timezone.utc).isoformat()

        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._project_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._state.model_dump(), indent=2))
        os.replace(str(tmp), str(self._project_path))

    def _check_version(self, expected_version: int | None) -> None:
        """Raise StaleVersionError if expected_version mismatches."""
        if expected_version is not None and expected_version != self._state.version:
            raise StaleVersionError(
                f"Expected version {expected_version}, "
                f"current is {self._state.version}"
            )

    # -- Intent -------------------------------------------------------------

    def update_intent(
        self,
        text: str,
        locked: bool,
        expected_version: int | None = None,
    ) -> Intent:
        with self._lock:
            self._check_version(expected_version)
            self._state.intent = Intent(text=text, locked=locked)
            self._save()
            return self._state.intent

    # -- Contract -----------------------------------------------------------

    def update_contract(
        self,
        contract: Contract,
        expected_version: int | None = None,
    ) -> Contract:
        with self._lock:
            self._check_version(expected_version)
            self._state.contract = contract
            self._save()
            return self._state.contract

    # -- Plan phases --------------------------------------------------------

    def add_phase(self, name: str) -> Phase:
        with self._lock:
            phase = Phase(name=name)
            self._state.plan.phases.append(phase)
            self._save()
            return phase

    def update_phase(
        self,
        idx: int,
        name: str | None = None,
        locked: bool | None = None,
    ) -> Phase:
        with self._lock:
            phases = self._state.plan.phases
            if idx < 0 or idx >= len(phases):
                raise ValueError(f"Phase index {idx} out of range")
            if name is not None:
                phases[idx].name = name
            if locked is not None:
                phases[idx].locked = locked
            self._save()
            return phases[idx]

    # -- Plan items ---------------------------------------------------------

    def add_plan_item(self, phase_idx: int, text: str) -> PlanItem:
        with self._lock:
            phases = self._state.plan.phases
            if phase_idx < 0 or phase_idx >= len(phases):
                raise ValueError(f"Phase index {phase_idx} out of range")

            phase = phases[phase_idx]
            seq = len(phase.items)
            item = PlanItem(id=f"p{phase_idx}-{seq}", text=text)
            phase.items.append(item)
            self._save()
            return item

    def _find_item(self, item_id: str) -> tuple[int, int, PlanItem]:
        """Find item by id. Returns (phase_idx, item_idx, item)."""
        for pi, phase in enumerate(self._state.plan.phases):
            for ii, item in enumerate(phase.items):
                if item.id == item_id:
                    return pi, ii, item
        raise ValueError(f"Plan item '{item_id}' not found")

    def update_item_status(
        self,
        item_id: str,
        status: PlanItemStatus,
        expected_version: int | None = None,
    ) -> PlanItem:
        with self._lock:
            self._check_version(expected_version)
            pi, ii, item = self._find_item(item_id)

            # Rejected is always allowed (escape hatch / cancel)
            if status != PlanItemStatus.rejected:
                # Validate transition
                allowed = VALID_TRANSITIONS.get(item.status, set())
                if status not in allowed:
                    raise ValueError(
                        f"Invalid transition: {item.status.value} -> {status.value}"
                    )

                # Phase gating
                phase = self._state.plan.phases[pi]
                if phase.locked:
                    raise ValueError(
                        f"Phase '{phase.name}' is locked"
                    )

                # Previous phase must be complete
                if pi > 0:
                    prev_phase = self._state.plan.phases[pi - 1]
                    pending = [
                        it for it in prev_phase.items
                        if it.status not in TERMINAL_STATUSES
                    ]
                    if pending:
                        raise ValueError(
                            f"Previous phase '{prev_phase.name}' "
                            f"has {len(pending)} incomplete item(s)"
                        )
            else:
                # Even for rejected, validate it's a legal transition
                # (can't reject something already completed or rejected)
                allowed = VALID_TRANSITIONS.get(item.status, set())
                if PlanItemStatus.rejected not in allowed and item.status in TERMINAL_STATUSES:
                    raise ValueError(
                        f"Cannot reject item in terminal state: {item.status.value}"
                    )

            item.status = status
            self._save()
            return item

    # -- Files --------------------------------------------------------------

    def get_file_content(self, path: str) -> str | None:
        """Read file content from workspace."""
        resolved = _validate_path(path, self._workspace, self._allowed_extensions)
        if resolved.exists():
            return resolved.read_text()
        return None

    def get_file_prev(self, path: str) -> str | None:
        """Read previous version (.prev) of a file."""
        resolved = _validate_path(path, self._workspace, self._allowed_extensions)
        prev_path = resolved.with_suffix(resolved.suffix + ".prev")
        if prev_path.exists():
            return prev_path.read_text()
        return None

    def put_file(
        self,
        path: str,
        content: str,
        turn_id: str | None = None,
    ) -> FileEntry:
        """Write file to workspace, maintain one-deep .prev backup."""
        with self._lock:
            resolved = _validate_path(path, self._workspace, self._allowed_extensions)
            resolved.parent.mkdir(parents=True, exist_ok=True)

            # Move current to .prev if exists
            prev_path = resolved.with_suffix(resolved.suffix + ".prev")
            if resolved.exists():
                # Use os.replace for atomicity
                os.replace(str(resolved), str(prev_path))

            # Write new content
            resolved.write_text(content)

            # Update metadata
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            now = datetime.now(timezone.utc).isoformat()

            existing = self._state.files.get(path)
            version = (existing.version + 1) if existing else 1

            entry = FileEntry(
                version=version,
                accepted_at=now,
                accepted_turn_id=turn_id,
                content_hash=content_hash,
            )
            self._state.files[path] = entry
            self._save()
            return entry

    def list_files(self) -> dict[str, dict]:
        """Return file metadata for all tracked files."""
        return {
            path: entry.model_dump()
            for path, entry in self._state.files.items()
        }

    # -- Full state ---------------------------------------------------------

    def get_state(self) -> dict:
        """Full state dump for sidebar polling."""
        return self._state.model_dump()
