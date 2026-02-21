# SPDX-License-Identifier: Apache-2.0
"""
Versioned artifact store — file-backed, crash-safe, per-artifact concurrency.

File layout:
    <governor_dir>/.governor/artifacts/
        index.json                  # ArtifactIndex (metadata only)
        content/<artifact_id>/
            v1.txt                  # Immutable content snapshot
            v2.txt
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field


# ============================================================================
# Exceptions
# ============================================================================


class ArtifactStoreError(Exception):
    """Base exception for artifact store operations."""


class ArtifactNotFoundError(ArtifactStoreError):
    """Raised when an artifact ID is not in the index."""

    def __init__(self, artifact_id: str) -> None:
        self.artifact_id = artifact_id
        super().__init__(f"Artifact not found: {artifact_id}")


class ArtifactVersionNotFoundError(ArtifactStoreError):
    """Raised when a specific version does not exist for an artifact."""

    def __init__(self, artifact_id: str, version: int) -> None:
        self.artifact_id = artifact_id
        self.version = version
        super().__init__(f"Artifact {artifact_id} version {version} not found")


class StaleArtifactVersionError(ArtifactStoreError):
    """Raised on optimistic concurrency conflict (PUT with wrong version)."""

    def __init__(
        self,
        artifact_id: str,
        expected_current_version: int,
        current_version: int,
        index_version: int,
    ) -> None:
        self.artifact_id = artifact_id
        self.expected_current_version = expected_current_version
        self.current_version = current_version
        self.index_version = index_version
        super().__init__(
            f"Artifact {artifact_id}: expected version {expected_current_version}, "
            f"but current is {current_version}"
        )


class ArtifactValidationError(ArtifactStoreError):
    """Raised for invalid inputs (bad kind, oversized content, empty title)."""


class ArtifactContentMissingError(ArtifactStoreError):
    """Raised when a content file is expected on disk but missing."""

    def __init__(self, artifact_id: str, version: int, path: str) -> None:
        self.artifact_id = artifact_id
        self.version = version
        self.path = path
        super().__init__(
            f"Content file missing: artifact {artifact_id} v{version} at {path}"
        )


# ============================================================================
# Pydantic models
# ============================================================================


class ArtifactVersion(BaseModel):
    version: int
    created_at: str  # ISO8601 UTC
    content_hash: str  # SHA256 hex[:16]
    source: str = "manual"  # "promote" | "manual" | "edit"
    message_id: str | None = None


class ArtifactMeta(BaseModel):
    id: str  # uuid4().hex[:12]
    title: str
    kind: str = "text"  # "text" | "markdown" | "code"
    language: str = ""
    current_version: int = 1
    versions: list[ArtifactVersion] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


class ArtifactIndex(BaseModel):
    version: int = 1
    artifacts: dict[str, ArtifactMeta] = Field(default_factory=dict)
    updated_at: str = ""


class ArtifactSummary(BaseModel):
    """Not persisted — returned by list_all()."""

    id: str
    title: str
    kind: str
    language: str
    current_version: int
    created_at: str
    updated_at: str


# ============================================================================
# Store
# ============================================================================


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


class ArtifactStore:
    """File-backed artifact store with per-artifact optimistic concurrency."""

    ALLOWED_KINDS = {"text", "markdown", "code"}
    ALLOWED_SOURCES = {"promote", "manual", "edit"}

    def __init__(
        self,
        governor_dir: str | Path,
        *,
        max_title_len: int = 200,
        max_content_bytes: int = 1_000_000,
    ) -> None:
        base = Path(governor_dir) / ".governor" / "artifacts"
        self._base = base
        self._index_path = base / "index.json"
        self._content_dir = base / "content"
        self._max_title_len = max_title_len
        self._max_content_bytes = max_content_bytes
        self._lock = threading.Lock()

        # Ensure directories exist
        self._base.mkdir(parents=True, exist_ok=True)
        self._content_dir.mkdir(parents=True, exist_ok=True)

        # Bootstrap index if missing
        if not self._index_path.exists():
            self._write_index(ArtifactIndex(updated_at=_now_iso()))

    # ------------------------------------------------------------------
    # Index I/O
    # ------------------------------------------------------------------

    def _load_index(self) -> ArtifactIndex:
        data = json.loads(self._index_path.read_text(encoding="utf-8"))
        return ArtifactIndex.model_validate(data)

    def _write_index(self, index: ArtifactIndex) -> None:
        """Atomic write: tmp file + rename."""
        fd, tmp = tempfile.mkstemp(dir=str(self._base), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(index.model_dump_json(indent=2))
            os.replace(tmp, str(self._index_path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Content I/O
    # ------------------------------------------------------------------

    def _content_path(self, artifact_id: str, version: int) -> Path:
        return self._content_dir / artifact_id / f"v{version}.txt"

    def _write_content(self, artifact_id: str, version: int, content: str) -> None:
        """Crash-safe: write content file via tmp + replace."""
        dest = self._content_path(artifact_id, version)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, str(dest))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _read_content(self, artifact_id: str, version: int) -> str:
        p = self._content_path(artifact_id, version)
        if not p.exists():
            raise ArtifactContentMissingError(artifact_id, version, str(p))
        return p.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_kind(self, kind: str) -> None:
        if kind not in self.ALLOWED_KINDS:
            raise ArtifactValidationError(
                f"Invalid kind '{kind}'. Allowed: {sorted(self.ALLOWED_KINDS)}"
            )

    def _validate_source(self, source: str) -> None:
        if source not in self.ALLOWED_SOURCES:
            raise ArtifactValidationError(
                f"Invalid source '{source}'. Allowed: {sorted(self.ALLOWED_SOURCES)}"
            )

    def _validate_title(self, title: str) -> None:
        if not title or not title.strip():
            raise ArtifactValidationError("Title must not be empty")
        if len(title) > self._max_title_len:
            raise ArtifactValidationError(
                f"Title exceeds {self._max_title_len} characters"
            )

    def _validate_content(self, content: str) -> None:
        size = len(content.encode("utf-8"))
        if size > self._max_content_bytes:
            raise ArtifactValidationError(
                f"Content size {size} bytes exceeds limit of {self._max_content_bytes}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        title: str,
        content: str,
        kind: str = "text",
        language: str = "",
        message_id: str | None = None,
        source: str = "manual",
    ) -> tuple[ArtifactMeta, str, int]:
        """Create a new artifact. Returns (meta, content, index_version)."""
        # 1. Validate
        self._validate_kind(kind)
        self._validate_source(source)
        self._validate_title(title)
        self._validate_content(content)

        now = _now_iso()
        artifact_id = uuid4().hex[:12]
        chash = _content_hash(content)

        ver = ArtifactVersion(
            version=1,
            created_at=now,
            content_hash=chash,
            source=source,
            message_id=message_id,
        )
        meta = ArtifactMeta(
            id=artifact_id,
            title=title,
            kind=kind,
            language=language,
            current_version=1,
            versions=[ver],
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            # 2. Load index
            index = self._load_index()
            # 3. Mutate in-memory
            index.artifacts[artifact_id] = meta
            # 4. Write content first
            self._write_content(artifact_id, 1, content)
            # 5. Bump index version + timestamp
            index.version += 1
            index.updated_at = now
            # 6. Write index
            self._write_index(index)

        return meta, content, index.version

    def update(
        self,
        artifact_id: str,
        *,
        content: str,
        title: str | None = None,
        expected_current_version: int | None = None,
        source: str = "manual",
        message_id: str | None = None,
    ) -> tuple[ArtifactMeta, str, int]:
        """Update an artifact, creating a new version. Returns (meta, content, index_version)."""
        self._validate_source(source)
        self._validate_content(content)
        if title is not None:
            self._validate_title(title)

        with self._lock:
            index = self._load_index()
            meta = index.artifacts.get(artifact_id)
            if meta is None:
                raise ArtifactNotFoundError(artifact_id)

            # Per-artifact concurrency check
            if (
                expected_current_version is not None
                and expected_current_version != meta.current_version
            ):
                raise StaleArtifactVersionError(
                    artifact_id=artifact_id,
                    expected_current_version=expected_current_version,
                    current_version=meta.current_version,
                    index_version=index.version,
                )

            now = _now_iso()
            new_version = meta.current_version + 1
            chash = _content_hash(content)

            ver = ArtifactVersion(
                version=new_version,
                created_at=now,
                content_hash=chash,
                source=source,
                message_id=message_id,
            )

            # Mutate in-memory metadata
            meta.current_version = new_version
            meta.versions.append(ver)
            meta.updated_at = now
            if title is not None:
                meta.title = title

            # Write content first, then index
            self._write_content(artifact_id, new_version, content)
            index.version += 1
            index.updated_at = now
            self._write_index(index)

        return meta, content, index.version

    def get(self, artifact_id: str) -> tuple[ArtifactMeta, str, int]:
        """Get artifact metadata + latest content. Returns (meta, content, index_version)."""
        with self._lock:
            index = self._load_index()
            meta = index.artifacts.get(artifact_id)
            if meta is None:
                raise ArtifactNotFoundError(artifact_id)

        content = self._read_content(artifact_id, meta.current_version)
        return meta, content, index.version

    def get_version(self, artifact_id: str, version: int) -> str:
        """Get content for a specific version."""
        with self._lock:
            index = self._load_index()
            meta = index.artifacts.get(artifact_id)
            if meta is None:
                raise ArtifactNotFoundError(artifact_id)

        # Check version exists in metadata
        version_exists = any(v.version == version for v in meta.versions)
        if not version_exists:
            raise ArtifactVersionNotFoundError(artifact_id, version)

        return self._read_content(artifact_id, version)

    def list_all(self) -> tuple[list[ArtifactSummary], int]:
        """List all artifacts (metadata only, sorted by updated_at desc). Returns (summaries, index_version)."""
        with self._lock:
            index = self._load_index()

        summaries = [
            ArtifactSummary(
                id=m.id,
                title=m.title,
                kind=m.kind,
                language=m.language,
                current_version=m.current_version,
                created_at=m.created_at,
                updated_at=m.updated_at,
            )
            for m in index.artifacts.values()
        ]
        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries, index.version

    def delete(self, artifact_id: str) -> tuple[bool, int]:
        """Delete artifact from index (content files left on disk). Returns (True, index_version)."""
        with self._lock:
            index = self._load_index()
            if artifact_id not in index.artifacts:
                raise ArtifactNotFoundError(artifact_id)

            # Index-only delete — content files left on disk
            # TODO: archive/cleanup content files later
            del index.artifacts[artifact_id]
            index.version += 1
            index.updated_at = _now_iso()
            self._write_index(index)

        return True, index.version

    def get_state(self) -> dict:
        """Quick state for polling. Returns {version, updated_at, count}."""
        with self._lock:
            index = self._load_index()
        return {
            "version": index.version,
            "updated_at": index.updated_at,
            "count": len(index.artifacts),
        }

    def exists(self, artifact_id: str) -> bool:
        """Check if an artifact exists in the index."""
        with self._lock:
            index = self._load_index()
        return artifact_id in index.artifacts
