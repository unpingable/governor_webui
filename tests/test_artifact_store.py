# SPDX-License-Identifier: Apache-2.0
"""Tests for the versioned artifact store."""

import os
import threading

import pytest

from gov_webui.artifact_store import (
    ArtifactContentMissingError,
    ArtifactNotFoundError,
    ArtifactStore,
    ArtifactValidationError,
    ArtifactVersionNotFoundError,
    StaleArtifactVersionError,
)


@pytest.fixture
def store(tmp_path):
    """Create an ArtifactStore rooted in a temp directory."""
    return ArtifactStore(tmp_path)


# ------------------------------------------------------------------
# 1. create writes metadata and v1 content
# ------------------------------------------------------------------

def test_create_artifact_writes_metadata_and_v1_content(store):
    meta, content, idx_ver = store.create(
        title="Hello", content="world", kind="text", language="", source="manual"
    )
    assert meta.id
    assert meta.title == "Hello"
    assert meta.kind == "text"
    assert meta.current_version == 1
    assert len(meta.versions) == 1
    assert meta.versions[0].version == 1
    assert meta.versions[0].source == "manual"
    assert content == "world"
    assert idx_ver >= 2  # bootstrap=1, after create >=2


# ------------------------------------------------------------------
# 2. update creates new version; old version remains readable
# ------------------------------------------------------------------

def test_update_creates_new_version_and_old_version_remains_readable(store):
    meta, _, _ = store.create(
        title="Draft", content="v1 content", kind="markdown", language=""
    )
    aid = meta.id

    meta2, content2, _ = store.update(
        aid, content="v2 content", expected_current_version=1
    )
    assert meta2.current_version == 2
    assert content2 == "v2 content"
    assert len(meta2.versions) == 2

    # Old version still readable
    old = store.get_version(aid, 1)
    assert old == "v1 content"


# ------------------------------------------------------------------
# 3. stale version raises structured error
# ------------------------------------------------------------------

def test_stale_version_raises_structured_error(store):
    meta, _, _ = store.create(title="Doc", content="a", kind="text", language="")
    aid = meta.id

    # Update to v2
    store.update(aid, content="b", expected_current_version=1)

    # Now try with stale version 1
    with pytest.raises(StaleArtifactVersionError) as exc_info:
        store.update(aid, content="c", expected_current_version=1)

    err = exc_info.value
    assert err.artifact_id == aid
    assert err.expected_current_version == 1
    assert err.current_version == 2
    assert err.index_version >= 3


# ------------------------------------------------------------------
# 4. list_all returns metadata only, sorted by updated_at desc
# ------------------------------------------------------------------

def test_list_all_returns_metadata_only_sorted_by_updated_desc(store):
    store.create(title="First", content="a", kind="text", language="")
    store.create(title="Second", content="b", kind="text", language="")
    store.create(title="Third", content="c", kind="text", language="")

    summaries, idx_ver = store.list_all()
    assert len(summaries) == 3
    # Most recently created should be first
    assert summaries[0].title == "Third"
    assert summaries[-1].title == "First"
    # Summaries should not have versions list (it's ArtifactSummary, not ArtifactMeta)
    assert not hasattr(summaries[0], "versions")


# ------------------------------------------------------------------
# 5. delete removes from index
# ------------------------------------------------------------------

def test_delete_removes_from_index(store):
    meta, _, _ = store.create(title="Gone", content="bye", kind="text", language="")
    aid = meta.id

    deleted, idx_ver = store.delete(aid)
    assert deleted is True

    with pytest.raises(ArtifactNotFoundError):
        store.get(aid)

    summaries, _ = store.list_all()
    assert len(summaries) == 0


# ------------------------------------------------------------------
# 6. get specific version returns correct content
# ------------------------------------------------------------------

def test_get_specific_version_returns_correct_content(store):
    meta, _, _ = store.create(title="Doc", content="v1", kind="text", language="")
    aid = meta.id
    store.update(aid, content="v2", expected_current_version=1)
    store.update(aid, content="v3", expected_current_version=2)

    assert store.get_version(aid, 1) == "v1"
    assert store.get_version(aid, 2) == "v2"
    assert store.get_version(aid, 3) == "v3"


# ------------------------------------------------------------------
# 7. persist and reload roundtrip
# ------------------------------------------------------------------

def test_persist_and_reload_roundtrip(tmp_path):
    store1 = ArtifactStore(tmp_path)
    meta, _, _ = store1.create(
        title="Persist", content="data", kind="markdown", language="md"
    )
    aid = meta.id

    # Create a new store from the same directory
    store2 = ArtifactStore(tmp_path)
    meta2, content2, _ = store2.get(aid)
    assert meta2.title == "Persist"
    assert meta2.kind == "markdown"
    assert content2 == "data"


# ------------------------------------------------------------------
# 8. create with invalid kind raises validation error
# ------------------------------------------------------------------

def test_create_invalid_kind_raises_validation_error(store):
    with pytest.raises(ArtifactValidationError, match="Invalid kind"):
        store.create(title="Bad", content="x", kind="spreadsheet", language="")


# ------------------------------------------------------------------
# 9. create with oversized content raises validation error
# ------------------------------------------------------------------

def test_create_oversized_content_raises_validation_error(tmp_path):
    store = ArtifactStore(tmp_path, max_content_bytes=100)
    with pytest.raises(ArtifactValidationError, match="Content size"):
        store.create(title="Big", content="x" * 200, kind="text", language="")


# ------------------------------------------------------------------
# 10. get missing artifact raises artifact not found
# ------------------------------------------------------------------

def test_get_missing_artifact_raises_artifact_not_found(store):
    with pytest.raises(ArtifactNotFoundError) as exc_info:
        store.get("nonexistent123")
    assert exc_info.value.artifact_id == "nonexistent123"


# ------------------------------------------------------------------
# 11. missing content file raises content missing
# ------------------------------------------------------------------

def test_missing_content_file_raises_content_missing(store):
    meta, _, _ = store.create(title="Orphan", content="here", kind="text", language="")
    aid = meta.id

    # Delete the content file behind the store's back
    content_path = store._content_path(aid, 1)
    os.unlink(content_path)

    with pytest.raises(ArtifactContentMissingError) as exc_info:
        store.get(aid)
    err = exc_info.value
    assert err.artifact_id == aid
    assert err.version == 1


# ------------------------------------------------------------------
# 12. thread safety: concurrent creates do not corrupt index
# ------------------------------------------------------------------

def test_thread_safety_concurrent_creates_do_not_corrupt_index(tmp_path):
    store = ArtifactStore(tmp_path)
    errors = []
    count = 20

    def create_one(i):
        try:
            store.create(
                title=f"Thread-{i}", content=f"content-{i}", kind="text", language=""
            )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=create_one, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Errors during concurrent creates: {errors}"

    summaries, _ = store.list_all()
    assert len(summaries) == count


# ------------------------------------------------------------------
# 13. promote source provenance
# ------------------------------------------------------------------

def test_promote_source_provenance(store):
    meta, _, _ = store.create(
        title="From Chat",
        content="promoted text",
        kind="markdown",
        language="",
        source="promote",
        message_id="msg-abc-123",
    )

    assert meta.versions[0].source == "promote"
    assert meta.versions[0].message_id == "msg-abc-123"

    # Reload and verify persistence
    meta2, _, _ = store.get(meta.id)
    assert meta2.versions[0].source == "promote"
    assert meta2.versions[0].message_id == "msg-abc-123"
