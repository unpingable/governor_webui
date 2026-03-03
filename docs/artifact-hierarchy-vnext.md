# Hierarchical Artifact Model — vNext Spec

**Status:** Design only — no code changes
**Scope:** Additive, backward-compatible data model for hierarchical artifacts
**Target system:** Phosphor artifact store (currently flat `id -> version -> blob`)

---

## 1. Problem

The flat artifact store works for short-form outputs but breaks down at
long-form scale (chapter books, long papers).  Three failure modes:

1. **No update locality** — revising section 4 re-saves the entire blob.
2. **No dependency visibility** — no way to know which sections depend on
   stale definitions, canon, or style changes.
3. **No scoped regeneration** — regeneration is all-or-nothing.

This document defines a hierarchical model that preserves file-backed,
versioned, optimistic-concurrency semantics while adding node-level
structure, explicit dependencies, build receipts, derived-artifact caches,
and staleness tracking.

Implementation is deferred until real friction appears in production use.

---

## 2. Goals and Non-Goals

### Goals

- Tree-of-nodes structure with cross-node references.
- Root-level artifact identity with node-level versioning.
- Invariants (definitions, canon, terminology, style) as first-class versioned objects.
- Build receipts on every node generation / revision / reconciliation.
- Partial regeneration and staleness tracking.
- File-backed, hash-addressed, optimistically concurrent.
- Backward-compatible with the current flat store (additive layering).

### Non-Goals

This spec is not an implementation plan, a migration script, a timeline
commitment, a replacement for the flat store, or a UI / route design.

---

## 3. Conventions (Grounded Against Codebase)

These conventions match what the existing stores already use.

| Convention | Current pattern | Source |
|---|---|---|
| IDs | `uuid4().hex[:12]` | `artifact_store.py:279` |
| Content hashes | SHA-256 `hex[:16]` | `artifact_store.py:141` |
| Integrity hashes | SHA-256 full hex | `receipt_v1/canonical.py` |
| Timestamps | `str` ISO 8601 UTC | `artifact_store.py:96` |
| Models | Pydantic `BaseModel` | `artifact_store.py:94` |
| List/dict defaults | `Field(default_factory=...)` | `artifact_store.py:109` |
| Atomic writes | `tempfile.mkstemp` + `os.replace` | `artifact_store.py:184-193` |
| Version counters | `int`, monotonic | `artifact_store.py:108` |
| Concurrency | optimistic via `expected_current_version` | `artifact_store.py:339-349` |

New objects in this spec follow the same patterns unless noted.

---

## 4. Core Objects

### 4.1 Artifact (root)

Top-level identity and active hierarchy state.

```python
class HierarchicalArtifact(BaseModel):
    artifact_id: str                          # hex[:12]
    kind: str                                 # "fiction" | "paper" | "note" | "code" | "other"
    title: str
    status: str = "draft"                     # "draft" | "working" | "review" | "published" | "archived"

    # Version semantics
    root_version: int                         # bumps on any active-state change
    manifest_version: int                     # bumps on topology / config changes only
    current_snapshot_hash: str                # SHA-256 full hex over canonical active state

    # Active bindings
    active_invariant_set_id: str | None = None
    active_invariant_set_version: int | None = None
    active_style_profile: str | None = None   # style_policy profile name

    # Structure
    root_node_id: str                         # hex[:12]
    node_order_policy: str = "tree_order"     # "tree_order" | "explicit_order"
    derived_node_ids: list[str] = Field(default_factory=list)

    # Metadata
    tags: list[str] = Field(default_factory=list)
    created_at: str = ""                      # ISO 8601 UTC
    updated_at: str = ""

    # Concurrency / compatibility
    etag: str = ""                            # SHA-256 over root state
    compatibility_mode: str = "flat"          # "flat" | "hierarchical_shadow" | "hierarchical"
```

**`root_version`** is the user-visible "artifact has changed" counter.
**`manifest_version`** is narrower — structure / config only; ordinary node
content edits do not bump it.
**`current_snapshot_hash`** is computed over a canonical serialization of
active node tip hashes, edges, invariant version, derived tips, and ordering.

### 4.2 Node

A structural unit of content (chapter, section, scene, appendix, etc.).
Content is versioned independently.

```python
class Node(BaseModel):
    node_id: str                              # hex[:12]
    artifact_id: str

    kind: str                                 # see Node Kinds (sec 5.1)
    title: str

    # Computed / cached — not authoritative input fields
    slug: str = ""                            # derived from title
    depth: int = 0                            # cached from edge graph
    path: str = ""                            # cached, e.g. "ch03/sec02"

    ordinal: int | None = None                # sibling ordering
    current_version: int = 1
    current_content_hash: str = ""            # SHA-256 hex[:16]

    # Optional overrides
    style_profile_override: str | None = None
    invariant_scope_override: str | None = None

    # State / staleness
    state: str = "active"                     # "active" | "deleted" | "superseded"
    staleness: str = "fresh"                  # "fresh" | "soft_stale" | "hard_stale"
    staleness_reasons: list[str] = Field(default_factory=list)

    created_at: str = ""
    updated_at: str = ""
```

> **Note:** `slug`, `depth`, and `path` are derived values that can be
> recomputed from the edge graph.  They are cached for fast lookups but
> are never authoritative inputs.

### 4.3 NodeVersion

Immutable content snapshot for a single node revision.  Mirrors the
existing `ArtifactVersion` pattern but adds structured metadata.

```python
class NodeVersion(BaseModel):
    artifact_id: str
    node_id: str
    version: int

    body_hash: str                            # SHA-256 hex[:16] of content file
    source: str = "manual"                    # "promote" | "manual" | "edit" | "generate"

    # Structured metadata (optional)
    summary: str | None = None
    labels: list[str] = Field(default_factory=list)
    anchor_refs: list[str] = Field(default_factory=list)
    cited_node_ids: list[str] = Field(default_factory=list)

    # Provenance
    build_receipt_id: str | None = None
    parent_version: int | None = None
    revision_reason: str | None = None

    created_at: str = ""
    message_id: str | None = None             # mirrors ArtifactVersion.message_id
    source_turn_seq: int | None = None        # mirrors ArtifactVersion.source_turn_seq
```

Content files live at `content/<artifact_id>/nodes/<node_id>/v<N>.md`,
following the existing `content/<artifact_id>/v<N>.txt` pattern.

### 4.4 Edge

Graph relationships between nodes.  Parent/child is an edge type, not
embedded pointers.

```python
class Edge(BaseModel):
    edge_id: str                              # hex[:12]
    artifact_id: str

    edge_type: str                            # see Edge Types (sec 5.2)
    src_node_id: str
    dst_node_id: str

    label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    active: bool = True
    introduced_in_manifest_version: int = 1
    retired_in_manifest_version: int | None = None

    created_at: str = ""
```

### 4.5 InvariantSet

Frozen, versioned constraint set.  The document-level equivalent of the
governor's "compiler policy" layer.  Bridges fiction canon/definitions,
paper terminology, and style profiles into a single addressable object.

```python
class InvariantSet(BaseModel):
    invariant_set_id: str                     # hex[:12]
    artifact_id: str

    name: str
    version: int
    status: str = "draft"                     # "draft" | "frozen" | "superseded"

    # Content
    definitions: dict[str, str] = Field(default_factory=dict)
    canon_facts: dict[str, str] = Field(default_factory=dict)
    prohibitions: list[str] = Field(default_factory=list)
    terminology: dict[str, str] = Field(default_factory=dict)
    thesis_statements: list[str] = Field(default_factory=list)
    style_profile: str | None = None          # style_policy profile name
    style_overrides: dict[str, Any] = Field(default_factory=dict)

    # Integrity
    content_hash: str = ""                    # SHA-256 hex[:16] over canonical serialization
    created_at: str = ""
    supersedes_version: int | None = None
```

**Freeze semantics:** draft -> frozen -> superseded.  Changing an invariant
after freeze creates a new version.  Dependent nodes may be marked stale
(see sec 8).

**Relationship to existing systems:**

| InvariantSet field | Existing system | Mapping |
|---|---|---|
| `definitions` | fiction `DEFINITION` anchors | direct |
| `canon_facts` | fiction `CANON` anchors | direct |
| `prohibitions` | fiction `PROHIBITION` anchors | direct |
| `style_profile` | `style_policy.MODE_PROFILES` | profile name |
| `terminology` | (new for papers) | — |
| `thesis_statements` | (new for papers) | — |

### 4.6 BuildReceipt

Node-scoped generation provenance.  Mirrors `receipt_v1` principles
(explicit inputs, explicit outputs, hashes, actor, replayability) but
scoped to a single node or reconciliation pass rather than a tool call.

```python
class BuildReceipt(BaseModel):
    receipt_id: str                           # hex[:12]
    artifact_id: str
    node_id: str | None = None                # None for artifact-level ops

    receipt_kind: str                         # see sec 7.1

    # Inputs
    input_node_versions: list[tuple[str, int]] = Field(default_factory=list)
    input_invariant_set: tuple[str, int] | None = None
    input_style_profile: str | None = None
    input_prompts_hash: str | None = None     # SHA-256 hex[:16]

    # Outputs
    output_node_version: tuple[str, int] | None = None
    output_content_hash: str | None = None    # SHA-256 hex[:16]

    # Execution
    actor: str = ""                           # user or system identity
    executor: str = ""                        # model / tool identity
    started_at: str = ""
    completed_at: str = ""
    duration_ms: int | None = None

    status: str = "ok"                        # "ok" | "warning" | "error"
    warnings: list[str] = Field(default_factory=list)
    notes: str | None = None

    # Concurrency / replay
    expected_root_version: int | None = None
    expected_node_version: int | None = None
    replayable: bool = True
```

### 4.7 DerivedNode

Cached computation over source nodes — summary, abstract, teaser, outline.
First-class because summaries are transformations with policies and
omissions, not neutral pass-throughs.

```python
class DerivedNode(BaseModel):
    derived_node_id: str                      # hex[:12]
    artifact_id: str

    kind: str                                 # "artifact_abstract" | "chapter_summary" | "teaser" | "outline" | ...
    title: str
    target_scope: dict[str, Any] = Field(default_factory=dict)

    current_version: int = 1
    current_body_hash: str = ""               # SHA-256 hex[:16]

    # Derivation provenance
    source_node_versions: list[tuple[str, int]] = Field(default_factory=list)
    source_invariant_set: tuple[str, int] | None = None
    compression_policy: str = ""
    omission_classes: list[str] = Field(default_factory=list)
    as_of_root_version: int = 0
    build_receipt_id: str = ""

    staleness: str = "fresh"                  # "fresh" | "stale"
    staleness_reasons: list[str] = Field(default_factory=list)

    created_at: str = ""
    updated_at: str = ""
```

---

## 5. Node Kinds and Edge Types

### 5.1 Node Kinds (extensible strings)

**Structural:** `artifact_root`, `part`, `chapter`, `section`,
`subsection`, `scene`, `appendix`, `footnote_block`, `bibliography`

**Semantic:** `definition_block`, `canon_block`, `thesis_block`,
`method_block`, `figure_caption`, `table_notes`

### 5.2 Edge Types

**Structural:**
`PARENT_CHILD` (canonical hierarchy),
`NEXT_SIBLING` (explicit sequencing when `ordinal` is insufficient)

**Semantic:**
`REFERS_TO`, `CITES`, `DEPENDS_ON`, `DEFINES`, `USES_TERM`,
`USES_CANON`, `CONTRADICTS` (advisory, from reconciliation)

**Derived:**
`SUMMARIZES`, `COMPRESSES_FROM`, `BUILT_FROM`

**Governance (optional):**
`GOVERNED_BY_INVARIANT_SET`, `STYLE_PROFILE_BINDING`

---

## 6. Version Semantics

### 6.1 Node-level (local)

Each node has an independent version counter.  Editing content produces a
new `NodeVersion`; `Node.current_version` and `current_content_hash`
advance.  A `BuildReceipt` is recorded.

### 6.2 Root-level (global)

| Counter | Bumps on | Purpose |
|---|---|---|
| `root_version` | Any active-state change (node tip, topology, invariants, derived tips) | User-visible "artifact has changed" |
| `manifest_version` | Topology / config changes only (nodes, edges, invariant bindings) | Structural integrity |

Ordinary node content edits bump `root_version` but **not**
`manifest_version`.

### 6.3 Snapshot hash

`current_snapshot_hash` is computed over a canonical serialization of:
artifact root metadata (excluding mutable timestamps), active node tip
hashes, active edges, active invariant version hash, active derived tips,
and ordering metadata.

### 6.4 Optimistic concurrency

Writes supply expected versions (`expected_root_version`,
`expected_manifest_version` for topology edits, `expected_node_version`
for node edits).  Mismatch rejects the write — same pattern as
`StaleArtifactVersionError` in the current store.

---

## 7. Build Receipt Model

### 7.1 Receipt kinds by pipeline phase

| Phase | `receipt_kind` | Scope |
|---|---|---|
| Spine | `spine_plan` | Creates/updates outline; often `node_id=None` |
| Node generation | `node_generate` | First draft of a node |
| Node revision | `node_revise` | Targeted rewrite |
| Reconciliation | `reconcile` | Cross-node consistency pass |
| Compression | `compress` | Produces `DerivedNode` outputs |
| Manual edit | `manual_edit` | Human edit with actor + reason |
| Invariant update | `invariant_update` | Freeze / supersede an invariant set |

### 7.2 Input normalization

For reproducibility: prompts are normalized before hashing, input lists are
sorted canonically, tool/model identifiers are explicit, invariant version
and style profile are always included when relevant.

### 7.3 Relationship to receipt_v1

`BuildReceipt` is a document-system analogue of `receipt_v1.Receipt`.
It shares the same principles (explicit inputs/outputs, hashes, actor,
chain linkage) but is scoped to artifact operations rather than tool calls.

Where a `receipt_v1.Receipt` records "tool X was called with args Y and
produced result Z," a `BuildReceipt` records "node N was generated from
inputs [parent context, invariant set v3, style profile P] and produced
content hash H."

---

## 8. Staleness Rules

### 8.1 Categories

- **`fresh`** — current with all inputs.
- **`soft_stale`** — indirect dependency changed; may still be valid.
- **`hard_stale`** — direct dependency changed; likely needs regeneration.

### 8.2 Triggers

**Node staleness:**

| Trigger | Severity |
|---|---|
| Referenced node changed (`DEPENDS_ON`, `USES_CANON`, etc.) | hard |
| Bound invariant definition / canon / prohibition changed | hard |
| Style profile changed in ways affecting rendering | hard |
| Terminology guidance changed (no explicit usage mapping) | soft |
| Parent node changed (if node depends on parent context) | soft |
| Reconciliation flagged contradiction | soft |

**Derived node staleness:** any source node version changed, compression
policy changed, or relevant invariant binding changed.

### 8.3 Propagation

- **Node content edit** — mark downstream dependents `soft_stale`; mark
  directly affected summaries `stale`.
- **Invariant change** — mark explicitly dependent nodes `hard_stale`.
- **Topology change** — mark affected subtree summaries `stale`.

### 8.4 Resolution options

Stale does not necessarily mean regenerate:

- **Revalidate** — confirm still valid, clear staleness without rewrite.
- **Regenerate** — produce new node version.
- **Waive** — acknowledge and dismiss with a manual receipt.

---

## 9. Partial Regeneration Contract

### 9.1 Scopes

`NODE_ONLY`, `SUBTREE`, `UPWARD_SUMMARIES`, `RECONCILE_ONLY`,
`FULL_ARTIFACT`

### 9.2 Request contract (conceptual)

A partial regeneration request specifies: target artifact + node/subtree,
scope, expected versions, invariant binding mode (active set vs pinned
version), style policy mode (inherited vs override), and whether to run
reconciliation and/or refresh derived nodes afterward.

### 9.3 Output contract

Returns: changed node IDs + new versions, generated `BuildReceipt` IDs,
updated `root_version`, updated staleness statuses, remaining stale
derived nodes, and conflict info if the operation could not apply cleanly.

### 9.4 Invariant safety

Partial regeneration must not silently violate invariant consistency.
Active invariant set version is always included in the build receipt.
Reconciliation can detect contradiction drift.  Caller can choose "strict"
mode to fail on unresolved invariant conflicts.

---

## 10. Pipeline Mapping

### 10.1 Spine pass (planning)

Outputs: skeleton nodes, structural edges, optional invariant drafts,
`spine_plan` receipt.

### 10.2 Node pass (generation / revision)

Inputs: target node, bounded parent/sibling context, active invariant set,
style profile.
Outputs: new `NodeVersion`, `node_generate` or `node_revise` receipt.

### 10.3 Reconciliation pass

Checks: terminology drift, canon contradictions, thesis inconsistency,
cross-reference integrity, sequence continuity.
Outputs: staleness flags, advisory `CONTRADICTS` edges, optional patch
suggestions, `reconcile` receipt.

### 10.4 Compression pass

Produces: `DerivedNode` outputs (abstracts, summaries, teasers),
`compress` receipts, explicit staleness/freshness for caches.

---

## 11. File Layout on Disk

### 11.1 Hierarchical mode

Extends the current `content/<artifact_id>/` pattern.

```text
<governor_dir>/.governor/artifacts/
  index.json                                # ArtifactIndex (flat artifacts)
  content/<artifact_id>/
    artifact.json                           # HierarchicalArtifact root object
    manifest.json                           # canonical topology snapshot (stable hash)

    nodes/<node_id>/
      node.json                             # Node object
      v1.md                                 # content body (immutable)
      v1.meta.json                          # NodeVersion metadata
      v2.md
      v2.meta.json

    edges/
      edges.json                            # atomic-replace (matches current index.json pattern)

    invariants/<invariant_set_id>/
      v1.json
      v2.json

    derived/<derived_node_id>/
      derived.json                          # DerivedNode object
      v1.md
      v1.meta.json

    receipts/
      <receipt_id>.json                     # BuildReceipt store
```

### 11.2 Storage notes

- **Edge storage:** the current codebase uses JSON files with atomic
  replace (`tempfile` + `os.replace`).  JSONL append is an option for
  high-frequency edge writes, but the default should match the existing
  pattern.  An `edges.jsonl` variant can be added later if append
  performance matters.
- **`manifest.json`** must be canonicalized for stable hashing.
- Receipts are stored in a flat directory per artifact; co-location under
  `nodes/` is not recommended.

---

## 12. Compatibility Modes

### 12.1 Mode A: `flat` (current)

Single blob content/version.  No hierarchy objects.  This is what ships
today.

### 12.2 Mode B: `hierarchical_shadow`

Flat blob remains authoritative for content.  A shadow manifest stores
structural metadata alongside it (see sec 13).  Used to collect hierarchy
metadata before full migration.

### 12.3 Mode C: `hierarchical`

Nodes, edges, invariants, and derived nodes are authoritative.  A flat
"whole artifact" blob may be materialized as a derived export.

### 12.4 Migration path (future, lazy)

When promoting flat -> hierarchical:

1. Create `HierarchicalArtifact` root.
2. Create root `Node` + `NodeVersion` from existing blob.
3. (Optional) Split by headings/anchors into child nodes.
4. Create `PARENT_CHILD` edges.
5. Create `InvariantSet` from existing continuity/style metadata.
6. Set `compatibility_mode = "hierarchical"`.
7. Preserve original flat blob hash in migration receipt.

No migration script is specified here.

---

## 13. Shadow Manifest (What the Flat Store Should Capture Now)

Even before hierarchical implementation, the flat store can capture
metadata that reduces migration pain later.

### 13.1 Schema

A flat artifact may optionally include a `hierarchy_shadow` field:

```python
class HierarchyShadow(BaseModel):
    schema_version: int = 1
    artifact_kind: str = ""                   # "fiction" | "paper" | etc.
    outline: list[dict[str, Any]] = Field(default_factory=list)  # ordered sections
    section_anchors: list[dict[str, str]] = Field(default_factory=list)  # id, title, marker
    invariant_bindings: dict[str, Any] = Field(default_factory=dict)
    style_profile: str | None = None
    derived_outputs: list[dict[str, Any]] = Field(default_factory=list)
    last_reconcile_at: str | None = None
    notes: str | None = None
    content_hash: str = ""                    # SHA-256 hex[:16], versioned with artifact
```

### 13.2 Purpose

This allows the flat system to start collecting structural intent,
invariant references, and summary provenance without committing to the
hierarchical store.  Shadow metadata is informational only — it does not
enable node-level edits.

### 13.3 Constraints

- Shadow manifest is metadata only.
- If shadow metadata becomes stale, it must be marked stale explicitly.
- Shadow manifest hash is included in the flat artifact's versioning.

---

## 14. Open Questions (Intentional Deferrals)

1. **Granularity defaults** — how aggressively to split imported flat
   content into nodes (headings only? paragraphs? scenes?).
2. **Edge indexing strategy** — JSON + cache may suffice; SQLite could be
   needed at scale.
3. **Reconciliation strictness** — advisory vs blocking for contradiction
   detection.
4. **Derived node retention** — keep all versions vs prune by policy.
5. **Cross-artifact references** — this spec is artifact-local;
   cross-artifact graph semantics are deferred.
6. **UI affordances** — node tree, staleness display, rebuild scopes,
   summary provenance views are out of scope.

---

## 15. Summary

This spec extends the flat Phosphor artifact store into a receipted,
node-scoped document graph for long fiction and long papers.

The key moves:

- **Hierarchy** for update locality.
- **Invariants** for frozen definitions / canon / style.
- **Build receipts** for generation provenance.
- **Derived nodes** for summaries as cached, receipted computations.
- **Staleness + partial regeneration** for practical maintenance at scale.

No implementation is proposed.  The purpose is to lock the data model and
vocabulary before real-world friction forces a rushed redesign.
