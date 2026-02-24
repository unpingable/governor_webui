# SPDX-License-Identifier: Apache-2.0
"""
Deterministic typography normalization — pure module, no external deps.

V1 scope: em dash, ellipsis, space-before-punctuation, multi-space collapse.
Protected spans: code fences, inline code, URLs, YAML frontmatter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern


# ============================================================================
# Data structures
# ============================================================================


@dataclass(frozen=True, slots=True)
class StyleCorrection:
    line: int  # 1-based
    col: int  # 0-based
    rule: str  # "em_dash", "ellipsis", etc.
    original: str
    replacement: str


# ============================================================================
# Profiles & mode mappings
# ============================================================================

PROFILES: dict[str, list[str]] = {
    "fiction_typography_v1": [
        "em_dash",
        "ellipsis",
        "space_before_punct",
        "multi_space",
    ],
    "research_typography_v1": [
        "em_dash",
        "ellipsis",
        "space_before_punct",
        "multi_space",
    ],
}

MODE_PROFILES: dict[str, str] = {
    "fiction": "fiction_typography_v1",
    "research": "research_typography_v1",
}

MODE_ACTIONS: dict[str, str] = {
    "fiction": "fix",
    "research": "warn",
}


# ============================================================================
# Rules
# ============================================================================

_RULES: list[tuple[str, Pattern[str], str]] = [
    ("em_dash", re.compile(r"(?<!-)--(?!-)"), "\u2014"),
    ("ellipsis", re.compile(r"\.\.\."), "\u2026"),
    ("space_before_punct", re.compile(r" +([,\.;:])"), r"\1"),
    ("multi_space", re.compile(r"(?<=\S) {2,}(?=\S)"), " "),
]

_RULE_MAP: dict[str, tuple[Pattern[str], str]] = {
    name: (pat, repl) for name, pat, repl in _RULES
}


# ============================================================================
# Protected spans
# ============================================================================

_CODE_FENCE = re.compile(r"^```[^\n]*\n.*?^```", re.MULTILINE | re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_URL = re.compile(r"https?://[^\s)>\]]+")
_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


def _protected_spans(content: str) -> list[tuple[int, int]]:
    """Return sorted, merged list of (start, end) spans to skip."""
    spans: list[tuple[int, int]] = []
    for pat in (_CODE_FENCE, _INLINE_CODE, _URL, _FRONTMATTER):
        for m in pat.finditer(content):
            spans.append((m.start(), m.end()))
    # Sort and merge overlapping
    if not spans:
        return []
    spans.sort()
    merged: list[tuple[int, int]] = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _in_protected(offset: int, spans: list[tuple[int, int]]) -> bool:
    """Check if offset falls inside any protected span."""
    for s, e in spans:
        if s <= offset < e:
            return True
        if s > offset:
            break
    return False


# ============================================================================
# Line/col helper
# ============================================================================


def _offset_to_line_col(content: str, offset: int) -> tuple[int, int]:
    """Convert a string offset to (1-based line, 0-based col)."""
    line = content[:offset].count("\n") + 1
    last_nl = content.rfind("\n", 0, offset)
    col = offset if last_nl == -1 else offset - last_nl - 1
    return line, col


# ============================================================================
# Public API
# ============================================================================


def profile_for_mode(mode: str) -> str | None:
    """Return the style profile name for a governor mode, or None."""
    return MODE_PROFILES.get(mode)


def action_for_mode(mode: str) -> str | None:
    """Return 'fix', 'warn', or None for a governor mode."""
    return MODE_ACTIONS.get(mode)


def check(content: str, profile: str) -> list[StyleCorrection]:
    """Find style violations without modifying content."""
    rule_names = PROFILES.get(profile)
    if not rule_names:
        return []
    protected = _protected_spans(content)
    corrections: list[StyleCorrection] = []
    for name in rule_names:
        pat, repl = _RULE_MAP[name]
        for m in pat.finditer(content):
            if _in_protected(m.start(), protected):
                continue
            line, col = _offset_to_line_col(content, m.start())
            # Compute actual replacement text (handles backrefs)
            actual_repl = m.expand(repl)
            corrections.append(StyleCorrection(
                line=line, col=col, rule=name,
                original=m.group(), replacement=actual_repl,
            ))
    corrections.sort(key=lambda c: (c.line, c.col))
    return corrections


def fix(content: str, profile: str) -> tuple[str, list[StyleCorrection]]:
    """Apply fixes and return (fixed_content, corrections_applied)."""
    rule_names = PROFILES.get(profile)
    if not rule_names:
        return content, []
    protected = _protected_spans(content)
    # Collect all matches with their offsets
    matches: list[tuple[int, int, str, str, str]] = []  # (start, end, rule, orig, repl)
    for name in rule_names:
        pat, repl = _RULE_MAP[name]
        for m in pat.finditer(content):
            if _in_protected(m.start(), protected):
                continue
            actual_repl = m.expand(repl)
            matches.append((m.start(), m.end(), name, m.group(), actual_repl))
    if not matches:
        return content, []
    # Sort by offset descending so replacements don't shift earlier offsets
    matches.sort(key=lambda t: t[0], reverse=True)
    # Build corrections in document order (before applying)
    corrections: list[StyleCorrection] = []
    for start, _end, rule, orig, repl in sorted(matches, key=lambda t: t[0]):
        line, col = _offset_to_line_col(content, start)
        corrections.append(StyleCorrection(
            line=line, col=col, rule=rule,
            original=orig, replacement=repl,
        ))
    # Apply in reverse offset order
    result = content
    for start, end, _rule, _orig, repl in matches:
        result = result[:start] + repl + result[end:]
    return result, corrections


def corrections_to_dicts(corrections: list[StyleCorrection]) -> list[dict]:
    """Serialize corrections for JSON response."""
    return [
        {
            "line": c.line,
            "col": c.col,
            "rule": c.rule,
            "original": c.original,
            "replacement": c.replacement,
        }
        for c in corrections
    ]
