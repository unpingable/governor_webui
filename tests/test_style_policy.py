# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the style_policy pure module."""

import pytest

from gov_webui.style_policy import (
    PROFILES,
    StyleCorrection,
    action_for_mode,
    check,
    corrections_to_dicts,
    fix,
    profile_for_mode,
)


# ============================================================================
# profile_for_mode / action_for_mode
# ============================================================================


def test_profile_for_mode_fiction():
    assert profile_for_mode("fiction") == "fiction_typography_v1"


def test_profile_for_mode_research():
    assert profile_for_mode("research") == "research_typography_v1"


def test_profile_for_mode_code_returns_none():
    assert profile_for_mode("code") is None


def test_profile_for_mode_general_returns_none():
    assert profile_for_mode("general") is None


def test_action_for_mode_fiction():
    assert action_for_mode("fiction") == "fix"


def test_action_for_mode_research():
    assert action_for_mode("research") == "warn"


def test_action_for_mode_code_returns_none():
    assert action_for_mode("code") is None


# ============================================================================
# em_dash rule
# ============================================================================


def test_em_dash_basic():
    content = "hello -- world"
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert fixed == "hello \u2014 world"
    assert len(corrections) == 1
    assert corrections[0].rule == "em_dash"
    assert corrections[0].original == "--"
    assert corrections[0].replacement == "\u2014"


def test_em_dash_no_mangle_hr():
    """Triple-dash markdown horizontal rule must not be mangled."""
    content = "---"
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert fixed == "---"
    assert not corrections


def test_em_dash_no_mangle_quadruple():
    """Four dashes must not be mangled (frontmatter delimiter)."""
    content = "----"
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert fixed == "----"
    assert not corrections


def test_em_dash_triple_dash_in_text():
    """Triple-dash in flowing text stays unchanged."""
    content = "wait---really"
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert fixed == "wait---really"
    assert not corrections


def test_em_dash_multiple():
    content = "one -- two -- three"
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert fixed == "one \u2014 two \u2014 three"
    assert len(corrections) == 2


# ============================================================================
# ellipsis rule
# ============================================================================


def test_ellipsis():
    content = "wait..."
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert fixed == "wait\u2026"
    assert len(corrections) == 1
    assert corrections[0].rule == "ellipsis"


def test_ellipsis_mid_sentence():
    content = "I thought... maybe"
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert fixed == "I thought\u2026 maybe"
    assert len(corrections) == 1


# ============================================================================
# space_before_punct rule
# ============================================================================


def test_space_before_punct_comma():
    content = "hello ,world"
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert fixed == "hello,world"
    assert len(corrections) == 1
    assert corrections[0].rule == "space_before_punct"


def test_space_before_punct_period():
    content = "done ."
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert fixed == "done."
    assert len(corrections) == 1


def test_space_before_punct_semicolon():
    content = "clause ;next"
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert fixed == "clause;next"


def test_space_before_punct_colon():
    content = "label :value"
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert fixed == "label:value"


# ============================================================================
# multi_space rule
# ============================================================================


def test_multi_space():
    content = "hello  world"
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert fixed == "hello world"
    assert len(corrections) == 1
    assert corrections[0].rule == "multi_space"


def test_multi_space_three_spaces():
    content = "hello   world"
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert fixed == "hello world"


def test_multi_space_preserves_indent():
    """Leading spaces (indentation) must not be collapsed."""
    content = "  hello"
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert fixed == "  hello"
    assert not corrections


def test_multi_space_preserves_indent_four():
    content = "    code line"
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert fixed == "    code line"
    assert not corrections


# ============================================================================
# Protected spans
# ============================================================================


def test_protected_code_fence():
    content = "before\n```\nhello -- world\n```\nafter -- done"
    fixed, corrections = fix(content, "fiction_typography_v1")
    # Only the outside -- should be fixed
    assert "hello -- world" in fixed
    assert "after \u2014 done" in fixed
    assert len(corrections) == 1


def test_protected_inline_code():
    content = "Use `foo -- bar` for dashes"
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert "`foo -- bar`" in fixed
    assert not corrections  # no non-protected violations


def test_protected_inline_code_with_surrounding():
    content = "Say -- `foo -- bar` -- done"
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert "`foo -- bar`" in fixed  # protected
    assert "\u2014" in fixed  # surrounding fixed
    assert len(corrections) == 2


def test_protected_url():
    content = "Visit https://example.com/foo--bar for info"
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert "https://example.com/foo--bar" in fixed
    assert not corrections


def test_protected_frontmatter():
    content = "---\ntitle: foo--bar\n---\nHello -- world"
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert "title: foo--bar" in fixed  # frontmatter protected
    assert "Hello \u2014 world" in fixed  # body fixed
    assert len(corrections) == 1


def test_protected_frontmatter_only_at_start():
    """--- block not at file start is NOT frontmatter."""
    content = "some text\n---\ntitle: foo--bar\n---\n"
    fixed, corrections = fix(content, "fiction_typography_v1")
    # The -- in foo--bar is inside --- which is an HR, not frontmatter
    # But foo--bar has no bare -- (it's surrounded by non-dash), so it matches
    assert len([c for c in corrections if c.rule == "em_dash"]) == 1


# ============================================================================
# check() vs fix()
# ============================================================================


def test_check_returns_without_modifying():
    content = "hello -- world..."
    corrections = check(content, "fiction_typography_v1")
    assert len(corrections) == 2
    # Content is never modified by check()
    rules = {c.rule for c in corrections}
    assert "em_dash" in rules
    assert "ellipsis" in rules


def test_fix_returns_corrections():
    content = "hello -- world..."
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert fixed == "hello \u2014 world\u2026"
    assert len(corrections) == 2


# ============================================================================
# No profile / unknown mode
# ============================================================================


def test_no_profile_returns_empty():
    corrections = check("hello -- world", "nonexistent_profile")
    assert corrections == []


def test_fix_no_profile_returns_unchanged():
    content = "hello -- world"
    fixed, corrections = fix(content, "nonexistent_profile")
    assert fixed == content
    assert corrections == []


# ============================================================================
# Combined rules
# ============================================================================


def test_combined_rules():
    content = "He said -- wait...  really ,folks"
    fixed, corrections = fix(content, "fiction_typography_v1")
    assert "\u2014" in fixed  # em dash
    assert "\u2026" in fixed  # ellipsis
    assert "  " not in fixed.replace("  ", "")  # multi_space may appear
    assert " ,folks" not in fixed  # space_before_punct
    assert len(corrections) >= 3


# ============================================================================
# Idempotence
# ============================================================================


def test_idempotence():
    """fix(fix(content)) == fix(content) for all rule combinations."""
    samples = [
        "hello -- world... done ,really  now",
        "wait -- no...  stop ,please",
        "---\ntitle: test\n---\nfoo -- bar...",
        "some `code -- here` and -- outside",
        "A -- B -- C...  D ,E ;F :G",
    ]
    for profile in PROFILES:
        for sample in samples:
            first_pass, _ = fix(sample, profile)
            second_pass, corrections = fix(first_pass, profile)
            assert second_pass == first_pass, (
                f"Not idempotent for profile={profile}: "
                f"{sample!r} -> {first_pass!r} -> {second_pass!r}"
            )
            assert corrections == [], (
                f"Second pass found corrections for profile={profile}: {corrections}"
            )


# ============================================================================
# corrections_to_dicts
# ============================================================================


def test_corrections_to_dicts():
    corrections = [
        StyleCorrection(line=1, col=6, rule="em_dash", original="--", replacement="\u2014"),
        StyleCorrection(line=1, col=17, rule="ellipsis", original="...", replacement="\u2026"),
    ]
    dicts = corrections_to_dicts(corrections)
    assert len(dicts) == 2
    assert dicts[0]["rule"] == "em_dash"
    assert dicts[0]["line"] == 1
    assert dicts[0]["col"] == 6
    assert dicts[1]["rule"] == "ellipsis"


# ============================================================================
# Correction line/col accuracy
# ============================================================================


def test_correction_line_col():
    content = "line one\nfoo -- bar"
    corrections = check(content, "fiction_typography_v1")
    em = [c for c in corrections if c.rule == "em_dash"]
    assert len(em) == 1
    assert em[0].line == 2
    assert em[0].col == 4  # "foo " = 4 chars, -- starts at col 4
