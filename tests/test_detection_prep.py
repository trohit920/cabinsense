"""Tests for the detection-dataset merge logic (alias resolution + label remap).

Pure, deterministic, no I/O. Covers case-insensitive alias matching, source->unified
id mapping, and label-line remapping that drops unmapped classes (so negatives such as
the seatbelt set's ``person-noseatbelt`` become background).
"""

from __future__ import annotations

from src.data.detection_prep import (
    build_source_id_map,
    remap_label_lines,
    resolve_alias,
)

ALIASES = {
    "smoking": ("cigarette", "vape", "smoking"),
    "seatbelt": ("seatbelt", "person-seatbelt"),
    "cap": ("cap", "hat"),
}
UNIFIED_INDEX = {"smoking": 0, "seatbelt": 1, "cap": 2}


def test_resolve_alias_is_case_insensitive() -> None:
    """Source names resolve regardless of case."""
    assert resolve_alias("Cigarette", ALIASES) == "smoking"
    assert resolve_alias("VAPE", ALIASES) == "smoking"
    assert resolve_alias("person-seatbelt", ALIASES) == "seatbelt"


def test_resolve_alias_unknown_returns_none() -> None:
    """An unmapped source class (e.g. the seatbelt negative) resolves to None."""
    assert resolve_alias("person-noseatbelt", ALIASES) is None


def test_build_source_id_map_smoking_two_to_one() -> None:
    """Two source classes (Cigarette, Vape) both map to the single smoking id."""
    src_names = {0: "Cigarette", 1: "Vape"}
    assert build_source_id_map(src_names, ALIASES, UNIFIED_INDEX) == {0: 0, 1: 0}


def test_build_source_id_map_drops_negative() -> None:
    """The seatbelt set keeps person-seatbelt and drops person-noseatbelt."""
    src_names = {0: "person-noseatbelt", 1: "person-seatbelt"}
    assert build_source_id_map(src_names, ALIASES, UNIFIED_INDEX) == {1: 1}


def test_remap_label_lines_rewrites_and_drops() -> None:
    """Mapped boxes are rewritten to unified ids; unmapped boxes are dropped."""
    lines = [
        "0 0.5 0.5 0.1 0.2",   # source person-noseatbelt -> dropped
        "1 0.4 0.4 0.2 0.3",   # source person-seatbelt -> unified id 1
    ]
    id_map = {1: 1}
    assert remap_label_lines(lines, id_map) == ["1 0.4 0.4 0.2 0.3"]


def test_remap_label_lines_ignores_malformed() -> None:
    """Lines without exactly five fields are skipped."""
    assert remap_label_lines(["1 0.4 0.4", "garbage"], {1: 1}) == []
