"""Unit tests for temporal smoothing (majority vote, ratio window, hysteresis)."""

from __future__ import annotations

import pytest

from src.core.temporal import HysteresisFlag, RatioWindow, majority_label


def test_majority_label_picks_most_frequent() -> None:
    """Majority vote returns the most common label."""
    assert majority_label(["road", "road", "left"]) == "road"


def test_majority_label_empty_raises() -> None:
    """An empty window is an error."""
    with pytest.raises(ValueError):
        majority_label([])


def test_ratio_window_tracks_recent_fraction() -> None:
    """The window reports the True-fraction over the most recent entries."""
    window = RatioWindow(window_size=4)
    for value in (True, True, False, False):
        window.update(value)
    assert window.ratio() == 0.5
    window.update(True)  # evicts the oldest True -> [T, F, F, T]
    assert window.ratio() == 0.5


def test_ratio_window_ramps_in_from_empty() -> None:
    """An unfilled window divides by capacity, so one True is not a full ratio."""
    window = RatioWindow(window_size=4)
    assert window.update(True) == 0.25          # not 1.0: three slots unobserved
    assert window.filled() is False
    for _ in range(3):
        window.update(True)
    assert window.ratio() == 1.0
    assert window.filled() is True


def test_hysteresis_debounces_flag() -> None:
    """The flag turns on only above on_ratio and off only below off_ratio."""
    flag = HysteresisFlag(window_size=5, on_ratio=0.6, off_ratio=0.4)
    states = [flag.update(True) for _ in range(3)]
    assert states[-1] is True  # 3/5 = 0.6 reaches on_ratio
    # A couple of False frames keep it on (ratio stays above off_ratio).
    assert flag.update(False) is True
    for _ in range(4):
        flag.update(False)
    assert flag.state is False


def test_hysteresis_rejects_bad_thresholds() -> None:
    """Off ratio above on ratio is rejected."""
    with pytest.raises(ValueError):
        HysteresisFlag(window_size=5, on_ratio=0.3, off_ratio=0.6)
