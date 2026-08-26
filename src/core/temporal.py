"""Temporal smoothing utilities (pure, C++-portable, stateful).

Turns noisy per-frame predictions into stable outputs: majority voting over a
window for categorical states, hysteresis for boolean flags (so a flag does not
flicker), and a ratio window for PERCLOS-style fractions. Plain deque/counter
logic that maps directly to a C++ ring buffer.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Sequence


def majority_label(labels: Sequence[str]) -> str:
    """Most frequent label in a window (ties resolved by first occurrence)."""
    if not labels:
        raise ValueError("labels must be non-empty")
    counts = Counter(labels)
    best = max(counts.values())
    for label in labels:  # preserve first-seen order on ties
        if counts[label] == best:
            return label
    return labels[0]


class RatioWindow:
    """Sliding window tracking the fraction of True values (e.g. eyes closed).

    The fraction is taken over the window's full capacity, not over the samples seen
    so far. PERCLOS means "the share of the last N seconds spent with eyes closed";
    if only one second has been observed, the other N-1 have not been observed closed.
    Dividing by the sample count instead makes the very first True frame report a
    ratio of 1.0, which latches every downstream flag on the first frame of a stream.
    Counting unfilled slots as False ramps the ratio in over the window instead, which
    errs toward "not yet triggered" - the safe direction for an alerting system.
    """

    def __init__(self, window_size: int) -> None:
        """Create a window holding the most recent ``window_size`` booleans."""
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self._capacity = window_size
        self._buffer: deque[bool] = deque(maxlen=window_size)

    def update(self, value: bool) -> float:
        """Append a value and return the current True-fraction."""
        self._buffer.append(bool(value))
        return self.ratio()

    def ratio(self) -> float:
        """Fraction of the window that is True, counting unfilled slots as False."""
        return sum(self._buffer) / self._capacity

    def filled(self) -> bool:
        """True once the window holds a full set of observations."""
        return len(self._buffer) == self._capacity


class HysteresisFlag:
    """Boolean flag with separate on/off thresholds to prevent flicker."""

    def __init__(self, window_size: int, on_ratio: float, off_ratio: float) -> None:
        """Configure the window and the on/off switching fractions."""
        if not 0.0 <= off_ratio <= on_ratio <= 1.0:
            raise ValueError("require 0 <= off_ratio <= on_ratio <= 1")
        self._window = RatioWindow(window_size)
        self._on_ratio = on_ratio
        self._off_ratio = off_ratio
        self._state = False

    def update(self, active: bool) -> bool:
        """Feed a per-frame observation and return the debounced flag state."""
        ratio = self._window.update(active)
        if not self._state and ratio >= self._on_ratio:
            self._state = True
        elif self._state and ratio <= self._off_ratio:
            self._state = False
        return self._state

    @property
    def state(self) -> bool:
        """Current debounced flag state."""
        return self._state
