#
# time_tools.py: timing utilities
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements simple timing utility classes.
#

import time, threading
from typing import Optional, Tuple


class Timer:
    """Simple timer class."""

    def __init__(self):
        """Constructor. Records start time."""
        self._start_time = time.time_ns()

    def __call__(self) -> float:
        """Get elapsed time since timer creation.

        Returns:
            Time elapsed in seconds since object construction.
        """
        return (time.time_ns() - self._start_time) * 1e-9


class Watchdog:
    """Monitors activity rate and timing using tick events and a filtered TPS estimate.

    Tracks the frequency of `tick()` calls and the time since the last one. The `check()` method
    evaluates whether the activity is recent enough and meets a minimum TPS (ticks per second) threshold,
    using a single-pole low-pass filter to smooth TPS estimation.
    """

    def __init__(self, time_limit: float, tps_threshold: float, smoothing: float = 0.9):
        """Initializes the Watchdog.

        Args:
            time_limit (float): Maximum allowed time (in seconds) since the last tick.
            tps_threshold (float): Minimum required filtered ticks per second.
            smoothing (float): Smoothing factor for the low-pass filter (0 < smoothing < 1).
        """

        self._time_limit = time_limit
        self._tps_threshold = tps_threshold
        self._smoothing = smoothing
        self._last_tick: Optional[float] = None
        self._average_tick = -1.0
        self._lock = threading.Lock()

    def tick(self):
        """Records the current timestamp and updates the filtered TPS estimate.

        Should be called regularly to track system activity. Uses the time between ticks to calculate
        instantaneous TPS and applies a low-pass filter to smooth the estimate.
        """

        with self._lock:
            now = time.time()
            if self._last_tick is not None:
                dt = now - self._last_tick
                self._average_tick = (
                    dt
                    if self._average_tick < 0
                    else (
                        self._smoothing * self._average_tick
                        + (1 - self._smoothing) * dt
                    )
                )
            self._last_tick = now

    def check(self) -> Tuple[bool, float]:
        """Checks whether the watchdog is within the allowed timing and TPS threshold.

        Returns:
            Tuple[bool, float]: A tuple containing:
                - bool: True if the watchdog is active (recent enough and meets TPS threshold), False otherwise.
                - float: The current TPS value.

        """
        with self._lock:
            if self._last_tick is None:
                return True, 0  # No ticks yet, consider it active
            age = time.time() - self._last_tick
            tps = 1 / self._average_tick if self._average_tick > 0 else 0
            return age <= self._time_limit and tps >= self._tps_threshold, tps
