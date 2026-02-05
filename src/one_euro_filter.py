"""One Euro Filter implementation for low-latency signal smoothing.

The One Euro Filter is designed specifically for human input tracking.
It adapts smoothing based on movement speed:
- Stationary input -> heavy smoothing -> eliminates jitter
- Fast movement -> light smoothing -> near-zero latency

Reference: Casiez, G., Roussel, N., & Vogel, D. (2012). 1€ Filter: A Simple
Speed-based Low-pass Filter for Noisy Input in Interactive Systems.
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class FilterState:
    """Internal state for a single One Euro Filter instance."""

    x: float  # Filtered value
    dx: float  # Filtered derivative
    last_time: float  # Timestamp of last update


class OneEuroFilter:
    """
    One Euro Filter for adaptive smoothing of noisy signals.

    The filter adapts its cutoff frequency based on the rate of change
    of the input signal. Fast movements get less smoothing (lower latency),
    while slow/stationary input gets heavy smoothing (less jitter).

    Parameters:
        min_cutoff: Minimum cutoff frequency in Hz. Lower = more smoothing
                   when stationary. Default 1.0 works well for hand tracking.
        beta: Speed coefficient. Higher = less lag during fast movements.
              0.0 = constant smoothing. Default 0.7 for hand tracking.
        d_cutoff: Cutoff frequency for the derivative filter. Default 1.0.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.7,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._state: Optional[FilterState] = None

    def _smoothing_factor(self, cutoff: float, dt: float) -> float:
        """
        Compute the exponential smoothing factor alpha.

        Args:
            cutoff: Cutoff frequency in Hz
            dt: Time delta in seconds

        Returns:
            Smoothing factor alpha in range (0, 1]
        """
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def _exponential_smoothing(self, alpha: float, x: float, x_prev: float) -> float:
        """Apply exponential smoothing."""
        return alpha * x + (1.0 - alpha) * x_prev

    def filter(self, x: float, timestamp: float) -> float:
        """
        Filter a single value.

        Args:
            x: Raw input value
            timestamp: Current timestamp in seconds

        Returns:
            Filtered value
        """
        if self._state is None:
            # Initialize on first sample
            self._state = FilterState(x=x, dx=0.0, last_time=timestamp)
            return x

        # Compute time delta
        dt = timestamp - self._state.last_time
        if dt <= 0:
            # Avoid division by zero or negative time
            return self._state.x

        # Estimate derivative (rate of change)
        dx = (x - self._state.x) / dt

        # Filter the derivative
        alpha_d = self._smoothing_factor(self.d_cutoff, dt)
        dx_filtered = self._exponential_smoothing(alpha_d, dx, self._state.dx)

        # Compute adaptive cutoff frequency based on filtered derivative
        # Higher speed = higher cutoff = less smoothing = lower latency
        cutoff = self.min_cutoff + self.beta * abs(dx_filtered)

        # Filter the value
        alpha = self._smoothing_factor(cutoff, dt)
        x_filtered = self._exponential_smoothing(alpha, x, self._state.x)

        # Update state
        self._state = FilterState(x=x_filtered, dx=dx_filtered, last_time=timestamp)

        return x_filtered

    def reset(self) -> None:
        """Reset filter state."""
        self._state = None

    @property
    def is_initialized(self) -> bool:
        """Check if filter has been initialized with at least one sample."""
        return self._state is not None

    @property
    def last_derivative(self) -> float:
        """Get the last filtered derivative (velocity estimate)."""
        if self._state is None:
            return 0.0
        return self._state.dx


class OneEuroFilter2D:
    """
    Two-dimensional One Euro Filter for tracking 2D positions.

    Wraps two independent OneEuroFilter instances for X and Y coordinates.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.7,
        d_cutoff: float = 1.0,
    ):
        self.filter_x = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.filter_y = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def filter(self, x: float, y: float, timestamp: float) -> tuple[float, float]:
        """
        Filter a 2D position.

        Args:
            x: Raw X coordinate
            y: Raw Y coordinate
            timestamp: Current timestamp in seconds

        Returns:
            Tuple of (filtered_x, filtered_y)
        """
        return (
            self.filter_x.filter(x, timestamp),
            self.filter_y.filter(y, timestamp),
        )

    def reset(self) -> None:
        """Reset both filter states."""
        self.filter_x.reset()
        self.filter_y.reset()

    @property
    def is_initialized(self) -> bool:
        """Check if filters have been initialized."""
        return self.filter_x.is_initialized and self.filter_y.is_initialized

    @property
    def velocity(self) -> tuple[float, float]:
        """Get the velocity estimate (dx, dy) in units per second."""
        return (self.filter_x.last_derivative, self.filter_y.last_derivative)
