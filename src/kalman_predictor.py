"""Kalman filter for velocity estimation and position prediction.

This module provides a constant-velocity Kalman filter suitable for
tracking fingertip positions. It can:
- Estimate velocity from noisy position measurements
- Predict positions during brief occlusions (1-3 frames)
- Provide velocity for adaptive outlier thresholds
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class KalmanState:
    """State vector and covariance for 2D position tracking."""

    # State vector: [x, y, vx, vy]
    state: np.ndarray
    # 4x4 covariance matrix
    covariance: np.ndarray
    last_time: float


class KalmanPredictor:
    """
    Constant-velocity Kalman filter for 2D position tracking.

    State vector: [x, y, vx, vy] where (x, y) is position and (vx, vy) is velocity.

    This filter is used for:
    1. Velocity estimation - provides smooth velocity estimates
    2. Prediction during occlusion - can predict forward when no measurement available
    3. Outlier detection - predicted position helps identify impossible jumps

    Parameters:
        process_noise: Process noise variance (Q). Higher = more responsive to changes.
        measurement_noise: Measurement noise variance (R). Higher = trusts measurements less.
        initial_covariance: Initial state covariance. Higher = more uncertain initially.
    """

    def __init__(
        self,
        process_noise: float = 100.0,
        measurement_noise: float = 10.0,
        initial_covariance: float = 1000.0,
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initial_covariance = initial_covariance
        self._state: Optional[KalmanState] = None
        self._frames_without_measurement = 0
        self._max_prediction_frames = 3  # Max frames to predict without measurement

    def _make_transition_matrix(self, dt: float) -> np.ndarray:
        """
        Create state transition matrix F for time step dt.

        For constant velocity model:
        x' = x + vx * dt
        y' = y + vy * dt
        vx' = vx
        vy' = vy
        """
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    def _make_process_noise_matrix(self, dt: float) -> np.ndarray:
        """
        Create process noise matrix Q for time step dt.

        Uses discrete white noise model for constant velocity.
        """
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        q = self.process_noise
        return np.array([
            [dt4/4, 0, dt3/2, 0],
            [0, dt4/4, 0, dt3/2],
            [dt3/2, 0, dt2, 0],
            [0, dt3/2, 0, dt2],
        ]) * q

    def _make_measurement_matrix(self) -> np.ndarray:
        """
        Create measurement matrix H.

        We only measure position (x, y), not velocity.
        """
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

    def _make_measurement_noise_matrix(self) -> np.ndarray:
        """Create measurement noise matrix R."""
        r = self.measurement_noise
        return np.array([
            [r, 0],
            [0, r],
        ])

    def update(self, x: float, y: float, timestamp: float) -> tuple[float, float]:
        """
        Update the filter with a new measurement.

        Args:
            x: Measured X position
            y: Measured Y position
            timestamp: Current timestamp in seconds

        Returns:
            Tuple of (filtered_x, filtered_y)
        """
        self._frames_without_measurement = 0

        if self._state is None:
            # Initialize state on first measurement
            self._state = KalmanState(
                state=np.array([x, y, 0.0, 0.0]),
                covariance=np.eye(4) * self.initial_covariance,
                last_time=timestamp,
            )
            return (x, y)

        # Compute time delta
        dt = timestamp - self._state.last_time
        if dt <= 0:
            return (self._state.state[0], self._state.state[1])

        # Prediction step
        F = self._make_transition_matrix(dt)
        Q = self._make_process_noise_matrix(dt)

        predicted_state = F @ self._state.state
        predicted_covariance = F @ self._state.covariance @ F.T + Q

        # Update step
        H = self._make_measurement_matrix()
        R = self._make_measurement_noise_matrix()

        measurement = np.array([x, y])
        innovation = measurement - H @ predicted_state
        innovation_covariance = H @ predicted_covariance @ H.T + R

        # Kalman gain
        K = predicted_covariance @ H.T @ np.linalg.inv(innovation_covariance)

        # Update state and covariance
        updated_state = predicted_state + K @ innovation
        updated_covariance = (np.eye(4) - K @ H) @ predicted_covariance

        self._state = KalmanState(
            state=updated_state,
            covariance=updated_covariance,
            last_time=timestamp,
        )

        return (updated_state[0], updated_state[1])

    def predict(self, timestamp: float) -> Optional[tuple[float, float]]:
        """
        Predict position without a measurement (for occlusion handling).

        Args:
            timestamp: Current timestamp in seconds

        Returns:
            Predicted (x, y) position, or None if prediction not possible
        """
        if self._state is None:
            return None

        self._frames_without_measurement += 1
        if self._frames_without_measurement > self._max_prediction_frames:
            return None

        dt = timestamp - self._state.last_time
        if dt <= 0:
            return (self._state.state[0], self._state.state[1])

        # Prediction step only (no update)
        F = self._make_transition_matrix(dt)
        Q = self._make_process_noise_matrix(dt)

        predicted_state = F @ self._state.state
        predicted_covariance = F @ self._state.covariance @ F.T + Q

        # Update internal state with prediction
        self._state = KalmanState(
            state=predicted_state,
            covariance=predicted_covariance,
            last_time=timestamp,
        )

        return (predicted_state[0], predicted_state[1])

    def reset(self) -> None:
        """Reset filter state."""
        self._state = None
        self._frames_without_measurement = 0

    @property
    def is_initialized(self) -> bool:
        """Check if filter has been initialized."""
        return self._state is not None

    @property
    def velocity(self) -> tuple[float, float]:
        """Get estimated velocity (vx, vy) in pixels per second."""
        if self._state is None:
            return (0.0, 0.0)
        return (self._state.state[2], self._state.state[3])

    @property
    def speed(self) -> float:
        """Get estimated speed magnitude in pixels per second."""
        vx, vy = self.velocity
        return math.sqrt(vx * vx + vy * vy)

    @property
    def position(self) -> Optional[tuple[float, float]]:
        """Get current filtered position."""
        if self._state is None:
            return None
        return (self._state.state[0], self._state.state[1])

    @property
    def can_predict(self) -> bool:
        """Check if filter can still make predictions."""
        return (
            self._state is not None
            and self._frames_without_measurement < self._max_prediction_frames
        )

    def get_prediction_confidence(self) -> float:
        """
        Get confidence in current prediction (0.0 to 1.0).

        Confidence decreases with each frame of prediction without measurement.
        """
        if self._state is None:
            return 0.0
        if self._frames_without_measurement == 0:
            return 1.0
        # Linear decay over max prediction frames
        return max(
            0.0,
            1.0 - (self._frames_without_measurement / (self._max_prediction_frames + 1))
        )
