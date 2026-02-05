"""Temporal smoothing module using One Euro Filter and Kalman prediction.

This module provides advanced smoothing for hand tracking that:
- Uses One Euro Filter for adaptive smoothing (less lag during fast movements)
- Uses Kalman Filter for velocity estimation and occlusion prediction
- Applies adaptive outlier thresholds based on hand size and velocity
- Maintains per-landmark confidence weighting
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from . import config
from .hand_tracker import HandData, FingertipPosition
from .one_euro_filter import OneEuroFilter2D
from .kalman_predictor import KalmanPredictor


@dataclass
class FingertipState:
    """Tracking state for a single fingertip."""

    finger_name: str
    hand_label: str
    one_euro: OneEuroFilter2D
    kalman: KalmanPredictor
    last_raw_x: float = 0.0
    last_raw_y: float = 0.0
    last_filtered_x: float = 0.0
    last_filtered_y: float = 0.0
    confidence: float = 0.0


@dataclass
class HandState:
    """Tracks per-hand state across frames for temporal smoothing."""

    handedness: str
    fingertip_states: dict[str, FingertipState] = field(default_factory=dict)
    is_tracking: bool = False
    frames_without_detection: int = 0
    last_hand_confidence: float = 0.0
    # Hand size estimate (distance from wrist to middle fingertip)
    hand_size: float = 200.0  # Default estimate in pixels


class TemporalSmoother:
    """
    Applies temporal smoothing using One Euro Filter and Kalman prediction.

    Key features:
    - Adaptive smoothing: less smoothing during fast movements (One Euro Filter)
    - Velocity estimation: tracks fingertip velocity for prediction (Kalman)
    - Occlusion handling: predicts positions during brief occlusions
    - Adaptive outlier rejection: threshold scales with hand size and velocity
    """

    def __init__(
        self,
        min_cutoff: float = config.ONE_EURO_MIN_CUTOFF,
        beta: float = config.ONE_EURO_BETA,
        d_cutoff: float = config.ONE_EURO_D_CUTOFF,
        acquire_threshold: float = config.CONFIDENCE_ACQUIRE_THRESHOLD,
        lose_threshold: float = config.CONFIDENCE_LOSE_THRESHOLD,
        persistence_frames: int = config.PERSISTENCE_FRAMES,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.acquire_threshold = acquire_threshold
        self.lose_threshold = lose_threshold
        self.persistence_frames = persistence_frames

        # State for each hand (keyed by handedness: "Left" or "Right")
        self.hand_states: dict[str, HandState] = {}

    def _create_fingertip_state(self, finger_name: str, hand_label: str) -> FingertipState:
        """Create a new fingertip state with fresh filters."""
        return FingertipState(
            finger_name=finger_name,
            hand_label=hand_label,
            one_euro=OneEuroFilter2D(
                min_cutoff=self.min_cutoff,
                beta=self.beta,
                d_cutoff=self.d_cutoff,
            ),
            kalman=KalmanPredictor(
                process_noise=config.KALMAN_PROCESS_NOISE,
                measurement_noise=config.KALMAN_MEASUREMENT_NOISE,
                initial_covariance=config.KALMAN_INITIAL_COVARIANCE,
            ),
        )

    def _estimate_hand_size(self, hand_data: HandData) -> float:
        """
        Estimate hand size from landmarks.

        Uses distance from wrist (landmark 0) to middle fingertip (landmark 12).
        """
        if len(hand_data.all_landmarks) < 13:
            return 200.0  # Default estimate

        wrist = hand_data.all_landmarks[0]
        middle_tip = hand_data.all_landmarks[12]

        dx = middle_tip[0] - wrist[0]
        dy = middle_tip[1] - wrist[1]
        return math.sqrt(dx * dx + dy * dy)

    def _get_adaptive_outlier_threshold(
        self, hand_state: HandState, fingertip_state: FingertipState
    ) -> float:
        """
        Compute adaptive outlier threshold based on hand size and velocity.

        Larger hands at closer distances get larger thresholds.
        Fast-moving fingertips get larger thresholds.
        """
        base = config.BASE_OUTLIER_THRESHOLD

        # Scale by hand size (normalized to typical size of 200px)
        size_factor = hand_state.hand_size / 200.0

        # Scale by velocity
        speed = fingertip_state.kalman.speed
        velocity_factor = 1.0 + config.OUTLIER_VELOCITY_FACTOR * speed

        return base * size_factor * velocity_factor

    def _is_outlier(
        self,
        new_x: float,
        new_y: float,
        fingertip_state: FingertipState,
        hand_state: HandState,
    ) -> bool:
        """Check if a new position is an outlier based on adaptive threshold."""
        if not fingertip_state.kalman.is_initialized:
            return False

        # Get predicted position from Kalman filter
        predicted = fingertip_state.kalman.position
        if predicted is None:
            return False

        dx = new_x - predicted[0]
        dy = new_y - predicted[1]
        distance = math.sqrt(dx * dx + dy * dy)

        threshold = self._get_adaptive_outlier_threshold(hand_state, fingertip_state)
        return distance > threshold

    def process(self, hands_data: list[HandData], timestamp: float) -> list[HandData]:
        """
        Process raw hand data and return smoothed results.

        Args:
            hands_data: Raw hand detection data from HandTracker
            timestamp: Current timestamp in seconds

        Returns:
            Smoothed hand data with temporal filtering applied
        """
        if not config.SMOOTHING_ENABLED:
            return hands_data

        # Build a lookup of detected hands by handedness
        detected_hands: dict[str, HandData] = {}
        for hand in hands_data:
            detected_hands[hand.handedness] = hand

        # Update state for each potentially tracked hand
        result = []

        for handedness in ["Left", "Right"]:
            if handedness in detected_hands:
                # Hand is detected this frame
                hand = detected_hands[handedness]
                smoothed_hand = self._process_detected_hand(handedness, hand, timestamp)
                if smoothed_hand:
                    result.append(smoothed_hand)
            elif handedness in self.hand_states:
                # Hand was tracked but not detected this frame - try prediction
                smoothed_hand = self._process_missing_hand(handedness, timestamp)
                if smoothed_hand:
                    result.append(smoothed_hand)

        return result

    def _process_detected_hand(
        self, handedness: str, hand_data: HandData, timestamp: float
    ) -> Optional[HandData]:
        """Process a detected hand and return smoothed data."""
        # Initialize state if needed
        if handedness not in self.hand_states:
            self.hand_states[handedness] = HandState(handedness=handedness)

        state = self.hand_states[handedness]

        # Update hand size estimate
        state.hand_size = self._estimate_hand_size(hand_data)
        state.last_hand_confidence = hand_data.hand_confidence
        state.frames_without_detection = 0

        # Hysteresis for tracking state
        if not state.is_tracking:
            if hand_data.hand_confidence >= self.acquire_threshold:
                state.is_tracking = True
            else:
                return None
        elif hand_data.hand_confidence < self.lose_threshold:
            # Below lose threshold but still detected - keep tracking for now
            pass

        # Process each fingertip
        smoothed_fingertips = []

        for fingertip in hand_data.fingertips:
            finger_key = fingertip.finger_name

            # Initialize fingertip state if needed
            if finger_key not in state.fingertip_states:
                state.fingertip_states[finger_key] = self._create_fingertip_state(
                    fingertip.finger_name, fingertip.hand_label
                )

            ft_state = state.fingertip_states[finger_key]

            # Check for outliers
            if self._is_outlier(fingertip.x, fingertip.y, ft_state, state):
                # Outlier detected - use prediction instead or reset
                if ft_state.kalman.can_predict:
                    predicted = ft_state.kalman.predict(timestamp)
                    if predicted:
                        smoothed_fingertips.append(
                            FingertipPosition(
                                x=int(round(predicted[0])),
                                y=int(round(predicted[1])),
                                finger_name=fingertip.finger_name,
                                hand_label=fingertip.hand_label,
                                visibility=fingertip.visibility * 0.5,  # Reduced confidence
                                presence=fingertip.presence * 0.5,
                            )
                        )
                        continue
                else:
                    # Can't predict - reset filters and use raw
                    ft_state.one_euro.reset()
                    ft_state.kalman.reset()

            # Apply One Euro Filter for smooth tracking
            filtered_x, filtered_y = ft_state.one_euro.filter(
                fingertip.x, fingertip.y, timestamp
            )

            # Update Kalman filter for velocity estimation
            ft_state.kalman.update(filtered_x, filtered_y, timestamp)

            # Store state
            ft_state.last_raw_x = fingertip.x
            ft_state.last_raw_y = fingertip.y
            ft_state.last_filtered_x = filtered_x
            ft_state.last_filtered_y = filtered_y
            ft_state.confidence = fingertip.visibility * fingertip.presence

            smoothed_fingertips.append(
                FingertipPosition(
                    x=int(round(filtered_x)),
                    y=int(round(filtered_y)),
                    finger_name=fingertip.finger_name,
                    hand_label=fingertip.hand_label,
                    visibility=fingertip.visibility,
                    presence=fingertip.presence,
                )
            )

        if not smoothed_fingertips:
            return None

        return HandData(
            fingertips=smoothed_fingertips,
            handedness=handedness,
            all_landmarks=hand_data.all_landmarks,
            hand_confidence=hand_data.hand_confidence,
        )

    def _process_missing_hand(
        self, handedness: str, timestamp: float
    ) -> Optional[HandData]:
        """Process a hand that was not detected - use Kalman prediction."""
        state = self.hand_states[handedness]

        if not state.is_tracking:
            return None

        state.frames_without_detection += 1

        # Check if we've exceeded persistence frames
        if state.frames_without_detection > self.persistence_frames:
            state.is_tracking = False
            # Reset all fingertip filters
            for ft_state in state.fingertip_states.values():
                ft_state.one_euro.reset()
                ft_state.kalman.reset()
            return None

        # Try to predict positions using Kalman filter
        predicted_fingertips = []

        for finger_key, ft_state in state.fingertip_states.items():
            if not ft_state.kalman.can_predict:
                continue

            predicted = ft_state.kalman.predict(timestamp)
            if predicted is None:
                continue

            # Confidence decreases with prediction
            confidence = ft_state.kalman.get_prediction_confidence()

            predicted_fingertips.append(
                FingertipPosition(
                    x=int(round(predicted[0])),
                    y=int(round(predicted[1])),
                    finger_name=ft_state.finger_name,
                    hand_label=ft_state.hand_label,
                    visibility=confidence,
                    presence=confidence,
                )
            )

        if not predicted_fingertips:
            state.is_tracking = False
            return None

        return HandData(
            fingertips=predicted_fingertips,
            handedness=handedness,
            all_landmarks=[],  # No landmarks during prediction
            hand_confidence=state.last_hand_confidence * 0.5,
        )

    def reset(self) -> None:
        """Reset all tracking state."""
        self.hand_states.clear()
