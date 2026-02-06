"""Snap gesture detector for thumb-index pinch + quick release."""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from . import config
from .hand_tracker import HandData


class SnapState(Enum):
    """Snap gesture detection states."""
    IDLE = "idle"
    PINCHING = "pinching"
    RELEASED = "released"


@dataclass
class FingerPosition:
    """Position with timestamp for velocity calculation."""
    x: float
    y: float
    time: float


class SnapGestureDetector:
    """
    Detects snap gesture (thumb-index pinch + quick release).

    Different from spread gesture which uses all fingers.
    This only tracks thumb and index finger.
    """

    def __init__(
        self,
        pinch_threshold: float = config.SNAP_PINCH_THRESHOLD,
        release_threshold: float = config.SNAP_RELEASE_THRESHOLD,
        release_speed: float = config.SNAP_RELEASE_SPEED,
        min_time: float = 0.05,
        max_time: float = 0.4,
    ):
        self.pinch_threshold = pinch_threshold
        self.release_threshold = release_threshold
        self.release_speed = release_speed
        self.min_time = min_time
        self.max_time = max_time

        # State tracking
        self._state = SnapState.IDLE
        self._pinch_start_time: Optional[float] = None
        self._last_thumb_pos: Optional[FingerPosition] = None
        self._last_index_pos: Optional[FingerPosition] = None

        # Adaptive thresholds based on hand size
        self._current_hand_size: float = 0.0

    def _estimate_hand_size(self, hand: HandData) -> float:
        """Estimate hand size from wrist to middle fingertip."""
        if len(hand.all_landmarks) >= 13:
            wrist = hand.all_landmarks[0]
            middle_tip = hand.all_landmarks[12]
            dx = middle_tip[0] - wrist[0]
            dy = middle_tip[1] - wrist[1]
            return math.sqrt(dx * dx + dy * dy)
        return config.SPREAD_DEFAULT_HAND_SIZE

    def _get_finger_positions(
        self, hand: HandData
    ) -> tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]:
        """Get thumb and index fingertip positions."""
        thumb_pos = None
        index_pos = None

        for ft in hand.fingertips:
            if ft.finger_name == "Thumb":
                thumb_pos = (ft.x, ft.y)
            elif ft.finger_name == "Index":
                index_pos = (ft.x, ft.y)

        return thumb_pos, index_pos

    def _distance(self, p1: tuple[int, int], p2: tuple[int, int]) -> float:
        """Calculate distance between two points."""
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def _calculate_release_speed(
        self,
        current_thumb: tuple[int, int],
        current_index: tuple[int, int],
        current_time: float,
    ) -> float:
        """Calculate the speed at which fingers are moving apart."""
        if self._last_thumb_pos is None or self._last_index_pos is None:
            return 0.0

        # Calculate previous and current distance between fingers
        prev_dist = self._distance(
            (self._last_thumb_pos.x, self._last_thumb_pos.y),
            (self._last_index_pos.x, self._last_index_pos.y),
        )
        curr_dist = self._distance(current_thumb, current_index)

        # Calculate time delta
        dt = current_time - self._last_thumb_pos.time
        if dt <= 0:
            return 0.0

        # Speed = change in distance / time
        return (curr_dist - prev_dist) / dt

    def process(
        self, hands_data: list[HandData], current_time: float
    ) -> bool:
        """
        Process hand data to detect snap gesture.

        Args:
            hands_data: List of HandData from hand tracker
            current_time: Current timestamp in seconds

        Returns:
            True if snap gesture was detected this frame
        """
        # Find right hand (primary interaction hand)
        right_hand = None
        for hand in hands_data:
            if hand.handedness == "Right":
                right_hand = hand
                break

        if right_hand is None:
            self._reset_state()
            return False

        # Get finger positions
        thumb_pos, index_pos = self._get_finger_positions(right_hand)
        if thumb_pos is None or index_pos is None:
            self._reset_state()
            return False

        # Estimate hand size for adaptive thresholds
        self._current_hand_size = self._estimate_hand_size(right_hand)

        # Scale thresholds by hand size (relative to default 200px)
        scale = self._current_hand_size / config.SPREAD_DEFAULT_HAND_SIZE
        pinch_thresh = self.pinch_threshold * scale
        release_thresh = self.release_threshold * scale
        release_speed_thresh = self.release_speed * scale

        # Calculate current distance between thumb and index
        distance = self._distance(thumb_pos, index_pos)

        # Calculate release speed
        release_speed = self._calculate_release_speed(
            thumb_pos, index_pos, current_time
        )

        # State machine
        snap_detected = False

        if self._state == SnapState.IDLE:
            # Check for pinch start
            if distance < pinch_thresh:
                self._state = SnapState.PINCHING
                self._pinch_start_time = current_time

        elif self._state == SnapState.PINCHING:
            # Check for release
            if distance > release_thresh and release_speed > release_speed_thresh:
                # Check timing
                if self._pinch_start_time is not None:
                    elapsed = current_time - self._pinch_start_time
                    if self.min_time <= elapsed <= self.max_time:
                        snap_detected = True

                self._state = SnapState.IDLE
                self._pinch_start_time = None

            elif distance > pinch_thresh * 1.5:
                # Pinch was released too slowly, reset
                if self._pinch_start_time is not None:
                    elapsed = current_time - self._pinch_start_time
                    if elapsed > self.max_time:
                        self._state = SnapState.IDLE
                        self._pinch_start_time = None

        # Update last positions
        self._last_thumb_pos = FingerPosition(thumb_pos[0], thumb_pos[1], current_time)
        self._last_index_pos = FingerPosition(index_pos[0], index_pos[1], current_time)

        return snap_detected

    def _reset_state(self):
        """Reset detection state."""
        self._state = SnapState.IDLE
        self._pinch_start_time = None
        self._last_thumb_pos = None
        self._last_index_pos = None

    def get_state_info(self) -> dict:
        """Get current detection state for debugging."""
        return {
            "state": self._state.value,
            "pinch_start_time": self._pinch_start_time,
            "hand_size": self._current_hand_size,
        }

    def is_pinching(self) -> bool:
        """Check if currently in pinch state."""
        return self._state == SnapState.PINCHING

    def reset(self):
        """Reset detector state."""
        self._reset_state()
