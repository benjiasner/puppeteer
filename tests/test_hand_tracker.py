"""Tests for the hand tracker module."""

import numpy as np
import pytest

from src.hand_tracker import FingertipPosition, HandData, HandTracker


class TestFingertipPosition:
    """Tests for FingertipPosition dataclass."""

    def test_creation(self):
        """Test creating a FingertipPosition."""
        fingertip = FingertipPosition(x=100, y=200, finger_name="Index", hand_label="Right")
        assert fingertip.x == 100
        assert fingertip.y == 200
        assert fingertip.finger_name == "Index"
        assert fingertip.hand_label == "Right"


class TestHandData:
    """Tests for HandData dataclass."""

    def test_creation(self):
        """Test creating HandData with fingertips."""
        fingertips = [
            FingertipPosition(x=100, y=100, finger_name="Thumb", hand_label="Right"),
            FingertipPosition(x=200, y=100, finger_name="Index", hand_label="Right"),
        ]
        hand = HandData(
            fingertips=fingertips,
            handedness="Right",
            all_landmarks=[(50, 50), (100, 100)],
        )
        assert len(hand.fingertips) == 2
        assert hand.handedness == "Right"
        assert len(hand.all_landmarks) == 2


class TestHandTracker:
    """Tests for HandTracker class."""

    def test_initialization(self):
        """Test HandTracker initializes without errors."""
        tracker = HandTracker()
        assert tracker.landmarker is not None
        tracker.close()

    def test_initialization_with_custom_params(self):
        """Test HandTracker with custom parameters."""
        tracker = HandTracker(
            max_num_hands=1,
            detection_confidence=0.8,
            tracking_confidence=0.6,
        )
        assert tracker.landmarker is not None
        tracker.close()

    def test_process_empty_frame(self):
        """Test processing a frame with no hands."""
        tracker = HandTracker()
        # Create a blank black frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hands_data = tracker.process_frame(frame)
        assert isinstance(hands_data, list)
        assert len(hands_data) == 0
        tracker.close()

    def test_process_frame_returns_list(self):
        """Test that process_frame always returns a list."""
        tracker = HandTracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tracker.process_frame(frame)
        assert isinstance(result, list)
        tracker.close()
