"""Tests for the visualizer module."""

import numpy as np
import pytest

from src.hand_tracker import FingertipPosition, HandData
from src.visualizer import draw_all_fingertips, draw_debug_info, draw_fingertip_box


class TestDrawFingertipBox:
    """Tests for draw_fingertip_box function."""

    def test_modifies_frame(self):
        """Test that drawing modifies the frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original_sum = frame.sum()
        draw_fingertip_box(frame, 320, 240)
        assert frame.sum() > original_sum

    def test_with_custom_color(self):
        """Test drawing with custom color."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_fingertip_box(frame, 320, 240, color=(0, 255, 0))
        # Check that green channel has values
        assert frame[:, :, 1].sum() > 0

    def test_with_custom_size(self):
        """Test drawing with custom size."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_fingertip_box(frame, 320, 240, half_size=50)
        # Larger box should have more non-zero pixels
        large_box_sum = frame.sum()

        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_fingertip_box(frame2, 320, 240, half_size=10)
        small_box_sum = frame2.sum()

        assert large_box_sum > small_box_sum


class TestDrawAllFingertips:
    """Tests for draw_all_fingertips function."""

    def test_empty_hands_list(self):
        """Test drawing with no hands detected."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original_sum = frame.sum()
        draw_all_fingertips(frame, [])
        assert frame.sum() == original_sum

    def test_draws_boxes_for_fingertips(self):
        """Test that boxes are drawn for each fingertip."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fingertips = [
            FingertipPosition(x=100, y=100, finger_name="Thumb", hand_label="Right"),
            FingertipPosition(x=200, y=100, finger_name="Index", hand_label="Right"),
        ]
        hands_data = [
            HandData(fingertips=fingertips, handedness="Right", all_landmarks=[])
        ]
        draw_all_fingertips(frame, hands_data)
        assert frame.sum() > 0

    def test_multiple_hands(self):
        """Test drawing boxes for multiple hands."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hands_data = [
            HandData(
                fingertips=[
                    FingertipPosition(x=100, y=100, finger_name="Index", hand_label="Right")
                ],
                handedness="Right",
                all_landmarks=[],
            ),
            HandData(
                fingertips=[
                    FingertipPosition(x=400, y=100, finger_name="Index", hand_label="Left")
                ],
                handedness="Left",
                all_landmarks=[],
            ),
        ]
        draw_all_fingertips(frame, hands_data)
        # Should have drawn 2 boxes
        assert frame.sum() > 0


class TestDrawDebugInfo:
    """Tests for draw_debug_info function."""

    def test_draws_text(self):
        """Test that debug info is drawn."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original_sum = frame.sum()
        draw_debug_info(frame, fps=30.0, num_hands=1)
        assert frame.sum() > original_sum

    def test_with_zero_values(self):
        """Test debug info with zero values."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_debug_info(frame, fps=0.0, num_hands=0)
        assert frame.sum() > 0
