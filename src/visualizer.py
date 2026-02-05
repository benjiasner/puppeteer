"""Visualization module for drawing on frames."""

import cv2
import numpy as np

from . import config
from .hand_tracker import FingertipPosition, HandData


def draw_fingertip_box(
    frame: np.ndarray,
    x: int,
    y: int,
    color: tuple[int, int, int] = config.BOX_COLOR,
    half_size: int = config.BOX_HALF_SIZE,
    thickness: int = config.BOX_THICKNESS,
) -> None:
    """
    Draw a rectangle centered on the given coordinates.

    Args:
        frame: Image to draw on (modified in place)
        x: Center x coordinate
        y: Center y coordinate
        color: BGR color tuple
        half_size: Distance from center to edge
        thickness: Line thickness
    """
    top_left = (x - half_size, y - half_size)
    bottom_right = (x + half_size, y + half_size)
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)


def draw_all_fingertips(
    frame: np.ndarray,
    hands_data: list[HandData],
    color: tuple[int, int, int] = config.BOX_COLOR,
    half_size: int = config.BOX_HALF_SIZE,
    thickness: int = config.BOX_THICKNESS,
) -> None:
    """
    Draw boxes around all detected fingertips.

    Args:
        frame: Image to draw on (modified in place)
        hands_data: List of HandData objects from hand tracker
        color: BGR color tuple
        half_size: Distance from center to edge
        thickness: Line thickness
    """
    for hand in hands_data:
        for fingertip in hand.fingertips:
            draw_fingertip_box(frame, fingertip.x, fingertip.y, color, half_size, thickness)


def draw_debug_info(
    frame: np.ndarray,
    fps: float,
    num_hands: int,
    color: tuple[int, int, int] = config.DEBUG_TEXT_COLOR,
    scale: float = config.DEBUG_TEXT_SCALE,
    thickness: int = config.DEBUG_TEXT_THICKNESS,
) -> None:
    """
    Draw debug information overlay on frame.

    Args:
        frame: Image to draw on (modified in place)
        fps: Current frames per second
        num_hands: Number of detected hands
        color: BGR color tuple for text
        scale: Font scale
        thickness: Text thickness
    """
    # FPS counter in top-left
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    # Hand count below FPS
    hands_text = f"Hands: {num_hands}"
    cv2.putText(frame, hands_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
