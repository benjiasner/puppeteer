"""Visualization module for drawing on frames."""

from typing import TYPE_CHECKING

import cv2
import numpy as np

from . import config
from .hand_tracker import FingertipPosition, HandData

if TYPE_CHECKING:
    from .hand_commands import CommandLogEntry


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

def draw_fingertip_dot(
    frame: np.ndarray,
    x: int,
    y: int,
    color: tuple[int, int, int] = config.BOX_COLOR,
    radius: int = 5,  # Adjust this value to make dots bigger/smaller
) -> None:
    """
    Draw a solid dot (filled circle) at the given coordinates.

    Args:
        frame: Image to draw on (modified in place)
        x: Center x coordinate
        y: Center y coordinate
        color: BGR color tuple
        radius: Radius of the dot in pixels
    """
    # Thickness -1 fills the circle to make it a solid dot
    cv2.circle(frame, (x, y), radius, color, thickness=-1)

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
            #draw_fingertip_box(frame, fingertip.x, fingertip.y, color, half_size, thickness)
            draw_fingertip_dot(frame, fingertip.x, fingertip.y, color, 3*thickness)


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


def draw_command_log(
    frame: np.ndarray,
    log_entries: list["CommandLogEntry"],
    panel_width: int = config.LOG_PANEL_WIDTH,
    bg_color: tuple[int, int, int] = config.LOG_PANEL_BG_COLOR,
    text_color: tuple[int, int, int] = config.LOG_PANEL_TEXT_COLOR,
    highlight_color: tuple[int, int, int] = config.LOG_PANEL_HIGHLIGHT_COLOR,
) -> np.ndarray:
    """
    Draw a command log panel on the right side of the frame.

    Args:
        frame: Image to draw on
        log_entries: List of CommandLogEntry objects to display
        panel_width: Width of the log panel
        bg_color: BGR color for panel background
        text_color: BGR color for normal text
        highlight_color: BGR color for recent entries

    Returns:
        New frame with panel added (wider than original)
    """
    frame_height, frame_width = frame.shape[:2]

    # Create a new wider frame with log panel
    new_width = frame_width + panel_width
    new_frame = np.zeros((frame_height, new_width, 3), dtype=np.uint8)

    # Copy original frame
    new_frame[:, :frame_width] = frame

    # Fill log panel with background
    new_frame[:, frame_width:] = bg_color

    # Draw panel header
    header_text = "Command Log"
    cv2.putText(
        new_frame,
        header_text,
        (frame_width + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    # Draw separator line
    cv2.line(
        new_frame,
        (frame_width + 5, 45),
        (new_width - 5, 45),
        (100, 100, 100),
        1,
    )

    # Draw log entries (most recent at top)
    y_offset = 70
    line_height = 25
    max_entries_shown = (frame_height - 70) // line_height

    # Get entries in reverse order (most recent first)
    entries_to_show = list(reversed(log_entries))[:max_entries_shown]

    import time

    current_time = time.time()

    for i, entry in enumerate(entries_to_show):
        # Highlight recent entries (within last 2 seconds)
        age = current_time - entry.timestamp
        if age < 2.0:
            color = highlight_color
        else:
            color = text_color

        # Format: "HH:MM:SS - action"
        text = f"{entry.formatted_time} - {entry.action}"

        cv2.putText(
            new_frame,
            text,
            (frame_width + 10, y_offset + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    # Draw hint at bottom
    hint_text = "Press 'l' to hide"
    cv2.putText(
        new_frame,
        hint_text,
        (frame_width + 10, frame_height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (120, 120, 120),
        1,
    )

    return new_frame


def draw_gesture_feedback(
    frame: np.ndarray,
    gesture_state: dict,
    color: tuple[int, int, int] = (0, 200, 255),
) -> None:
    """
    Draw visual feedback for gesture detection state.

    Args:
        frame: Image to draw on (modified in place)
        gesture_state: Dictionary with gesture detection state
        color: BGR color for feedback elements
    """
    if gesture_state.get("was_pinched") and gesture_state.get("pinch_center"):
        center = gesture_state["pinch_center"]
        # Draw a pulsing circle at pinch center to indicate gesture in progress
        cv2.circle(frame, center, 30, color, 2)
        cv2.putText(
            frame,
            "PINCH",
            (center[0] - 25, center[1] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def draw_spread_debug(
    frame: np.ndarray,
    spread_info: dict,
    x_offset: int = 10,
    y_offset: int = 100,
) -> None:
    """
    Draw spread gesture debug overlay showing adaptive threshold info.

    Args:
        frame: Image to draw on (modified in place)
        spread_info: Dictionary from HandCommandDetector.get_spread_debug_info()
        x_offset: X position for debug text
        y_offset: Starting Y position for debug text
    """
    if not spread_info:
        return

    # Colors based on status
    status = spread_info.get("status", "NO HAND")
    if status == "PINCHING":
        status_color = (0, 165, 255)  # Orange
    elif status == "SPREAD!":
        status_color = (0, 255, 0)  # Green
    elif status == "READY":
        status_color = (255, 255, 255)  # White
    else:
        status_color = (128, 128, 128)  # Gray

    line_height = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    y = y_offset

    # Status line (larger)
    cv2.putText(
        frame,
        f"Spread: {status}",
        (x_offset, y),
        font,
        0.7,
        status_color,
        2,
    )
    y += line_height + 5

    # Hand size
    hand_size = spread_info.get("hand_size", 0)
    cv2.putText(
        frame,
        f"Hand size: {hand_size:.0f}px",
        (x_offset, y),
        font,
        font_scale,
        (200, 200, 200),
        thickness,
    )
    y += line_height

    # Current spread with percentage
    avg_spread = spread_info.get("avg_spread", 0)
    spread_pct = spread_info.get("spread_pct", 0)
    cv2.putText(
        frame,
        f"Spread: {avg_spread:.0f}px ({spread_pct:.0f}%)",
        (x_offset, y),
        font,
        font_scale,
        (200, 200, 200),
        thickness,
    )
    y += line_height

    # Thresholds
    pinch_thresh = spread_info.get("pinch_threshold", 0)
    spread_thresh = spread_info.get("spread_threshold", 0)
    pinch_ratio = spread_info.get("pinch_ratio", 0)
    spread_ratio = spread_info.get("spread_ratio", 0)
    cv2.putText(
        frame,
        f"Pinch < {pinch_thresh:.0f}px ({pinch_ratio:.0f}%)",
        (x_offset, y),
        font,
        font_scale,
        (0, 165, 255),  # Orange
        thickness,
    )
    y += line_height

    cv2.putText(
        frame,
        f"Spread > {spread_thresh:.0f}px ({spread_ratio:.0f}%)",
        (x_offset, y),
        font,
        font_scale,
        (0, 255, 0),  # Green
        thickness,
    )
    y += line_height + 5

    # Progress bar
    bar_width = 150
    bar_height = 15
    bar_x = x_offset
    bar_y = y

    # Background
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (50, 50, 50),
        -1,
    )

    # Calculate fill based on spread percentage relative to thresholds
    if hand_size > 0:
        # Map spread from 0 to spread_threshold onto 0-100%
        fill_pct = min(1.0, avg_spread / spread_thresh) if spread_thresh > 0 else 0
        fill_width = int(bar_width * fill_pct)

        # Color based on state
        if avg_spread < pinch_thresh:
            bar_color = (0, 165, 255)  # Orange - pinched
        elif avg_spread > spread_thresh:
            bar_color = (0, 255, 0)  # Green - spread
        else:
            bar_color = (255, 255, 255)  # White - in between

        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + fill_width, bar_y + bar_height),
            bar_color,
            -1,
        )

        # Threshold markers
        pinch_x = int(bar_x + bar_width * (pinch_thresh / spread_thresh)) if spread_thresh > 0 else bar_x
        cv2.line(frame, (pinch_x, bar_y - 2), (pinch_x, bar_y + bar_height + 2), (0, 165, 255), 2)

        spread_x = bar_x + bar_width
        cv2.line(frame, (spread_x, bar_y - 2), (spread_x, bar_y + bar_height + 2), (0, 255, 0), 2)

    # Border
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (100, 100, 100),
        1,
    )
