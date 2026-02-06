"""Hand command detection module for gesture-based system commands."""

import math
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

from . import config
from .hand_tracker import HandData


@dataclass
class CommandLogEntry:
    """Represents a logged command action."""

    action: str
    timestamp: float
    formatted_time: str = field(init=False)

    def __post_init__(self):
        self.formatted_time = time.strftime("%H:%M:%S", time.localtime(self.timestamp))


class HandCommandDetector:
    """Detects hand gestures and triggers system commands."""

    def __init__(
        self,
        pinch_ratio: float = config.SPREAD_PINCH_RATIO,
        spread_ratio: float = config.SPREAD_DISTANCE_RATIO,
        min_time: float = config.SPREAD_MIN_TIME,
        max_time: float = config.SPREAD_MAX_TIME,
        required_fingers: int = config.SPREAD_REQUIRED_FINGERS,
    ):
        self.pinch_ratio = pinch_ratio
        self.spread_ratio = spread_ratio
        self.min_time = min_time
        self.max_time = max_time
        self.required_fingers = required_fingers

        # State tracking
        self.pinch_start_time: Optional[float] = None
        self.pinch_center: Optional[tuple[int, int]] = None
        self.was_pinched = False

        # Debug state (updated each frame for visualization)
        self.current_hand_size: float = 0.0
        self.current_avg_spread: float = 0.0
        self.current_pinch_threshold: float = 0.0
        self.current_spread_threshold: float = 0.0

        # Command log
        self.log: list[CommandLogEntry] = []
        self.max_log_entries = config.MAX_LOG_ENTRIES

    def _estimate_hand_size(self, hands_data: list[HandData]) -> float:
        """
        Estimate hand size from landmarks.

        Uses distance from wrist (landmark 0) to middle fingertip (landmark 12).
        Returns the average hand size if multiple hands, or default if no landmarks.
        """
        sizes = []
        for hand in hands_data:
            if len(hand.all_landmarks) >= 13:
                wrist = hand.all_landmarks[0]
                middle_tip = hand.all_landmarks[12]
                dx = middle_tip[0] - wrist[0]
                dy = middle_tip[1] - wrist[1]
                sizes.append(math.sqrt(dx * dx + dy * dy))

        if sizes:
            return sum(sizes) / len(sizes)
        return config.SPREAD_DEFAULT_HAND_SIZE

    def _calculate_fingertip_spread(
        self, fingertips: list[tuple[int, int]]
    ) -> tuple[float, tuple[int, int]]:
        """
        Calculate the average distance from center for all fingertips.

        Returns:
            Tuple of (average_spread_distance, center_point)
        """
        if not fingertips:
            return 0.0, (0, 0)

        # Calculate centroid
        center_x = sum(pt[0] for pt in fingertips) // len(fingertips)
        center_y = sum(pt[1] for pt in fingertips) // len(fingertips)

        # Calculate average distance from center
        total_dist = 0.0
        for x, y in fingertips:
            dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            total_dist += dist

        avg_spread = total_dist / len(fingertips)
        return avg_spread, (center_x, center_y)

    def _execute_spread_command(self):
        """Execute macOS Mission Control (spread windows)."""
        try:
            # Use AppleScript to trigger Mission Control
            # This is equivalent to two-finger double-click on Magic Mouse
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to key code 160',
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            # Fallback: try using open command with Mission Control
            try:
                subprocess.run(
                    ["open", "-a", "Mission Control"],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                pass  # Silently fail if Mission Control can't be triggered

    def _log_command(self, action: str):
        """Add a command to the log."""
        entry = CommandLogEntry(action=action, timestamp=time.time())
        self.log.append(entry)

        # Trim log if too long
        if len(self.log) > self.max_log_entries:
            self.log = self.log[-self.max_log_entries :]

    def process(
        self, hands_data: list[HandData], current_time: float
    ) -> Optional[str]:
        """
        Process hand data to detect commands.

        Args:
            hands_data: List of HandData from hand tracker
            current_time: Current timestamp in seconds

        Returns:
            Name of detected command, or None
        """
        # Collect all fingertips from all hands
        all_fingertips = []
        for hand in hands_data:
            for fingertip in hand.fingertips:
                all_fingertips.append((fingertip.x, fingertip.y))

        # Need enough fingertips to detect gestures
        if len(all_fingertips) < self.required_fingers:
            self.was_pinched = False
            self.pinch_start_time = None
            # Reset debug state
            self.current_hand_size = 0.0
            self.current_avg_spread = 0.0
            self.current_pinch_threshold = 0.0
            self.current_spread_threshold = 0.0
            return None

        # Estimate hand size for adaptive thresholds
        hand_size = self._estimate_hand_size(hands_data)
        pinch_threshold = hand_size * self.pinch_ratio
        spread_threshold = hand_size * self.spread_ratio

        # Store for debug display
        self.current_hand_size = hand_size
        self.current_pinch_threshold = pinch_threshold
        self.current_spread_threshold = spread_threshold

        # Calculate current spread
        avg_spread, center = self._calculate_fingertip_spread(all_fingertips)
        self.current_avg_spread = avg_spread

        # Check for pinch state (fingers close together)
        is_pinched = avg_spread < pinch_threshold

        # Check for spread state (fingers far apart)
        is_spread = avg_spread > spread_threshold

        # State machine for spread detection
        if is_pinched and not self.was_pinched:
            # Just entered pinch state
            self.pinch_start_time = current_time
            self.pinch_center = center
            self.was_pinched = True

        elif is_spread and self.was_pinched and self.pinch_start_time is not None:
            # Transitioned from pinch to spread
            elapsed = current_time - self.pinch_start_time

            if self.min_time <= elapsed <= self.max_time:
                # Valid spread gesture detected
                self._execute_spread_command()
                self._log_command("spread")

                # Reset state
                self.was_pinched = False
                self.pinch_start_time = None
                self.pinch_center = None

                return "spread"

            # If timing was wrong, still reset
            self.was_pinched = False
            self.pinch_start_time = None

        elif not is_pinched and self.was_pinched:
            # Check if we've exceeded max time without proper spread
            if self.pinch_start_time is not None:
                elapsed = current_time - self.pinch_start_time
                if elapsed > self.max_time:
                    # Gesture timed out
                    self.was_pinched = False
                    self.pinch_start_time = None

        return None

    def get_log_entries(self, count: int = 10) -> list[CommandLogEntry]:
        """Get the most recent log entries."""
        return self.log[-count:]

    def get_state_info(self) -> dict:
        """Get current detection state for debugging."""
        return {
            "was_pinched": self.was_pinched,
            "pinch_start_time": self.pinch_start_time,
            "pinch_center": self.pinch_center,
        }

    def get_spread_debug_info(self) -> dict:
        """Get spread detection debug info for visualization."""
        # Calculate normalized percentage (0-100 scale relative to thresholds)
        if self.current_hand_size > 0:
            spread_pct = (self.current_avg_spread / self.current_hand_size) * 100
        else:
            spread_pct = 0.0

        # Determine status
        if self.current_hand_size == 0:
            status = "NO HAND"
        elif self.current_avg_spread < self.current_pinch_threshold:
            status = "PINCHING"
        elif self.current_avg_spread > self.current_spread_threshold:
            status = "SPREAD!"
        else:
            status = "READY"

        return {
            "hand_size": self.current_hand_size,
            "avg_spread": self.current_avg_spread,
            "spread_pct": spread_pct,
            "pinch_threshold": self.current_pinch_threshold,
            "spread_threshold": self.current_spread_threshold,
            "pinch_ratio": self.pinch_ratio * 100,
            "spread_ratio": self.spread_ratio * 100,
            "status": status,
            "was_pinched": self.was_pinched,
        }

    def reset(self):
        """Reset detection state."""
        self.was_pinched = False
        self.pinch_start_time = None
        self.pinch_center = None
