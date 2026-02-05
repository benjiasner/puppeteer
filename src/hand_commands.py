"""Hand command detection module for gesture-based system commands."""

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
        pinch_threshold: int = config.SPREAD_PINCH_THRESHOLD,
        spread_threshold: int = config.SPREAD_DISTANCE_THRESHOLD,
        min_time: float = config.SPREAD_MIN_TIME,
        max_time: float = config.SPREAD_MAX_TIME,
        required_fingers: int = config.SPREAD_REQUIRED_FINGERS,
    ):
        self.pinch_threshold = pinch_threshold
        self.spread_threshold = spread_threshold
        self.min_time = min_time
        self.max_time = max_time
        self.required_fingers = required_fingers

        # State tracking
        self.pinch_start_time: Optional[float] = None
        self.pinch_center: Optional[tuple[int, int]] = None
        self.was_pinched = False

        # Command log
        self.log: list[CommandLogEntry] = []
        self.max_log_entries = config.MAX_LOG_ENTRIES

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
        center_x = sum(x for x, y in fingertips) // len(fingertips)
        center_y = sum(y for x, y in fingertips) // len(fingertips)

        # Calculate average distance from center
        total_dist = 0.0
        for x, y in fingertips:
            dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            total_dist += dist

        avg_spread = total_dist / len(fingertips)
        return avg_spread, (center_x, center_y)

    def _check_fingertips_intersect(
        self, fingertips: list[tuple[int, int]], threshold: int
    ) -> bool:
        """
        Check if a quorum of fingertips are within threshold distance of each other.

        This checks if fingertips are clustered together (pinched state).
        """
        if len(fingertips) < self.required_fingers:
            return False

        avg_spread, _ = self._calculate_fingertip_spread(fingertips)
        return avg_spread < threshold

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
            return None

        # Calculate current spread
        avg_spread, center = self._calculate_fingertip_spread(all_fingertips)

        # Check for pinch state (fingers close together)
        is_pinched = avg_spread < self.pinch_threshold

        # Check for spread state (fingers far apart)
        is_spread = avg_spread > self.spread_threshold

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

    def reset(self):
        """Reset detection state."""
        self.was_pinched = False
        self.pinch_start_time = None
        self.pinch_center = None
