"""Hand command detection module for gesture-based system commands."""

import math
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional, Dict

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


class HandState:
    """Tracks the gesture state for a single hand to prevent jitter."""
    def __init__(self):
        self.is_pinched: bool = False
        self.pinch_start_time: Optional[float] = None
        self.last_command_time: float = 0.0


class HandCommandDetector:
    """Detects hand gestures and triggers system commands."""

    def __init__(
        self,
        # RATIOS relative to hand size (not raw pixels)
        # 0.5 means average finger distance is half the palm size
        pinch_ratio_threshold: float = 0.6, 
        spread_ratio_threshold: float = 1.1,
        cooldown_time: float = 1.0,  # Seconds between commands
        required_fingers: int = config.SPREAD_REQUIRED_FINGERS,
    ):
        self.pinch_ratio_threshold = pinch_ratio_threshold
        self.spread_ratio_threshold = spread_ratio_threshold
        self.cooldown_time = cooldown_time
        self.required_fingers = required_fingers

        # State tracking: Map hand index/label to HandState
        self.hand_states: Dict[str, HandState] = {}

        # Command log
        self.log: list[CommandLogEntry] = []
        self.max_log_entries = config.MAX_LOG_ENTRIES

    def _get_hand_size(self, landmarks: list[tuple[int, int]]) -> float:
        """
        Calculates the physical size of the hand in the image.
        Uses distance from Wrist (0) to Middle Finger MCP (9).
        """
        if not landmarks or len(landmarks) <= 9:
            return 1.0  # Fallback to avoid division by zero
            
        x0, y0 = landmarks[0] # Wrist
        x9, y9 = landmarks[9] # Middle Finger Knuckle (MCP)
        
        distance = math.hypot(x9 - x0, y9 - y0)
        return max(distance, 1.0) # Ensure we don't return 0

    def _calculate_spread_ratio(
        self, 
        fingertips: list[tuple[int, int]], 
        hand_size: float
    ) -> tuple[float, tuple[int, int]]:
        """
        Calculates how spread out the fingers are relative to hand size.
        Returns: (ratio, center_point)
        """
        if not fingertips:
            return 0.0, (0, 0)

        # 1. Calculate centroid of fingertips
        center_x = sum(x for x, y in fingertips) // len(fingertips)
        center_y = sum(y for x, y in fingertips) // len(fingertips)

        # 2. Calculate average distance of fingertips from that centroid
        total_dist = 0.0
        for x, y in fingertips:
            dist = math.hypot(x - center_x, y - center_y)
            total_dist += dist

        avg_absolute_spread = total_dist / len(fingertips)
        
        # 3. Normalize by hand size (The Magic Fix)
        ratio = avg_absolute_spread / hand_size
        
        return ratio, (center_x, center_y)

    def _execute_spread_command(self):
        """Execute macOS Mission Control (spread windows)."""
        print(">>> SPREAD COMMAND DETECTED <<<")
        try:
            # AppleScript to trigger Mission Control (Key Code 160)
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
            try:
                # Fallback: Open Mission Control app
                subprocess.run(
                    ["open", "-a", "Mission Control"],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                pass

    def _log_command(self, action: str):
        entry = CommandLogEntry(action=action, timestamp=time.time())
        self.log.append(entry)
        if len(self.log) > self.max_log_entries:
            self.log = self.log[-self.max_log_entries :]

    def process(
        self, hands_data: list[HandData], current_time: float
    ) -> Optional[str]:
        """
        Process hand data to detect commands per hand.
        """
        detected_command = None
        
        # Track which hands are currently active to manage state cleanup
        active_hand_ids = set()

        for i, hand in enumerate(hands_data):
            # Create a unique ID for the hand based on index + handedness
            # (Simple index works well if number of hands is constant, 
            # handedness helps if hands cross)
            hand_id = f"{hand.handedness}_{i}"
            active_hand_ids.add(hand_id)

            if hand_id not in self.hand_states:
                self.hand_states[hand_id] = HandState()
            
            state = self.hand_states[hand_id]

            # 1. Extract Data
            fingertip_coords = [(ft.x, ft.y) for ft in hand.fingertips]
            
            if len(fingertip_coords) < self.required_fingers:
                state.is_pinched = False
                continue

            # 2. Calculate Normalized Metrics
            hand_size = self._get_hand_size(hand.all_landmarks)
            spread_ratio, center = self._calculate_spread_ratio(fingertip_coords, hand_size)

            # 3. Check Cooldown
            if current_time - state.last_command_time < self.cooldown_time:
                continue

            # 4. State Machine Logic
            is_currently_pinched = spread_ratio < self.pinch_ratio_threshold
            is_currently_spread = spread_ratio > self.spread_ratio_threshold

            if is_currently_pinched:
                if not state.is_pinched:
                    state.is_pinched = True
                    state.pinch_start_time = current_time
            
            elif is_currently_spread:
                # Only trigger if we were PREVIOUSLY pinched
                if state.is_pinched:
                    self._execute_spread_command()
                    self._log_command("spread")
                    detected_command = "spread"
                    
                    # Reset state and set cooldown
                    state.is_pinched = False
                    state.pinch_start_time = None
                    state.last_command_time = current_time
            
            else:
                # "In-between" state (handoff zone)
                # We do NOT reset state here. This allows the hand to travel 
                # from pinched -> spread without losing the tracking in the middle.
                pass

        # Cleanup states for hands that disappeared
        # (Use list(keys) to avoid runtime error during iteration)
        for hand_id in list(self.hand_states.keys()):
            if hand_id not in active_hand_ids:
                del self.hand_states[hand_id]

        return detected_command

    def get_log_entries(self, count: int = 10) -> list[CommandLogEntry]:
        return self.log[-count:]

    def get_state_info(self) -> dict:
        """Returns debug info for the first active hand."""
        if not self.hand_states:
            return {"was_pinched": False, "pinch_start_time": None}
        
        # Just grab the first state found for visualization
        first_state = next(iter(self.hand_states.values()))
        return {
            "was_pinched": first_state.is_pinched,
            "pinch_start_time": first_state.pinch_start_time,
            "pinch_center": None 
        }

    def reset(self):
        self.hand_states.clear()