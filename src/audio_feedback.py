"""Audio feedback for gesture confirmation."""

import subprocess
import threading
import os
from pathlib import Path


# Path to custom snap sound (if we create one)
SOUND_DIR = Path(__file__).parent.parent / "sounds"


def _play_sound_async(command: list[str]):
    """Run sound command in background thread."""
    def _run():
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=2
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            pass

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def play_snap_sound():
    """
    Play a sharp snap/click sound for gesture feedback.

    Uses macOS system sounds or afplay command.
    Non-blocking (runs in background thread).
    """
    # Option 1: Use macOS system sound (Tink is a nice sharp click)
    # Available sounds: Tink, Pop, Glass, Ping, Sosumi, etc.
    system_sounds = [
        "/System/Library/Sounds/Tink.aiff",
        "/System/Library/Sounds/Pop.aiff",
        "/System/Library/Sounds/Glass.aiff",
    ]

    # Find first available system sound
    for sound_path in system_sounds:
        if os.path.exists(sound_path):
            _play_sound_async(["afplay", "-v", "2.0", sound_path])
            return

    # Option 2: Use say command for a click-like sound (fallback)
    # This creates a very short beep-like sound
    _play_sound_async([
        "osascript", "-e",
        'beep 1'
    ])


def play_hover_sound():
    """
    Play a subtle sound when hovering over widget.

    Softer than snap sound for less intrusive feedback.
    """
    system_sounds = [
        "/System/Library/Sounds/Pop.aiff",
        "/System/Library/Sounds/Tink.aiff",
    ]

    for sound_path in system_sounds:
        if os.path.exists(sound_path):
            # Lower volume for hover
            _play_sound_async(["afplay", "-v", "0.3", sound_path])
            return


def play_expand_sound():
    """
    Play a sound when widget expands.

    Slightly different tone to indicate state change.
    """
    sound_path = "/System/Library/Sounds/Glass.aiff"
    if os.path.exists(sound_path):
        _play_sound_async(["afplay", "-v", "1.5", sound_path])
    else:
        # Fallback
        play_snap_sound()


def play_collapse_sound():
    """
    Play a sound when widget collapses.

    Lower pitch or different sound than expand.
    """
    sound_path = "/System/Library/Sounds/Tink.aiff"
    if os.path.exists(sound_path):
        _play_sound_async(["afplay", "-v", "1.0", sound_path])
    else:
        play_snap_sound()
