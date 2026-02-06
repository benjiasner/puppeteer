"""Main entry point for the hand tracking application."""

import time

import cv2

from . import config
from .hand_commands import HandCommandDetector
from .hand_tracker import HandTracker
from .smoother import TemporalSmoother
from .visualizer import (
    draw_all_fingertips,
    draw_command_log,
    draw_debug_info,
    draw_gesture_feedback,
    draw_spread_debug,
)


def main():
    """Run the hand tracking application."""
    # Initialize components
    tracker = HandTracker()
    smoother = TemporalSmoother()
    command_detector = HandCommandDetector()

    # Open webcam
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # State variables
    show_debug = False
    use_smoothing = True
    show_log = False
    prev_time = time.time()
    fps = 0.0

    # Timestamp tracking for VIDEO mode (must be monotonically increasing)
    start_time = time.perf_counter()

    print("Hand Tracking Started")
    print("Press 'q' to quit, 'd' to toggle debug info, 's' to toggle smoothing")
    print("Press 'p' to toggle preprocessing, 'l' to toggle command log")
    print("Gesture: Pinch fingers together then spread quickly to trigger Mission Control")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Mirror the frame horizontally for natural interaction
            frame = cv2.flip(frame, 1)

            # Calculate timestamp in milliseconds for MediaPipe VIDEO mode
            current_time = time.perf_counter()
            timestamp_ms = int((current_time - start_time) * 1000)
            timestamp_sec = current_time - start_time

            # Process frame for hand detection (VIDEO mode requires timestamp)
            hands_data = tracker.process_frame(frame, timestamp_ms)

            # Apply temporal smoothing if enabled
            if use_smoothing:
                hands_data = smoother.process(hands_data, timestamp_sec)

            # Process hand commands (spread gesture detection)
            detected_command = command_detector.process(hands_data, timestamp_sec)
            if detected_command:
                print(f"Command detected: {detected_command}")

            # Draw fingertip boxes
            draw_all_fingertips(frame, hands_data)

            # Draw gesture feedback (shows when pinch is detected)
            if show_debug:
                gesture_state = command_detector.get_state_info()
                draw_gesture_feedback(frame, gesture_state)
                # Draw spread debug overlay
                spread_info = command_detector.get_spread_debug_info()
                draw_spread_debug(frame, spread_info)

            # Calculate FPS
            fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time

            # Draw debug info if enabled
            if show_debug:
                draw_debug_info(frame, fps, len(hands_data))

            # Add command log panel if enabled
            display_frame = frame
            if show_log:
                log_entries = command_detector.get_log_entries(count=20)
                display_frame = draw_command_log(frame, log_entries)

            # Display the frame
            cv2.imshow("Hand Tracking", display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                show_debug = not show_debug
            elif key == ord("s"):
                use_smoothing = not use_smoothing
                smoother.reset()
                print(f"Smoothing: {'ON' if use_smoothing else 'OFF'}")
            elif key == ord("p"):
                config.PREPROCESSING_ENABLED = not config.PREPROCESSING_ENABLED
                print(f"Preprocessing: {'ON' if config.PREPROCESSING_ENABLED else 'OFF'}")
            elif key == ord("l"):
                show_log = not show_log
                print(f"Command Log: {'ON' if show_log else 'OFF'}")

    finally:
        # Cleanup
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Hand Tracking Stopped")


if __name__ == "__main__":
    main()
