"""Main entry point for the hand tracking application."""

import time

import cv2

from . import config
from .hand_tracker import HandTracker
from .visualizer import draw_all_fingertips, draw_debug_info


def main():
    """Run the hand tracking application."""
    # Initialize components
    tracker = HandTracker()

    # Open webcam
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # State variables
    show_debug = False
    prev_time = time.time()
    fps = 0.0

    print("Hand Tracking Started")
    print("Press 'q' to quit, 'd' to toggle debug info")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Mirror the frame horizontally for natural interaction
            frame = cv2.flip(frame, 1)

            # Process frame for hand detection
            hands_data = tracker.process_frame(frame)

            # Draw fingertip boxes
            draw_all_fingertips(frame, hands_data)

            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time

            # Draw debug info if enabled
            if show_debug:
                draw_debug_info(frame, fps, len(hands_data))

            # Display the frame
            cv2.imshow("Hand Tracking", frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                show_debug = not show_debug

    finally:
        # Cleanup
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Hand Tracking Stopped")


if __name__ == "__main__":
    main()
