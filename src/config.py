"""Configuration constants for the hand tracking application."""

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# MediaPipe settings
MAX_NUM_HANDS = 2
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.5

# Visualization settings
# Light blue in BGR format (OpenCV uses BGR, not RGB)
BOX_COLOR = (255, 191, 0)
BOX_HALF_SIZE = 20  # pixels from center to edge
BOX_THICKNESS = 2

# Debug text settings
DEBUG_TEXT_COLOR = (0, 255, 0)  # Green
DEBUG_TEXT_SCALE = 0.7
DEBUG_TEXT_THICKNESS = 2

# MediaPipe hand landmark indices for fingertips
# Reference: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
FINGERTIP_INDICES = {
    4: "Thumb",
    8: "Index",
    12: "Middle",
    16: "Ring",
    20: "Pinky",
}
