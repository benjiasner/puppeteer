"""Configuration constants for the hand tracking application."""

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# MediaPipe settings
MAX_NUM_HANDS = 2
DETECTION_CONFIDENCE = 0.5  # Increased from 0.2 to reduce false positives
TRACKING_CONFIDENCE = 0.5  # Increased from 0.4 for more reliable tracking
HAND_PRESENCE_CONFIDENCE = 0.5  # Individual landmark confidence

# Image preprocessing for distance detection
PREPROCESSING_ENABLED = False  # Disabled by default - only enable in poor lighting
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8

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

# Temporal smoothing settings
SMOOTHING_ENABLED = True

# One Euro Filter parameters
# These adapt smoothing based on movement speed for low-latency tracking
ONE_EURO_MIN_CUTOFF = 1.0  # Hz - baseline smoothing (lower = more smoothing when stationary)
ONE_EURO_BETA = 0.7  # Speed coefficient (higher = less lag during fast movements)
ONE_EURO_D_CUTOFF = 1.0  # Derivative filter cutoff

# Kalman filter parameters
# Used for velocity estimation and prediction during occlusions
KALMAN_PROCESS_NOISE = 100.0  # Higher = more responsive to changes
KALMAN_MEASUREMENT_NOISE = 10.0  # Higher = trusts measurements less
KALMAN_INITIAL_COVARIANCE = 1000.0  # Initial uncertainty

# Hand tracking state machine
CONFIDENCE_ACQUIRE_THRESHOLD = 0.6  # Confidence to start tracking a hand
CONFIDENCE_LOSE_THRESHOLD = 0.3  # Lower threshold prevents flickering
PERSISTENCE_FRAMES = 3  # Frames to predict during occlusion (reduced due to Kalman prediction)

# Adaptive outlier rejection
# Base threshold scales with estimated hand size
BASE_OUTLIER_THRESHOLD = 100  # Base pixels for outlier detection
OUTLIER_VELOCITY_FACTOR = 0.1  # Factor to add based on velocity

# Hand command settings (spread gesture)
# Adaptive thresholds - ratios relative to hand size for distance-independent detection
SPREAD_PINCH_RATIO = 0.16  # avg_spread < 25% of hand_size = pinched
SPREAD_DISTANCE_RATIO = 0.35  # avg_spread > 55% of hand_size = spread
SPREAD_MIN_TIME = 0.1  # Minimum seconds between pinch and spread
SPREAD_MAX_TIME = 0.7  # Maximum seconds between pinch and spread (more time for deliberate gestures)
SPREAD_REQUIRED_FINGERS = 4  # Minimum fingertips needed for gesture
SPREAD_DEFAULT_HAND_SIZE = 200  # Fallback hand size in pixels if no landmarks

# Command log settings
MAX_LOG_ENTRIES = 50  # Maximum entries to keep in command log
LOG_PANEL_WIDTH = 200  # Width of log panel in pixels
LOG_PANEL_BG_COLOR = (30, 30, 30)  # Dark gray background
LOG_PANEL_TEXT_COLOR = (200, 200, 200)  # Light gray text
LOG_PANEL_HIGHLIGHT_COLOR = (100, 200, 100)  # Green for recent entries

# Weather Widget settings
WEATHER_WIDGET_MARGIN = 20  # Margin from screen edge
WEATHER_WIDGET_COMPACT_SIZE = (160, 80)  # (width, height) for compact view
WEATHER_WIDGET_EXPANDED_SIZE = (350, 280)  # (width, height) for expanded view
WEATHER_CACHE_DURATION = 600  # 10 minutes cache for weather data

# Snap gesture settings (thumb-index pinch + quick release)
SNAP_PINCH_THRESHOLD = 30.0  # Distance in pixels for pinch detection
SNAP_RELEASE_THRESHOLD = 60.0  # Distance in pixels for release detection
SNAP_RELEASE_SPEED = 300.0  # Speed in pixels/second for valid snap
