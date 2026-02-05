"""Hand tracking module using MediaPipe."""

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from . import config


# Model download URL and path
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"


def _ensure_model_exists():
    """Download the hand landmarker model if it doesn't exist."""
    if MODEL_PATH.exists():
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model to {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully.")


@dataclass
class FingertipPosition:
    """Represents a single fingertip position."""

    x: int
    y: int
    finger_name: str
    hand_label: str


@dataclass
class HandData:
    """Represents detected hand data."""

    fingertips: list[FingertipPosition]
    handedness: str
    all_landmarks: list[tuple[int, int]]


class HandTracker:
    """Tracks hands and extracts fingertip positions using MediaPipe."""

    def __init__(
        self,
        max_num_hands: int = config.MAX_NUM_HANDS,
        detection_confidence: float = config.DETECTION_CONFIDENCE,
        tracking_confidence: float = config.TRACKING_CONFIDENCE,
    ):
        _ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray) -> list[HandData]:
        """
        Process a frame and return detected hand data.

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of HandData objects for each detected hand
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect(mp_image)

        hands_data = []
        frame_height, frame_width = frame.shape[:2]

        if result.hand_landmarks and result.handedness:
            for hand_landmarks, handedness_info in zip(
                result.hand_landmarks, result.handedness
            ):
                hand_label = handedness_info[0].category_name
                hand_data = self._extract_hand_data(
                    hand_landmarks, hand_label, frame_width, frame_height
                )
                hands_data.append(hand_data)

        return hands_data

    def _extract_hand_data(
        self,
        hand_landmarks,
        hand_label: str,
        frame_width: int,
        frame_height: int,
    ) -> HandData:
        """Extract fingertip positions and all landmarks from a hand."""
        fingertips = []
        all_landmarks = []

        for idx, landmark in enumerate(hand_landmarks):
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            all_landmarks.append((x, y))

            # Check if this is a fingertip
            if idx in config.FINGERTIP_INDICES:
                fingertips.append(
                    FingertipPosition(
                        x=x,
                        y=y,
                        finger_name=config.FINGERTIP_INDICES[idx],
                        hand_label=hand_label,
                    )
                )

        return HandData(
            fingertips=fingertips,
            handedness=hand_label,
            all_landmarks=all_landmarks,
        )

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()
