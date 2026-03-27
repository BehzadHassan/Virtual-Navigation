"""
gesture_detector.py  (mediapipe.tasks version)
────────────────────────────────────────────────
Uses the new MediaPipe Tasks API (mediapipe.tasks.python.vision.HandLandmarker)
which works on Python 3.14 / mediapipe 0.10+.

Model file:  hand_landmarker.task  (must sit next to this script)
Download:    https://storage.googleapis.com/mediapipe-models/hand_landmarker/
             hand_landmarker/float16/latest/hand_landmarker.task
"""

import os
import cv2
import math
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers import landmark as _lm_mod

# ── Colour constants ─────────────────────────────────────────────────────────
_JOINT_COLOR = (0, 245, 212)    # cyan-green
_BONE_COLOR  = (180, 58, 228)   # purple
_TIP_COLOR   = (255, 255, 0)    # yellow

# Hand connection pairs (same indices as mp.solutions.hands)
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (9,10),(10,11),(11,12),           # middle
    (5,9),(9,13),(13,17),(0,17),      # palm
    (13,14),(14,15),(15,16),          # ring
    (17,18),(18,19),(19,20),          # pinky
]

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


class GestureDetector:
    """Real-time single-hand landmark detector — mediapipe.tasks API."""

    def __init__(
        self,
        model_path: str = _MODEL_PATH,
        num_hands:  int = 1,
        min_hand_detection_confidence: float = 0.6,
        min_tracking_confidence:       float = 0.5,
    ):
        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,   # per-frame sync
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    # ------------------------------------------------------------------ #

    def detect(self, bgr_frame: np.ndarray):
        """
        Process one BGR frame.

        Returns
        -------
        annotated_frame : np.ndarray  – frame with skeleton
        landmarks       : list | None – list of NormalizedLandmark (21) or None
        handedness      : str  | None – 'Left' / 'Right' or None
        """
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect(mp_image)

        annotated = bgr_frame.copy()

        if not result.hand_landmarks:
            return annotated, None, None

        raw_lms = result.hand_landmarks[0]   # list of NormalizedLandmark

        # ── Draw skeleton ─────────────────────────────────────────────────
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in raw_lms]

        # Glow pass (thick)
        for a, b in _CONNECTIONS:
            cv2.line(annotated, pts[a], pts[b], _BONE_COLOR, 5, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(annotated, pt, 6, _JOINT_COLOR, -1, cv2.LINE_AA)

        # Sharp pass (thin)
        for a, b in _CONNECTIONS:
            cv2.line(annotated, pts[a], pts[b], _BONE_COLOR, 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(annotated, pt, 3, _JOINT_COLOR, -1, cv2.LINE_AA)

        # Fingertip highlights
        for tip_idx in [4, 8, 12, 16, 20]:
            pt = pts[tip_idx]
            cv2.circle(annotated, pt, 9, _TIP_COLOR, -1, cv2.LINE_AA)
            cv2.circle(annotated, pt, 11, (255, 255, 255), 1, cv2.LINE_AA)

        handedness = None
        if result.handedness:
            handedness = result.handedness[0][0].display_name   # 'Left' / 'Right'

        return annotated, raw_lms, handedness

    def release(self):
        self._landmarker.close()
