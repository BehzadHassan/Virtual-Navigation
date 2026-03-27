"""
gesture_detector.py  —  v3 (GestureRecognizer neural-net model)
═══════════════════════════════════════════════════════════════════
Uses MediaPipe GestureRecognizer — a trained neural network that directly
classifies hand poses, far more robust than rule-based landmark geometry.

Model file: gesture_recognizer.task (must be next to this script)
Download:   https://storage.googleapis.com/mediapipe-models/gesture_recognizer/
            gesture_recognizer/float16/latest/gesture_recognizer.task

GestureRecognizer returns:
  gestures[0][0].category_name  →  one of:
    None | Closed_Fist | Open_Palm | Pointing_Up |
    Thumb_Down | Thumb_Up | Victory | ILoveYou
  hand_landmarks[0]              →  21 NormalizedLandmark objects
  handedness[0][0].display_name  →  'Left' / 'Right'
"""

import os
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Drawing colours ───────────────────────────────────────────────────────────
_JOINT_COLOR = (0, 245, 212)
_BONE_COLOR  = (180, 58, 228)
_TIP_COLOR   = (255, 220, 0)

_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "gesture_recognizer.task")


class GestureDetector:
    """
    Wraps MediaPipe GestureRecognizer.
    Returns the model's gesture label + hand landmarks for pinch overlay.
    """

    def __init__(
        self,
        model_path: str   = _MODEL_PATH,
        num_hands:  int   = 1,
        min_hand_detection_confidence: float = 0.70,
        min_tracking_confidence:       float = 0.60,
        min_hand_presence_confidence:  float = 0.65,
    ):
        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.GestureRecognizerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
        )
        self._recognizer = mp_vision.GestureRecognizer.create_from_options(opts)

    # ────────────────────────────────────────────────────────────────────

    def detect(self, bgr_frame: np.ndarray):
        """
        Process one BGR frame.

        Returns
        -------
        annotated   : np.ndarray  – BGR frame with skeleton overlay
        landmarks   : list | None – 21 NormalizedLandmark, or None
        mp_gesture  : str  | None – model's gesture label, or None
                      (one of: None, Closed_Fist, Open_Palm, Pointing_Up,
                       Thumb_Down, Thumb_Up, Victory, ILoveYou)
        """
        h, w = bgr_frame.shape[:2]
        rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._recognizer.recognize(mp_img)

        annotated = bgr_frame.copy()

        if not result.hand_landmarks:
            return annotated, None, None

        lms = result.hand_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

        # Glow pass then sharp pass
        for a, b in _CONNECTIONS:
            cv2.line(annotated, pts[a], pts[b], _BONE_COLOR, 6, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(annotated, pt, 7, _JOINT_COLOR, -1, cv2.LINE_AA)
        for a, b in _CONNECTIONS:
            cv2.line(annotated, pts[a], pts[b], _BONE_COLOR, 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(annotated, pt, 3, _JOINT_COLOR, -1, cv2.LINE_AA)
        for tip_idx in (4, 8, 12, 16, 20):
            cv2.circle(annotated, pts[tip_idx], 10, _TIP_COLOR, -1, cv2.LINE_AA)
            cv2.circle(annotated, pts[tip_idx], 12, (255,255,255),  1, cv2.LINE_AA)

        mp_gesture = None
        if result.gestures:
            mp_gesture = result.gestures[0][0].category_name   # e.g. "Victory"

        return annotated, lms, mp_gesture

    def release(self):
        try:
            self._recognizer.close()
        except Exception:
            pass
