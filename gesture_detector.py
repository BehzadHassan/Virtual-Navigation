"""
gesture_detector.py
────────────────────
Wraps the MediaPipe Hands pipeline.

Usage
-----
    detector = GestureDetector()
    annotated_frame, landmarks, handedness = detector.detect(bgr_frame)
"""

import cv2
import mediapipe as mp
import numpy as np


class GestureDetector:
    """Real-time single-hand landmark detector using MediaPipe Hands."""

    # Neon palette for skeleton overlay
    _JOINT_COLOR  = (0, 245, 212)   # cyan-green  #00f5d4
    _BONE_COLOR   = (180, 58, 228)  # purple      #b43ae4
    _TIP_COLOR    = (255, 255, 0)   # yellow for fingertips

    # Fingertip & knuckle landmark indices
    _TIPS    = [4, 8, 12, 16, 20]
    _MCPS    = [1, 5, 9, 13, 17]

    def __init__(
        self,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
    ):
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Custom drawing specs
        self._joint_spec = self._mp_draw.DrawingSpec(
            color=self._JOINT_COLOR, thickness=2, circle_radius=4
        )
        self._bone_spec = self._mp_draw.DrawingSpec(
            color=self._BONE_COLOR, thickness=2, circle_radius=2
        )

    # ------------------------------------------------------------------ #

    def detect(self, bgr_frame: np.ndarray):
        """
        Process one BGR frame.

        Returns
        -------
        annotated_frame : np.ndarray  – frame with skeleton drawn on it
        landmarks_list  : list | None – list of 21 NormalizedLandmarks or None
        handedness      : str  | None – 'Left' / 'Right' or None
        """
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True
        annotated = bgr_frame.copy()

        if not results.multi_hand_landmarks:
            return annotated, None, None

        # Take first detected hand only
        hand_lms = results.multi_hand_landmarks[0]

        # Draw skeleton with glow effect (draw thick layer first)
        glow_bone = self._mp_draw.DrawingSpec(
            color=self._BONE_COLOR, thickness=6, circle_radius=2
        )
        glow_joint = self._mp_draw.DrawingSpec(
            color=self._JOINT_COLOR, thickness=6, circle_radius=6
        )
        self._mp_draw.draw_landmarks(
            annotated, hand_lms,
            self._mp_hands.HAND_CONNECTIONS,
            glow_joint, glow_bone,
        )
        self._mp_draw.draw_landmarks(
            annotated, hand_lms,
            self._mp_hands.HAND_CONNECTIONS,
            self._joint_spec, self._bone_spec,
        )

        # Highlight fingertips with bright circles
        for tip_idx in self._TIPS:
            lm = hand_lms.landmark[tip_idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 8, self._TIP_COLOR, -1)
            cv2.circle(annotated, (cx, cy), 10, (255, 255, 255), 1)

        handedness_label = (
            results.multi_handedness[0].classification[0].label
            if results.multi_handedness else None
        )

        return annotated, hand_lms.landmark, handedness_label

    def release(self):
        """Release MediaPipe resources."""
        self._hands.close()
