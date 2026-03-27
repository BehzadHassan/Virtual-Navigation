"""
gesture_classifier.py
──────────────────────
Maps 21 MediaPipe hand landmarks → a named gesture string.

Gesture catalogue
─────────────────
  "MOVE"          – index finger pointing up, others curled
  "LEFT_CLICK"    – pinch: index tip close to thumb tip
  "RIGHT_CLICK"   – middle tip close to thumb tip
  "DOUBLE_CLICK"  – quick release after left-click pinch
  "SCROLL"        – index + middle both extended (V-sign)
  "DRAG"          – fist (all fingers curled)
  "PAUSE"         – open palm (all 5 fingers extended)
  "NONE"          – unrecognised / transitioning
"""

import math
import time
import numpy as np


# ── landmark index constants ────────────────────────────────────────────────
WRIST       = 0
THUMB_TIP   = 4
INDEX_MCP   = 5;  INDEX_PIP  = 6;  INDEX_TIP  = 8
MIDDLE_MCP  = 9;  MIDDLE_PIP = 10; MIDDLE_TIP = 12
RING_MCP    = 13; RING_PIP   = 14; RING_TIP   = 16
PINKY_MCP   = 17; PINKY_PIP  = 18; PINKY_TIP  = 20


def _dist(a, b) -> float:
    """Euclidean distance between two landmarks (normalised coords)."""
    return math.hypot(a.x - b.x, a.y - b.y)


def _is_extended(tip_lm, pip_lm, mcp_lm) -> bool:
    """
    Finger is 'extended' if the tip is further from the wrist Y than the MCP,
    using a relative-height heuristic that is rotation-aware.
    """
    # Compare tip.y vs pip.y — in image coords, smaller y = higher on screen
    return tip_lm.y < pip_lm.y - 0.01


def _thumb_extended(landmarks) -> bool:
    """Thumb extended = tip is far from index MCP."""
    return _dist(landmarks[THUMB_TIP], landmarks[INDEX_MCP]) > 0.08


class GestureClassifier:
    """
    Classifies a list of 21 MediaPipe NormalizedLandmarks into a gesture name.

    Parameters
    ----------
    pinch_threshold : float
        Maximum normalised distance between index-tip & thumb-tip to count as a pinch.
    """

    def __init__(self, pinch_threshold: float = 0.055):
        self._pinch_thresh = pinch_threshold
        self._click_thresh  = 0.060   # right-click: middle+thumb
        self._double_click_window = 0.40   # seconds

        self._last_pinch_time: float = 0.0
        self._was_pinching: bool     = False
        self._pending_double: bool   = False

    # ------------------------------------------------------------------ #

    def classify(self, landmarks) -> str:
        """
        Parameters
        ----------
        landmarks : list of NormalizedLandmark (length 21)

        Returns
        -------
        str – gesture name
        """
        if landmarks is None:
            return "NONE"

        lms = landmarks   # shorthand

        # ── finger states ───────────────────────────────────────────────
        idx_ext    = _is_extended(lms[INDEX_TIP],  lms[INDEX_PIP],  lms[INDEX_MCP])
        mid_ext    = _is_extended(lms[MIDDLE_TIP], lms[MIDDLE_PIP], lms[MIDDLE_MCP])
        ring_ext   = _is_extended(lms[RING_TIP],   lms[RING_PIP],   lms[RING_MCP])
        pinky_ext  = _is_extended(lms[PINKY_TIP],  lms[PINKY_PIP],  lms[PINKY_MCP])
        thumb_ext  = _thumb_extended(lms)

        n_extended = sum([thumb_ext, idx_ext, mid_ext, ring_ext, pinky_ext])

        # ── pinch distances ─────────────────────────────────────────────
        idx_pinch  = _dist(lms[INDEX_TIP],  lms[THUMB_TIP])
        mid_pinch  = _dist(lms[MIDDLE_TIP], lms[THUMB_TIP])

        # ── OPEN PALM – all 5 extended ───────────────────────────────────
        if n_extended >= 4 and thumb_ext:
            return "PAUSE"

        # ── V-SIGN / SCROLL – index + middle up, ring + pinky down ──────
        if idx_ext and mid_ext and not ring_ext and not pinky_ext:
            return "SCROLL"

        # ── FIST / DRAG – all fingers curled ────────────────────────────
        if not idx_ext and not mid_ext and not ring_ext and not pinky_ext:
            return "DRAG"

        # ── LEFT CLICK / DOUBLE CLICK – index + thumb pinch ─────────────
        if idx_pinch < self._pinch_thresh:
            now = time.time()
            if not self._was_pinching:   # leading edge of pinch
                self._was_pinching = True
                gap = now - self._last_pinch_time
                self._last_pinch_time = now
                if gap < self._double_click_window and self._pending_double:
                    self._pending_double = False
                    return "DOUBLE_CLICK"
                self._pending_double = True
                return "LEFT_CLICK"
            return "LEFT_CLICK"
        else:
            self._was_pinching = False

        # ── RIGHT CLICK – middle + thumb pinch ──────────────────────────
        if mid_pinch < self._click_thresh and not idx_ext:
            return "RIGHT_CLICK"

        # ── MOVE – only index finger extended ───────────────────────────
        if idx_ext and not mid_ext and not ring_ext and not pinky_ext:
            return "MOVE"

        return "NONE"
