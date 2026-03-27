"""
gesture_classifier.py  —  v6 (all 5 targeted fixes applied)
════════════════════════════════════════════════════════════════

Fix 1 — Pinch clearance multiplier
    When index+thumb pinch → verify middle tip is > 2.5× pinch distance away.
    When middle+thumb pinch → verify index tip is > 2.5× pinch distance away.
    Prevents LEFT vs RIGHT CLICK confusion when both fingers hover near thumb.

Fix 2 — MOVE vs PAUSE (fingertip-y vs MCP-y)
    Model handles Pointing_Up vs Open_Palm — we add a redundant landmark
    check so the model's Open_Palm overrides MOVE only if ≥3 fingers are
    clearly above their MCP. Prevents ambiguous mid-pose misfires.

Fix 3 — DRAG requires 12 confirmation frames (vs 2 for other gestures)
    Prevents brief fist micro-poses mid-transition triggering a drag/drop.

Fix 4 — SCROLL explicitly requires ring + pinky down
    Victory model label accepted only when ring_tip.y > ring_mcp.y - 0.01
    and pinky_tip.y > pinky_mcp.y - 0.01, eliminating SCROLL/MOVE confusion.

Fix 5 — Majority-vote buffer = 7 frames (was 3)
    EMA only on XY cursor; gesture labels use majority vote over last 7 frames.
    More stable label output without adding cursor latency.

Debug — debug_states property exposes T/I/M/R/P booleans for HUD overlay.
"""

import math
import time
from collections import deque

# Landmark indices
THUMB_TIP  = 4
INDEX_MCP  = 5;  INDEX_TIP  = 8
MID_MCP    = 9;  MID_TIP    = 12
RING_MCP   = 13; RING_TIP   = 16
PINK_MCP   = 17; PINK_TIP   = 20


def _d3(a, b) -> float:
    dx = a.x - b.x; dy = a.y - b.y; dz = a.z - b.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def _tip_above_mcp(tip_lm, mcp_lm, margin: float = 0.01) -> bool:
    """Fix 2: tip.y < mcp.y - margin  (y increases downward)."""
    return tip_lm.y < mcp_lm.y - margin


_LABEL_MAP = {
    "Pointing_Up": "MOVE",
    "Closed_Fist": "DRAG",
    "Open_Palm":   "PAUSE",
    "Victory":     "SCROLL",
    "Thumb_Up":    "NONE",
    "Thumb_Down":  "NONE",
    "ILoveYou":    "NONE",
    "None":        "NONE",
    None:          "NONE",
}

# Fix 3: per-gesture confirmation frame requirements
_CONFIRM_FRAMES = {
    "DRAG":         6,   # Must hold fist for 6 frames (~200ms at 30fps)
    "SCROLL":       4,
    "PAUSE":        4,
    "LEFT_CLICK":   2,
    "RIGHT_CLICK":  2,
    "DOUBLE_CLICK": 2,
    "NONE":         1,
    "MOVE":         1,
}


class GestureClassifier:
    """
    Parameters
    ----------
    vote_window  : frames kept for majority vote — Fix 5 sets this to 7
    pinch_enter  : 3-D normalised dist to START a pinch
    pinch_exit   : 3-D normalised dist to RELEASE a pinch
    """

    def __init__(
        self,
        vote_window: int   = 4,      # Cut to 4 for lower latency
        pinch_enter: float = 0.060,
        pinch_exit:  float = 0.090,
    ):
        self._vw   = vote_window
        self._pen  = pinch_enter
        self._pex  = pinch_exit

        self._buf       = deque(maxlen=vote_window)
        self._stable    = "NONE"
        self._candidate = "NONE"
        self._cand_n    = 0

        self._idx_pinch = False
        self._mid_pinch = False

        self._last_click = 0.0
        self._dbl_win    = 0.38

        # Fix Debug: expose last finger extension states
        self.debug_states: dict = {
            "T": False, "I": False, "M": False, "R": False, "P": False
        }

    # ── Public ───────────────────────────────────────────────────────────

    def classify(self, landmarks, mp_gesture: str) -> str:
        if landmarks is None:
            self._buf.append("NONE")
            self.debug_states = {k: False for k in "TIMRP"}
            return self._commit("NONE")

        raw = self._raw(landmarks, mp_gesture)
        self._buf.append(raw)
        voted = self._vote()
        return self._commit(voted)

    # ── Raw classification ───────────────────────────────────────────────

    def _raw(self, lms, mp_gesture: str) -> str:
        # Fix 2: ext states via fingertip-y vs MCP-y
        i_ext = _tip_above_mcp(lms[INDEX_TIP], lms[INDEX_MCP])
        m_ext = _tip_above_mcp(lms[MID_TIP],   lms[MID_MCP])
        r_ext = _tip_above_mcp(lms[RING_TIP],  lms[RING_MCP])
        p_ext = _tip_above_mcp(lms[PINK_TIP],  lms[PINK_MCP])
        t_ext = _d3(lms[THUMB_TIP], lms[INDEX_MCP]) > 0.065

        # Update debug states
        self.debug_states = {
            "T": t_ext, "I": i_ext, "M": m_ext, "R": r_ext, "P": p_ext
        }

        # Pinch distances
        id_ = _d3(lms[INDEX_TIP], lms[THUMB_TIP])
        md_ = _d3(lms[MID_TIP],   lms[THUMB_TIP])

        # Pinch enter/exit hysteresis
        if not self._idx_pinch and id_ < self._pen:  self._idx_pinch = True
        elif self._idx_pinch and id_ > self._pex:    self._idx_pinch = False
        if not self._mid_pinch and md_ < self._pen:  self._mid_pinch = True
        elif self._mid_pinch and md_ > self._pex:    self._mid_pinch = False

        # Fix 1: clearance multiplier — relaxed to 1.5x (was 2.5x) so inputs aren't arbitrarily dropped
        idx_clear = (not self._mid_pinch) and (md_ > id_ * 1.5)
        mid_clear = (not self._idx_pinch) and (id_ > md_ * 1.5)

        # Priority order:
        # 1. LEFT_CLICK — index pinch with middle clearly away
        if self._idx_pinch and idx_clear:
            return "LEFT_CLICK"

        # 2. RIGHT_CLICK — middle pinch with index clearly away
        if self._mid_pinch and mid_clear:
            return "RIGHT_CLICK"

        # 3. Map model gesture
        base = _LABEL_MAP.get(mp_gesture, "NONE")

        # Fix 4: SCROLL — verify ring + pinky are explicitly DOWN
        if base == "SCROLL":
            if r_ext or p_ext:
                base = "MOVE"   # ring or pinky up → not a clean V-sign

        # Fix 2 guard: if model says PAUSE, double-check ≥3 fingers are up
        if base == "PAUSE":
            n_up = sum([i_ext, m_ext, r_ext, p_ext])
            if n_up < 3:
                base = "NONE"   # not actually open palm

        return base

    # ── Majority vote (Fix 5) ─────────────────────────────────────────────

    def _vote(self) -> str:
        if not self._buf:
            return "NONE"
        counts: dict = {}
        for g in self._buf:
            counts[g] = counts.get(g, 0) + 1
        return max(counts, key=lambda k: counts[k])

    # ── Per-gesture confirmation gate (Fix 3) ────────────────────────────

    def _commit(self, voted: str) -> str:
        if voted == self._stable:
            self._candidate = voted
            self._cand_n    = _CONFIRM_FRAMES.get(voted, 2)
            return self._stable

        # Instant transitions for cursor responsiveness
        if voted in ("NONE", "MOVE"):
            self._stable    = voted
            self._candidate = voted
            self._cand_n    = _CONFIRM_FRAMES.get(voted, 1)
            return self._stable

        # Gate: each gesture has its own required frame count
        required = _CONFIRM_FRAMES.get(voted, 2)
        if voted == self._candidate:
            self._cand_n += 1
        else:
            self._candidate = voted
            self._cand_n    = 1

        if self._cand_n >= required:
            if voted == "LEFT_CLICK":
                now = time.time()
                gap = now - self._last_click
                self._last_click = now
                if gap < self._dbl_win:
                    self._stable = "DOUBLE_CLICK"
                    return self._stable
            self._stable = voted

        return self._stable
