"""
mouse_controller.py
────────────────────
Translates gesture commands into actual OS mouse actions via PyAutoGUI.

Features
────────
  • Exponential moving-average cursor smoothing
  • Per-action debounce timers (prevents jitter-clicks)
  • PAUSE gesture freezes cursor in place
  • Safe-fail: PyAutoGUI FAILSAFE is kept ON (move to corner to abort)
"""

import time
import pyautogui
import numpy as np

# Disable the tiny pause PyAutoGUI inserts between actions (improves latency)
pyautogui.PAUSE = 0.0
# Keep fail-safe: moving mouse to top-left corner will raise an exception
pyautogui.FAILSAFE = True


class MouseController:
    """
    Parameters
    ----------
    screen_w, screen_h  : int   – display resolution (pixels)
    cam_w,   cam_h      : int   – webcam frame size (pixels)
    smoothing           : float – EMA alpha (0 = max smooth, 1 = no smooth)
    sensitivity         : float – multiplier on cursor delta (0.5–2.0)
    scroll_speed        : int   – lines to scroll per gesture tick
    click_debounce      : float – min seconds between consecutive clicks
    """

    def __init__(
        self,
        screen_w: int   = 1920,
        screen_h: int   = 1080,
        cam_w:    int   = 640,
        cam_h:    int   = 480,
        smoothing:     float = 0.25,
        sensitivity:   float = 1.0,
        scroll_speed:  int   = 3,
        click_debounce: float = 0.40,
    ):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.cam_w    = cam_w
        self.cam_h    = cam_h
        self.smoothing     = smoothing
        self.sensitivity   = sensitivity
        self.scroll_speed  = scroll_speed
        self.click_debounce = click_debounce

        # Smoothed cursor position (screen pixels)
        cur_x, cur_y = pyautogui.position()
        self._sx = float(cur_x)
        self._sy = float(cur_y)

        # Timestamps for debounce
        self._last_left_click:   float = 0.0
        self._last_right_click:  float = 0.0
        self._last_double_click: float = 0.0
        self._last_scroll:       float = 0.0

        # Drag state
        self._dragging: bool = False

        # Pause state
        self._paused: bool = False

    # ────────────────────────────────────────────────────────────────────── #
    #  Public interface                                                       #
    # ────────────────────────────────────────────────────────────────────── #

    def update_settings(self, smoothing: float = None, sensitivity: float = None):
        if smoothing   is not None: self.smoothing   = max(0.02, min(1.0, smoothing))
        if sensitivity is not None: self.sensitivity = max(0.3,  min(3.0, sensitivity))

    def handle_gesture(self, gesture: str, landmark_x: float, landmark_y: float):
        """
        Main dispatch — call this every frame with the current gesture and
        the normalised (0-1) position of the index-finger tip.
        """
        # Always update smoothed position from index tip
        target_sx = np.clip(landmark_x * self.screen_w * self.sensitivity *
                            (self.screen_w / (self.cam_w or 1)) *
                            (self.screen_w / self.screen_w), 0, self.screen_w - 1)

        # Simpler direct mapping: map cam → screen with optional flip
        target_sx = np.clip((1.0 - landmark_x) * self.screen_w, 0, self.screen_w - 1)
        target_sy = np.clip(landmark_y          * self.screen_h, 0, self.screen_h - 1)

        # EMA smoothing
        alpha = self.smoothing
        self._sx = alpha * target_sx + (1 - alpha) * self._sx
        self._sy = alpha * target_sy + (1 - alpha) * self._sy

        if gesture == "PAUSE":
            self._paused = True
            if self._dragging:
                pyautogui.mouseUp(button="left")
                self._dragging = False
            return

        self._paused = False

        if gesture == "MOVE":
            self._move()

        elif gesture == "LEFT_CLICK":
            self._move()
            self._left_click()

        elif gesture == "RIGHT_CLICK":
            self._move()
            self._right_click()

        elif gesture == "DOUBLE_CLICK":
            self._move()
            self._double_click()

        elif gesture == "SCROLL":
            # Use vertical position — upper half = scroll up, lower = down
            if landmark_y < 0.45:
                self._scroll(self.scroll_speed)
            elif landmark_y > 0.55:
                self._scroll(-self.scroll_speed)

        elif gesture == "DRAG":
            self._move()
            if not self._dragging:
                pyautogui.mouseDown(button="left")
                self._dragging = True

        elif gesture in ("NONE", None):
            if self._dragging:
                pyautogui.mouseUp(button="left")
                self._dragging = False

    # ────────────────────────────────────────────────────────────────────── #
    #  Private helpers                                                        #
    # ────────────────────────────────────────────────────────────────────── #

    def _move(self):
        if not self._paused:
            try:
                pyautogui.moveTo(int(self._sx), int(self._sy), duration=0)
            except pyautogui.FailSafeException:
                pass

    def _left_click(self):
        now = time.time()
        if now - self._last_left_click > self.click_debounce:
            self._last_left_click = now
            try:
                pyautogui.click(button="left")
            except pyautogui.FailSafeException:
                pass

    def _right_click(self):
        now = time.time()
        if now - self._last_right_click > self.click_debounce:
            self._last_right_click = now
            try:
                pyautogui.click(button="right")
            except pyautogui.FailSafeException:
                pass

    def _double_click(self):
        now = time.time()
        if now - self._last_double_click > self.click_debounce:
            self._last_double_click = now
            try:
                pyautogui.doubleClick(button="left")
            except pyautogui.FailSafeException:
                pass

    def _scroll(self, amount: int):
        now = time.time()
        if now - self._last_scroll > 0.08:
            self._last_scroll = now
            try:
                pyautogui.scroll(amount)
            except pyautogui.FailSafeException:
                pass

    @property
    def cursor_pos(self):
        return int(self._sx), int(self._sy)

    @property
    def is_dragging(self):
        return self._dragging

    @property
    def is_paused(self):
        return self._paused
