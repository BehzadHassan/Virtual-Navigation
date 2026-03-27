"""
mouse_controller.py  —  v4 (pure hardware input)
════════════════════════════════════════════════
Replacing all PyAutoGUI logic with pure Windows SendInput ctypes.
This eliminates all internal library lag, fail-safe overhead, and thread blocks.
Gestures are edge-triggered (e.g. state transition from MOVE -> CLICK) which
makes them respond instantly on the first frame rather than relying on a timer.
"""

import time
import ctypes
import math

# Windows mouse_event flags
_MOUSEEVENTF_MOVE     = 0x0001
_MOUSEEVENTF_ABSOLUTE = 0x8000
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP   = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP  = 0x0010
_MOUSEEVENTF_WHEEL    = 0x0800

class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx",          ctypes.c_long),
        ("dy",          ctypes.c_long),
        ("mouseData",   ctypes.c_ulong),
        ("dwFlags",     ctypes.c_ulong),
        ("time",        ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class _INPUT(ctypes.Structure):
    class _I(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT)]
    _anonymous_ = ("_i",)
    _fields_    = [("type", ctypes.c_ulong), ("_i", _I)]

def _send_mouse(x=0, y=0, flags=0, mouseData=0, screen_w=1920, screen_h=1080):
    """Direct Windows hardware input block."""
    inp = _INPUT()
    inp.type = 0  # INPUT_MOUSE
    if flags & _MOUSEEVENTF_ABSOLUTE:
        # Scale 0-65535 absolute coordinates
        inp.mi.dx = int(x * 65535 / max(screen_w - 1, 1))
        inp.mi.dy = int(y * 65535 / max(screen_h - 1, 1))
    else:
        inp.mi.dx = x
        inp.mi.dy = y
    
    # Needs to handle negative values properly for scrolling as 32-bit unsigned
    if mouseData < 0:
        inp.mi.mouseData = (1 << 32) + mouseData
    else:
        inp.mi.mouseData = mouseData
        
    inp.mi.dwFlags = flags
    inp.mi.time = 0
    inp.mi.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


class MouseController:
    def __init__(
        self,
        screen_w: int   = 1920,
        screen_h: int   = 1080,
        cam_w:    int   = 640,
        cam_h:    int   = 480,
        smoothing:      float = 0.35,
        sensitivity:    float = 1.0,
        scroll_speed:   int   = 3,
    ):
        self.screen_w     = screen_w
        self.screen_h     = screen_h
        self.smoothing    = smoothing
        self.sensitivity  = sensitivity
        self.scroll_speed = scroll_speed

        self._sx = screen_w / 2.0
        self._sy = screen_h / 2.0
        self._prev_tx = self._sx
        self._prev_ty = self._sy

        self._prev_gesture = "NONE"
        self._dragging = False
        self._paused   = False
        self._last_scroll = 0.0

    def update_settings(self, smoothing=None, sensitivity=None):
        if smoothing   is not None: self.smoothing   = max(0.02, min(1.0, smoothing))
        if sensitivity is not None: self.sensitivity = max(0.3,  min(3.0, sensitivity))

    def handle_gesture(self, gesture: str, lm_x: float, lm_y: float):
        # Map landmark → screen (mirror X)
        target_sx = (1.0 - lm_x) * self.screen_w * self.sensitivity
        target_sy =        lm_y  * self.screen_h * self.sensitivity
        target_sx = max(0, min(self.screen_w - 1, target_sx))
        target_sy = max(0, min(self.screen_h - 1, target_sy))

        # Adaptive EMA smoothing
        speed = math.hypot(target_sx - self._prev_tx, target_sy - self._prev_ty) / max(self.screen_w, 1)
        self._prev_tx, self._prev_ty = target_sx, target_sy
        speed_scale = min(speed / 0.02, 1.0)
        alpha = max(0.08, min(self.smoothing * (0.3 + 1.7 * speed_scale), 1.0))

        self._sx = alpha * target_sx + (1 - alpha) * self._sx
        self._sy = alpha * target_sy + (1 - alpha) * self._sy

        # Continually move the mouse (unless paused)
        if gesture == "PAUSE":
            self._paused = True
        else:
            self._paused = False
            _send_mouse(x=self._sx, y=self._sy, flags=_MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE, 
                        screen_w=self.screen_w, screen_h=self.screen_h)

        # Edge-triggered actions (react immediately when gesture changes)
        if gesture != self._prev_gesture:
            if self._prev_gesture == "DRAG":
                _send_mouse(flags=_MOUSEEVENTF_LEFTUP)
                self._dragging = False

            if not self._paused:
                if gesture == "LEFT_CLICK":
                    _send_mouse(flags=_MOUSEEVENTF_LEFTDOWN | _MOUSEEVENTF_LEFTUP)
                elif gesture == "RIGHT_CLICK":
                    _send_mouse(flags=_MOUSEEVENTF_RIGHTDOWN | _MOUSEEVENTF_RIGHTUP)
                elif gesture == "DOUBLE_CLICK":
                    _send_mouse(flags=_MOUSEEVENTF_LEFTDOWN | _MOUSEEVENTF_LEFTUP)
                    time.sleep(0.05)  # Required to separate the clicks at OS level
                    _send_mouse(flags=_MOUSEEVENTF_LEFTDOWN | _MOUSEEVENTF_LEFTUP)
                elif gesture == "DRAG":
                    _send_mouse(flags=_MOUSEEVENTF_LEFTDOWN)
                    self._dragging = True

        self._prev_gesture = gesture

        # Continuous scroll action (fires repeatedly while held)
        if gesture == "SCROLL":
            now = time.time()
            if now - self._last_scroll > 0.06:
                magnitude = 120 * self.scroll_speed
                if lm_y < 0.40:
                    _send_mouse(flags=_MOUSEEVENTF_WHEEL, mouseData=magnitude)
                    self._last_scroll = now
                elif lm_y > 0.60:
                    _send_mouse(flags=_MOUSEEVENTF_WHEEL, mouseData=-magnitude)
                    self._last_scroll = now

    @property
    def cursor_pos(self): return int(self._sx), int(self._sy)
    @property
    def is_dragging(self): return self._dragging
    @property
    def is_paused(self): return self._paused
