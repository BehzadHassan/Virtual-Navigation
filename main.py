"""
main.py  —  Virtual Navigation  v5 (dual-thread, low-latency)
══════════════════════════════════════════════════════════════
Architecture:
  Thread A (camera)    : reads webcam at full speed → stores latest frame
  Thread B (inference) : picks latest frame → MediaPipe → gesture → mouse
  Main thread          : only drives Tkinter UI at ~30fps

The two bg-threads are DECOUPLED so MediaPipe inference time (30-60ms)
never causes the camera to buffer stale frames.

Run:  python main.py
"""

import tkinter as tk
import threading
import time
import queue
import math

import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
import pyautogui

from gesture_detector   import GestureDetector
from gesture_classifier import GestureClassifier
from mouse_controller   import MouseController

# ── Theme ────────────────────────────────────────────────────────────────────
ACCENT       = "#00f5d4"
ACCENT2      = "#b43ae4"
ACCENT3      = "#ff6b35"
BG_DARK      = "#080c14"
BG_PANEL     = "#0d1520"
BG_CARD      = "#111c2e"
FG_PRIMARY   = "#e8f4fd"
FG_DIM       = "#4a7a9b"
BORDER_COLOR = "#1a2e4a"

GESTURE_COLORS = {
    "MOVE":         "#00f5d4",
    "LEFT_CLICK":   "#00d4ff",
    "RIGHT_CLICK":  "#ff6b35",
    "DOUBLE_CLICK": "#ffe066",
    "SCROLL":       "#b43ae4",
    "DRAG":         "#ff4d6a",
    "PAUSE":        "#4a7a9b",
    "NONE":         "#1a2e4a",
}
GESTURE_ICONS = {
    "MOVE":         "☝",
    "LEFT_CLICK":   "🤏",
    "RIGHT_CLICK":  "🖱",
    "DOUBLE_CLICK": "⚡",
    "SCROLL":       "✌",
    "DRAG":         "✊",
    "PAUSE":        "🖐",
    "NONE":         "·",
}
GESTURE_DESC = {
    "MOVE":         "Index finger up  →  move cursor",
    "LEFT_CLICK":   "Index + Thumb pinch  →  left click",
    "RIGHT_CLICK":  "Middle + Thumb pinch  →  right click",
    "DOUBLE_CLICK": "Quick double pinch  →  double click",
    "SCROLL":       "V-sign  ↕  →  scroll",
    "DRAG":         "Fist  →  drag",
    "PAUSE":        "Open palm  →  freeze cursor",
}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ═══════════════════════════════════════════════════════════════════════════ #
#  Main Window                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class VirtualNavApp(ctk.CTk):

    CAM_W     = 640
    CAM_H     = 480
    DISPLAY_W = 560
    DISPLAY_H = 420

    def __init__(self):
        super().__init__()
        self.title("VIRTUAL NAVIGATION")
        try:
            self.iconbitmap("icon.ico")
        except Exception:
            pass  # Fail gracefully if icon is missing
        self.configure(fg_color=BG_DARK)
        self.minsize(1060, 700)
        self.resizable(True, True)

        # ── Shared state (bg threads write, main thread reads) ────────────
        self._alive           = True
        self._cam_running     = False
        self._inf_running     = False
        self._cam_idx         = 0
        self._cap             = None

        # Latest raw frame (camera thread writes, inference thread reads)
        self._latest_frame      = None
        self._latest_frame_lock = threading.Lock()
        self._new_frame_event   = threading.Event()

        # Latest annotated frame for display (inference thread writes PIL image)
        self._display_frame     = None
        self._display_lock      = threading.Lock()

        self._stats = {
            "fps": 0.0, "gesture": "NONE",
            "cursor_x": 0, "cursor_y": 0,
            "total_clicks": 0, "scroll_dir": "—",
        }
        self._stats_lock      = threading.Lock()
        self._anim_phase      = 0.0
        self._tracking_active = False

        # Mouse / gesture engines
        sw, sh = pyautogui.size()
        self._mouse_ctrl = MouseController(
            screen_w=sw, screen_h=sh,
            cam_w=self.CAM_W, cam_h=self.CAM_H,
            smoothing=0.30, sensitivity=1.0,
        )
        self._detector   = GestureDetector()
        self._classifier = GestureClassifier()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(120, self._refresh_ui)

    # ──────────────────────────────────────────────────────────────────── #
    #  UI                                                                   #
    # ──────────────────────────────────────────────────────────────────── #

    def _build_ui(self):
        tb = tk.Frame(self, bg=BG_PANEL, height=52)
        tb.pack(fill="x", side="top"); tb.pack_propagate(False)
        tk.Label(tb, text="⬡  VIRTUAL NAVIGATION",
                 font=("Consolas", 15, "bold"), bg=BG_PANEL, fg=ACCENT
                 ).pack(side="left", padx=18, pady=12)
        self._fps_var = tk.StringVar(value="FPS: --")
        self._cam_var = tk.StringVar(value="CAM: 0")
        tk.Label(tb, textvariable=self._fps_var,
                 font=("Consolas", 11), bg=BG_PANEL, fg=ACCENT2
                 ).pack(side="right", padx=8)
        tk.Label(tb, textvariable=self._cam_var,
                 font=("Consolas", 11), bg=BG_PANEL, fg=FG_DIM
                 ).pack(side="right", padx=8)
        tk.Frame(self, bg=BORDER_COLOR, height=1).pack(fill="x")

        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True)
        self._build_camera_panel(body)
        self._build_right_panel(body)
        self._build_status_bar()

    def _build_camera_panel(self, parent):
        col = tk.Frame(parent, bg=BG_DARK)
        col.pack(side="left", fill="both", expand=True, padx=16, pady=14)

        bw = self.DISPLAY_W + 14; bh = self.DISPLAY_H + 14
        self._border_canvas = tk.Canvas(col, width=bw, height=bh,
                                         bg=BG_DARK, highlightthickness=0)
        self._border_canvas.pack()
        self._cam_label = tk.Label(self._border_canvas, bg="#000000",
                                   width=self.DISPLAY_W, height=self.DISPLAY_H)
        self._cam_label.place(x=7, y=7)

        gr = tk.Frame(col, bg=BG_DARK); gr.pack(pady=(8, 0))
        self._icon_var = tk.StringVar(value="·")
        self._name_var = tk.StringVar(value="NONE")
        self._icon_lbl = tk.Label(gr, textvariable=self._icon_var,
                                  font=("Segoe UI Emoji", 26), bg=BG_DARK, fg=FG_DIM)
        self._icon_lbl.pack(side="left", padx=(0, 8))
        self._name_lbl = tk.Label(gr, textvariable=self._name_var,
                                  font=("Consolas", 14, "bold"), bg=BG_DARK, fg=FG_DIM,
                                  width=14, anchor="w")
        self._name_lbl.pack(side="left")

        self._cursor_var = tk.StringVar(value="CURSOR   X: ----   Y: ----")
        tk.Label(col, textvariable=self._cursor_var,
                 font=("Consolas", 10), bg=BG_DARK, fg=FG_DIM).pack(pady=(4, 0))

        self._toggle_btn = ctk.CTkButton(
            col, text="▶  START TRACKING",
            font=ctk.CTkFont("Consolas", 13, weight="bold"),
            fg_color=ACCENT, hover_color=ACCENT2, text_color=BG_DARK,
            corner_radius=8, height=40, width=230,
            command=self._toggle_tracking)
        self._toggle_btn.pack(pady=(10, 0))

    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg=BG_DARK, width=350)
        right.pack(side="right", fill="y", padx=(0, 16), pady=14)
        right.pack_propagate(False)

        self._section(right, "GESTURE LEGEND")
        card = self._card(right)
        for g, desc in GESTURE_DESC.items():
            row = tk.Frame(card, bg=BG_CARD); row.pack(fill="x", padx=10, pady=3)
            tk.Label(row, text=GESTURE_ICONS[g], font=("Segoe UI Emoji", 13),
                     bg=BG_CARD, fg=GESTURE_COLORS[g], width=3).pack(side="left")
            tk.Label(row, text=desc, font=("Consolas", 9),
                     bg=BG_CARD, fg=FG_PRIMARY, anchor="w"
                     ).pack(side="left", fill="x", expand=True)

        self._section(right, "SETTINGS")
        sc = self._card(right)
        self._sens_var   = tk.DoubleVar(value=1.0)
        self._smooth_var = tk.DoubleVar(value=0.30)
        self._slider_row(sc, "Sensitivity", self._sens_var,
                         0.3, 2.5, self._on_sensitivity_change)
        self._slider_row(sc, "Smoothing", self._smooth_var,
                         0.05, 1.0, self._on_smooth_change)

        cr = tk.Frame(sc, bg=BG_CARD); cr.pack(fill="x", padx=12, pady=(4, 10))
        tk.Label(cr, text="Camera", font=("Consolas", 10),
                 bg=BG_CARD, fg=FG_DIM).pack(side="left")
        self._cam_menu = ctk.CTkOptionMenu(
            cr, values=["CAM 0", "CAM 1", "CAM 2"],
            font=ctk.CTkFont("Consolas", 10),
            fg_color=BG_PANEL, button_color=ACCENT2,
            dropdown_fg_color=BG_CARD, text_color=FG_PRIMARY,
            command=self._on_cam_change)
        self._cam_menu.pack(side="right")

        self._section(right, "SESSION STATS")
        stc = self._card(right)
        self._clicks_var = tk.StringVar(value="Clicks:       0")
        self._scroll_var = tk.StringVar(value="Scroll:       —")
        self._mode_var   = tk.StringVar(value="Mode:    IDLE")
        for v in (self._clicks_var, self._scroll_var, self._mode_var):
            tk.Label(stc, textvariable=v, font=("Consolas", 10),
                     bg=BG_CARD, fg=FG_PRIMARY, anchor="w",
                     padx=12, pady=3).pack(fill="x")

    def _section(self, p, txt):
        r = tk.Frame(p, bg=BG_DARK); r.pack(fill="x", pady=(8, 2))
        tk.Label(r, text=txt, font=("Consolas", 9, "bold"),
                 bg=BG_DARK, fg=ACCENT2).pack(side="left")
        tk.Frame(r, bg=BORDER_COLOR, height=1).pack(
            side="left", fill="x", expand=True, padx=(6, 0), pady=6)

    def _card(self, p):
        f = tk.Frame(p, bg=BG_CARD,
                     highlightbackground=BORDER_COLOR, highlightthickness=1)
        f.pack(fill="x", pady=(0, 6)); return f

    def _slider_row(self, p, label, var, from_, to, cmd):
        row = tk.Frame(p, bg=BG_CARD); row.pack(fill="x", padx=12, pady=4)
        tk.Label(row, text=label, font=("Consolas", 10), bg=BG_CARD,
                 fg=FG_DIM, width=11, anchor="w").pack(side="left")
        vl = tk.Label(row, text=f"{var.get():.2f}", font=("Consolas", 10),
                      bg=BG_CARD, fg=ACCENT, width=5)
        vl.pack(side="right")
        def _u(v): vl.configure(text=f"{float(v):.2f}"); cmd(float(v))
        ctk.CTkSlider(row, from_=from_, to=to, variable=var, command=_u,
                      fg_color=BORDER_COLOR, progress_color=ACCENT2,
                      button_color=ACCENT, button_hover_color=ACCENT2,
                      width=120).pack(side="right", padx=(0, 6))

    def _build_status_bar(self):
        tk.Frame(self, bg=BORDER_COLOR, height=1).pack(fill="x", side="bottom")
        sb = tk.Frame(self, bg=BG_PANEL, height=32)
        sb.pack(fill="x", side="bottom"); sb.pack_propagate(False)
        self._status_dot = tk.Label(sb, text="●", font=("Consolas", 11),
                                    bg=BG_PANEL, fg=FG_DIM)
        self._status_dot.pack(side="left", padx=(10, 4), pady=6)
        self._status_var = tk.StringVar(value="IDLE  —  Start tracking to begin")
        tk.Label(sb, textvariable=self._status_var,
                 font=("Consolas", 10), bg=BG_PANEL, fg=FG_DIM).pack(side="left")
        tk.Label(sb, text="Virtual Navigation  v5",
                 font=("Consolas", 9), bg=BG_PANEL, fg=FG_DIM
                 ).pack(side="right", padx=12)

    # ──────────────────────────────────────────────────────────────────── #
    #  Tracking — TWO BACKGROUND THREADS                                   #
    # ──────────────────────────────────────────────────────────────────── #

    def _toggle_tracking(self):
        if self._cam_running:
            self._stop_tracking()
        else:
            self._start_tracking()

    def _start_tracking(self):
        self._cap = cv2.VideoCapture(self._cam_idx, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.CAM_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_H)
        self._cap.set(cv2.CAP_PROP_FPS,          60)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        if not self._cap.isOpened():
            self._status_var.set("ERROR  —  Cannot open camera")
            self._cap = None; return

        self._cam_running = True
        self._inf_running = True
        self._tracking_active = True
        self._stats["total_clicks"] = 0
        self._toggle_btn.configure(text="■  STOP TRACKING", fg_color=ACCENT3)

        # Thread A: camera capture (lightweight, just reads frames)
        threading.Thread(target=self._camera_loop, daemon=True).start()
        # Thread B: inference (MediaPipe + classify + mouse)
        threading.Thread(target=self._inference_loop, daemon=True).start()

    def _stop_tracking(self):
        self._cam_running = False
        self._inf_running = False
        self._tracking_active = False
        self._new_frame_event.set()   # unblock inference thread
        self._toggle_btn.configure(text="▶  START TRACKING", fg_color=ACCENT)
        if self._cap:
            self._cap.release(); self._cap = None
        self._cam_label.configure(image="", text="[ NO SIGNAL ]",
                                  font=("Consolas", 16), fg=FG_DIM, compound="none")

    # ── Thread A: Camera capture ─────────────────────────────────────────

    def _camera_loop(self):
        """
        Reads frames as fast as the camera allows.
        Stores ONLY the most recent frame — never queues stale ones.
        """
        while self._cam_running:
            if not self._cap:
                break
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.005); continue

            # Overwrite latest frame (no queue, always fresh)
            with self._latest_frame_lock:
                self._latest_frame = frame
            self._new_frame_event.set()   # signal inference thread

    # ── Thread B: Inference ──────────────────────────────────────────────

    def _inference_loop(self):
        """
        Waits for a new frame → runs MediaPipe → classifies → moves mouse.
        Decoupled from camera so inference latency doesn't cause frame buildup.
        """
        fps_buf   = []
        prev_time = time.time()

        while self._inf_running:
            # Wait for a new frame (with timeout so we can exit cleanly)
            if not self._new_frame_event.wait(timeout=0.5):
                continue
            self._new_frame_event.clear()

            if not self._inf_running:
                break

            with self._latest_frame_lock:
                frame = self._latest_frame
            if frame is None:
                continue

            # FPS tracking
            now = time.time()
            dt  = max(now - prev_time, 1e-6)
            prev_time = now
            fps_buf.append(1.0 / dt)
            if len(fps_buf) > 30: fps_buf.pop(0)
            fps = sum(fps_buf) / len(fps_buf)

            # MediaPipe detection (returns model's gesture label too)
            annotated, landmarks, mp_gesture = self._detector.detect(frame)

            # Gesture classification (model label + pinch detection)
            gesture = self._classifier.classify(landmarks, mp_gesture)

            # Mouse control
            if landmarks:
                ix = landmarks[8].x
                iy = landmarks[8].y
                self._mouse_ctrl.handle_gesture(gesture, ix, iy)

                with self._stats_lock:
                    if gesture in ("LEFT_CLICK", "DOUBLE_CLICK"):
                        self._stats["total_clicks"] += 1
                    if gesture == "SCROLL":
                        self._stats["scroll_dir"] = "↑" if iy < 0.42 else "↓"
            else:
                # No hand: stop drag if active
                if self._mouse_ctrl.is_dragging:
                    self._mouse_ctrl.handle_gesture("NONE", 0.5, 0.5)

            # Draw HUD overlay
            self._draw_hud(annotated, gesture, fps, landmarks)

            # Pre-process frame for display (heavy work done HERE, not on UI thread)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.DISPLAY_W, self.DISPLAY_H),
                             interpolation=cv2.INTER_LINEAR)
            pil = Image.fromarray(rgb)

            cx, cy = self._mouse_ctrl.cursor_pos
            with self._stats_lock:
                self._stats.update({
                    "fps": fps, "gesture": gesture,
                    "cursor_x": cx, "cursor_y": cy,
                })

            with self._display_lock:
                self._display_frame = pil

    # ── Hand-size distance estimation ────────────────────────────────────
    # We measure wrist (0) → middle MCP (9) in pixels.
    # Empirical ranges for a typical 640x480 webcam at arm's-length:
    #   < 55 px  → too far
    #   55-110 px → perfect zone
    #   > 110 px  → too close
    _DIST_TOO_FAR   = 55
    _DIST_TOO_CLOSE = 110
    _DIST_IDEAL_LO  = 60
    _DIST_IDEAL_HI  = 100

    def _hand_size_px(self, landmarks, fw, fh):
        """Pixel distance wrist→middle-MCP. Returns None if no hand."""
        if landmarks is None:
            return None
        lm = landmarks
        x0, y0 = lm[0].x * fw, lm[0].y * fh   # WRIST
        x9, y9 = lm[9].x * fw, lm[9].y * fh   # MIDDLE MCP
        import math
        return math.hypot(x9 - x0, y9 - y0)

    def _draw_hud(self, frame, gesture, fps, landmarks=None):
        h, w = frame.shape[:2]

        # ── 1. Virtual Interaction Plane ─────────────────────────────────
        # Centre rectangle — hand should be inside this area
        margin_x = int(w * 0.12)
        margin_y = int(h * 0.10)
        px1, py1 = margin_x, margin_y
        px2, py2 = w - margin_x, h - margin_y - 28   # leave room for bottom bar

        hand_size = self._hand_size_px(landmarks, w, h)

        if hand_size is None:
            # No hand detected — draw guide in dim cyan
            zone_col  = (80, 180, 160)
            dist_label = "SHOW YOUR HAND"
            dist_col   = (80, 180, 160)
            depth_frac = 0.0
        elif hand_size < self._DIST_TOO_FAR:
            zone_col  = (60, 60, 255)       # red-ish (BGR)
            dist_label = "MOVE CLOSER"
            dist_col   = (60, 60, 255)
            depth_frac = hand_size / self._DIST_IDEAL_LO
        elif hand_size > self._DIST_TOO_CLOSE:
            zone_col  = (30, 30, 255)
            dist_label = "MOVE BACK"
            dist_col   = (30, 30, 255)
            depth_frac = 1.0
        else:
            # Perfect zone
            t = (hand_size - self._DIST_IDEAL_LO) / max(self._DIST_IDEAL_HI - self._DIST_IDEAL_LO, 1)
            t = max(0.0, min(1.0, t))
            zone_col  = (0, 245, 212)       # neon cyan — perfect
            dist_label = "\u2713 PERFECT ZONE"
            dist_col   = (0, 220, 100)
            depth_frac = 0.3 + 0.7 * t

        # Draw dashed glowing rectangle (draw in segments)
        def _dashed_rect(img, p1, p2, color, thickness=2, dash=14, gap=7):
            x1, y1 = p1; x2, y2 = p2
            # Glow pass
            for pts in [((x1,y1),(x2,y1)), ((x2,y1),(x2,y2)),
                        ((x2,y2),(x1,y2)), ((x1,y2),(x1,y1))]:
                cv2.line(img, pts[0], pts[1], color, thickness+3, cv2.LINE_AA)
            # Top
            x = x1
            while x < x2:
                ex = min(x + dash, x2)
                cv2.line(img, (x, y1), (ex, y1), color, thickness, cv2.LINE_AA)
                x += dash + gap
            # Bottom
            x = x1
            while x < x2:
                ex = min(x + dash, x2)
                cv2.line(img, (x, y2), (ex, y2), color, thickness, cv2.LINE_AA)
                x += dash + gap
            # Left
            y = y1
            while y < y2:
                ey = min(y + dash, y2)
                cv2.line(img, (x1, y), (x1, ey), color, thickness, cv2.LINE_AA)
                y += dash + gap
            # Right
            y = y1
            while y < y2:
                ey = min(y + dash, y2)
                cv2.line(img, (x2, y), (x2, ey), color, thickness, cv2.LINE_AA)
                y += dash + gap

        # Semi-transparent fill inside the plane
        ov = frame.copy()
        cv2.rectangle(ov, (px1, py1), (px2, py2), zone_col, -1)
        cv2.addWeighted(ov, 0.07, frame, 0.93, 0, frame)

        _dashed_rect(frame, (px1, py1), (px2, py2), zone_col, thickness=2)

        # Corner brackets
        cs = 20
        bk_col = zone_col; bk_t = 3
        for cx_, cy_, dx, dy in [
            (px1, py1,  1,  1), (px2, py1, -1,  1),
            (px1, py2,  1, -1), (px2, py2, -1, -1),
        ]:
            cv2.line(frame, (cx_, cy_), (cx_ + dx*cs, cy_),        bk_col, bk_t, cv2.LINE_AA)
            cv2.line(frame, (cx_, cy_), (cx_,          cy_ + dy*cs), bk_col, bk_t, cv2.LINE_AA)

        # ── 2. Virtual Plane label (inside the box, top-centre) ───────────
        lbl_plane = "VIRTUAL INTERACTION PLANE"
        (lw, lh_), _ = cv2.getTextSize(lbl_plane, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        lx = (px1 + px2) // 2 - lw // 2
        cv2.putText(frame, lbl_plane,
                    (lx, py1 + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, zone_col, 1, cv2.LINE_AA)

        # ── 3. Distance metre (left edge, vertical bar) ───────────────────
        bar_x = 10; bar_y1 = py1 + 30; bar_y2 = py2 - 10
        bar_h  = bar_y2 - bar_y1
        fill_h = int(bar_h * min(max(depth_frac, 0.0), 1.0))

        # Background track
        cv2.rectangle(frame, (bar_x, bar_y1), (bar_x + 12, bar_y2),
                      (30, 30, 30), -1)
        # Filled portion (bottom to top = far to close)
        if fill_h > 0:
            cv2.rectangle(frame,
                          (bar_x, bar_y2 - fill_h),
                          (bar_x + 12, bar_y2),
                          zone_col, -1)
        # Ideal zone indicator lines
        ideal_lo_y = bar_y2 - int(bar_h * 0.30)
        ideal_hi_y = bar_y2 - int(bar_h * 0.80)
        cv2.line(frame, (bar_x - 3, ideal_lo_y), (bar_x + 15, ideal_lo_y),
                 (0, 245, 212), 1, cv2.LINE_AA)
        cv2.line(frame, (bar_x - 3, ideal_hi_y), (bar_x + 15, ideal_hi_y),
                 (0, 245, 212), 1, cv2.LINE_AA)
        # Labels
        cv2.putText(frame, "FAR",  (bar_x - 2, bar_y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120,120,120), 1, cv2.LINE_AA)
        cv2.putText(frame, "NEAR", (bar_x - 2, bar_y2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120,120,120), 1, cv2.LINE_AA)

        # ── 4. Distance status label (below the plane) ────────────────────
        ov2 = frame.copy()
        cv2.rectangle(ov2, (px1, py2 - 2), (px2, py2 + 22), (0, 0, 0), -1)
        cv2.addWeighted(ov2, 0.55, frame, 0.45, 0, frame)
        (dlw, _), _ = cv2.getTextSize(dist_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        dlx = (px1 + px2) // 2 - dlw // 2
        cv2.putText(frame, dist_label, (dlx, py2 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, dist_col, 2, cv2.LINE_AA)

        # ── 5. Top-left info box (gesture + fps + T/I/M/R/P debug) ────────
        ov3 = frame.copy()
        cv2.rectangle(ov3, (0, 0), (260, 80), (0, 0, 0), -1)
        cv2.addWeighted(ov3, 0.55, frame, 0.45, 0, frame)
        gcol = tuple(reversed(hex_to_rgb(GESTURE_COLORS.get(gesture, FG_DIM))))
        cv2.putText(frame, f"GESTURE: {gesture}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, gcol, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.0f}",
                    (190, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 220, 80), 1, cv2.LINE_AA)

        # Debug: T/I/M/R/P finger states (Fix 6 debug overlay)
        dbg = self._classifier.debug_states
        labels = [("T", dbg.get("T")), ("I", dbg.get("I")), ("M", dbg.get("M")),
                  ("R", dbg.get("R")), ("P", dbg.get("P"))]
        dx = 14
        for idx_f, (lbl, state) in enumerate(labels):
            dot_col  = (0, 220, 80) if state else (60, 60, 220)
            tx = 12 + idx_f * 46
            cv2.circle(frame, (tx, 50), 8, dot_col, -1, cv2.LINE_AA)
            cv2.putText(frame, lbl, (tx - 5, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "TIMRP",
                    (240, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(frame, "fingers",
                    (12, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 80, 80), 1, cv2.LINE_AA)

        # ── 6. Bottom cursor bar ──────────────────────────────────────────
        cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 0), -1)
        cx, cy = self._mouse_ctrl.cursor_pos
        tag = "[PAUSED]" if self._mouse_ctrl.is_paused else ""
        cv2.putText(frame, f"  CURSOR  X:{cx:5d}  Y:{cy:5d}  {tag}",
                    (4, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, (0, 245, 212), 1, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────────── #
    #  UI refresh (main thread — ~30fps)                                   #
    # ──────────────────────────────────────────────────────────────────── #

    def _refresh_ui(self):
        if not self._alive:
            return

        self._draw_border()

        # Grab latest pre-processed frame
        with self._display_lock:
            pil = self._display_frame
            self._display_frame = None

        if pil is not None:
            photo = ImageTk.PhotoImage(pil)
            self._cam_label.configure(image=photo, text="", compound="none")
            self._cam_label._photo = photo

        # Stats
        with self._stats_lock:
            s = dict(self._stats)

        gesture = s["gesture"]
        col     = GESTURE_COLORS.get(gesture, FG_DIM)

        self._fps_var.set(f"FPS: {s['fps']:.0f}")
        self._cam_var.set(f"CAM: {self._cam_idx}")
        self._cursor_var.set(f"CURSOR   X: {s['cursor_x']:5d}   Y: {s['cursor_y']:5d}")
        self._icon_var.set(GESTURE_ICONS.get(gesture, "·"))
        self._name_var.set(gesture)

        if self._tracking_active:
            self._icon_lbl.configure(fg=col)
            self._name_lbl.configure(fg=col)
            mode = "PAUSED" if self._mouse_ctrl.is_paused else gesture
            self._mode_var.set(f"Mode:    {mode}")
            self._status_dot.configure(fg=col)
            self._status_var.set(
                f"TRACKING ACTIVE   Gesture: {gesture:<14s}  "
                f"Clicks: {s['total_clicks']:4d}   Scroll: {s['scroll_dir']}"
            )
        else:
            self._icon_lbl.configure(fg=FG_DIM)
            self._name_lbl.configure(fg=FG_DIM)
            self._mode_var.set("Mode:    IDLE")
            self._status_dot.configure(fg=FG_DIM)
            self._status_var.set("IDLE  —  Start tracking to begin")

        self._clicks_var.set(f"Clicks:       {s['total_clicks']}")
        self._scroll_var.set(f"Scroll:       {s['scroll_dir']}")

        if self._alive:
            self.after(33, self._refresh_ui)

    def _draw_border(self):
        try:
            c = self._border_canvas
            if not c.winfo_exists(): return
        except Exception:
            return
        self._anim_phase = (self._anim_phase + 0.09) % (2 * math.pi)
        a = (math.sin(self._anim_phase) + 1) / 2
        g = self._stats.get("gesture", "NONE")
        base = GESTURE_COLORS.get(g, ACCENT) if self._tracking_active else FG_DIM
        r2, g2, b2 = hex_to_rgb(base)
        col   = f"#{int(r2*(0.3+0.7*a)):02x}{int(g2*(0.3+0.7*a)):02x}{int(b2*(0.3+0.7*a)):02x}"
        thick = int(2 + a * 4) if self._tracking_active else 1
        bw = self.DISPLAY_W + 14; bh = self.DISPLAY_H + 14
        c.delete("border")
        p = 2; cs = 22
        c.create_rectangle(p, p, bw-p, bh-p, outline=col, width=thick, tags="border")
        for x1,y1,x2,y2 in [
            (p,p,p+cs,p),(p,p,p,p+cs),(bw-p-cs,p,bw-p,p),(bw-p,p,bw-p,p+cs),
            (p,bh-p,p+cs,bh-p),(p,bh-p-cs,p,bh-p),(bw-p-cs,bh-p,bw-p,bh-p),
            (bw-p,bh-p-cs,bw-p,bh-p),
        ]:
            c.create_line(x1,y1,x2,y2, fill=col, width=thick+1, tags="border")

    # ──────────────────────────────────────────────────────────────────── #
    #  Settings                                                             #
    # ──────────────────────────────────────────────────────────────────── #

    def _on_sensitivity_change(self, v): self._mouse_ctrl.update_settings(sensitivity=v)
    def _on_smooth_change(self, v):      self._mouse_ctrl.update_settings(smoothing=v)

    def _on_cam_change(self, choice):
        idx = int(choice.split()[-1])
        if idx != self._cam_idx:
            was = self._cam_running
            if was: self._stop_tracking(); time.sleep(0.3)
            self._cam_idx = idx
            if was: self._start_tracking()

    # ──────────────────────────────────────────────────────────────────── #
    #  Cleanup                                                              #
    # ──────────────────────────────────────────────────────────────────── #

    def _on_close(self):
        self._alive       = False
        self._cam_running = False
        self._inf_running = False
        self._new_frame_event.set()
        if self._cap:
            self._cap.release()
        try: self._detector.release()
        except: pass
        self.destroy()


if __name__ == "__main__":
    app = VirtualNavApp()
    app.mainloop()
