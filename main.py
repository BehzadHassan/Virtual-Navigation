"""
main.py  –  Virtual Navigation
════════════════════════════════════════════════════════════════════════════════
Futuristic hand-gesture mouse controller.
Webcam → MediaPipe → gesture → OS mouse actions.

Run:  python main.py
"""

import tkinter as tk
import threading
import time
import queue
import math
import sys

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
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
    "SCROLL":       "V-sign  ↕ move hand  →  scroll",
    "DRAG":         "Fist  →  drag / hold",
    "PAUSE":        "Open palm  →  freeze cursor",
}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


# ── Helpers ──────────────────────────────────────────────────────────────────

def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def add_scanlines(img: Image.Image, alpha: int = 14) -> Image.Image:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for y in range(0, img.height, 3):
        draw.line([(0, y), (img.width, y)], fill=(0, 0, 0, alpha))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def dim_color(hex_col: str, factor: float) -> str:
    r, g, b = hex_to_rgb(hex_col)
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"


# ═══════════════════════════════════════════════════════════════════════════ #
#  Main Application                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

class VirtualNavApp(ctk.CTk):

    CAM_W     = 640
    CAM_H     = 480
    DISPLAY_W = 560
    DISPLAY_H = 420

    def __init__(self):
        super().__init__()

        self.title("VIRTUAL NAVIGATION")
        self.configure(fg_color=BG_DARK)
        self.minsize(1060, 700)
        self.resizable(True, True)

        # ── Internal state ───────────────────────────────────────────────
        self._alive           = True
        self._running         = False
        self._cam_idx         = 0
        self._cap             = None
        self._thread          = None
        self._frame_q: queue.Queue = queue.Queue(maxsize=2)
        self._anim_phase      = 0.0
        self._stats = {
            "fps": 0.0, "gesture": "NONE",
            "cursor_x": 0, "cursor_y": 0,
            "total_clicks": 0, "scroll_dir": "—",
        }
        self._tracking_active = False

        sw, sh = pyautogui.size()
        self._mouse_ctrl = MouseController(
            screen_w=sw, screen_h=sh,
            cam_w=self.CAM_W, cam_h=self.CAM_H,
            smoothing=0.25, sensitivity=1.0,
        )
        self._detector   = GestureDetector()
        self._classifier = GestureClassifier()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # Delay first refresh so all widgets are fully initialised
        self.after(100, self._refresh_ui)

    # ──────────────────────────────────────────────────────────────────── #
    #  UI Build                                                             #
    # ──────────────────────────────────────────────────────────────────── #

    def _build_ui(self):
        # Title bar
        tb = tk.Frame(self, bg=BG_PANEL, height=52)
        tb.pack(fill="x", side="top")
        tb.pack_propagate(False)

        tk.Label(tb, text="⬡  VIRTUAL NAVIGATION",
                 font=("Consolas", 15, "bold"),
                 bg=BG_PANEL, fg=ACCENT).pack(side="left", padx=18, pady=12)

        self._fps_var = tk.StringVar(value="FPS: --")
        self._cam_var = tk.StringVar(value="CAM: 0")
        tk.Label(tb, textvariable=self._fps_var,
                 font=("Consolas", 11), bg=BG_PANEL, fg=ACCENT2
                 ).pack(side="right", padx=8)
        tk.Label(tb, textvariable=self._cam_var,
                 font=("Consolas", 11), bg=BG_PANEL, fg=FG_DIM
                 ).pack(side="right", padx=8)

        tk.Frame(self, bg=BORDER_COLOR, height=1).pack(fill="x")

        # Body
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True)

        self._build_camera_panel(body)
        self._build_right_panel(body)
        self._build_status_bar()

    # ── Camera panel ─────────────────────────────────────────────────────

    def _build_camera_panel(self, parent):
        cam_col = tk.Frame(parent, bg=BG_DARK)
        cam_col.pack(side="left", fill="both", expand=True, padx=16, pady=14)

        # Border canvas (drawn by _refresh_ui, NOT by its own after loop)
        bw = self.DISPLAY_W + 14
        bh = self.DISPLAY_H + 14
        self._border_canvas = tk.Canvas(
            cam_col, width=bw, height=bh,
            bg=BG_DARK, highlightthickness=0
        )
        self._border_canvas.pack()

        # Camera image label placed inside the border canvas
        self._cam_label = tk.Label(
            self._border_canvas, bg="#000000",
            width=self.DISPLAY_W, height=self.DISPLAY_H,
        )
        self._cam_label.place(x=7, y=7)

        # Gesture indicator row
        gest_row = tk.Frame(cam_col, bg=BG_DARK)
        gest_row.pack(pady=(8, 0))

        self._icon_var = tk.StringVar(value="·")
        self._name_var = tk.StringVar(value="NONE")
        self._icon_lbl = tk.Label(gest_row, textvariable=self._icon_var,
                                  font=("Segoe UI Emoji", 26), bg=BG_DARK, fg=FG_DIM)
        self._icon_lbl.pack(side="left", padx=(0, 8))
        self._name_lbl = tk.Label(gest_row, textvariable=self._name_var,
                                  font=("Consolas", 14, "bold"), bg=BG_DARK, fg=FG_DIM,
                                  width=14, anchor="w")
        self._name_lbl.pack(side="left")

        # Cursor readout
        self._cursor_var = tk.StringVar(value="CURSOR   X: ----   Y: ----")
        tk.Label(cam_col, textvariable=self._cursor_var,
                 font=("Consolas", 10), bg=BG_DARK, fg=FG_DIM).pack(pady=(4, 0))

        # Start/Stop button
        self._toggle_btn = ctk.CTkButton(
            cam_col,
            text="▶  START TRACKING",
            font=ctk.CTkFont("Consolas", 13, weight="bold"),
            fg_color=ACCENT, hover_color=ACCENT2,
            text_color=BG_DARK,
            corner_radius=8, height=40, width=230,
            command=self._toggle_tracking,
        )
        self._toggle_btn.pack(pady=(10, 0))

    # ── Right panel ───────────────────────────────────────────────────────

    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg=BG_DARK, width=350)
        right.pack(side="right", fill="y", padx=(0, 16), pady=14)
        right.pack_propagate(False)

        # Gesture Legend
        self._section(right, "GESTURE LEGEND")
        card = self._card(right)
        for gesture, desc in GESTURE_DESC.items():
            row = tk.Frame(card, bg=BG_CARD)
            row.pack(fill="x", padx=10, pady=3)
            tk.Label(row, text=GESTURE_ICONS[gesture],
                     font=("Segoe UI Emoji", 13),
                     bg=BG_CARD, fg=GESTURE_COLORS[gesture], width=3
                     ).pack(side="left")
            tk.Label(row, text=desc, font=("Consolas", 9),
                     bg=BG_CARD, fg=FG_PRIMARY, anchor="w"
                     ).pack(side="left", fill="x", expand=True)

        # Settings
        self._section(right, "SETTINGS")
        scard = self._card(right)

        self._sens_var   = tk.DoubleVar(value=1.0)
        self._smooth_var = tk.DoubleVar(value=0.25)
        self._slider_row(scard, "Sensitivity", self._sens_var,
                         0.3, 2.5, self._on_sensitivity_change)
        self._slider_row(scard, "Smoothing",   self._smooth_var,
                         0.05, 1.0, self._on_smooth_change)

        cam_row = tk.Frame(scard, bg=BG_CARD)
        cam_row.pack(fill="x", padx=12, pady=(4, 10))
        tk.Label(cam_row, text="Camera", font=("Consolas", 10),
                 bg=BG_CARD, fg=FG_DIM).pack(side="left")
        self._cam_menu = ctk.CTkOptionMenu(
            cam_row, values=["CAM 0", "CAM 1", "CAM 2"],
            font=ctk.CTkFont("Consolas", 10),
            fg_color=BG_PANEL, button_color=ACCENT2,
            dropdown_fg_color=BG_CARD, text_color=FG_PRIMARY,
            command=self._on_cam_change,
        )
        self._cam_menu.pack(side="right")

        # Stats
        self._section(right, "SESSION STATS")
        stcard = self._card(right)
        self._clicks_var = tk.StringVar(value="Clicks:       0")
        self._scroll_var = tk.StringVar(value="Scroll:       —")
        self._mode_var   = tk.StringVar(value="Mode:    IDLE")
        for var in (self._clicks_var, self._scroll_var, self._mode_var):
            tk.Label(stcard, textvariable=var, font=("Consolas", 10),
                     bg=BG_CARD, fg=FG_PRIMARY, anchor="w",
                     padx=12, pady=3).pack(fill="x")

    def _section(self, parent, text):
        row = tk.Frame(parent, bg=BG_DARK)
        row.pack(fill="x", pady=(8, 2))
        tk.Label(row, text=text, font=("Consolas", 9, "bold"),
                 bg=BG_DARK, fg=ACCENT2).pack(side="left")
        tk.Frame(row, bg=BORDER_COLOR, height=1).pack(
            side="left", fill="x", expand=True, padx=(6, 0), pady=6)

    def _card(self, parent):
        f = tk.Frame(parent, bg=BG_CARD,
                     highlightbackground=BORDER_COLOR, highlightthickness=1)
        f.pack(fill="x", pady=(0, 6))
        return f

    def _slider_row(self, parent, label, var, from_, to, cmd):
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill="x", padx=12, pady=4)
        tk.Label(row, text=label, font=("Consolas", 10),
                 bg=BG_CARD, fg=FG_DIM, width=11, anchor="w").pack(side="left")
        val_lbl = tk.Label(row, text=f"{var.get():.2f}",
                           font=("Consolas", 10), bg=BG_CARD, fg=ACCENT, width=5)
        val_lbl.pack(side="right")

        def _upd(v):
            val_lbl.configure(text=f"{float(v):.2f}")
            cmd(float(v))

        ctk.CTkSlider(
            row, from_=from_, to=to, variable=var, command=_upd,
            fg_color=BORDER_COLOR, progress_color=ACCENT2,
            button_color=ACCENT, button_hover_color=ACCENT2, width=120,
        ).pack(side="right", padx=(0, 6))

    # ── Status bar ────────────────────────────────────────────────────────

    def _build_status_bar(self):
        tk.Frame(self, bg=BORDER_COLOR, height=1).pack(fill="x", side="bottom")
        sb = tk.Frame(self, bg=BG_PANEL, height=32)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)

        self._status_dot = tk.Label(sb, text="●", font=("Consolas", 11),
                                    bg=BG_PANEL, fg=FG_DIM)
        self._status_dot.pack(side="left", padx=(10, 4), pady=6)

        self._status_var = tk.StringVar(value="IDLE  —  Start tracking to begin")
        tk.Label(sb, textvariable=self._status_var,
                 font=("Consolas", 10), bg=BG_PANEL, fg=FG_DIM).pack(side="left")

        tk.Label(sb, text="Virtual Navigation  v1.0",
                 font=("Consolas", 9), bg=BG_PANEL, fg=FG_DIM
                 ).pack(side="right", padx=12)

    # ──────────────────────────────────────────────────────────────────── #
    #  Tracking                                                             #
    # ──────────────────────────────────────────────────────────────────── #

    def _toggle_tracking(self):
        if self._running:
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
            self._cap = None
            return

        self._running = True
        self._tracking_active = True
        self._stats["total_clicks"] = 0
        self._toggle_btn.configure(text="■  STOP TRACKING", fg_color=ACCENT3)

        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()

    def _stop_tracking(self):
        self._running = False
        self._tracking_active = False
        self._toggle_btn.configure(text="▶  START TRACKING", fg_color=ACCENT)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._cam_label.configure(image="", text="[ NO SIGNAL ]",
                                  font=("Consolas", 16), fg=FG_DIM,
                                  compound="none")

    def _tracking_loop(self):
        fps_buf   = []
        prev_time = time.time()

        while self._running:
            if self._cap is None:
                break
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            now = time.time()
            dt  = max(now - prev_time, 1e-6)
            prev_time = now
            fps_buf.append(1.0 / dt)
            if len(fps_buf) > 20:
                fps_buf.pop(0)
            fps = sum(fps_buf) / len(fps_buf)

            annotated, landmarks, _ = self._detector.detect(frame)
            gesture = self._classifier.classify(landmarks)

            if landmarks:
                ix = landmarks[8].x
                iy = landmarks[8].y
                self._mouse_ctrl.handle_gesture(gesture, ix, iy)
                if gesture in ("LEFT_CLICK", "DOUBLE_CLICK"):
                    self._stats["total_clicks"] += 1
                if gesture == "SCROLL":
                    self._stats["scroll_dir"] = "↑" if iy < 0.45 else "↓"

            self._draw_hud(annotated, gesture, fps)

            cx, cy = self._mouse_ctrl.cursor_pos
            self._stats.update({
                "fps": fps, "gesture": gesture,
                "cursor_x": cx, "cursor_y": cy,
            })

            try:
                self._frame_q.put_nowait(annotated)
            except queue.Full:
                pass

    def _draw_hud(self, frame, gesture, fps):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (250, 58), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        col = tuple(reversed(hex_to_rgb(GESTURE_COLORS.get(gesture, FG_DIM))))
        cv2.putText(frame, f"GESTURE: {gesture}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 1, cv2.LINE_AA)

        cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 0), -1)
        cx, cy = self._mouse_ctrl.cursor_pos
        tag = "[PAUSED]" if self._mouse_ctrl.is_paused else ""
        cv2.putText(frame, f"  CURSOR  X:{cx:5d}  Y:{cy:5d}  {tag}",
                    (4, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, (0, 245, 212), 1, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────────── #
    #  UI Refresh loop (main thread only)                                   #
    # ──────────────────────────────────────────────────────────────────── #

    def _refresh_ui(self):
        if not self._alive:
            return

        # Draw animated border
        self._draw_border()

        # Push latest frame to camera label
        frame = None
        while not self._frame_q.empty():
            try:
                frame = self._frame_q.get_nowait()
            except queue.Empty:
                break

        if frame is not None:
            img   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = cv2.resize(img, (self.DISPLAY_W, self.DISPLAY_H))
            pil   = add_scanlines(Image.fromarray(img))
            photo = ImageTk.PhotoImage(pil)
            self._cam_label.configure(image=photo, text="", compound="none")
            self._cam_label._photo = photo

        # Stats
        s = self._stats
        gesture = s["gesture"]
        fps     = s["fps"]
        col     = GESTURE_COLORS.get(gesture, FG_DIM)

        self._fps_var.set(f"FPS: {fps:.0f}")
        self._cam_var.set(f"CAM: {self._cam_idx}")
        self._cursor_var.set(f"CURSOR   X: {s['cursor_x']:5d}   Y: {s['cursor_y']:5d}")

        # Gesture flash
        self._icon_var.set(GESTURE_ICONS.get(gesture, "·"))
        self._name_var.set(gesture)
        if self._tracking_active:
            self._icon_lbl.configure(fg=col)
            self._name_lbl.configure(fg=col)
        else:
            self._icon_lbl.configure(fg=FG_DIM)
            self._name_lbl.configure(fg=FG_DIM)

        self._clicks_var.set(f"Clicks:       {s['total_clicks']}")
        self._scroll_var.set(f"Scroll:       {s['scroll_dir']}")

        if self._tracking_active:
            mode = "PAUSED" if self._mouse_ctrl.is_paused else gesture
            self._mode_var.set(f"Mode:    {mode}")
            self._status_dot.configure(fg=col)
            self._status_var.set(
                f"TRACKING ACTIVE   Gesture: {gesture:<14s}   "
                f"Clicks: {s['total_clicks']:4d}   Scroll: {s['scroll_dir']}"
            )
        else:
            self._mode_var.set("Mode:    IDLE")
            self._status_dot.configure(fg=FG_DIM)
            self._status_var.set("IDLE  —  Start tracking to begin")

        self.after(33, self._refresh_ui)

    def _draw_border(self):
        """Animate the neon border on the camera canvas from the main loop."""
        try:
            c = self._border_canvas
            if not c.winfo_exists():
                return
        except Exception:
            return

        self._anim_phase = (self._anim_phase + 0.08) % (2 * math.pi)
        alpha_f = (math.sin(self._anim_phase) + 1) / 2

        gesture = self._stats["gesture"]
        base_col = GESTURE_COLORS.get(gesture, ACCENT) if self._tracking_active else FG_DIM
        r, g, b = hex_to_rgb(base_col)
        glow_r = int(r * (0.3 + 0.7 * alpha_f))
        glow_g = int(g * (0.3 + 0.7 * alpha_f))
        glow_b = int(b * (0.3 + 0.7 * alpha_f))
        col   = f"#{glow_r:02x}{glow_g:02x}{glow_b:02x}"
        thick = int(2 + alpha_f * 4) if self._tracking_active else 1

        bw = self.DISPLAY_W + 14
        bh = self.DISPLAY_H + 14

        c.delete("border")
        p = 2
        c.create_rectangle(p, p, bw - p, bh - p,
                           outline=col, width=thick, tags="border")
        cs_ = 22
        corners = [
            (p, p, p+cs_, p),     (p, p, p, p+cs_),
            (bw-p-cs_, p, bw-p, p), (bw-p, p, bw-p, p+cs_),
            (p, bh-p, p+cs_, bh-p), (p, bh-p-cs_, p, bh-p),
            (bw-p-cs_, bh-p, bw-p, bh-p), (bw-p, bh-p-cs_, bw-p, bh-p),
        ]
        for x1, y1, x2, y2 in corners:
            c.create_line(x1, y1, x2, y2, fill=col, width=thick+1, tags="border")

    # ──────────────────────────────────────────────────────────────────── #
    #  Settings callbacks                                                   #
    # ──────────────────────────────────────────────────────────────────── #

    def _on_sensitivity_change(self, val):
        self._mouse_ctrl.update_settings(sensitivity=val)

    def _on_smooth_change(self, val):
        self._mouse_ctrl.update_settings(smoothing=val)

    def _on_cam_change(self, choice):
        idx = int(choice.split()[-1])
        if idx != self._cam_idx:
            was = self._running
            if was:
                self._stop_tracking()
                time.sleep(0.3)
            self._cam_idx = idx
            if was:
                self._start_tracking()

    # ──────────────────────────────────────────────────────────────────── #
    #  Cleanup                                                              #
    # ──────────────────────────────────────────────────────────────────── #

    def _on_close(self):
        self._alive   = False
        self._running = False
        if self._cap:
            self._cap.release()
        try:
            self._detector.release()
        except Exception:
            pass
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    app = VirtualNavApp()
    app.mainloop()
