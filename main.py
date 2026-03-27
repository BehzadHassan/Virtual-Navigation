"""
main.py  –  Virtual Navigation
════════════════════════════════════════════════════════════════════════════════
Futuristic hand-gesture mouse controller.
Controls the system cursor via webcam + MediaPipe hand tracking.

Run:
    python main.py
"""

import tkinter as tk
import tkinter.font as tkfont
import threading
import time
import queue
import math
import sys

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import customtkinter as ctk
import pyautogui

from gesture_detector   import GestureDetector
from gesture_classifier import GestureClassifier
from mouse_controller   import MouseController

# ── Theme constants ──────────────────────────────────────────────────────────
ACCENT       = "#00f5d4"   # neon cyan-green
ACCENT2      = "#b43ae4"   # neon purple
ACCENT3      = "#ff6b35"   # neon orange
BG_DARK      = "#080c14"   # near-black background
BG_PANEL     = "#0d1520"   # panel background
BG_CARD      = "#111c2e"   # card / frame background
FG_PRIMARY   = "#e8f4fd"   # primary text
FG_DIM       = "#4a7a9b"   # dim text
BORDER_COLOR = "#1a2e4a"   # subtle border

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
    "MOVE":         "Index finger up → moves cursor",
    "LEFT_CLICK":   "Index + Thumb pinch → left click",
    "RIGHT_CLICK":  "Middle + Thumb pinch → right click",
    "DOUBLE_CLICK": "Quick double pinch → double click",
    "SCROLL":       "V-sign (↕ move hand) → scroll",
    "DRAG":         "Fist → drag / hold",
    "PAUSE":        "Open palm → freeze cursor",
}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ═══════════════════════════════════════════════════════════════════════════ #
#  Helpers                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def add_scanlines(img: Image.Image, alpha: int = 18) -> Image.Image:
    """Draw subtle horizontal scanlines over a PIL image."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    for y in range(0, img.height, 3):
        draw.line([(0, y), (img.width, y)], fill=(0, 0, 0, alpha))
    base = img.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


# ═══════════════════════════════════════════════════════════════════════════ #
#  PulsingBorder canvas widget                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class PulsingBorder(tk.Canvas):
    """Animates a glowing neon border around the camera feed."""

    def __init__(self, master, width, height, **kw):
        super().__init__(master, width=width, height=height,
                         bg=BG_DARK, highlightthickness=0, **kw)
        self._w = width
        self._h = height
        self._phase   = 0.0
        self._active  = True
        self._color   = ACCENT
        self._animate()

    def set_active(self, active: bool, color: str = ACCENT):
        self._active = active
        self._color  = color

    def _animate(self):
        self.delete("border")
        self._phase = (self._phase + 0.05) % (2 * math.pi)
        alpha_f = (math.sin(self._phase) + 1) / 2          # 0 → 1
        thick   = int(2 + alpha_f * 4) if self._active else 1

        r, g, b = hex_to_rgb(self._color if self._active else FG_DIM)
        br = int(r * (0.4 + 0.6 * alpha_f))
        bg_ = int(g * (0.4 + 0.6 * alpha_f))
        bb = int(b * (0.4 + 0.6 * alpha_f))
        col = f"#{br:02x}{bg_:02x}{bb:02x}"

        pad = 2
        self.create_rectangle(
            pad, pad, self._w - pad, self._h - pad,
            outline=col, width=thick, tags="border"
        )
        # Corner accents
        cs = 20
        for x1, y1, x2, y2 in [
            (pad, pad, pad + cs, pad),
            (pad, pad, pad, pad + cs),
            (self._w-pad-cs, pad, self._w-pad, pad),
            (self._w-pad, pad, self._w-pad, pad+cs),
            (pad, self._h-pad, pad+cs, self._h-pad),
            (pad, self._h-pad-cs, pad, self._h-pad),
            (self._w-pad-cs, self._h-pad, self._w-pad, self._h-pad),
            (self._w-pad, self._h-pad-cs, self._w-pad, self._h-pad),
        ]:
            self.create_line(x1, y1, x2, y2, fill=col, width=thick + 1, tags="border")

        self.after(30, self._animate)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Background grid canvas                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

class GridBackground(tk.Canvas):
    """Static perspective-grid background for a futuristic look."""

    def __init__(self, master, **kw):
        super().__init__(master, bg=BG_DARK, highlightthickness=0, **kw)
        self.bind("<Configure>", self._draw)

    def _draw(self, event=None):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2 or h < 2:
            return
        r, g, b = hex_to_rgb(ACCENT)
        dim_col = f"#{r//8:02x}{g//8:02x}{b//8:02x}"

        # Horizontal lines
        spacing = 40
        for y in range(0, h, spacing):
            self.create_line(0, y, w, y, fill=dim_col, width=1)
        # Vertical lines
        for x in range(0, w, spacing):
            self.create_line(x, 0, x, h, fill=dim_col, width=1)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Gesture flash label                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

class GestureFlash(tk.Frame):
    """A label that flashes neon when a gesture fires."""

    def __init__(self, master, **kw):
        super().__init__(master, bg=BG_DARK, **kw)
        self._icon_var = tk.StringVar(value="·")
        self._name_var = tk.StringVar(value="NONE")
        self._flash_id = None

        self._icon_lbl = tk.Label(
            self, textvariable=self._icon_var,
            font=("Segoe UI Emoji", 28), bg=BG_DARK, fg=FG_DIM,
        )
        self._icon_lbl.pack()
        self._name_lbl = tk.Label(
            self, textvariable=self._name_var,
            font=("Consolas", 13, "bold"), bg=BG_DARK, fg=FG_DIM,
            width=14,
        )
        self._name_lbl.pack()

    def flash(self, gesture: str):
        color = GESTURE_COLORS.get(gesture, FG_DIM)
        icon  = GESTURE_ICONS.get(gesture, "·")
        self._icon_var.set(icon)
        self._name_var.set(gesture)
        self._icon_lbl.configure(fg=color)
        self._name_lbl.configure(fg=color)
        if self._flash_id:
            self.after_cancel(self._flash_id)
        self._flash_id = self.after(600, self._dim)

    def _dim(self):
        self._icon_lbl.configure(fg=FG_DIM)
        self._name_lbl.configure(fg=FG_DIM)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Main Application Window                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

class VirtualNavApp(ctk.CTk):

    CAM_W = 640
    CAM_H = 480
    DISPLAY_W = 560
    DISPLAY_H = 420

    def __init__(self):
        super().__init__()

        # ── Window setup ─────────────────────────────────────────────────
        self.title("VIRTUAL NAVIGATION")
        self.configure(fg_color=BG_DARK)
        self.minsize(1060, 680)
        self.resizable(True, True)

        # Try to set icon
        try:
            self.iconbitmap("assets/icon.ico")
        except Exception:
            pass

        # ── State ────────────────────────────────────────────────────────
        self._running      = False
        self._cam_idx      = 0
        self._cap          = None
        self._thread       = None
        self._frame_q: queue.Queue = queue.Queue(maxsize=2)
        self._stats        = {
            "fps": 0.0, "gesture": "NONE",
            "cursor_x": 0, "cursor_y": 0,
            "total_clicks": 0, "scroll_dir": "—",
        }
        self._tracking_active = False

        # Screen resolution
        sw, sh = pyautogui.size()
        self._mouse_ctrl = MouseController(
            screen_w=sw, screen_h=sh,
            cam_w=self.CAM_W, cam_h=self.CAM_H,
            smoothing=0.25, sensitivity=1.0,
        )
        self._detector    = GestureDetector()
        self._classifier  = GestureClassifier()

        # Build UI
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start periodic UI refresh
        self._refresh_ui()

    # ──────────────────────────────────────────────────────────────────── #
    #  UI Construction                                                      #
    # ──────────────────────────────────────────────────────────────────── #

    def _build_ui(self):
        # ── Title bar ────────────────────────────────────────────────────
        tb = tk.Frame(self, bg=BG_PANEL, height=52)
        tb.pack(fill="x", side="top")
        tb.pack_propagate(False)

        tk.Label(
            tb, text="⬡  VIRTUAL NAVIGATION",
            font=("Consolas", 15, "bold"),
            bg=BG_PANEL, fg=ACCENT,
        ).pack(side="left", padx=18, pady=12)

        # FPS & CAM badge top-right
        self._fps_var = tk.StringVar(value="FPS: --")
        self._cam_var = tk.StringVar(value="CAM: 0")
        tk.Label(tb, textvariable=self._fps_var,
                 font=("Consolas", 11), bg=BG_PANEL, fg=ACCENT2
                 ).pack(side="right", padx=8)
        tk.Label(tb, textvariable=self._cam_var,
                 font=("Consolas", 11), bg=BG_PANEL, fg=FG_DIM
                 ).pack(side="right", padx=8)

        # Separator line
        sep = tk.Frame(self, bg=BORDER_COLOR, height=1)
        sep.pack(fill="x")

        # ── Body ─────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True)

        # LEFT – camera feed
        self._build_camera_panel(body)

        # RIGHT – info panels
        self._build_right_panel(body)

        # ── Status bar ───────────────────────────────────────────────────
        self._build_status_bar()

    # ── Camera panel ─────────────────────────────────────────────────────

    def _build_camera_panel(self, parent):
        cam_frame = tk.Frame(parent, bg=BG_DARK)
        cam_frame.pack(side="left", fill="both", expand=True, padx=16, pady=14)

        # Pulsing border canvas (wraps the image label)
        self._border = PulsingBorder(cam_frame, self.DISPLAY_W + 10, self.DISPLAY_H + 10)
        self._border.pack()

        self._cam_label = tk.Label(
            self._border, bg=BG_DARK,
            width=self.DISPLAY_W, height=self.DISPLAY_H,
        )
        self._cam_label.place(x=5, y=5)

        # Overlay HUD inside camera area
        hud = tk.Frame(self._border, bg=BG_DARK, bg="#00000000")
        # Use a transparent-ish frame trick – just overlay text on top
        self._gesture_flash = GestureFlash(cam_frame)
        self._gesture_flash.pack(pady=(6, 0))

        # Cursor readout
        self._cursor_var = tk.StringVar(value="CURSOR  X: ----  Y: ----")
        tk.Label(
            cam_frame, textvariable=self._cursor_var,
            font=("Consolas", 10), bg=BG_DARK, fg=FG_DIM,
        ).pack()

        # Toggle button
        self._toggle_btn = ctk.CTkButton(
            cam_frame,
            text="▶  START TRACKING",
            font=ctk.CTkFont("Consolas", 13, weight="bold"),
            fg_color=ACCENT, hover_color=ACCENT2,
            text_color=BG_DARK,
            corner_radius=8, height=38, width=220,
            command=self._toggle_tracking,
        )
        self._toggle_btn.pack(pady=(10, 0))

    # ── Right panel ───────────────────────────────────────────────────────

    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg=BG_DARK, width=340)
        right.pack(side="right", fill="y", padx=(0, 16), pady=14)
        right.pack_propagate(False)

        # ── Gesture Legend card ──────────────────────────────────────────
        self._add_section_header(right, "GESTURE LEGEND")
        legend_card = tk.Frame(right, bg=BG_CARD,
                               highlightbackground=BORDER_COLOR,
                               highlightthickness=1)
        legend_card.pack(fill="x", pady=(0, 10))

        for gesture, desc in GESTURE_DESC.items():
            icon  = GESTURE_ICONS[gesture]
            color = GESTURE_COLORS[gesture]
            row   = tk.Frame(legend_card, bg=BG_CARD)
            row.pack(fill="x", padx=10, pady=3)
            tk.Label(row, text=icon, font=("Segoe UI Emoji", 14),
                     bg=BG_CARD, fg=color, width=3).pack(side="left")
            tk.Label(row, text=desc, font=("Consolas", 9),
                     bg=BG_CARD, fg=FG_PRIMARY, anchor="w",
                     justify="left").pack(side="left", fill="x", expand=True)

        # ── Settings card ────────────────────────────────────────────────
        self._add_section_header(right, "SETTINGS")
        settings_card = tk.Frame(right, bg=BG_CARD,
                                 highlightbackground=BORDER_COLOR,
                                 highlightthickness=1)
        settings_card.pack(fill="x", pady=(0, 10))

        # Sensitivity
        self._sens_var = tk.DoubleVar(value=1.0)
        self._add_slider(settings_card, "Sensitivity", self._sens_var,
                         0.3, 2.5, self._on_sensitivity_change)

        # Smoothing
        self._smooth_var = tk.DoubleVar(value=0.25)
        self._add_slider(settings_card, "Smoothing", self._smooth_var,
                         0.05, 1.0, self._on_smooth_change)

        # Camera selector
        cam_row = tk.Frame(settings_card, bg=BG_CARD)
        cam_row.pack(fill="x", padx=12, pady=(4, 10))
        tk.Label(cam_row, text="Camera", font=("Consolas", 10),
                 bg=BG_CARD, fg=FG_DIM).pack(side="left")
        self._cam_menu = ctk.CTkOptionMenu(
            cam_row,
            values=["CAM 0", "CAM 1", "CAM 2"],
            font=ctk.CTkFont("Consolas", 10),
            fg_color=BG_PANEL, button_color=ACCENT2,
            dropdown_fg_color=BG_CARD,
            text_color=FG_PRIMARY,
            command=self._on_cam_change,
        )
        self._cam_menu.pack(side="right")

        # ── Stats card ───────────────────────────────────────────────────
        self._add_section_header(right, "SESSION STATS")
        stats_card = tk.Frame(right, bg=BG_CARD,
                              highlightbackground=BORDER_COLOR,
                              highlightthickness=1)
        stats_card.pack(fill="x", pady=(0, 10))

        self._clicks_var = tk.StringVar(value="Clicks:      0")
        self._scroll_var = tk.StringVar(value="Scroll:      —")
        self._mode_var   = tk.StringVar(value="Mode:   IDLE")

        for var in (self._clicks_var, self._scroll_var, self._mode_var):
            tk.Label(stats_card, textvariable=var,
                     font=("Consolas", 10), bg=BG_CARD,
                     fg=FG_PRIMARY, anchor="w",
                     padx=12, pady=3).pack(fill="x")

    def _add_section_header(self, parent, text: str):
        row = tk.Frame(parent, bg=BG_DARK)
        row.pack(fill="x", pady=(6, 2))
        tk.Label(row, text=text, font=("Consolas", 9, "bold"),
                 bg=BG_DARK, fg=ACCENT2).pack(side="left")
        tk.Frame(row, bg=BORDER_COLOR, height=1).pack(
            side="left", fill="x", expand=True, padx=(6, 0), pady=6)

    def _add_slider(self, parent, label: str, var, from_, to, command):
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill="x", padx=12, pady=4)
        tk.Label(row, text=label, font=("Consolas", 10),
                 bg=BG_CARD, fg=FG_DIM, width=11, anchor="w").pack(side="left")
        val_lbl = tk.Label(row, text=f"{var.get():.2f}",
                           font=("Consolas", 10),
                           bg=BG_CARD, fg=ACCENT, width=5)
        val_lbl.pack(side="right")

        def _update(v):
            val_lbl.configure(text=f"{float(v):.2f}")
            command(float(v))

        slider = ctk.CTkSlider(
            row, from_=from_, to=to, variable=var,
            command=_update,
            fg_color=BORDER_COLOR, progress_color=ACCENT2,
            button_color=ACCENT, button_hover_color=ACCENT2,
            width=120,
        )
        slider.pack(side="right", padx=(0, 6))

    # ── Status bar ────────────────────────────────────────────────────────

    def _build_status_bar(self):
        sb = tk.Frame(self, bg=BG_PANEL, height=32)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)

        tk.Frame(self, bg=BORDER_COLOR, height=1).pack(fill="x", side="bottom")

        self._status_dot  = tk.Label(sb, text="●", font=("Consolas", 11),
                                     bg=BG_PANEL, fg=FG_DIM)
        self._status_dot.pack(side="left", padx=(10, 4), pady=6)

        self._status_var  = tk.StringVar(value="IDLE  —  Start tracking to begin")
        tk.Label(sb, textvariable=self._status_var,
                 font=("Consolas", 10), bg=BG_PANEL, fg=FG_DIM,
                 ).pack(side="left")

        # right side
        self._sb_right_var = tk.StringVar(value="Virtual Navigation  v1.0")
        tk.Label(sb, textvariable=self._sb_right_var,
                 font=("Consolas", 9), bg=BG_PANEL, fg=FG_DIM,
                 ).pack(side="right", padx=12)

    # ──────────────────────────────────────────────────────────────────── #
    #  Tracking thread                                                      #
    # ──────────────────────────────────────────────────────────────────── #

    def _toggle_tracking(self):
        if self._running:
            self._stop_tracking()
        else:
            self._start_tracking()

    def _start_tracking(self):
        self._cap = cv2.VideoCapture(self._cam_idx, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAM_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_H)
        self._cap.set(cv2.CAP_PROP_FPS, 60)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            self._status_var.set("ERROR  —  Cannot open camera")
            self._cap = None
            return

        self._running         = True
        self._tracking_active = True
        self._toggle_btn.configure(
            text="■  STOP TRACKING",
            fg_color=ACCENT3,
        )
        self._border.set_active(True, ACCENT)
        self._stats["total_clicks"] = 0

        self._thread = threading.Thread(
            target=self._tracking_loop, daemon=True
        )
        self._thread.start()

    def _stop_tracking(self):
        self._running         = False
        self._tracking_active = False
        self._toggle_btn.configure(
            text="▶  START TRACKING",
            fg_color=ACCENT,
        )
        self._border.set_active(False)
        if self._cap:
            self._cap.release()
            self._cap = None
        # Show placeholder
        self._cam_label.configure(image=None, text="  [ NO SIGNAL ]",
                                  font=("Consolas", 16), fg=FG_DIM)

    def _tracking_loop(self):
        """Background thread: capture → detect → classify → act."""
        fps_buf   = []
        prev_time = time.time()
        prev_gesture = "NONE"

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # FPS
            now      = time.time()
            dt       = now - prev_time or 1e-6
            prev_time = now
            fps_buf.append(1.0 / dt)
            if len(fps_buf) > 20:
                fps_buf.pop(0)
            fps = sum(fps_buf) / len(fps_buf)

            # Detect landmarks
            annotated, landmarks, _ = self._detector.detect(frame)

            # Classify gesture
            gesture = self._classifier.classify(landmarks)

            # Index-tip position for mouse
            if landmarks:
                ix = landmarks[8].x
                iy = landmarks[8].y
                self._mouse_ctrl.handle_gesture(gesture, ix, iy)

                if gesture in ("LEFT_CLICK", "DOUBLE_CLICK"):
                    self._stats["total_clicks"] += 1
                if gesture == "SCROLL":
                    self._stats["scroll_dir"] = "↑" if iy < 0.45 else "↓"

            # Draw gesture HUD on frame
            self._draw_hud(annotated, gesture, fps)

            # Update stats
            cx, cy = self._mouse_ctrl.cursor_pos
            self._stats.update({
                "fps":      fps,
                "gesture":  gesture,
                "cursor_x": cx,
                "cursor_y": cy,
            })
            prev_gesture = gesture

            # Push frame to queue (drop if full)
            try:
                self._frame_q.put_nowait(annotated)
            except queue.Full:
                pass

    def _draw_hud(self, frame: np.ndarray, gesture: str, fps: float):
        """Draw semi-transparent HUD text onto the OpenCV frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Top-left box
        cv2.rectangle(overlay, (0, 0), (240, 56), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        color = tuple(reversed(hex_to_rgb(GESTURE_COLORS.get(gesture, FG_DIM))))
        cv2.putText(frame, f"GESTURE: {gesture}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (100, 200, 100), 1, cv2.LINE_AA)

        # Bottom bar
        cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 0), -1)
        cx, cy = self._mouse_ctrl.cursor_pos
        cv2.putText(frame,
                    f"  CURSOR  X:{cx:5d}  Y:{cy:5d}   "
                    f"{'[PAUSED]' if self._mouse_ctrl.is_paused else ''}",
                    (4, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 245, 212), 1, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────────── #
    #  Periodic UI refresh (runs on main thread)                           #
    # ──────────────────────────────────────────────────────────────────── #

    def _refresh_ui(self):
        # Drain frame queue
        while not self._frame_q.empty():
            try:
                frame = self._frame_q.get_nowait()
                img   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img   = cv2.resize(img, (self.DISPLAY_W, self.DISPLAY_H))
                pil   = Image.fromarray(img)
                pil   = add_scanlines(pil, alpha=12)
                photo = ImageTk.PhotoImage(pil)
                self._cam_label.configure(image=photo, text="")
                self._cam_label._photo = photo   # keep reference
            except queue.Empty:
                break

        # Update widgets from stats
        s = self._stats
        gesture = s["gesture"]
        fps     = s["fps"]

        self._fps_var.set(f"FPS: {fps:.0f}")
        self._cam_var.set(f"CAM: {self._cam_idx}")
        self._cursor_var.set(
            f"CURSOR   X: {s['cursor_x']:5d}   Y: {s['cursor_y']:5d}"
        )
        self._gesture_flash.flash(gesture)
        self._clicks_var.set(f"Clicks:      {s['total_clicks']}")
        self._scroll_var.set(f"Scroll:      {s['scroll_dir']}")

        if self._tracking_active:
            mode     = "PAUSED" if self._mouse_ctrl.is_paused else gesture
            dot_col  = GESTURE_COLORS.get(gesture, ACCENT)
            self._status_dot.configure(fg=dot_col)
            self._status_var.set(
                f"TRACKING ACTIVE   Gesture: {gesture:<14s}   "
                f"Clicks: {s['total_clicks']:4d}   Scroll: {s['scroll_dir']}"
            )
            self._mode_var.set(f"Mode:   {mode}")
            # Change border colour per gesture
            self._border.set_active(True, dot_col)
        else:
            self._status_dot.configure(fg=FG_DIM)
            self._status_var.set("IDLE  —  Start tracking to begin")
            self._mode_var.set("Mode:   IDLE")

        self.after(33, self._refresh_ui)   # ≈30 fps UI refresh

    # ──────────────────────────────────────────────────────────────────── #
    #  Setting callbacks                                                    #
    # ──────────────────────────────────────────────────────────────────── #

    def _on_sensitivity_change(self, val: float):
        self._mouse_ctrl.update_settings(sensitivity=val)

    def _on_smooth_change(self, val: float):
        self._mouse_ctrl.update_settings(smoothing=val)

    def _on_cam_change(self, choice: str):
        idx = int(choice.split()[-1])
        if idx != self._cam_idx:
            was_running = self._running
            if was_running:
                self._stop_tracking()
                time.sleep(0.2)
            self._cam_idx = idx
            if was_running:
                self._start_tracking()

    # ──────────────────────────────────────────────────────────────────── #
    #  Cleanup                                                              #
    # ──────────────────────────────────────────────────────────────────── #

    def _on_close(self):
        self._running = False
        if self._cap:
            self._cap.release()
        self._detector.release()
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════ #
#  Entry point                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    app = VirtualNavApp()
    app.mainloop()
