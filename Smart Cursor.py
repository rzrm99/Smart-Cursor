

"""
NoseMouse — camera-based hands‑free mouse with eye‑wink clicks (v1.3.0)

This build focuses on RELIABLE detection and clicks and is ready to run.

What’s new vs prior build:
• Clean shutdown (releases camera + closes MediaPipe)
• Overlay toggle (O) + clearer status banners (PAUSED / SCROLL / DRAG:HOLD|TOGGLE / HELD)
• **Drag without mode toggling** (new default: hold-to-drag with your left eye)
• Extra CLI flags (—overlay/—no-overlay, —debug, —config, —drag-behavior=[hold|toggle])
• Safer camera open + better error messages
• Persist + load config (same ~/.nosemouse.json by default)
• Minor smoothing + key mapping polish

Hotkeys: Q/ESC quit • C center • S span wizard • L/R test click • D cycle drag behavior • M scroll toggle • P pause • H help • O overlay • B blink‑calibrate
Requirements: pip install opencv-python mediapipe numpy pyautogui pynput
macOS: allow Accessibility for your terminal/Python. Linux/Wayland: if pyautogui fails, it will fall back to pynput; for best results use X11.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

try:
    import cv2
except Exception as e:
    print("[FATAL] OpenCV (cv2) is required: pip install opencv-python", file=sys.stderr)
    raise

import numpy as np

try:
    import mediapipe as mp
except Exception:
    print("[ERROR] mediapipe is required: pip install mediapipe", file=sys.stderr)
    raise

try:
    import pyautogui
    pyautogui.FAILSAFE = False
except Exception:
    print("[WARN] pyautogui failed; will try pynput fallback only.")
    pyautogui = None  # type: ignore

try:
    from pynput.mouse import Controller as PynputMouse, Button as PynputButton
    _pynput_mouse = PynputMouse()
    _has_pynput = True
except Exception:
    _has_pynput = False

APP_NAME = "nosemouse"
DEFAULT_CFG_PATH = Path.home() / f".{APP_NAME}.json"

DEFAULTS = dict(
    debug=False,
    show_overlay=True,
    cam_index=None,           # None → auto-pick first available
    mirror=True,
    max_fps=60,

    # Cursor motion
    ema_alpha=0.35,
    mouse_sensitivity=1.2,
    deadzone=0.02,
    calibrated_span=0.35,

    # Blink thresholds (adaptive around baseline EAR)
    ear_close_ratio=0.72,     # consider closed if EAR < ratio * open_baseline
    ear_open_ratio=0.84,      # reopen if EAR > ratio * open_baseline
    hold_min_duration=0.65,   # seconds holding one eye closed → action

    # Drag & Scroll
    scroll_pixels=80,
    drag_behavior="hold",     # 'hold' (default) or 'toggle'
)

# MediaPipe face-landmark indices
LEFT_EYE_CORNERS  = (33, 133)
LEFT_EYE_VERTS    = [(159, 145), (158, 153)]
RIGHT_EYE_CORNERS = (362, 263)
RIGHT_EYE_VERTS   = [(386, 374), (385, 380)]
NOSE_TIP = 4
LEFT_EYE_RING  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 33]
RIGHT_EYE_RING = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 263]

# --------------------------- helpers ---------------------------
def clamp01(x: float) -> float: return max(0.0, min(1.0, x))

def lerp(a: float, b: float, t: float) -> float: return a + (b - a) * t

def eye_aspect_ratio(landmarks, corners, vert_pairs, img_w, img_h) -> float:
    l_idx, r_idx = corners
    l = np.array([landmarks[l_idx].x * img_w, landmarks[l_idx].y * img_h], dtype=np.float32)
    r = np.array([landmarks[r_idx].x * img_w, landmarks[r_idx].y * img_h], dtype=np.float32)
    horiz = float(np.linalg.norm(r - l) + 1e-6)
    vds = []
    for up_idx, dn_idx in vert_pairs:
        up = np.array([landmarks[up_idx].x * img_w, landmarks[up_idx].y * img_h], dtype=np.float32)
        dn = np.array([landmarks[dn_idx].x * img_w, landmarks[dn_idx].y * img_h], dtype=np.float32)
        vds.append(float(np.linalg.norm(up - dn)))
    return float((vds[0] + vds[1]) / (2.0 * horiz))

class MouseIO:
    def __init__(self):
        self.screen_w, self.screen_h = self._get_size()
        self.dragging = False
    def _get_size(self):
        if pyautogui is not None:
            try:
                return pyautogui.size()
            except Exception:
                pass
        # last resort
        return (1920, 1080)
    def move(self, x: int, y: int):
        x = max(0, min(self.screen_w - 1, int(x)))
        y = max(0, min(self.screen_h - 1, int(y)))
        if pyautogui is not None:
            try:
                pyautogui.moveTo(x, y, duration=0)
                return
            except Exception:
                pass
        if _has_pynput:
            try:
                _pynput_mouse.position = (x, y)
            except Exception:
                pass
    def click(self, button: str = "left"):
        if pyautogui is not None:
            try:
                pyautogui.click(button=button)
                return
            except Exception:
                pass
        if _has_pynput:
            try:
                _pynput_mouse.click(PynputButton.left if button == "left" else PynputButton.right, 1)
            except Exception:
                pass
    def press(self, button: str = "left"):
        if pyautogui is not None:
            try:
                pyautogui.mouseDown(button=button)
                return
            except Exception:
                pass
        if _has_pynput:
            try:
                _pynput_mouse.press(PynputButton.left if button == "left" else PynputButton.right)
            except Exception:
                pass
    def release(self, button: str = "left"):
        if pyautogui is not None:
            try:
                pyautogui.mouseUp(button=button)
                return
            except Exception:
                pass
        if _has_pynput:
            try:
                _pynput_mouse.release(PynputButton.left if button == "left" else PynputButton.right)
            except Exception:
                pass
    def scroll(self, dy_pixels: int):
        if pyautogui is not None:
            try:
                pyautogui.scroll(int(-dy_pixels))
                return
            except Exception:
                pass
        if _has_pynput:
            try:
                # pynput uses 'clicks', roughly 120 px per notch on many systems
                _pynput_mouse.scroll(0, int(-dy_pixels / 120))
            except Exception:
                pass

class WinkState:
    def __init__(self, hold_s: float, close_ratio: float, open_ratio: float):
        self.hold_s = hold_s
        self.close_ratio = close_ratio
        self.open_ratio = open_ratio
        self.closed = False
        self.hold_start: float | None = None
        self.fired = False
        self.baseline = 0.3  # updated from running open EAR
    def reset(self):
        self.closed = False; self.hold_start=None; self.fired=False
    def update(self, ear: float, now: float) -> bool:
        # update baseline with EMA when not closed
        if not self.closed:
            self.baseline = 0.9*self.baseline + 0.1*ear
        close_thr = self.baseline * self.close_ratio
        open_thr  = self.baseline * self.open_ratio
        if not self.closed:
            if ear < close_thr:
                self.closed = True; self.hold_start = now; self.fired = False
        else:
            if ear > open_thr:
                self.closed = False; self.hold_start = None; self.fired = False
        if self.closed and not self.fired and self.hold_start is not None and (now - self.hold_start) >= self.hold_s:
            self.fired = True; return True
        return False

class NoseMouse:
    def __init__(self, cfg: dict, cfg_path: Path):
        self.cfg = cfg
        self.cfg_path = cfg_path
        self.debug = bool(cfg.get("debug", False))
        self.overlay = bool(cfg.get("show_overlay", True))
        self.mirror = bool(cfg.get("mirror", True))
        self.max_fps = int(cfg.get("max_fps", 60))
        self.ema_alpha = float(cfg.get("ema_alpha", 0.35))
        self.sens = float(cfg.get("mouse_sensitivity", 1.2))
        self.deadzone = float(cfg.get("deadzone", 0.02))
        self.calibrated_span = float(cfg.get("calibrated_span", 0.35))
        self.scroll_pixels = int(cfg.get("scroll_pixels", 80))
        self.close_ratio = float(cfg.get("ear_close_ratio", 0.72))
        self.open_ratio  = float(cfg.get("ear_open_ratio",  0.84))
        self.hold_s      = float(cfg.get("hold_min_duration", 0.65))
        self.drag_behavior = str(cfg.get("drag_behavior", "hold")).lower()
        self.help_shown = True
        self.paused = False
        self.scroll_mode = False
        self.drag_mode_toggle = False  # legacy toggle state if drag_behavior=="toggle"
        self.mouse = MouseIO()
        self.screen_w, self.screen_h = self.mouse.screen_w, self.mouse.screen_h
        self.cap = self._open_camera(cfg.get("cam_index"))
        # bump camera resolution for better eye landmarks
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, min(60, self.max_fps))
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.6, min_tracking_confidence=0.6,
        )
        self.calibrated_center: tuple[float, float] | None = None
        self.ema_nx: float | None = None
        self.ema_ny: float | None = None
        self.left_wink = WinkState(self.hold_s, self.close_ratio, self.open_ratio)
        self.right_wink = WinkState(self.hold_s, self.close_ratio, self.open_ratio)
        self.hist_nose = deque(maxlen=4)
        self.last_fps_t = time.time(); self.fps = 0.0; self._fps_count=0
        self.face_ok = False; self.eyes_ok = False
        self.win_name = "Nose Mouse"

    def _open_camera(self, preferred_index):
        tried = []
        if preferred_index is not None:
            tried.append(int(preferred_index))
        tried += [0,1,2,3]
        errors = []
        for idx in tried:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"[INFO] Using camera index {idx}")
                return cap
            else:
                errors.append(idx)
        raise RuntimeError(f"Could not open any camera (tried indices: {errors}). Try --cam-index N")

    # ---------- calibration ----------
    def calibrate_center(self, nx: float, ny: float):
        self.calibrated_center = (nx, ny)
        print(f"[CALIBRATE] center={self.calibrated_center} span={self.calibrated_span:.3f}")
        self._save_cfg()

    def recalibrate_blinks(self):
        self.left_wink.baseline = 0.3
        self.right_wink.baseline = 0.3
        self.left_wink.reset(); self.right_wink.reset()
        print("[CALIBRATE] blink thresholds reset — keep eyes OPEN for 2s, then try a slow wink.")

    def auto_span_wizard(self):
        print("[WIZARD] Span: look LEFT, RIGHT, UP, DOWN as guided…")
        prompts = ["Look LEFT edge","Look RIGHT edge","Look UP edge","Look DOWN edge"]
        samples=[]; step=0; last=time.time(); start=last
        while step < 4:
            ok, frm = self.cap.read()
            if not ok:
                print("[WARN] Camera frame not available during span wizard.")
                break
            if self.mirror: frm = cv2.flip(frm, 1)
            hh, ww = frm.shape[:2]
            res = self.face_mesh.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                fl = res.multi_face_landmarks[0].landmark
                nx, ny = clamp01(fl[NOSE_TIP].x), clamp01(fl[NOSE_TIP].y)
                txt = prompts[step]
                cv2.putText(frm, f"Span Wizard: {txt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.circle(frm, (int(nx*ww), int(ny*hh)), 4, (0,255,0), -1)
                cv2.imshow(self.win_name, frm)
                if cv2.waitKey(1) & 0xFF in (27, ord('q'), ord('Q')):
                    print("[WIZARD] cancelled")
                    return
                now=time.time()
                if now-last>0.2:
                    samples.append((nx,ny)); last=now
                    if now-start >= (step+1)*1.2: step+=1
            else:
                cv2.putText(frm, "Face not found — hold still", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,100,255), 2)
                cv2.imshow(self.win_name, frm)
                cv2.waitKey(1)
        if samples:
            xs=[s[0] for s in samples]; ys=[s[1] for s in samples]
            span_x=max(xs)-min(xs); span_y=max(ys)-min(ys)
            self.calibrated_span=float(max(0.15,min(0.60,0.40*max(span_x,span_y))))
            print(f"[CALIBRATE] auto span={self.calibrated_span:.3f}")
            self._save_cfg()

    def _save_cfg(self):
        data = {
            "calibrated_center": self.calibrated_center,
            "calibrated_span": self.calibrated_span,
            "ear_close_ratio": self.close_ratio,
            "ear_open_ratio": self.open_ratio,
            "hold_min_duration": self.hold_s,
            "mouse_sensitivity": self.sens,
            "deadzone": self.deadzone,
            "ema_alpha": self.ema_alpha,
            "mirror": self.mirror,
            "max_fps": self.max_fps,
            "drag_behavior": self.drag_behavior,
        }
        try:
            with open(self.cfg_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[WARN] Could not save config to {self.cfg_path}: {e}")

    def _load_cfg(self):
        if not self.cfg_path.exists():
            return
        try:
            data = json.load(open(self.cfg_path))
        except Exception as e:
            print(f"[WARN] Could not load config {self.cfg_path}: {e}")
            return
        self.calibrated_center = tuple(data.get("calibrated_center")) if data.get("calibrated_center") else None
        self.calibrated_span = float(data.get("calibrated_span", self.calibrated_span))
        self.close_ratio = float(data.get("ear_close_ratio", self.close_ratio))
        self.open_ratio  = float(data.get("ear_open_ratio",  self.open_ratio))
        self.hold_s = float(data.get("hold_min_duration", self.hold_s))
        self.sens = float(data.get("mouse_sensitivity", self.sens))
        self.deadzone = float(data.get("deadzone", self.deadzone))
        self.ema_alpha = float(data.get("ema_alpha", self.ema_alpha))
        self.mirror = bool(data.get("mirror", self.mirror))
        self.max_fps = int(data.get("max_fps", self.max_fps))
        self.drag_behavior = str(data.get("drag_behavior", self.drag_behavior)).lower()

    # ----------------- main -----------------
    def run(self):
        self._load_cfg()
        if self.calibrated_center is None:
            self.calibrated_center = (0.5, 0.5)
        print("Controls: C center • S span • B blink-cal • L/R click • D cycle drag‑behavior • M scroll • P pause • O overlay • H help • Q/ESC quit")
        frame_interval = 1.0 / max(1, self.max_fps)
        last_time = time.time()
        try:
            while True:
                now=time.time()
                if now-last_time < frame_interval:
                    time.sleep(0.001)
                last_time=now
                ok, frame = self.cap.read()
                if not ok:
                    print("[ERROR] Camera frame not available. Is the camera in use by another app?")
                    break
                if self.mirror: frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                res = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.face_ok = bool(res.multi_face_landmarks)
                self.eyes_ok = False

                if self.face_ok:
                    fl = res.multi_face_landmarks[0].landmark
                    pts = [fl[i] for i in [LEFT_EYE_CORNERS[0], LEFT_EYE_CORNERS[1], RIGHT_EYE_CORNERS[0], RIGHT_EYE_CORNERS[1]]]
                    vis = all(0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0 for p in pts)
                    self.eyes_ok = vis
                    nx_raw, ny_raw = clamp01(fl[NOSE_TIP].x), clamp01(fl[NOSE_TIP].y)
                    if self.ema_nx is None:
                        self.ema_nx, self.ema_ny = nx_raw, ny_raw
                    else:
                        self.ema_nx = lerp(self.ema_nx, nx_raw, self.ema_alpha)
                        self.ema_ny = lerp(self.ema_ny, ny_raw, self.ema_alpha)
                    self.hist_nose.append((self.ema_nx, self.ema_ny))
                    nx = sum(p[0] for p in self.hist_nose)/len(self.hist_nose)
                    ny = sum(p[1] for p in self.hist_nose)/len(self.hist_nose)

                    cx, cy = self.calibrated_center
                    dx = (nx - cx) / max(self.calibrated_span, 1e-6)
                    dy = (ny - cy) / max(self.calibrated_span, 1e-6)
                    if abs(dx) < self.deadzone: dx = 0.0
                    if abs(dy) < self.deadzone: dy = 0.0
                    dx = max(-1.0, min(1.0, dx))
                    dy = max(-1.0, min(1.0, dy))
                    target_x = int((0.5 + dx * self.sens * 0.5) * self.screen_w)
                    target_y = int((0.5 + dy * self.sens * 0.5) * self.screen_h)
                    if not self.paused:
                        if self.scroll_mode:
                            self.mouse.scroll(int(dy * self.scroll_pixels))
                        else:
                            self.mouse.move(target_x, target_y)

                    if self.eyes_ok:
                        left_ear  = eye_aspect_ratio(fl, LEFT_EYE_CORNERS, LEFT_EYE_VERTS, w, h)
                        right_ear = eye_aspect_ratio(fl, RIGHT_EYE_CORNERS, RIGHT_EYE_VERTS, w, h)
                        left_fired = self.left_wink.update(left_ear, now)
                        right_fired = self.right_wink.update(right_ear, now)

                        if self.drag_behavior == 'hold':
                            # Hold-to-drag: after hold_s of left wink → press; release on eye open
                            if self.left_wink.closed and left_fired and not self.mouse.dragging:
                                self.mouse.press('left'); self.mouse.dragging = True
                            if (not self.left_wink.closed) and self.mouse.dragging:
                                self.mouse.release('left'); self.mouse.dragging = False
                            # Right eye long-wink as right-click
                            if right_fired:
                                self.mouse.click('right')
                        else:
                            # Toggle behavior (legacy): long left wink toggles hold; long right wink clicks right
                            if left_fired:
                                if not self.mouse.dragging:
                                    self.mouse.press('left'); self.mouse.dragging = True
                                else:
                                    self.mouse.release('left'); self.mouse.dragging = False
                            if right_fired:
                                self.mouse.click('right')

                    if self.overlay:
                        self._draw_overlay(frame, fl, w, h, nx, ny)
                else:
                    if self.overlay:
                        self._draw_text_bg(frame, "NO FACE", (10, 30), scale=0.8, color=(0,100,255))

                # FPS
                self._fps_count += 1
                t=time.time()
                if t-self.last_fps_t >= 0.5:
                    self.fps = self._fps_count/(t-self.last_fps_t)
                    self._fps_count=0; self.last_fps_t=t

                # keys
                cv2.imshow(self.win_name, frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q'), ord('Q')): break
                elif k in (ord('c'), ord('C')) and self.ema_nx is not None:
                    self.calibrate_center(self.ema_nx, self.ema_ny)
                elif k in (ord('l'), ord('L')): self.mouse.click('left')
                elif k in (ord('r'), ord('R')): self.mouse.click('right')
                elif k in (ord('p'), ord('P')): self.paused = not self.paused
                elif k in (ord('d'), ord('D')):
                    # Cycle drag behavior between hold and toggle at runtime
                    self.drag_behavior = 'toggle' if self.drag_behavior == 'hold' else 'hold'
                    print(f"[MODE] drag_behavior={self.drag_behavior}")
                elif k in (ord('m'), ord('M')): self.scroll_mode = not self.scroll_mode
                elif k in (ord('h'), ord('H')): self.help_shown = not self.help_shown
                elif k in (ord('s'), ord('S')): self.auto_span_wizard()
                elif k in (ord('b'), ord('B')): self.recalibrate_blinks()
                elif k in (ord('o'), ord('O')): self.overlay = not self.overlay
        except KeyboardInterrupt:
            pass
        finally:
            if self.mouse.dragging:
                self.mouse.release('left'); self.mouse.dragging=False
            try:
                self.cap.release()
            except Exception:
                pass
            try:
                self.face_mesh.close()
            except Exception:
                pass
            cv2.destroyAllWindows()

    # ---------- drawing ----------
    def _draw_text_bg(self, frame, text, org, scale=0.6, color=(220,220,220), thickness=1, bg=(0,0,0)):
        (x,y) = org
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(frame, (x-4, y-th-6), (x+tw+4, y+2), bg, -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def _draw_overlay(self, frame, fl, w, h, nx, ny):
        # nose + box
        cv2.circle(frame, (int(nx*w), int(ny*h)), 4, (0,255,0), -1)
        cx, cy = int(self.calibrated_center[0]*w), int(self.calibrated_center[1]*h)
        span_w, span_h = int(self.calibrated_span*w), int(self.calibrated_span*h)
        cv2.rectangle(frame, (cx-span_w, cy-span_h), (cx+span_w, cy+span_h), (255,255,0), 1)
        # status lamps
        def lamp(x,y,text,on):
            color=(0,255,0) if on else (60,60,60)
            cv2.circle(frame,(x,y),7,color,-1)
            cv2.putText(frame,text,(x+12,y+4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(220,220,220),1)
        lamp(10,20,"FACE", self.face_ok)
        lamp(90,20,"EYES", self.eyes_ok)
        lamp(170,20,"L-WINK", self.left_wink.closed)
        lamp(270,20,"R-WINK", self.right_wink.closed)
        # modes banner
        mode_txt = []
        if self.paused: mode_txt.append("PAUSED")
        if self.scroll_mode: mode_txt.append("SCROLL")
        if self.drag_behavior == 'toggle': mode_txt.append("DRAG:TOGGLE")
        else: mode_txt.append("DRAG:HOLD")
        if self.mouse.dragging: mode_txt.append("HELD")
        if mode_txt:
            self._draw_text_bg(frame, " ".join(f"[{m}]" for m in mode_txt), (10, 46), scale=0.6, bg=(25,25,25))
        # debug draw eye rings
        if self.debug and self.eyes_ok:
            for idx in LEFT_EYE_RING:
                p = fl[idx]
                cv2.circle(frame, (int(p.x*w), int(p.y*h)), 1, (0,180,255), -1)
            for idx in RIGHT_EYE_RING:
                p = fl[idx]
                cv2.circle(frame, (int(p.x*w), int(p.y*h)), 1, (0,180,255), -1)
        # help + fps
        self._draw_text_bg(frame, f"FPS {self.fps:4.1f}", (w-140, 20), scale=0.6, bg=(25,25,25))
        if self.help_shown:
            help1 = "C center  S span  B blink-cal  D drag-behavior  M scroll  P pause  O overlay  L/R click  H help  Q quit"
            self._draw_text_bg(frame, help1, (10, h-12), scale=0.5, bg=(25,25,25))


# ---------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Hands-free camera mouse using nose tracking + wink clicks")
    p.add_argument('--cam-index', type=int, default=None, help='Camera index (default: auto)')
    p.add_argument('--mirror', dest='mirror', action='store_true', help='Mirror preview (default on)')
    p.add_argument('--no-mirror', dest='mirror', action='store_false', help='Disable mirrored preview')
    p.set_defaults(mirror=True)
    p.add_argument('--overlay', dest='overlay', action='store_true', help='Show overlay (default on)')
    p.add_argument('--no-overlay', dest='overlay', action='store_false', help='Hide overlay UI')
    p.set_defaults(overlay=True)
    p.add_argument('--max-fps', type=int, default=None, help='FPS cap (default 60)')
    p.add_argument('--sens', type=float, default=None, help='Mouse sensitivity multiplier')
    p.add_argument('--deadzone', type=float, default=None, help='Deadzone (0..1)')
    p.add_argument('--span', type=float, default=None, help='Normalized span around center (0.15..0.60)')
    p.add_argument('--hold', type=float, default=None, help='Hold duration for wink actions (seconds)')
    p.add_argument('--debug', action='store_true', help='Debug draw eye rings')
    p.add_argument('--drag-behavior', choices=['hold','toggle'], default='hold', help='Drag style: hold-to-drag (default) or toggle press/release')
    p.add_argument('--config', type=Path, default=DEFAULT_CFG_PATH, help=f'Config path (default {DEFAULT_CFG_PATH})')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = DEFAULTS.copy()
    cfg['cam_index'] = args.cam_index
    cfg['mirror'] = bool(args.mirror)
    cfg['show_overlay'] = bool(args.overlay)
    if args.max_fps is not None: cfg['max_fps'] = max(15, int(args.max_fps))
    if args.sens is not None: cfg['mouse_sensitivity'] = float(args.sens)
    if args.deadzone is not None: cfg['deadzone'] = float(args.deadzone)
    if args.span is not None: cfg['calibrated_span'] = float(args.span)
    if args.hold is not None: cfg['hold_min_duration'] = float(args.hold)
    if args.debug: cfg['debug'] = True
    cfg['drag_behavior'] = args.drag_behavior

    try:
        app = NoseMouse(cfg, args.config)
        app.run()
    except RuntimeError as e:
        print(f"[FATAL] {e}")
        if sys.platform.startswith('darwin'):
            print("macOS: grant Accessibility permission to your terminal/Python (System Settings → Privacy & Security → Accessibility).")
        elif os.environ.get('XDG_SESSION_TYPE', '').lower().startswith('wayland'):
            print("Linux Wayland: if movement/clicks fail, try X11 or rely on pynput fallback.")

if __name__ == '__main__':
    main()
