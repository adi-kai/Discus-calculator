# Standard library
import math
import threading
import urllib.request
import os

# Third-party
try:
    import cv2
except ImportError:
    raise SystemExit("Run: pip install opencv-python")

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    raise SystemExit("Run: pip install mediapipe==0.10.35")

try:
    import numpy as np
except ImportError:
    raise SystemExit("Run: pip install numpy")

try:
    from PIL import Image, ImageTk
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog, ttk

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not found — graphs disabled")

# ── Colors (BGR) 
COL_DISC = (0,   230,  80)
COL_VECTOR = (0,   200, 255)
COL_RING = (255, 100,  40)
COL_TRAIL = (50,  220, 255)
COL_PREDICT = (0,     0, 220)
COL_PHASE = (255, 255, 255)

MIN_DISC_PTS = 8

# ── Pose landmark indices 
_LM = {
    "NOSE": 0,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,      "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,     "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,    "RIGHT_ANKLE": 28,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32,
}

# Skeleton connections for drawing bones between joints
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,31),
    (24,26),(26,28),(28,32),
]

MODEL_PATH = "pose_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "pose_landmarker/pose_landmarker_full/float16/latest/"
              "pose_landmarker_full.task")


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose model (~20 MB)…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


def make_landmarker():
    ensure_model()
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)

#  ROI SELECTOR  shows first frame, user drags a box, returns (x,y,w,h)
#  Returns None if user presses ESC or closes without selecting

def select_roi(video_path):
    """
    Opens the first frame in an OpenCV window.
    User drags a rectangle over the disc's flight zone.
    Returns (x, y, w, h) in pixels, or None to skip ROI.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    # Scale down if too large for the screen
    max_dim = 900
    h, w = frame.shape[:2]
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

    instructions = "Draw a box around the disc flight zone. Press ENTER/SPACE to confirm, ESC to skip."
    # Put instructions on frame
    disp = frame.copy()
    cv2.putText(disp, instructions, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
    cv2.putText(disp, instructions, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 80), 1)

    roi = cv2.selectROI("Select ROI — draw box then press ENTER", disp,
                        fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, rw, rh = roi
    if rw == 0 or rh == 0:
        return None  # user pressed ESC or drew nothing

    # Scale back to original frame coords if we resized
    if scale < 1.0:
        x  = int(x  / scale)
        y  = int(y  / scale)
        rw = int(rw / scale)
        rh = int(rh / scale)

    return (x, y, rw, rh)


def draw_landmarks_on_frame(frame, landmarks_list):
    """Draw skeleton on frame using new API landmarks."""
    if not landmarks_list:
        return
    lms = landmarks_list[0]
    h, w = frame.shape[:2]

    # Bones
    for a, b in POSE_CONNECTIONS:
        if a >= len(lms) or b >= len(lms):
            continue
        pa, pb = lms[a], lms[b]
        if pa.visibility < 0.4 or pb.visibility < 0.4:
            continue
        x1, y1 = int(pa.x * w), int(pa.y * h)
        x2, y2 = int(pb.x * w), int(pb.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Joints
    for lm in lms:
        if lm.visibility < 0.4:
            continue
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (80, 220, 255), -1)



#  DISC DETECTOR
class DiscDetector:
    def __init__(self):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=False
        )
        self.min_area = 250
        self.max_area = 9000
        self.min_circularity = 0.40
        self._frames_seen = 0
        self.WARMUP_FRAMES = 25

    def detect(self, frame, roi=None):
        """
        roi: (x, y, w, h) bounding box to restrict detection, or None for full frame.
        Always runs bg_sub on the full frame to keep the model well-trained,
        but only searches for contours inside the ROI.
        """
        fg = self.bg_sub.apply(frame)
        self._frames_seen += 1
        if self._frames_seen <= self.WARMUP_FRAMES:
            return None

        # Mask everything outside ROI before contour search
        if roi is not None:
            rx, ry, rw, rh = roi
            mask = np.zeros(fg.shape, dtype=np.uint8)
            mask[ry:ry+rh, rx:rx+rw] = 255
            fg = cv2.bitwise_and(fg, mask)

        k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area < area < self.max_area):
                continue
            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circ = (4 * math.pi * area) / (perim ** 2)
            if circ < self.min_circularity:
                continue
            score = circ * area
            if score > best_score:
                best_score = score
                best = cnt
        if best is None:
            return None
        M = cv2.moments(best)
        if M["m00"] == 0:
            return None
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


#  PHASE CLASSIFIER  
def classify_phase(landmarks_list, frame_idx, total_frames):
    if not landmarks_list:
        return "—"
    lms = landmarks_list[0]

    def g(name):
        return lms[_LM[name]]

    rf = g("RIGHT_FOOT_INDEX")
    lf = g("LEFT_FOOT_INDEX")
    rk = g("RIGHT_KNEE")
    lk = g("LEFT_KNEE")
    rw = g("RIGHT_WRIST")

    rf_up = (rf.y - rk.y) < -0.05 and rf.visibility > 0.4
    lf_up = (lf.y - lk.y) < -0.05 and lf.visibility > 0.4

    safe_total = max(total_frames - 1, 1) if total_frames and total_frames > 1 else 9999
    progress   = frame_idx / safe_total

    if progress < 0.15:
        return "Backswing"
    elif rf_up and not lf_up:
        return "Right foot lifts"
    elif lf_up and not rf_up:
        return "Left foot lifts"
    elif not rf_up and not lf_up and progress < 0.55:
        return "Both feet land"
    elif rw.y < 0.35 and progress > 0.55:
        return "Release"
    elif progress > 0.80:
        return "Follow-through"
    else:
        return "Approach"


#  THROWING RING
def draw_throwing_ring(frame, landmarks_list, frame_w, frame_h):
    if not landmarks_list:
        return
    lms = landmarks_list[0]

    def g(name):
        return lms[_LM[name]]

    rh = g("RIGHT_HIP");  lh = g("LEFT_HIP")
    rs = g("RIGHT_SHOULDER"); ls = g("LEFT_SHOULDER")

    if rh.visibility < 0.4 or lh.visibility < 0.4:
        return

    cx = int(((rh.x + lh.x) / 2) * frame_w)
    cy = int(((rh.y + lh.y) / 2) * frame_h)

    shoulder_px = abs(rs.x - ls.x) * frame_w
    ring_radius = max(30, int(shoulder_px * 2.5))

    overlay = frame.copy()
    cv2.ellipse(overlay, (cx, cy + ring_radius // 3),
                (ring_radius, ring_radius // 3), 0, 0, 360, COL_RING, 2)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "Throwing ring",
                (cx - 55, cy + ring_radius // 3 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_RING, 1)



#VELOCITY VECTORS

def draw_velocity_vectors(frame, curr_list, prev_list, frame_w, frame_h):
    if not prev_list or not curr_list:
        return
    curr_lms = curr_list[0]
    prev_lms = prev_list[0]
    joint_names = ["RIGHT_WRIST","LEFT_WRIST","RIGHT_SHOULDER",
                   "LEFT_SHOULDER","RIGHT_HIP","LEFT_HIP"]
    scale = 60
    for name in joint_names:
        idx = _LM[name]
        if idx >= len(curr_lms) or idx >= len(prev_lms):
            continue
        c = curr_lms[idx]; p = prev_lms[idx]
        if c.visibility < 0.4 or p.visibility < 0.4:
            continue
        dx = (c.x - p.x) * frame_w
        dy = (c.y - p.y) * frame_h
        mag = math.sqrt(dx*dx + dy*dy)
        if mag < 1.5:
            continue
        x1 = int(c.x * frame_w); y1 = int(c.y * frame_h)
        x2 = int(x1 + dx * scale / max(mag,1) * min(mag,20))
        y2 = int(y1 + dy * scale / max(mag,1) * min(mag,20))
        cv2.arrowedLine(frame, (x1,y1), (x2,y2), COL_VECTOR, 2, tipLength=0.35)



# TRAJECTORY TRACKER

class TrajectoryTracker:
    def __init__(self):
        self.positions  = []
        self.landing_px = None
        self._coeffs    = None
        self._frame_h   = None
        self._frame_w   = None

    def set_frame_size(self, frame_h, frame_w):
        self._frame_h = frame_h
        self._frame_w = frame_w

    def add(self, frame_idx, pt):
        self.positions.append((frame_idx, pt[0], pt[1]))
        if len(self.positions) >= MIN_DISC_PTS:
            self._fit()

    def _fit(self):
        xs = np.array([p[1] for p in self.positions], dtype=float)
        ys = np.array([p[2] for p in self.positions], dtype=float)
        try:
            c = np.polyfit(xs, ys, 2)
            if c[0] <= 0:
                return
            ground_y  = float(self._frame_h - 1) if self._frame_h else max(ys) * 1.05
            disc      = c[1]**2 - 4*c[0]*(c[2] - ground_y)
            if disc < 0:
                return
            x1 = (-c[1] + math.sqrt(disc)) / (2*c[0])
            x2 = (-c[1] - math.sqrt(disc)) / (2*c[0])
            launch_x  = xs[0]
            candidate = x1 if abs(x1-launch_x) > abs(x2-launch_x) else x2
            if candidate > launch_x:
                self._coeffs    = c
                self.landing_px = candidate
        except Exception:
            pass

    def draw_overlay(self, frame, frame_h, frame_w):
        for i, (_, x, y) in enumerate(self.positions):
            alpha = i / max(len(self.positions)-1, 1)
            col = (int(COL_TRAIL[0]*(1-alpha)), int(COL_TRAIL[1]),
                   int(COL_TRAIL[2]*alpha))
            cv2.circle(frame, (x, y), 5, col, -1)

        if self._coeffs is not None and len(self.positions) >= MIN_DISC_PTS:
            x_start = min(p[1] for p in self.positions)
            x_end   = int(self.landing_px) if self.landing_px else x_start + 200
            x_end   = min(x_end, frame_w - 1)
            pts = []
            for x in range(x_start, x_end, 4):
                y = int(np.polyval(self._coeffs, x))
                if 0 <= y < frame_h:
                    pts.append((x, y))
            for i in range(len(pts)-1):
                cv2.line(frame, pts[i], pts[i+1], (0, 255, 200), 2)

        if self.landing_px is not None:
            lx = int(self.landing_px)
            if 0 <= lx < frame_w:
                cv2.line(frame, (lx, frame_h-50), (lx, frame_h), COL_PREDICT, 3)
                cv2.putText(frame, "PREDICTED", (max(0,lx-50), frame_h-55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_PREDICT, 2)

    def distance_meters(self, frame_w, real_width_m=20.0):
        if not self.positions or self.landing_px is None:
            return None
        launch_x = self.positions[0][1]
        px_dist  = abs(self.landing_px - launch_x)
        mpp      = real_width_m / frame_w
        return round(px_dist * mpp, 1)



# VIDEO PROCESSOR

class VideoProcessor:
    def __init__(self, path, progress_cb=None, roi=None):
        self.path        = path
        self.progress_cb = progress_cb
        self.roi         = roi   # (x, y, w, h) or None
        self.frames      = []
        self.raw_frames  = []
        self.disc_pts    = []
        self.landmarks   = []   # each entry: list of pose landmark lists, or []
        self.phases      = []
        self.tracker     = TrajectoryTracker()
        self.fps         = 30
        self.frame_w     = 0
        self.frame_h     = 0

    def process(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.path}")

        self.fps     = cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

        self.tracker.set_frame_size(self.frame_h, self.frame_w)

        detector   = DiscDetector()
        prev_lms   = None
        frame_idx  = 0

        landmarker = make_landmarker()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.raw_frames.append(frame.copy())
            annotated = frame.copy()

            # Pose (new API)
            rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp  = int(frame_idx * (1000 / self.fps))
            result     = landmarker.detect_for_video(mp_image, timestamp)
            lms_list   = result.pose_landmarks   # list of lists
            self.landmarks.append(lms_list)

            if lms_list:
                draw_landmarks_on_frame(annotated, lms_list)
                draw_throwing_ring(annotated, lms_list, self.frame_w, self.frame_h)
                draw_velocity_vectors(annotated, lms_list, prev_lms,
                                      self.frame_w, self.frame_h)

            # Phase
            phase = classify_phase(lms_list, frame_idx, total or 0)
            self.phases.append(phase)

            # Disc
            disc_pt = detector.detect(frame, self.roi)
            if disc_pt:
                self.tracker.add(frame_idx, disc_pt)
                self.disc_pts.append((frame_idx, disc_pt[0], disc_pt[1]))
                cv2.circle(annotated, disc_pt, 18, COL_DISC, 2)
                cv2.circle(annotated, disc_pt, 4,  COL_DISC, -1)
                cv2.putText(annotated, "DISC",
                            (disc_pt[0]+22, disc_pt[1]+6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_DISC, 2)
            else:
                self.disc_pts.append(None)

            # Draw ROI box so user can see it's active
            if self.roi is not None:
                rx, ry, rw, rh = self.roi
                cv2.rectangle(annotated, (rx, ry), (rx+rw, ry+rh),
                              (0, 180, 255), 2)
                cv2.putText(annotated, "ROI", (rx+4, ry+18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

            self.tracker.draw_overlay(annotated, self.frame_h, self.frame_w)

            # HUD
            dist    = self.tracker.distance_meters(self.frame_w)
            n_total = total if total else "?"
            hud = [
                f"Frame {frame_idx+1}/{n_total}  |  Phase: {phase}",
                f"Disc pts: {len(self.tracker.positions)}",
                f"Predicted: {dist}m" if dist else "Predicted: tracking...",
            ]
            for i, line in enumerate(hud):
                y = 28 + i * 26
                cv2.putText(annotated, line, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),   3)
                cv2.putText(annotated, line, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_PHASE, 1)

            self.frames.append(annotated)
            prev_lms  = lms_list
            frame_idx += 1

            if self.progress_cb and total:
                self.progress_cb(int(frame_idx / total * 100))

        cap.release()
        landmarker.close()
        if not self.frames:
            raise ValueError("No frames were processed — check the video file.")



#GRAPHS

def open_graphs(processor: VideoProcessor):
    if not HAS_MPL:
        print("matplotlib not available.")
        return
    tracker = processor.tracker
    if len(tracker.positions) < MIN_DISC_PTS:
        print(f"Not enough disc detections ({len(tracker.positions)}/{MIN_DISC_PTS}).")
        return

    xs_px = np.array([p[1] for p in tracker.positions], dtype=float)
    ys_px = np.array([p[2] for p in tracker.positions], dtype=float)
    ts    = np.array([p[0] for p in tracker.positions], dtype=float) / processor.fps

    mpp  = 20.0 / processor.frame_w
    xs_m = xs_px * mpp
    ys_m = -(ys_px * mpp); ys_m -= ys_m[0]

    heights = np.zeros_like(xs_m)
    try:
        c = np.polyfit(ts, ys_m, 2)
        heights = np.polyval(c, ts) - np.polyval(c, ts[0])
    except Exception:
        pass

    vx    = np.gradient(xs_m,    ts)
    vy    = np.gradient(heights, ts)
    speed = np.sqrt(vx**2 + vy**2)

    dist_m     = tracker.distance_meters(processor.frame_w)
    launch_x_m = xs_m[0]
    land_x_m   = (launch_x_m + dist_m) if dist_m else None

    fig = plt.figure(figsize=(16, 9), facecolor="#0e1117")
    fig.suptitle("Discus Throw Analysis", color="white",
                 fontsize=16, fontweight="bold", y=0.98)
    panel_kw = dict(facecolor="#1c2333")

    ax1 = fig.add_subplot(2,3,1,**panel_kw)
    ax1.set_title("Bird's Eye View", color="white")
    ax1.set_xlabel("Distance (m)", color="#aaa"); ax1.set_ylabel("Lateral (m)", color="#aaa")
    ax1.tick_params(colors="#aaa")
    for sp in ax1.spines.values(): sp.set_color("#333")
    lateral = np.sin(np.linspace(0, math.pi*0.3, len(xs_m))) * 0.8
    sc = ax1.scatter(xs_m, lateral, c=ts, cmap="plasma", s=40, zorder=3)
    ax1.plot(xs_m, lateral, color="#888", linewidth=1, alpha=0.5)
    ax1.scatter([xs_m[0]], [lateral[0]], color="#3B8BD4", s=100, zorder=5, label="Release")
    if land_x_m:
        ax1.axvline(land_x_m, color="#E24B4A", linewidth=1.5, linestyle="--", label=f"Landing ~{dist_m}m")
        ax1.scatter([land_x_m],[0], color="#E24B4A", s=100, zorder=5)
    ring = plt.Circle((xs_m[0],0),1.25,color="#FF6428",fill=False,linewidth=1.5,linestyle=":")
    ax1.add_patch(ring)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend(fontsize=7, labelcolor="white", facecolor="#111")
    plt.colorbar(sc, ax=ax1, label="Time (s)").ax.yaxis.label.set_color("white")

    ax2 = fig.add_subplot(2,3,2,**panel_kw)
    ax2.set_title("Side View (Trajectory Arc)", color="white")
    ax2.set_xlabel("Distance (m)", color="#aaa"); ax2.set_ylabel("Height (m)", color="#aaa")
    ax2.tick_params(colors="#aaa")
    for sp in ax2.spines.values(): sp.set_color("#333")
    ax2.plot(xs_m, heights, color="#EF9F27", linewidth=2.5, label="Disc path")
    ax2.scatter([xs_m[0]], [heights[0]], color="#3B8BD4", s=100, zorder=5, label="Release")
    ax2.axhline(0, color="#444", linewidth=1)
    if land_x_m:
        ax2.axvline(land_x_m, color="#E24B4A", linewidth=1.5, linestyle="--", label=f"~{dist_m}m")
        try:
            c2 = np.polyfit(xs_m, heights, 2)
            x_ext = np.linspace(xs_m[0], land_x_m, 120)
            ax2.plot(x_ext, np.polyval(c2,x_ext), color="#EF9F27", linewidth=1.5, linestyle="--", alpha=0.5)
        except Exception: pass
    ax2.fill_between(xs_m, heights, 0, where=(heights>0), alpha=0.15, color="#EF9F27")
    ax2.legend(fontsize=7, labelcolor="white", facecolor="#111")

    ax3 = fig.add_subplot(2,3,3,projection="3d")
    ax3.set_facecolor("#1c2333")
    ax3.set_title("3D Perspective", color="white")
    ax3.set_xlabel("Distance (m)", color="#aaa"); ax3.set_ylabel("Lateral (m)", color="#aaa"); ax3.set_zlabel("Height (m)", color="#aaa")
    ax3.tick_params(colors="#aaa")
    ax3.plot(xs_m, lateral, heights, color="#EF9F27", linewidth=2.5)
    ax3.scatter(xs_m, lateral, heights, c=ts, cmap="plasma", s=20)
    ax3.plot(xs_m, lateral, np.zeros_like(heights), color="#EF9F27", linewidth=1, linestyle="--", alpha=0.3)
    for i in range(0, len(xs_m), max(1, len(xs_m)//8)):
        ax3.plot([xs_m[i],xs_m[i]],[lateral[i],lateral[i]],[0,heights[i]], color="#555", linewidth=0.5)
    ax3.scatter([xs_m[0]],[lateral[0]],[heights[0]], color="#3B8BD4", s=80, zorder=5)
    if land_x_m:
        ax3.scatter([land_x_m],[0],[0], color="#E24B4A", s=80, zorder=5)
    theta = np.linspace(0, 2*math.pi, 60)
    ax3.plot(xs_m[0]+1.25*np.cos(theta), 1.25*np.sin(theta), np.zeros(60), color="#FF6428", linewidth=1.5, linestyle=":")
    ax3.xaxis.pane.fill = False; ax3.yaxis.pane.fill = False; ax3.zaxis.pane.fill = False

    ax4 = fig.add_subplot(2,3,4,**panel_kw)
    ax4.set_title("Velocity Vectors", color="white")
    ax4.set_xlabel("Distance (m)", color="#aaa"); ax4.set_ylabel("Height (m)", color="#aaa")
    ax4.tick_params(colors="#aaa")
    for sp in ax4.spines.values(): sp.set_color("#333")
    step = max(1, len(xs_m)//12)
    q = ax4.quiver(xs_m[::step], heights[::step], vx[::step], vy[::step], speed[::step],
                   cmap="cool", scale=None, scale_units="xy", angles="xy", width=0.006)
    ax4.plot(xs_m, heights, color="#444", linewidth=1, alpha=0.5)
    ax4.set_aspect("equal", adjustable="datalim")
    plt.colorbar(q, ax=ax4, label="Speed (m/s)").ax.yaxis.label.set_color("white")

    ax5 = fig.add_subplot(2,3,5,**panel_kw)
    ax5.set_title("Throwing Ring (top-down)", color="white")
    ax5.set_xlabel("m", color="#aaa"); ax5.set_ylabel("m", color="#aaa")
    ax5.tick_params(colors="#aaa")
    for sp in ax5.spines.values(): sp.set_color("#333")
    ax5.set_aspect("equal")
    ax5.add_patch(plt.Circle((0,0),1.25,color="#FF6428",fill=False,linewidth=2,label="Ring (2.5m dia)"))
    ax5.scatter([-0.2,0.2],[-0.3,-0.3], color="#3B8BD4", s=120, zorder=5, label="Feet")
    ax5.scatter([0],[0.3], color="#EF9F27", s=80, zorder=5, label="Release dir.")
    ax5.annotate("", xy=(0,0.9), xytext=(0,0.3), arrowprops=dict(arrowstyle="->",color="#EF9F27",lw=2))
    theta_arc = np.linspace(-math.pi*0.6, math.pi*0.4, 60)
    ax5.plot(0.7*np.cos(theta_arc), 0.7*np.sin(theta_arc), color="#aaa", linewidth=1.5, linestyle="--", alpha=0.7)
    ax5.set_xlim(-1.8,1.8); ax5.set_ylim(-1.8,1.8)
    ax5.legend(fontsize=7, labelcolor="white", facecolor="#111")

    ax6 = fig.add_subplot(2,3,6,**panel_kw)
    ax6.axis("off")
    ax6.set_title("Summary", color="white")
    summary = [
        ("Predicted distance",  f"{dist_m} m"            if dist_m      else "—"),
        ("Disc points tracked", str(len(tracker.positions))),
        ("Duration",            f"{ts[-1]:.2f} s"         if len(ts)     else "—"),
        ("Peak speed",          f"{speed.max():.1f} m/s"  if len(speed)  else "—"),
        ("Peak height (est.)",  f"{max(heights):.1f} m"   if len(heights) else "—"),
    ]
    for i, (label, val) in enumerate(summary):
        ax6.text(0.05, 0.82-i*0.16, label+":", color="#aaa",  fontsize=12, transform=ax6.transAxes)
        ax6.text(0.55, 0.82-i*0.16, val,        color="white", fontsize=13, fontweight="bold", transform=ax6.transAxes)

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

#GUI
class App:
    def __init__(self, root):
        self.root      = root
        self.root.title("🥏 Discus Throw Analyzer")
        self.root.configure(bg="#0e1117")
        self.root.geometry("960x680")
        self.processor = None
        self.cur_frame = 0
        self.playing   = False
        self._play_job = None
        self._build_ui()

    def _build_ui(self):
        top = tk.Frame(self.root, bg="#0e1117")
        top.pack(fill="x", padx=16, pady=(14,6))
        tk.Label(top, text="🥏 Discus Throw Analyzer",
                 font=("Courier",18,"bold"), bg="#0e1117", fg="#EF9F27").pack(side="left")
        self.load_btn = tk.Button(top, text="Open Video", command=self._load_video,
                                  font=("Courier",10,"bold"), bg="#3B8BD4", fg="white",
                                  relief="flat", padx=14, pady=6, cursor="hand2")
        self.load_btn.pack(side="right")
        self.graph_btn = tk.Button(top, text="Show Graphs", command=self._show_graphs,
                                   font=("Courier",10,"bold"), bg="#1c2333", fg="#EF9F27",
                                   relief="flat", padx=14, pady=6, cursor="hand2", state="disabled")
        self.graph_btn.pack(side="right", padx=(0,8))

        self.canvas = tk.Label(self.root, bg="#111827", text="Open a video to begin",
                               fg="#555", font=("Courier",13))
        self.canvas.pack(fill="both", expand=True, padx=16)

        self.prog_var = tk.IntVar(value=0)
        self.prog_bar = ttk.Progressbar(self.root, variable=self.prog_var, maximum=100, length=900)
        self.prog_bar.pack(fill="x", padx=16, pady=(4,0))
        self.prog_lbl = tk.Label(self.root, text="", font=("Courier",9), bg="#0e1117", fg="#888")
        self.prog_lbl.pack()

        scrub_frame = tk.Frame(self.root, bg="#0e1117")
        scrub_frame.pack(fill="x", padx=16, pady=(6,2))
        self.scrub = tk.Scale(scrub_frame, from_=0, to=0, orient="horizontal",
                              command=self._on_scrub, bg="#1c2333", fg="white",
                              troughcolor="#333", highlightthickness=0,
                              showvalue=False, sliderlength=16, bd=0)
        self.scrub.pack(fill="x")

        ctrl = tk.Frame(self.root, bg="#0e1117")
        ctrl.pack(pady=(2,10))
        btn_cfg = dict(font=("Courier",11,"bold"), bg="#1c2333", fg="white",
                       relief="flat", padx=12, pady=5, cursor="hand2")
        tk.Button(ctrl, text="◀◀", command=self._go_start,  **btn_cfg).pack(side="left", padx=4)
        tk.Button(ctrl, text="◀",  command=self._step_back, **btn_cfg).pack(side="left", padx=4)
        self.play_btn = tk.Button(ctrl, text="▶", command=self._toggle_play, **btn_cfg)
        self.play_btn.pack(side="left", padx=4)
        tk.Button(ctrl, text="▶",  command=self._step_fwd,  **btn_cfg).pack(side="left", padx=4)
        tk.Button(ctrl, text="▶▶", command=self._go_end,    **btn_cfg).pack(side="left", padx=4)
        self.frame_lbl = tk.Label(ctrl, text="—", font=("Courier",9), bg="#0e1117", fg="#888")
        self.frame_lbl.pack(side="left", padx=12)

    def _load_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video","*.mp4 *.mov *.avi *.mkv *.webm"),("All","*.*")])
        if not path: return
        self.load_btn.config(state="disabled")
        self.graph_btn.config(state="disabled")
        self.prog_var.set(0)

        # Ask user to draw ROI before processing
        self._set_status("Draw a box around the disc flight zone in the popup window…")
        self.root.update()
        roi = select_roi(path)
        if roi:
            self._set_status(f"ROI set: {roi} — processing video…")
        else:
            self._set_status("No ROI set — processing full frame…")

        threading.Thread(target=self._process, args=(path, roi), daemon=True).start()

    def _process(self, path, roi=None):
        try:
            proc = VideoProcessor(path, self._update_prog, roi=roi)
            proc.process()
            self.processor = proc
            self.cur_frame = 0
            n = len(proc.frames)
            self.root.after(0, lambda: self.scrub.config(to=max(0,n-1)))
            self.root.after(0, lambda: self._show_frame(0))
            self.root.after(0, lambda: self.graph_btn.config(state="normal"))
            self._set_status(f"Ready — {n} frames processed")
        except Exception as e:
            self._set_status(f"Error: {e}")
        finally:
            self.root.after(0, lambda: self.load_btn.config(state="normal"))

    def _show_frame(self, idx):
        if self.processor is None: return
        idx = max(0, min(idx, len(self.processor.frames)-1))
        self.cur_frame = idx
        frame = self.processor.frames[idx]
        cw = self.canvas.winfo_width()  or 880
        ch = self.canvas.winfo_height() or 460
        fh, fw = frame.shape[:2]
        scale = min(cw/fw, ch/fh, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(fw*scale), int(fh*scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.config(image=img, text="")
        self.canvas._img = img
        self.scrub.set(idx)
        n = len(self.processor.frames)
        self.frame_lbl.config(text=f"Frame {idx+1} / {n}  |  {self.processor.phases[idx]}")

    def _toggle_play(self):
        if self.playing:
            self.playing = False
            self.play_btn.config(text="▶")
            if self._play_job: self.root.after_cancel(self._play_job)
        else:
            self.playing = True
            self.play_btn.config(text="⏸")
            self._play_loop()

    def _play_loop(self):
        if not self.playing or self.processor is None: return
        n = len(self.processor.frames)
        if self.cur_frame >= n-1:
            self.playing = False
            self.play_btn.config(text="▶")
            return
        self._show_frame(self.cur_frame+1)
        delay = max(16, int(1000/self.processor.fps))
        self._play_job = self.root.after(delay, self._play_loop)

    def _on_scrub(self, val):
        if self.processor: self._show_frame(int(float(val)))

    def _step_fwd(self):
        if self.processor: self._show_frame(self.cur_frame+1)

    def _step_back(self):
        if self.processor: self._show_frame(self.cur_frame-1)

    def _go_start(self):
        if self.processor: self._show_frame(0)

    def _go_end(self):
        if self.processor: self._show_frame(len(self.processor.frames)-1)

    def _show_graphs(self):
        if self.processor:
            threading.Thread(target=open_graphs, args=(self.processor,), daemon=True).start()

    def _update_prog(self, val):
        self.root.after(0, lambda: self.prog_var.set(val))

    def _set_status(self, msg):
        self.root.after(0, lambda: self.prog_lbl.config(text=msg))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
