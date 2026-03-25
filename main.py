"""
main.py — People Counter Project

Unified launcher with:
- RTSP stream support with auto-reconnect
- Config file support (config.yaml)
- Graceful shutdown and final report
- Headless mode for server deployment
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import time
import argparse
import yaml
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ─── Defaults (overridden by config.yaml or CLI args) ─────────────────────────

DEFAULTS = {
    "source":      "0",
    "model":       "yolov8n.pt",
    "line_pos":    0.5,
    "conf":        0.4,
    "show":        True,
    "save":        None,
    "rtsp_retry":  5,      # seconds between RTSP reconnect attempts
    "max_retries": 10,     # max reconnect attempts before giving up
}

PERSON_CLASS_ID = 0
TRAIL_LEN       = 30
FONT            = cv2.FONT_HERSHEY_SIMPLEX

COLOR_IN   = (0, 200, 100)
COLOR_OUT  = (0, 140, 255)
COLOR_NEW  = (160, 160, 160)
COLOR_LINE = (0, 255, 255)
COLOR_TEXT = (255, 255, 255)


# ─── Config loader ────────────────────────────────────────────────────────────

def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULTS)
    if path and Path(path).exists():
        with open(path) as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in file_cfg.items() if v is not None})
        print(f"[INFO] Loaded config: {path}")
    return cfg


# ─── Track state ──────────────────────────────────────────────────────────────

class TrackState:
    def __init__(self):
        self.centers: dict[int, list] = {}
        self.crossed: dict[int, str | None] = {}

    def update(self, tid: int, cx: int, cy: int):
        if tid not in self.centers:
            self.centers[tid] = []
            self.crossed[tid] = None
        self.centers[tid].append((cx, cy))
        if len(self.centers[tid]) > TRAIL_LEN:
            self.centers[tid].pop(0)

    def check_crossing(self, tid: int, line_y: int) -> str | None:
        history = self.centers.get(tid, [])
        if len(history) < 2 or self.crossed[tid] is not None:
            return None
        prev_y, curr_y = history[-2][1], history[-1][1]
        if prev_y < line_y <= curr_y:
            self.crossed[tid] = "in"
            return "in"
        if prev_y > line_y >= curr_y:
            self.crossed[tid] = "out"
            return "out"
        return None


# ─── Drawing ──────────────────────────────────────────────────────────────────

def _status_color(status):
    return {"in": COLOR_IN, "out": COLOR_OUT, None: COLOR_NEW}[status]


def draw_line(frame, line_y):
    w = frame.shape[1]
    dash, gap = 20, 10
    x = 0
    while x < w:
        cv2.line(frame, (x, line_y), (min(x + dash, w), line_y), COLOR_LINE, 2)
        x += dash + gap
    cv2.putText(frame, "IN",  (10, line_y - 8),  FONT, 0.55, COLOR_LINE, 1, cv2.LINE_AA)
    cv2.putText(frame, "OUT", (10, line_y + 18), FONT, 0.55, COLOR_LINE, 1, cv2.LINE_AA)


def draw_box(frame, x1, y1, x2, y2, tid, conf, status):
    color = _status_color(status)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {tid}  {conf:.2f}"
    (lw, lh), bl = cv2.getTextSize(label, FONT, 0.52, 1)
    cv2.rectangle(frame, (x1, y1 - lh - bl - 4), (x1 + lw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - bl - 2), FONT, 0.52, COLOR_TEXT, 1, cv2.LINE_AA)


def draw_trail(frame, centers, status):
    n = len(centers)
    color = _status_color(status)
    for i in range(1, n):
        alpha = i / n
        c = tuple(int(ch * alpha) for ch in color)
        cv2.line(frame, centers[i - 1], centers[i], c, 1)


def draw_hud(frame, count_in, count_out, fps, active, source_label):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, f"IN:  {count_in}",  (20, 40),  FONT, 0.8, COLOR_IN,  2, cv2.LINE_AA)
    cv2.putText(frame, f"OUT: {count_out}", (20, 75),  FONT, 0.8, COLOR_OUT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{source_label}   FPS {fps:.1f}   active {active}",
                (20, 108), FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def draw_reconnecting(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "RECONNECTING...", (frame.shape[1] // 2 - 150, frame.shape[0] // 2),
                FONT, 1.0, COLOR_LINE, 2, cv2.LINE_AA)


# ─── Stream opener with retry ─────────────────────────────────────────────────

def open_stream(source, rtsp_retry: int, max_retries: int):
    """
    Open a VideoCapture. For RTSP sources, retries up to max_retries times.
    Returns (cap, width, height, fps) or raises RuntimeError.
    """
    is_rtsp = isinstance(source, str) and source.startswith("rtsp://")

    for attempt in range(1, max_retries + 1):
        cap = cv2.VideoCapture(source)

        # RTSP hint: reduce buffer to minimise latency
        if is_rtsp:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if cap.isOpened():
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            print(f"[INFO] Stream opened: {w}x{h} @ {fps:.1f} fps")
            return cap, w, h, fps

        cap.release()
        if not is_rtsp or attempt == max_retries:
            raise RuntimeError(f"Cannot open source after {attempt} attempt(s): {source}")

        print(f"[WARN] Cannot open stream (attempt {attempt}/{max_retries}). "
              f"Retrying in {rtsp_retry}s...")
        time.sleep(rtsp_retry)

    raise RuntimeError("Exhausted retries.")


# ─── Core loop ────────────────────────────────────────────────────────────────

def run(cfg: dict):
    source_raw = cfg["source"]
    source     = int(source_raw) if str(source_raw).isdigit() else source_raw
    is_rtsp    = isinstance(source, str) and source.startswith("rtsp://")

    source_label = (
        "webcam" if source == 0 else
        "RTSP"   if is_rtsp else
        Path(str(source)).name
    )

    model = YOLO(cfg["model"])
    print(f"[INFO] Model: {cfg['model']}")

    cap, width, height, fps_src = open_stream(
        source, cfg["rtsp_retry"], cfg["max_retries"]
    )
    line_y = int(height * cfg["line_pos"])
    print(f"[INFO] Counting line at y={line_y}  (line_pos={cfg['line_pos']})")

    writer = None
    if cfg["save"]:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cfg["save"], fourcc, fps_src, (width, height))
        print(f"[INFO] Saving to: {cfg['save']}")

    state      = TrackState()
    count_in   = 0
    count_out  = 0
    frame_idx  = 0
    fps_disp   = 0.0
    timer      = cv2.getTickCount()
    retries    = 0

    print("[INFO] Running — press Q to quit.\n")

    try:
        while True:
            ret, frame = cap.read()

            # ── Handle lost frames (RTSP drop / end of file) ──
            if not ret:
                if not is_rtsp:
                    print("[INFO] End of file.")
                    break

                retries += 1
                if retries > cfg["max_retries"]:
                    print("[ERROR] Too many failed reads. Giving up.")
                    break

                print(f"[WARN] Lost frame ({retries}/{cfg['max_retries']}). "
                      f"Reconnecting in {cfg['rtsp_retry']}s...")

                # Show "reconnecting" overlay on last frame if window is open
                if cfg["show"] and frame is not None:
                    draw_reconnecting(frame)
                    cv2.imshow("People Counter", frame)
                    cv2.waitKey(1)

                cap.release()
                time.sleep(cfg["rtsp_retry"])
                try:
                    cap, _, _, _ = open_stream(source, cfg["rtsp_retry"], 1)
                    retries = 0
                except RuntimeError:
                    continue
                continue

            retries = 0  # successful read resets the counter

            # ── Tracking ──
            results = model.track(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=cfg["conf"],
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )[0]

            boxes  = results.boxes
            active = 0

            if boxes is not None and boxes.id is not None:
                for box in boxes:
                    if box.id is None:
                        continue
                    tid = int(box.id[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    state.update(tid, cx, cy)
                    event = state.check_crossing(tid, line_y)

                    if event == "in":
                        count_in += 1
                        print(f"  → ID {tid:4d}  crossed IN   | IN={count_in}  OUT={count_out}")
                    elif event == "out":
                        count_out += 1
                        print(f"  → ID {tid:4d}  crossed OUT  | IN={count_in}  OUT={count_out}")

                    status = state.crossed[tid]
                    draw_trail(frame, state.centers[tid], status)
                    draw_box(frame, x1, y1, x2, y2, tid, conf, status)
                    cv2.circle(frame, (cx, cy), 3, COLOR_TEXT, -1)
                    active += 1

            # ── Overlays ──
            draw_line(frame, line_y)

            frame_idx += 1
            if frame_idx % 10 == 0:
                elapsed  = (cv2.getTickCount() - timer) / cv2.getTickFrequency()
                fps_disp = 10 / elapsed
                timer    = cv2.getTickCount()

            draw_hud(frame, count_in, count_out, fps_disp, active, source_label)

            if cfg["show"]:
                cv2.imshow("People Counter — press Q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Quit by user.")
                    break

            if writer:
                writer.write(frame)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C).")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # ── Final report ──
        print("\n" + "─" * 40)
        print(f"  Source  : {source_label}")
        print(f"  Model   : {cfg['model']}")
        print(f"  Frames  : {frame_idx}")
        print(f"  IN      : {count_in}")
        print(f"  OUT     : {count_out}")
        print(f"  Net     : {count_in - count_out:+d}")
        print("─" * 40)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="People counter — production entry point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",    default=None,          help="Path to config.yaml")
    p.add_argument("--source",    default=None,          help="Video file, '0' webcam, or rtsp://...")
    p.add_argument("--model",     default=None,          help="YOLOv8 weights file")
    p.add_argument("--line",      type=float,            help="Counting line position 0.0–1.0")
    p.add_argument("--conf",      type=float,            help="Detection confidence threshold")
    p.add_argument("--no-show",   action="store_true",   help="Disable live window (headless)")
    p.add_argument("--save",      default=None,          help="Save annotated video to file")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)

    # CLI args override config file
    if args.source   is not None: cfg["source"]   = args.source
    if args.model    is not None: cfg["model"]    = args.model
    if args.line     is not None: cfg["line_pos"] = args.line
    if args.conf     is not None: cfg["conf"]     = args.conf
    if args.no_show:               cfg["show"]    = False
    if args.save     is not None: cfg["save"]     = args.save

    run(cfg)
