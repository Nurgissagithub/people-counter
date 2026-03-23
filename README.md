# People / Object Counter

Real-time people counting from video streams using **YOLOv8** (detection) and **ByteTrack** (multi-object tracking).

Each person is assigned a persistent ID. When their path crosses a configurable virtual line, they are counted as **IN** or **OUT** based on direction.

---

## Features

- Person detection with YOLOv8 (nano → extra-large models)
- Multi-object tracking with ByteTrack — stable IDs across frames
- Virtual counting line with IN / OUT direction logic
- Color-coded bounding boxes and movement trails
- RTSP stream support with automatic reconnection
- Config file + CLI argument support
- Headless mode for server / embedded deployment
- Annotated video export

---

## Setup

```bash
conda create -n people-counter python=3.10 -y
conda activate people-counter
pip install -r requirements.txt
```

---

## Usage

### Webcam
```bash
python main.py --source 0
```

### Video file
```bash
python main.py --source video.mp4 --save output.mp4
```

### RTSP stream (IP camera)
```bash
python main.py --source rtsp://192.168.1.100:554/stream
```

### Config file
```bash
python main.py --config config.yaml
```
CLI args always override config file values.

### Headless (no display window)
```bash
python main.py --source rtsp://... --no-show --save output.mp4
```

---

## CLI reference

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | Video file, `0` for webcam, or `rtsp://...` URL |
| `--model` | `yolov8n.pt` | YOLOv8 weights (auto-downloaded on first use) |
| `--line` | `0.5` | Counting line position (0.0 = top, 1.0 = bottom) |
| `--conf` | `0.4` | Detection confidence threshold |
| `--no-show` | off | Disable live window |
| `--save` | off | Save annotated video to this path |
| `--config` | off | Load settings from a YAML file |

---

## Model sizes

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `yolov8n.pt` | 6 MB | fastest | good |
| `yolov8s.pt` | 22 MB | fast | better |
| `yolov8m.pt` | 52 MB | medium | best for CPU |
| `yolov8l.pt` | 87 MB | slow | requires GPU |

Models download automatically from Ultralytics on first run.

---

## Project structure

```
people-counter/
├── main.py          # Production entry point (Day 3)
├── tracker.py       # Standalone tracker script (Day 2)
├── detector.py      # Standalone detector script (Day 1)
├── config.yaml      # Configuration file
├── requirements.txt
└── README.md
```

---

## How it works

1. **Frame capture** — OpenCV reads frames from any source (file, webcam, RTSP).
2. **Detection** — YOLOv8 finds people in each frame (class 0 in COCO).
3. **Tracking** — ByteTrack assigns a consistent ID to each person across frames.
4. **Counting** — Each ID's center point is tracked. When it crosses the virtual line, the direction (up/down) determines IN or OUT. Each ID is counted only once.
5. **Output** — Annotated frames are displayed and/or written to a file.

---

## Technologies

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [ByteTrack](https://github.com/ifzhang/ByteTrack) (via Ultralytics)
- Python 3.10
