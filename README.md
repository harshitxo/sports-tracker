# 🏏 Predusk Sports Tracker
### Multi-Object Detection & Persistent ID Tracking in Public Sports Footage

> **Assignment:** Multi-Object Detection and Persistent ID Tracking  
> **Company:** Predusk Technology Pvt. Ltd. (ProcessVenue), Jaipur  
> **Type:** AI / Computer Vision / Data Science  

---

## 📌 Project Overview

This project implements a complete **real-time multi-object detection and persistent ID tracking pipeline** for public sports/event videos. It detects all moving subjects (players, athletes) in a video and assigns each a **unique, stable ID** that persists across the full video — even through occlusion, motion blur, and rapid movement.

**Architecture at a glance:**
```
Input Video → YOLOv8 Detection → ByteTrack ID Assignment → Annotated Output Video
                                        ↓
                          Analytics: Heatmap | ID Timeline | Trajectories
```

---

## 🎬 Source Video

| Detail | Info |
|--------|------|
| **Video Title** | *(Add your chosen video title here)* |
| **Public URL**  | *(Add YouTube / public link here)* |
| **Category**    | Cricket / Football / Marathon (choose one) |
| **Duration**    | *(e.g. ~3 minutes used)* |
| **Resolution**  | 720p |

> The video is publicly available and used solely for evaluation/academic purposes.

---

## 🧠 Model & Tracker Choices

| Component | Choice | Reason |
|-----------|--------|--------|
| **Detector** | YOLOv8m (Ultralytics) | Best speed-accuracy tradeoff; COCO-pretrained person class |
| **Tracker** | ByteTrack | State-of-the-art multi-object tracker; handles occlusion via low-confidence detection recovery |
| **Classes** | Person (COCO class 0) | Targets players/athletes |

---

## 📁 Project Structure

```
predusk_sports_tracker/
├── src/
│   ├── tracker.py          # Core detection + tracking class (SportsTracker)
│   ├── pipeline.py         # Entry-point: video I/O, frame loop, output writing
│   ├── visualize.py        # Analytics: heatmap, ID count chart, trajectory overlay
│   └── download_video.py   # Helper: download public video via yt-dlp
│
├── notebooks/
│   └── Sports_Tracker_Walkthrough.ipynb   # Full end-to-end notebook demo
│
├── output/                 # Generated output video + analytics (git-ignored)
│   ├── tracked.mp4
│   └── analytics/
│       ├── heatmap.png
│       ├── id_count.png
│       └── track_durations.png
│
├── screenshots/            # Sample result screenshots
├── reports/
│   └── technical_report.md
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- pip
- (Optional) NVIDIA GPU with CUDA for faster inference

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/predusk-sports-tracker.git
cd predusk-sports-tracker
```

### Step 2 — Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** YOLOv8 weights (`yolov8m.pt`) download automatically on first run (~52 MB).  
> PyTorch with CUDA support: visit https://pytorch.org/get-started/locally/ for GPU builds.

---

## 🚀 How to Run

### Option A — Command Line (Recommended)

**Step 1: Download a public sports video**
```bash
python src/download_video.py \
  --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
  --output data/input_video.mp4 \
  --max-height 720
```

**Step 2: Run the tracking pipeline**
```bash
python src/pipeline.py \
  --input data/input_video.mp4 \
  --output output/tracked.mp4 \
  --model yolov8m.pt \
  --conf 0.35 \
  --tracker bytetrack.yaml
```

**All CLI options:**
```
--input    / -i   Path to input video (required)
--output   / -o   Output video path (default: output/tracked.mp4)
--model    / -m   YOLOv8 model size: yolov8n/s/m/l/x.pt (default: yolov8m.pt)
--conf     / -c   Detection confidence threshold 0–1 (default: 0.35)
--skip     / -s   Frame skip interval — 1=every frame, 2=every other (default: 1)
--tracker  / -t   bytetrack.yaml or botsort.yaml (default: bytetrack.yaml)
--max-frames      Limit to first N frames (useful for quick testing)
--no-trajectory   Disable motion trail drawing
```

### Option B — Jupyter Notebook

```bash
jupyter notebook notebooks/Sports_Tracker_Walkthrough.ipynb
```

Run cells top-to-bottom. The notebook covers download → tracking → analytics → summary.

### Option C — Python API

```python
from src.pipeline import run_pipeline

summary = run_pipeline(
    input_path="data/input_video.mp4",
    output_path="output/tracked.mp4",
    config={
        "model": "yolov8m.pt",
        "conf_threshold": 0.35,
        "tracker": "bytetrack.yaml",
        "show_trajectory": True,
    }
)
print(f"Unique IDs: {summary['total_unique_ids']}")
```

---

## 📊 Output Description

### Annotated Video (`output/tracked.mp4`)
- **Rounded bounding box** per detected subject
- **ID label + confidence badge** (e.g. `ID 3  87%`)
- **Motion trajectory tail** — color-coded per ID, fades with distance
- **HUD overlay** — frame number, processing FPS, active/total ID count

### Analytics (`output/analytics/`)
| File | Description |
|------|-------------|
| `heatmap.png` | Spatial density of all subject movements (hot = high traffic) |
| `id_count.png` | Active subjects count over time (line chart) |
| `track_durations.png` | How long each unique ID was continuously tracked |

### Stats JSON (`output/run_stats.json`)
Machine-readable summary of the full run: total IDs, frame count, avg active, per-ID durations.

---

## 🔧 Configuration Reference

Key settings in `pipeline.py → DEFAULT_CONFIG`:

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `yolov8m.pt` | YOLOv8 variant (n/s/m/l/x — larger = more accurate, slower) |
| `conf_threshold` | `0.35` | Min detection confidence (lower = more detections, more FP) |
| `iou_threshold` | `0.50` | NMS IoU threshold |
| `tracker` | `bytetrack.yaml` | Tracking algorithm config |
| `frame_skip` | `1` | Process every Nth frame (increase for speed on long videos) |
| `trajectory_tail` | `40` | Number of past positions to draw as tail |
| `show_hud` | `True` | Enable stats overlay on video |

---

## 🧩 Assumptions & Design Decisions

1. **Person-only detection:** COCO class 0 (person) is targeted. For vehicle-based events, add class IDs 2 (car), 3 (motorbike) etc.
2. **ByteTrack over SORT/DeepSORT:** ByteTrack requires no external ReID model, uses both high- and low-confidence detections for better occlusion handling, and runs faster.
3. **YOLOv8m as default:** Nano/small models are faster but miss players in the distance; medium is the sweet spot for 720p sports footage.
4. **Frame skip = 1 by default:** Every frame processed for maximum ID stability. For videos > 5 minutes, use `--skip 2`.
5. **Trajectory memory capped at 500 points per ID:** Prevents memory growth on very long videos.

---

## ⚠️ Known Limitations

- **ID switches on severe occlusion:** When two players fully overlap for several frames, ByteTrack may assign a new ID post-occlusion.
- **No ReID / appearance model:** IDs are assigned purely by IoU + Kalman prediction. Adding a deep appearance model (e.g. OSNet) would improve re-identification after long occlusions.
- **Camera pan sensitivity:** Rapid camera movement causes Kalman predictions to diverge, temporarily increasing ID switches.
- **Small/distant subjects:** At 720p with many players (e.g. wide-angle football), small subjects may be missed at conf=0.35. Lowering threshold increases false positives.
- **Class-specific only:** Currently tracks humans. Does not distinguish team, role, or jersey number.

---

## 🚀 Possible Improvements

- Add **OSNet / FastReID** appearance embeddings for robust re-identification
- **Team clustering** via jersey color histogram (K-Means on HSV)
- **Speed estimation** using homography (pixel displacement × scale factor)
- **Bird's-eye view** projection using court/field homography
- **ONNX export** of YOLOv8 for faster CPU inference
- **Streamlit demo app** for browser-based video upload and tracking

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 detection + ByteTrack integration |
| `opencv-python` | Video I/O and frame annotation |
| `numpy` | Numerical operations |
| `matplotlib` | Analytics charts |
| `yt-dlp` | Public video download |
| `torch` | Deep learning backend |

---

## 👤 Author

Submitted as part of the Predusk Technology Paid Internship Assessment.  
**Assessment Type:** AI / Computer Vision  
**Contact:** *(Your email)*
