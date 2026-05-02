# Technical Report
## Multi-Object Detection and Persistent ID Tracking in Public Sports Footage
**Predusk Technology — AI/Computer Vision Internship Assessment**

---

## 1. Objective

The goal of this project is to build a robust computer vision pipeline that:
- Detects all moving subjects (players/athletes) in a public sports video
- Assigns each subject a **unique, persistent ID** across all frames
- Handles real-world challenges: occlusion, motion blur, camera movement, similar-looking subjects
- Produces an annotated output video and analytics visualizations

---

## 2. Model & Detector

### YOLOv8 (Ultralytics)

**Why YOLOv8?**

YOLOv8 is the current state-of-the-art single-stage object detector. It provides:
- Real-time inference (30–60 FPS on GPU at 720p)
- Excellent person detection accuracy on sports footage
- Native integration with ByteTrack and BotSORT trackers via the Ultralytics API
- Pretrained COCO weights that cover the "person" class (class 0) directly

**Variant used:** `yolov8m.pt` (medium)  
The medium variant was selected as the default because:
- Nano/small models miss distant or partially visible players
- Large/xlarge models are slower than necessary for 720p video
- Medium achieves a good balance — ~45 FPS on GPU, mAP ~50.2 on COCO

**Detection settings:**
- Confidence threshold: 0.35 (low enough to catch partially visible players)
- IoU threshold (NMS): 0.50
- Target class: 0 (person only)

---

## 3. Tracking Algorithm

### ByteTrack

**Why ByteTrack over alternatives?**

| Tracker | Approach | Requires ReID? | Speed | Occlusion Handling |
|---------|----------|----------------|-------|--------------------|
| SORT | IoU + Kalman | No | Very fast | Weak |
| DeepSORT | IoU + Kalman + Appearance | Yes (CNN) | Moderate | Good |
| **ByteTrack** | **IoU + Kalman (high+low conf)** | **No** | **Fast** | **Very Good** |
| BotSORT | IoU + Kalman + ReID | Optional | Moderate | Excellent |

ByteTrack's key innovation is using **both high-confidence and low-confidence detections** in a two-stage association. This means that when a player is partially occluded (lower detection confidence), ByteTrack still attempts to associate them with an existing track rather than dropping the ID — which is a common failure in SORT-based methods.

This makes ByteTrack ideal for sports footage where players frequently overlap, especially in cricket (fielder clusters), football (set pieces), and basketball (post play).

---

## 4. How ID Consistency Is Maintained

ID consistency relies on three layers:

**Layer 1 — Kalman Filter (motion prediction):**  
Each track maintains a Kalman state vector `[x, y, w, h, vx, vy]`. Between frames, the filter predicts where the subject will be based on velocity. This handles fast movement and brief disappearance without breaking the ID.

**Layer 2 — IoU Matching (two-stage):**  
ByteTrack first associates high-confidence detections (conf ≥ 0.5) with existing tracks via Hungarian algorithm on IoU scores. In the second stage, remaining unmatched tracks are given a second chance using lower-confidence detections (0.1–0.5 conf). This is the key step that prevents ID loss during occlusion.

**Layer 3 — Track lifecycle management:**  
Tracks are not deleted immediately when a detection is missed. A track stays "active" for N frames while the Kalman filter predicts its position. Only after N consecutive misses is the track marked lost. This prevents premature ID termination during a brief occlusion.

**Persistent state in code:**  
A `TrackState` object per ID stores its full trajectory, confidence history, frame range, and color. These persist in memory for the entire video run, allowing trajectory visualization and analytics.

---

## 5. Pipeline Architecture

```
Input Video
    │
    ▼
[Frame Reader] ─── frame_skip interval
    │
    ▼
[YOLOv8 Detection]
  - Person bounding boxes + confidence scores
    │
    ▼
[ByteTrack Association]
  - Stage 1: High-conf detections ↔ existing tracks (IoU + Kalman)
  - Stage 2: Low-conf detections ↔ remaining unmatched tracks
  - New tracks initialized for unmatched high-conf detections
    │
    ▼
[TrackState Update]
  - Update trajectory, box, conf, frame count per ID
    │
    ▼
[Frame Annotation]
  - Draw bounding box + ID label + trajectory tail + HUD
    │
    ▼
[Output Video Writer]
    │
    ▼
[Analytics Generator]
  - Heatmap, ID count over time, track duration chart
```

---

## 6. Challenges Faced

### 6.1 ID Switches on Full Occlusion
When two players fully overlap for more than ~8 frames, the Kalman prediction diverges enough that re-association fails. ByteTrack assigns a new ID on reappearance. This is the most common failure case observed.

**Partial mitigation:** Keeping conf_threshold low (0.35) and extending the track buffer helps, but doesn't fully solve full-body occlusion.

### 6.2 Camera Pan / Zoom
When the camera pans rapidly (common in cricket — following the ball), all Kalman predictions become inaccurate simultaneously. This causes momentary ID instability until the camera stabilizes. BotSORT with camera motion compensation handles this better, but adds overhead.

### 6.3 Similar-Looking Subjects
In sports with uniforms (e.g., cricket, football), players in the same team look very similar. Without an appearance/ReID model, the tracker relies purely on spatial proximity — which can cause confusions when players cross paths.

### 6.4 Distant/Small Subjects
In wide-angle shots, players in the background appear very small (< 20px height). YOLOv8m reliably detects subjects down to ~30px at conf=0.35, but smaller subjects are missed or assigned low confidence and may not survive NMS.

---

## 7. Failure Cases Observed

- **ID fragmentation:** A single player tracked as 2–3 different IDs across the video due to occlusion recovery failures
- **Ghost tracks:** Occasionally a Kalman-predicted track persists for a few frames in empty space after a player exits frame
- **Crowd merging:** In dense group scenes (e.g., celebrations, marathon starts), overlapping bounding boxes and similar positions cause ID instability
- **Motion blur:** Fast bowler / sprint causes blurred detections with lower confidence, leading to missed frames

---

## 8. Possible Improvements

| Improvement | Impact | Complexity |
|-------------|--------|------------|
| Add OSNet/FastReID appearance embeddings | High — solves ReID after occlusion | Medium |
| Camera motion compensation (BotSORT) | Medium — stabilizes IDs during pans | Low (just switch tracker) |
| Jersey color clustering for team separation | Medium — adds team context | Medium |
| Homography-based bird's-eye view | High — enables spatial analytics | High |
| Speed estimation (pixel/frame × scale) | Medium — useful stat | Medium |
| Multi-scale detection (SAHI tiling) | High — catches distant players | Medium |
| ONNX export for CPU deployment | Medium — deployment speed | Low |

---

## 9. Model Comparison

| Model | mAP@50 (COCO) | Speed (GPU 720p) | Notes |
|-------|---------------|------------------|-------|
| YOLOv8n | 37.3 | ~80 FPS | Misses distant players |
| YOLOv8s | 44.9 | ~65 FPS | Better, still misses small objects |
| **YOLOv8m** | **50.2** | **~45 FPS** | **Chosen — best tradeoff** |
| YOLOv8l | 52.9 | ~30 FPS | Marginal gain, slower |
| YOLOv8x | 53.9 | ~18 FPS | Overkill for 720p |

---

## 10. Conclusion

The YOLOv8 + ByteTrack combination proved to be a robust, practical choice for sports video tracking:
- No ReID model required — reduces complexity and dependencies
- Real-time capable on modern GPUs
- Handles moderate occlusion well via two-stage association
- Clean Ultralytics API enables rapid iteration

The main limitation is appearance-based re-identification — adding a lightweight ReID model (e.g. OSNet) as a future enhancement would significantly improve ID consistency across full occlusions, which is the primary failure mode in dense sports scenes.

---

*End of Technical Report*
