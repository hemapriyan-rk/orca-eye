# ORCA EYE — Baseline Assistive Vision Navigation System

> **RESEARCH PROTOTYPE — NOT FOR REAL-WORLD MOBILITY USE**

A modular, measurable baseline system for assistive-vision navigation research.
Every component is independently observable. Every failure is logged.

---

## Quick Start

### 1. Install Dependencies

```bash
# 1a. Install PyTorch with CUDA (RTX 4050 uses CUDA 11.8 or 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 1b. Install all other dependencies
pip install -r requirements.txt
```

### 2. Run with Webcam

```bash
python main.py --source 0
```

### 3. Run with Pre-recorded Video

```bash
python main.py --source data/sample.mp4
```

### 4. Run with Full Evaluation Metrics

```bash
python main.py --source data/sample.mp4 --evaluate
```

### 5. List Test Scenarios

```bash
python main.py --list-scenarios
```

### 6. Run Unit Tests

```bash
python -m pytest tests/ -v
```

---

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| Any other | Continue |

---

## Output Files

| Location | Contents |
|----------|----------|
| `logs/<session_id>/frames.jsonl` | Per-frame full pipeline state |
| `logs/<session_id>/metrics.csv` | Time-series metrics |
| `logs/<session_id>/performance_report.md` | Session summary |
| `failure_cases/frame_NNNNNN_SEVERITY.jpg` | Failure snapshots |
| `failure_cases/frame_NNNNNN_SEVERITY.json` | Failure state JSON |

---

## Project Structure

```
orca_eye/
│
├── main.py                    ← Entry point
├── config.yaml                ← ALL tunable parameters
├── requirements.txt
│
├── perception/
│   ├── camera.py              ← Stage 1: CameraSource
│   ├── detector.py            ← Stage 2: YOLODetector + ObjectState
│   ├── tracker.py             ← Stage 3: CentroidTracker + TrackedObject
│   ├── depth.py               ← Stage 4a: MiDaS depth estimator
│   └── freespace.py           ← Stage 4b: Hybrid free-space estimator
│
├── navigation/
│   ├── spatial_map.py         ← Stage 5: Grid-based scene representation
│   ├── path_generator.py      ← Stage 6: PathCandidate generator
│   ├── path_scorer.py         ← Stage 7: Weighted path scoring
│   └── decision.py            ← Stage 8: Safe-path selection + commands
│
├── visualization/
│   └── renderer.py            ← Stage 9: 4-panel OpenCV debug display
│
├── logging/
│   └── logger.py              ← Stage 10: JSONL + CSV research logger
│
├── failure_analysis/
│   └── detector.py            ← Stage 11: 12-condition failure flagging
│
├── evaluation/
│   ├── metrics.py             ← Stage 12a: Performance metrics
│   └── scenarios.py           ← Stage 12b: 15-scenario test suite
│
├── tests/
│   ├── test_detector.py
│   ├── test_tracker.py
│   ├── test_path_scorer.py
│   ├── test_spatial_map.py
│   ├── test_freespace.py
│   └── test_failure_detector.py
│
├── data/
│   └── scenarios/             ← Place test video clips here
│
├── logs/                      ← Runtime logs (gitignored)
└── failure_cases/             ← Failure snapshots (gitignored)
```

---

## Configuration

All parameters are in [`config.yaml`](config.yaml). Nothing is hard-coded in source modules.

Key sections:

| Section | Purpose |
|---------|---------|
| `camera` | Source, resolution, FPS target |
| `detection` | YOLO model, confidence, NMS, target classes |
| `tracking` | Max disappeared, distance, velocity alpha |
| `depth` | MiDaS model, device, enabled flag |
| `freespace` | Geometric prior, gradient threshold, obstacle inflation |
| `spatial_map` | Grid size, decay, uncertainty growth |
| `path_generation` | Candidate angles, arc depth, min width |
| `scoring` | w_c, w_f, w_p, w_r, w_u weights |
| `safety` | Min score, min clearance, hysteresis, oscillation window |
| `visualization` | Panel toggles, colors, TTS |
| `logging` | Log dir, flush interval, log size |
| `failure_detection` | All failure thresholds |
| `evaluation` | Scenario and results directories |

---

## Navigation Commands

| Command | Meaning |
|---------|---------|
| `STRAIGHT` | Continue forward |
| `SLIGHT_LEFT` | Gently veer left |
| `LEFT` | Turn left |
| `SLIGHT_RIGHT` | Gently veer right |
| `RIGHT` | Turn right |
| `STOP` | Halt — no safe path found |
| `CAUTION` | Scene too uncertain — halt |

---

## Path Scoring Formula

```
Score(P_i) = w_c·C_i + w_f·F_i + w_p·P_i - w_r·R_i - w_u·U_i

C_i = obstacle clearance
F_i = free-space availability
P_i = forward progress (penalized by curvature)
R_i = estimated collision risk
U_i = epistemic uncertainty
```

Default weights: `w_c=0.30, w_f=0.25, w_p=0.25, w_r=0.15, w_u=0.05`

---

## Test Scenarios

| ID | Name |
|----|------|
| A | Open corridor |
| B | Single static obstacle |
| C | Obstacle on left |
| D | Obstacle on right |
| E | Obstacle directly ahead |
| F | Narrow passage |
| G | Multiple obstacles |
| H | Moving pedestrian |
| I | Crossing pedestrian |
| J | Partial occlusion |
| K | Low-light scene |
| L | Strong illumination / glare |
| M | Crowded environment |
| N | Detection instability |
| O | Large unknown region |

Place video clips as `data/scenarios/scenario_A.mp4`, etc.

---

## Hardware Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (tested on RTX 4050)
- 16 GB RAM
- Webcam or video file

CPU-only mode works but will be slower than 15 FPS target.

---

## Known Limitations (Baseline)

See `ARCHITECTURE.md` for the complete list.

Short version:
1. Monocular depth is **relative only** — no metric distance
2. Free-space estimation uses heuristics — not ground-truth segmentation
3. Path planning is 2D grid-only — no 3D structure
4. No temporal prediction of dynamic obstacles
5. Tracker uses simple centroid matching — can swap in ByteTrack
6. No map memory across frames — purely reactive

---

## Research Note

This baseline is deliberately simple. Its failures are the research contribution.

After running the system, inspect:
- `failure_cases/` — automatic failure snapshots
- `logs/<session>/frames.jsonl` — full per-frame audit trail
- `logs/<session>/performance_report.md` — aggregate metrics

See `ARCHITECTURE.md` for the Baseline Failure Report.
