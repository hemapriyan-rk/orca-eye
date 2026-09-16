# ORCA EYE — Architecture & Technical Reference

> **RESEARCH PROTOTYPE — NOT FOR REAL-WORLD MOBILITY USE**

---

## System Data-Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         ORCA EYE                            │
│           Baseline Monocular Navigation System              │
└─────────────────────────────────────────────────────────────┘

          ┌─────────────┐
          │ Camera /    │  BGR frame (np.ndarray H×W×3)
          │ Video File  │  + timestamp (float)
          └──────┬──────┘
                 │ Stage 1 — CameraSource
                 ▼
          ┌─────────────────────────────────────┐
          │        YOLO Object Detection         │
          │        YOLOv8n (ultralytics)         │
          │  → List[ObjectState]                 │
          │  → DetectionResult (metadata)        │
          └────────────────┬────────────────────┘
                           │ Stage 2
                 ┌─────────┴──────────┐
                 │                    │
                 ▼                    ▼
    ┌────────────────────┐   ┌────────────────────┐
    │  Centroid Tracker  │   │  MiDaS Depth Est.  │
    │  IoU + distance    │   │  (relative only)   │
    │  → List[Tracked    │   │  → DepthResult     │
    │    Object]         │   │    depth_map[H,W]  │
    └────────┬───────────┘   └────────┬───────────┘
    Stage 3  │                        │ Stage 4a
             │               ┌────────┘
             │               ▼
             │   ┌───────────────────────────┐
             │   │  Free-Space Estimator     │
             │   │  Hybrid Option C:         │
             │   │    geometry + depth +     │
             │   │    YOLO obstacle boxes    │
             │   │  → FreeSpaceResult        │
             │   │    label_map: FREE/OBS/UNK│
             │   └────────────┬──────────────┘
             │                │ Stage 4b
             └────────────────┤
                              │
                              ▼
                ┌─────────────────────────┐
                │       Spatial Map        │
                │  Grid (12×20 default)   │
                │  Per cell:              │
                │   occupancy, free_prob  │
                │   depth_mean, semantic  │
                │   uncertainty           │
                └────────────┬────────────┘
                             │ Stage 5
                             ▼
                ┌─────────────────────────┐
                │   Candidate Path Gen    │
                │  Arc trajectories per  │
                │  angle: L/SL/S/SR/R    │
                │  + STOP                │
                │  → List[PathCandidate] │
                └────────────┬────────────┘
                             │ Stage 6
                             ▼
                ┌─────────────────────────┐
                │      Path Scorer        │
                │  Score = w_c·C + w_f·F │
                │        + w_p·P          │
                │        - w_r·R - w_u·U  │
                │  → sorted candidates   │
                └────────────┬────────────┘
                             │ Stage 7
                             ▼
                ┌─────────────────────────┐
                │    Decision Maker       │
                │  Safety checks:        │
                │   score < min → STOP   │
                │   clearance < min→STOP │
                │   uncertainty↑→CAUTION │
                │   hysteresis filter    │
                │  → NavigationDecision  │
                └────────────┬────────────┘
                             │ Stage 8
                    ┌────────┴──────────┐
                    │                   │
                    ▼                   ▼
          ┌──────────────────┐  ┌──────────────────┐
          │   Renderer       │  │  System Logger   │
          │  4-panel OpenCV  │  │  JSONL + CSV     │
          │  display         │  │  per frame       │
          └──────────────────┘  └──────────────────┘
          Stage 9                Stage 10
                    │
                    ▼
          ┌──────────────────┐
          │ Failure Detector │
          │ 12 conditions    │
          │ JPEG + JSON save │
          └──────────────────┘
          Stage 11
```

---

## Module Reference

### `perception/camera.py` — `CameraSource`

| Item | Detail |
|------|--------|
| Input | `source` (int or str), config dict |
| Output | `(success, frame, timestamp)` per `read()` |
| Key feature | Webcam reconnect loop, file EOF detection |
| Config section | `camera` |

### `perception/detector.py` — `YOLODetector`

| Item | Detail |
|------|--------|
| Input | BGR frame |
| Output | `DetectionResult` containing `List[ObjectState]` |
| Model | YOLOv8n (configurable) |
| Key feature | `ObjectState` is the canonical cross-module data contract. No other module imports `ultralytics` |
| Config section | `detection` |

**ObjectState fields:**
```python
class_id      : int
class_name    : str
confidence    : float
bbox          : [x1, y1, x2, y2]
center        : [cx, cy]
area          : float
track_id      : Optional[int]   # filled by tracker
```

### `perception/tracker.py` — `CentroidTracker`

| Item | Detail |
|------|--------|
| Input | `List[ObjectState]`, `frame_id` |
| Output | `List[TrackedObject]` |
| Algorithm | Centroid + IoU matching, greedy assignment |
| Velocity | Exponential moving average (configurable alpha) |
| Config section | `tracking` |

**TrackedObject fields:**
```python
track_id      : int
class_name    : str
confidence    : float
bbox          : [x1, y1, x2, y2]
center        : [float, float]
velocity      : [vx, vy]       # EMA-smoothed, pixels/frame
age           : int
last_seen     : int            # frame_id
is_stable     : bool           # age >= min_track_age
disappeared   : int            # consecutive unseen frames
center_history: List[...]      # last 50 positions
```

### `perception/depth.py` — `DepthEstimator`

| Item | Detail |
|------|--------|
| Input | BGR frame |
| Output | `DepthResult(depth_map, statistics, is_valid)` |
| Model | MiDaS v2.1 small (default) via `torch.hub` |
| Output convention | 0.0 = nearest, 1.0 = farthest (normalized) |
| **Critical** | Depth is **RELATIVE ONLY** — not metric. `is_relative=True` is set in statistics |
| Config section | `depth` |

### `perception/freespace.py` — `FreeSpaceEstimator`

| Item | Detail |
|------|--------|
| Input | frame, `DepthResult`, `List[ObjectState]` |
| Output | `FreeSpaceResult(label_map, free_prob_map, uncertainty, statistics)` |
| Method | Hybrid Option C: geometric prior + depth + YOLO boxes |
| Labels | FREE=0, OBSTACLE=1, UNKNOWN=2 |
| Config section | `freespace` |

**Steps:**
1. Geometric prior (bottom region = likely walkable)
2. Depth evidence (far = free, near = obstacle, high gradient = obstacle)
3. YOLO bbox masking (inflated boxes → OBSTACLE)
4. Threshold free-prob → label assignment

---

### `navigation/spatial_map.py` — `SpatialMap`

| Item | Detail |
|------|--------|
| Input | `FreeSpaceResult`, `DepthResult`, `List[TrackedObject]` |
| Output | Grid `(rows × cols)` of `GridCell` |
| Grid | Row 0 = far (top of image), Row N-1 = near (user's feet) |
| Temporal | Occupancy decays by factor per frame (configurable) |
| Config section | `spatial_map` |

**GridCell fields:**
```python
occupancy      : float [0,1]  — blocked probability
free_prob      : float [0,1]  — traversability probability
depth_mean     : float        — mean normalized depth
semantic_label : int          — dominant COCO class in cell
uncertainty    : float [0,1]  — epistemic uncertainty
```

### `navigation/path_generator.py` — `PathGenerator`

| Item | Detail |
|------|--------|
| Input | `SpatialMap` |
| Output | `List[PathCandidate]` |
| Candidates | Arc trajectories at configurable angle offsets |
| STOP | Always included as safe fallback |
| Config section | `path_generation` |

**PathCandidate fields:**
```python
direction    : str     — "LEFT", "SLIGHT_LEFT", "STRAIGHT", etc.
angle_deg    : float
points       : [(row, col), ...]  — grid cells along arc
clearance    : float   — min free_prob along path
width        : float   — lateral walkable width in grid cells
progress     : float   — fraction of planned depth reached
curvature    : float   — |angle| / 45.0
uncertainty  : float   — mean uncertainty along path
risk         : float   — mean occupancy along path
score        : float   — filled by PathScorer
```

### `navigation/path_scorer.py` — `PathScorer`

| Item | Detail |
|------|--------|
| Input | `List[PathCandidate]` |
| Output | Same list with `.score` populated, sorted descending |
| Formula | `w_c·C + w_f·F + w_p·P - w_r·R - w_u·U` |
| STOP | Fixed `stop_base_score` (config) |
| Config section | `scoring` |

### `navigation/decision.py` — `DecisionMaker`

| Item | Detail |
|------|--------|
| Input | Scored `List[PathCandidate]` |
| Output | `NavigationDecision` |
| Safety 1 | If `score < min_go_score` → STOP |
| Safety 2 | If `clearance < min_clearance` → STOP |
| Safety 3 | If `uncertainty > max_uncertainty` → CAUTION |
| Hysteresis | Requires `direction_hysteresis` margin to switch |
| Metrics | `N_switch`, `T_stable` tracked per session |
| Config section | `safety` |

---

### `visualization/renderer.py` — `Renderer`

4-panel composite OpenCV window:

| Panel | Content |
|-------|---------|
| Top-left | Camera feed + YOLO bboxes + track IDs + velocity arrows |
| Top-right | Free-space overlay (green=FREE, red=OBS, yellow=UNK) |
| Bottom-left | Navigation grid + candidate arc paths + selected path |
| Bottom-right | Command (large text) + score bar + all metrics |

Disclaimer banner: **"RESEARCH PROTOTYPE — NOT FOR REAL-WORLD MOBILITY USE"**

---

### `logging/logger.py` — `SystemLogger`

| Item | Detail |
|------|--------|
| Format | JSONL per frame + CSV time-series |
| Threading | Background writer thread with queue (non-blocking) |
| Contents | Full pipeline state — every number that was computed |
| Purpose | Enables post-hoc "why did it choose that path?" analysis |
| Config section | `logging` |

---

### `failure_analysis/detector.py` — `FailureDetector`

12 failure conditions checked every frame:

| # | Code | Trigger |
|---|------|---------|
| 1 | `NO_VALID_PATH` | STOP issued due to no safe path |
| 2 | `RAPID_PATH_CHANGE` | Direction changed in < `rapid_change_window` frames |
| 3 | `LOW_CLEARANCE` | Clearance < threshold while moving |
| 4 | `LOW_DETECTION_CONFIDENCE` | Mean conf below threshold |
| 5 | `FREESPACE_UNSTABLE` | Uncertainty > `high_uncertainty_threshold` |
| 6 | `DEPTH_INCONSISTENCY` | Frame-to-frame depth mean jump > threshold |
| 7 | `SUDDEN_APPEARANCE` | New track with age=1 |
| 8 | `SUDDEN_DISAPPEARANCE` | Previously tracked ID dropped |
| 9 | `DIRECTION_OSCILLATION` | Stability window changes >= max_changes |
| 10 | `LR_OSCILLATION` | Left/right alternation in last 6 frames |
| 11 | `UNKNOWN_DOMINANCE` | Unknown fraction > threshold |
| 12 | `PERCEPTION_PLANNER_DISAGREEMENT` | STOP despite high free fraction |

Saves: `failure_cases/frame_NNNNNN_SEVERITY.jpg` + `.json`

---

## Baseline Metrics

### Perception
- **FPS** (mean, min, max)
- **Latency** (mean, P95)
- **Detection count** (mean, max)
- **Mean confidence**

### Free-Space
- **Uncertainty** (mean, P95)
- **Free fraction** (mean)

### Navigation
- **Command distribution** (% of frames each command was issued)
- **STOP rate** — proportion of frames where no safe path exists
- **CAUTION rate**
- **Mean/min score**
- **Mean/min clearance**

### Stability
- **N_switch** — total direction changes per session
- **N_switch_window** — changes per `stability_window` frames
- **T_stable** — mean duration of a stable direction (seconds)
- **Oscillation events** — frames where `is_oscillating=True`

---

## BASELINE FAILURE REPORT

> This section will be populated after experimental runs.
> The following are **hypothesized** failure modes based on architectural analysis.
> Each must be verified by running the system and inspecting `failure_cases/`.

---

### Failure 1: Monocular Depth Scale Ambiguity

**Failure**
Incorrect obstacle distance estimation for objects at unusual scales.

**Reproduction condition**
Large object (e.g. truck) far away; small object (e.g. child) nearby.
MiDaS assigns similar relative depth values.

**Observed behavior**
Free-space estimator marks far truck as near-obstacle OR near child as walkable.

**Why current architecture fails**
MiDaS produces only ordinal depth — it cannot distinguish a 2m near object
from a 10m far object of similar apparent size.

**Responsible component**
`perception/depth.py` → `perception/freespace.py`

**Missing information**
Metric depth (stereo, ToF, or scale-calibrated monocular)

**Potential research question**
Can a monocular system recover metric-useful depth without hardware changes
using object-size priors from YOLO class detections?

---

### Failure 2: Free-Space/Obstacle Label Conflict at Object Boundaries

**Failure**
YOLO bbox includes walkable ground below the object.

**Reproduction condition**
Tall object (person, pole) with significant ground visible beneath bbox.

**Observed behavior**
Ground below object marked OBSTACLE. System avoids walkable region.
Unnecessary STOP or detour.

**Why current architecture fails**
Obstacle inflation + full-bbox masking does not distinguish the object
from the ground beneath it.

**Responsible component**
`perception/freespace.py` — obstacle masking step

**Missing information**
Instance-level segmentation mask (instead of bbox)

**Potential research question**
Can YOLO bounding box geometry + depth gradients provide sufficient
ground/object separation without full segmentation?

---

### Failure 3: Temporal Depth Inconsistency Under Camera Motion

**Failure**
Depth map changes dramatically between frames during camera pan or shake.

**Reproduction condition**
Fast head movement (wearable scenario) or hand-held camera shake.

**Observed behavior**
`DEPTH_INCONSISTENCY` failure flagged repeatedly.
Free-space map flickers between FREE and OBSTACLE.
Navigation oscillates.

**Why current architecture fails**
MiDaS is per-frame — no temporal coherence.
No optical flow or ego-motion compensation.

**Responsible component**
`perception/depth.py`

**Missing information**
Ego-motion estimate (IMU, optical flow) for depth stabilization

**Potential research question**
Does temporal smoothing of depth maps (EMA, Kalman on depth) reduce
navigation oscillation without introducing dangerous lag?

---

### Failure 4: Decision Oscillation in Symmetric Scenes

**Failure**
System alternates LEFT/RIGHT in scenes with symmetric obstacles.

**Reproduction condition**
Narrow corridor with equal obstacles on both sides.
Left and right path scores nearly identical.

**Observed behavior**
`LR_OSCILLATION` flagged. Navigation command alternates every 2–3 frames.

**Why current architecture fails**
Hysteresis margin is insufficient when left/right scores differ by < `direction_hysteresis`.
No memory of preferred direction from previous frames.

**Responsible component**
`navigation/decision.py` — hysteresis filter

**Missing information**
Goal-directed bias (preferred direction) or longer temporal memory

**Potential research question**
How much temporal memory (frame window) is needed to produce stable navigation
decisions in symmetric scenes?

---

### Failure 5: Unknown Region Dominates in Glare or Low-Light

**Failure**
Large fraction of frame classified UNKNOWN — system issues CAUTION or STOP
even when environment is physically safe.

**Reproduction condition**
Strong backlight, bright window, low-light indoor scene.

**Observed behavior**
`UNKNOWN_DOMINANCE` flagged. System conservatively stops.
False-positive STOP rate increases significantly.

**Why current architecture fails**
Free-space geometric prior cannot compensate for image-level degradation.
Depth estimation quality degrades drastically in low-light/glare.

**Responsible component**
`perception/depth.py` + `perception/freespace.py`

**Missing information**
Image quality assessment + adaptive confidence weighting

**Potential research question**
Can a lightweight image quality metric (sharpness, exposure) be used
to adaptively weight perception confidence and reduce false-positive STOPs?

---

### Failure 6: Sudden Object Appearance (Out-of-FOV Entry)

**Failure**
Object enters the frame abruptly from the side — tracker assigns age=1.
System may not react fast enough to prevent close approach.

**Reproduction condition**
Crossing pedestrian entering from outside FOV (Scenario I).

**Observed behavior**
`SUDDEN_APPEARANCE` flagged. First frame of appearance has only one data point.
Velocity estimate is [0, 0] — direction of crossing unknown.

**Why current architecture fails**
Centroid tracker provides no prediction before first observation.
No anticipatory planning.

**Responsible component**
`perception/tracker.py` — no future-state prediction

**Missing information**
Predictive motion model (Kalman prediction step beyond observed frames)

**Potential research question**
Does adding a Kalman prediction step (projecting object position one frame ahead)
reduce collision risk for suddenly-appearing dynamic obstacles?

---

### Failure 7: Crowded Scene Causes Navigation Paralysis

**Failure**
Multiple people fill all candidate path corridors → STOP issued.
System is "frozen" even though a slow, careful path might be feasible.

**Reproduction condition**
Scenario M — crowded environment.

**Observed behavior**
Repeated STOP. High STOP rate in crowded environments.
System cannot distinguish "momentarily blocked" from "permanently blocked".

**Why current architecture fails**
Path planning is purely reactive — no concept of "wait" vs "stop permanently".
No temporal prediction of whether obstacles will clear.

**Responsible component**
`navigation/decision.py` — no "WAIT" command

**Missing information**
Short-horizon prediction of whether the scene will open up

**Potential research question**
Should a "WAIT" command be added, and how should the system determine
when to wait vs. when to issue STOP?

---

### Failure 8: 2D Grid Cannot Represent Vertical Obstacles

**Failure**
Overhead obstacle (low beam, shelf) or ground-level obstacle (step, curb)
not distinguished from mid-height obstacles.

**Reproduction condition**
Low hanging sign. Floor-level step. Raised platform edge.

**Observed behavior**
Grid treats all obstacles as equivalent — cannot differentiate
something the user must duck under from something they must avoid laterally.

**Why current architecture fails**
Spatial map is a 2D top-down projection — vertical information is lost.

**Responsible component**
`navigation/spatial_map.py`

**Missing information**
3D spatial representation (voxel grid or height-map)

**Potential research question**
Is a 2.5D representation (height-map per cell) sufficient to catch the
most dangerous vertical hazards (low beams, curbs)?

---

### Failure 9: Path Scoring Ignores Dynamic Obstacle Trajectories

**Failure**
A path that is currently clear will be blocked by a moving obstacle
within 1–2 seconds — but the system chooses it as the best path.

**Reproduction condition**
Scenario H — moving pedestrian on a collision course.

**Observed behavior**
System selects STRAIGHT because current frame shows clear path.
Next frame: pedestrian has moved to block path.
Rapid command change flagged.

**Why current architecture fails**
Scoring uses instantaneous frame state — no prediction of obstacle movement.
Velocity is tracked but not used in path scoring.

**Responsible component**
`navigation/path_scorer.py` — velocity not incorporated into risk term

**Missing information**
Predicted future position of tracked objects (velocity × time horizon)

**Potential research question**
Can incorporating tracked-object velocity into the risk term R_i
reduce the rate of "wrong at the next frame" path selections?

---

### Failure 10: Perception/Planner Disagreement in Partial Occlusion

**Failure**
Free-space map shows clear path. YOLO detects no obstacle.
But an obstacle is partially hidden behind another object.

**Reproduction condition**
Scenario J — partial occlusion. Obstacle B hidden behind obstacle A.

**Observed behavior**
`PERCEPTION_PLANNER_DISAGREEMENT` NOT flagged (both agree: path is clear).
System recommends STRAIGHT into the hidden obstacle.

**Why current architecture fails**
Neither YOLO nor MiDaS has any model of occlusion.
Free-space estimation cannot infer what is behind a visible object.

**Responsible component**
Both `perception/detector.py` and `perception/freespace.py`

**Missing information**
Occlusion reasoning — marking the region immediately behind known obstacles
as UNKNOWN rather than FREE

**Potential research question**
Can projecting a simple "shadow" region behind each detected obstacle
(based on bbox depth and position) reduce dangerous path selections
in partial-occlusion scenarios?

---

## Research Agenda (Post-Baseline)

After running all 15 scenarios and collecting failure data:

1. Rank failures by frequency and severity in `failure_cases/`
2. Identify which failure mode causes the highest navigation risk
3. Target that specific failure for a focused research contribution
4. Design a minimal intervention (single module change) that measurably
   reduces that failure mode
5. Compare baseline vs. modified system using the same scenarios and metrics

**Do not claim novelty before step 4.**
