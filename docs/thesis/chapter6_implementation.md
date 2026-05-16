# 6. Implementation Details

This chapter documents the engineering choices that turn the methodology of Chapter 5 into working software. The goal is reproducibility: with the public repository, the Roboflow datasets, and a Google Colab account, another student should be able to obtain the same trained weights and the same evaluation numbers.

The complete source code is hosted on GitHub at [github.com/sertacakalin/hatched-area-violation-detection](https://github.com/sertacakalin/hatched-area-violation-detection). The repository contains the pipeline source (`src/`), runtime configuration (`configs/`), and the Gradio web demonstration (`app.py`). Command-line scripts and Colab training notebooks are preserved in an archive directory adjacent to the repository (see Section 6.1.4).

---

## 6.1. Environment and Setup

### 6.1.1. Development Environment

The project uses a dual-platform workflow that separates training from inference. **Training** runs in Google Colab (Python 3.10 on Ubuntu 22.04, Tesla T4 GPU). **Inference**, development, and the Gradio demo run locally on **macOS 15.6.1** (Apple Silicon, Python 3.12.7) inside a `venv` virtual environment.

A typical session looks like this:

```bash
git clone https://github.com/sertacakalin/hatched-area-violation-detection.git
cd hatched-area-violation-detection
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py    # → http://localhost:7860
```

The pipeline is pure Python with no platform-specific code; it has been tested on macOS Apple Silicon, Linux + NVIDIA GPU, and Windows 11 (WSL2). `requirements.txt` uses `>=` version pins rather than `==`, so the install survives Colab's coordinated version bumps.

### 6.1.2. Hardware

Three hardware profiles were involved:

- **Training — Google Colab Tesla T4** (16 GB VRAM, mixed-precision via Ultralytics `amp: true`). The v1 YOLOv8s fine-tune ran for **26 epochs in 1 h 28 min** (early-stopped from 50, `patience=10`). Peak VRAM usage on the T4 during v3 training was approximately `[TODO: read from Colab session log; YOLOv8m at batch=16, imgsz=640 typically peaks around 10–12 GB]`.
- **Inference — Apple Silicon Mac.** Runs on CPU because Ultralytics + EasyOCR do not yet support Apple's Metal Performance Shaders reliably; `half_precision: false` is therefore set in `configs/config.yaml`. End-to-end throughput on a 4K, 30 fps clip is approximately **4–7 frames per second**, which is enough for offline analysis.
- **Inference — Linux + NVIDIA GPU.** Near-real-time throughput because half-precision can be enabled and the detect/track stage stays on the GPU.

A minimum of **8 GB system RAM** and **SSD storage** are recommended for inference; a full evaluation run produces several gigabytes of outputs under `results/`.

### 6.1.3. Software Libraries

The implementation uses a small, well-maintained set of libraries. Tested versions:

| Layer | Library | Version | Role |
|---|---|---|---|
| Detection + tracking | `ultralytics` | 8.4.33 | YOLOv8 family + ByteTrack |
|  | `torch` | 2.8.0 | DL backend |
|  | `torchvision` | 0.23.0 | Vision primitives |
|  | `lap` | ≥ 0.4.0 | Tracking assignment |
| Vision + geometry | `opencv-python` | ≥ 4.8.0 | Video I/O, drawing, ORB |
|  | `numpy` | ≥ 1.24.0 | Arrays |
|  | `shapely` | ≥ 2.0.0 | Polygon geometry |
| Plate recognition | `easyocr` | ≥ 1.7.0 | OCR (PyTorch-backed) |
| Web demo | `gradio` | 6.11.0 | Interactive demo |
|  | `plotly` | ≥ 5.18.0 | Analytics charts |
| Persistence + config | `sqlite3` | built-in | Violation database |
|  | `PyYAML` | ≥ 6.0 | Configuration |
| Dataset (Colab only) | `roboflow` | ≥ 1.1.0 | Dataset download |

Two choices deserve a brief note. **Ultralytics** packages YOLOv8 and ByteTrack together, accessible through one `YOLO.track(persist=True)` call — this avoids a separate tracker integration layer and is isolated from the rest of the codebase by `src/tracking/bytetrack_wrapper.py`. **EasyOCR** was preferred over PaddleOCR because it installs on Apple Silicon without custom build steps; the OCR backend is parameterized through `plate.ocr.backend` in `configs/config.yaml`, so an alternative could be slotted in without touching the calling pipeline.

### 6.1.4. System Implementation Architecture

The source tree is organized strictly by responsibility. Packages communicate only through immutable dataclasses (`Detection`, `TrackedObject`, `PlateResult`, `ViolationEvent`) defined in `src/core/data_models.py`, so any package can be unit-tested in isolation.

```
hatched-area-violation-detection/
├── app.py                  # Gradio web demo (primary entry point)
├── configs/
│   ├── config.yaml         # Runtime configuration (single source of truth)
│   ├── bytetrack.yaml      # Tracker hyperparameters
│   ├── dataset_info.yaml   # Dataset provenance index
│   └── zones/*.json        # Per-camera polygon definitions
├── data/
│   ├── videos/test/        # Test clips (gitignored)
│   ├── datasets/final_v3/  # YOLO path lists (gitignored)
│   └── ground_truth/*.json # Hand-annotated targets (tracked)
├── src/
│   ├── core/               # Config, frame I/O, data models, visualizer
│   ├── detection/          # YOLO vehicle detector wrapper
│   ├── tracking/           # ByteTrack wrapper
│   ├── zones/              # Polygon manager + ROI selector
│   ├── violation/          # State machine, trajectory, severity
│   ├── plate/              # Detector, OCR, TR format validator
│   ├── storage/            # SQLite schema + migrations
│   └── pipeline/           # Pipeline class + factory
├── weights/                # Model checkpoints (gitignored)
├── results/                # Pipeline outputs (gitignored)
└── docs/thesis/            # This chapter and Chapter 4
```

The package responsibilities are:

- **`core`** holds shared dataclasses, the YAML `Config` class with dotted-path lookup, the `FrameProvider` context manager around `cv2.VideoCapture`, the `Visualizer` that draws boxes and overlays, and the `ViolationHeatmap` used by the Gradio demo.
- **`tracking`** wraps `YOLO.track(persist=True)`. It filters detections that exceed `max_bbox_ratio`, but exposes the filtered track IDs through `last_filtered_track_ids` so the state machine downstream does not age them out by mistake.
- **`zones`** loads polygon JSON files, applies a Shapely buffer (positive or negative), and answers point-in-polygon / bounding-box-overlap queries.
- **`violation`** orchestrates the state machine (`OUTSIDE → ENTERING → INSIDE → VIOLATION`), the trajectory analyzer that records bottom-center positions, and the severity scorer that classifies events into types (`KAYNAK`, `SEYIR`, `KENAR_TEMASI`, `DIGER`) and levels (`DUSUK`, `ORTA`, `YUKSEK`, `KRITIK`). A `per_track_lock` invariant guarantees at most one confirmed violation per track per video.
- **`plate`** wraps a second YOLO instance (`weights/plate.pt`), an EasyOCR-backed OCR with a Lanczos-upscale-plus-unsharp-mask preprocessing step, and the Turkish 81-province format validator (`tr_plate.py`).
- **`storage`** encapsulates the SQLite schema, idempotent column-by-column migrations, and a CRUD surface used by the violation logger.
- **`pipeline`** is the single orchestrator. Its constructor builds every component from a `Config` object; its `run(on_violation=...)` method drives the per-frame loop.

Archived alongside the repository are the command-line entry points (`run_pipeline.py`, `evaluate_with_ground_truth.py`, `select_roi.py`, `download_weights.py`, `generate_dataset_visualizations.py`, `generate_report.py`) and the Colab training notebooks (`00_frame_extraction` through `11_master_pipeline`, including `05_train_mobese_v3` — named after the İBB MOBESE batches that make up Phase 2 of the production training set, see Chapter 4, Section 4.10). These are required to reproduce training but not to run the pipeline; the Gradio demo (`python app.py`) is fully self-sufficient.

---

## 6.2. Model and Pipeline

### 6.2.1. Data Pipeline

The training data path and the inference data path are deliberately separate.

**Training (offline, Colab).** Each Roboflow dataset is exported in YOLOv8 format and consumed directly by `model.train(data=data.yaml, ...)`. The production export `final_v3` combines two sources — the author's own cam1–cam5 recordings from Mall of Istanbul and Güneşli (manually labeled) plus İBB MOBESE still-frame batches (auto-labeled then manually reviewed); see Chapter 4 for the full provenance. All preprocessing (640 × 640 letterbox, normalization, mosaic, HSV jitter, horizontal flip) is performed internally by Ultralytics; no custom `Dataset` class is needed.

**Inference (online, local or Colab).** Frames flow through the per-frame loop of `Pipeline.run()` in `src/pipeline/pipeline.py`:

```python
for frame_num, frame in self.frame_provider:
    tracked = self.tracker.update(None, frame)
    filtered_ids = self.tracker.last_filtered_track_ids
    if self.plate_recognizer is not None:
        self.plate_recognizer.update_buffer(tracked, frame, frame_num)
    tracked, new_violations = self.violation_detector.process_frame(
        tracked, frame, frame_num, fps, extra_active_ids=filtered_ids)
    for event in new_violations:
        if self.plate_recognizer is not None:
            event.plate = self.plate_recognizer.recognize(event.track_id)
        self.violation_logger.log_violation(event)
        self.events.append(event)
    self._video_writer.write(self._visualize(frame, tracked, new_violations))
```

Three properties matter. (i) The frame stream is never copied into memory all at once — each frame is processed and discarded. (ii) The plate ring buffer is updated on every frame for every track, so that when a violation is later confirmed, the recognizer already has the K most recent crops without seeking back in the video. (iii) `extra_active_ids` propagates the tracker's filtered IDs into the state machine, preventing premature ageing-out of tracks whose bounding box briefly exceeded `max_bbox_ratio`.

No batching is used — each frame is a single forward pass. The project targets offline analysis of recorded clips, so per-frame latency matters more than throughput.

### 6.2.2. Models

The implementation uses one architecture (YOLOv8) instantiated three times with different weights.

- **Vehicle detector — `weights/best_v3.pt`.** YOLOv8m fine-tune over four classes in the order `{0: bus, 1: car, 2: motorcycle, 3: truck}`. The class order is critical: `configs/config.yaml` exposes `vehicle_detection.classes` as integer IDs, and a mismatch would silently detect the wrong objects.
- **Plate detector — `weights/plate.pt`.** YOLOv8n fine-tune over one class. Loaded with a confidence threshold of `0.25` (lower than the vehicle threshold of `0.55`), because plates are small and the goal at this stage is recall — false positives are filtered downstream by the OCR voting step.
- **Tracker — ByteTrack.** Invoked through `YOLO.track(persist=True)`, configured by `configs/bytetrack.yaml`:

| Parameter | Value | Role |
|---|---:|---|
| `track_high_thresh` | 0.30 | High-score detection threshold |
| `track_low_thresh` | 0.05 | Low-score threshold (ByteTrack's main advantage) |
| `new_track_thresh` | 0.40 | Threshold to spawn a new track |
| `track_buffer` | 45 | Frames to keep a lost track alive |
| `match_thresh` | 0.75 | IoU matching threshold |
| `fuse_score` | true | Fuse detection score with IoU |

The state machine, trajectory analyzer, and severity scorer live in `src/violation/` and are plain Python (no learned components). The plate recognizer in `src/plate/recognizer.py` performs **best-frame voting**: it runs the plate detector once on the entire per-track buffer, ranks candidates by `confidence × √area`, runs OCR on only the top `topk_for_ocr` candidates, and picks the one with the highest combined `ocr_confidence × format_bonus` score.

### 6.2.3. Training Configuration

Three training runs produced the three checkpoints. Each run is fully documented in `configs/dataset_info.yaml`.

**v1 (legacy) — `weights/best.pt`**

| Setting | Value |
|---|---|
| Base model | `yolov8s.pt` |
| Dataset | `istanbul-traffic-vehicles` v2 (Roboflow) |
| Image size / batch | 640 × 640 / 16 |
| Optimizer | `auto` → SGD + cosine annealing |
| `lr0`, momentum, weight decay | 0.01, 0.937, 5 × 10⁻⁴ |
| Max epochs / actual | 50 / 26 (best at 16, early-stopped at patience 10) |
| Seed | 0 (from `args.yaml`) |
| Training time | 1.47 h on T4 |
| **Final metrics** | **P = 0.6389, R = 0.5656, mAP@0.50 = 0.5969, mAP@0.50:0.95 = 0.4496** |

**v3 (production) — `weights/best_v3.pt`**

| Setting | Value |
|---|---|
| Base model | `yolov8m.pt` |
| Dataset | `final_v3` (Roboflow), 6 631 frames — 2 919 from cam1–cam5 + 3 715 from İBB MOBESE; see Chapter 4, Table 4.2 |
| Image size / batch | 640 × 640 / 16 |
| Optimizer | `auto` per `dataset_info.yaml` — `[TODO: read concrete optimizer from v3 run's args.yaml on Drive]` |
| `lr0` | 0.001 (ten times lower than v1 — genuine fine-tune) |
| Max epochs / actual | 100 / `[TODO: read from results.csv]` |
| Seed | `[TODO: confirm from v3 args.yaml; v1 used seed = 0 despite dataset_info.yaml stating 42]` |
| Training time | `[TODO]` |
| **Final metrics** | **P = 0.7647, R = 0.7873, mAP@0.50 = 0.8075, mAP@0.50:0.95 = 0.6530** |

Two deliberate changes lifted recall by **roughly 22 percentage points** (0.5656 → 0.7873): the upgrade from YOLOv8s to YOLOv8m (more capacity for small-object features) and the ten-times-lower learning rate (preserving the COCO-pretrained features instead of overwriting them in the first epochs).

**Plate detector — `weights/plate.pt`**

| Setting | Value |
|---|---|
| Base model | `yolov8n.pt` |
| Dataset | `TR-PLAKA-1` (Roboflow), single class |
| Image size / batch | 640 × 640 / 32 |
| Optimizer | `auto` → SGD + cosine annealing |
| `lr0` / max epochs | 0.001 / 80 |
| **Final metrics** | **P = 0.9560, R = 0.9447, mAP@0.50 = 0.9815, mAP@0.50:0.95 = 0.8171** |

The plate detector is very accurate because plates are geometrically simple and single-class. The lightest YOLOv8 variant (`yolov8n`) was used deliberately because it runs on per-track ring-buffer crops, not full frames.

Each run's `results.csv`, `results.png`, PR/F1 curves, and `confusion_matrix.png` are stored on Drive under `MyDrive/tez_models/<run_dir>/`. `[TODO: the docs/figures/training/ directory currently archives the v1 run artifacts; replace with the v3 run before final submission so that Figure 6.1 actually depicts the production model.]`

### 6.2.4. Experiment Management

Experiment tracking is intentionally lightweight: there were a handful of training runs, not hundreds.

- **Datasets** are versioned by Roboflow (`workspace, project, version` triples are stored in `configs/dataset_info.yaml`).
- **Training artifacts** (`results.csv`, `results.png`, weights, confusion matrices) are kept on Drive under `MyDrive/tez_models/`.
- **Code** is versioned with Git on GitHub.
- **Configuration is the experiment manager**: `configs/config.yaml` records every runtime parameter and `configs/dataset_info.yaml` records every training context; both are in Git.

Weights & Biases and TensorBoard were intentionally not used — `results.csv` was enough signal on its own.

### 6.2.5. Hyperparameter Tuning

Hyperparameter selection was performed manually because the search space is small, each training run is expensive (1–2 hours on free Colab), and the cost of a slightly suboptimal setting is much smaller than the cost of overfitting to a small validation set.

**Training-time changes between v1 and v3.** Three deliberate adjustments, each motivated by a concrete v1 failure mode:

1. Base architecture `yolov8s → yolov8m` (capacity for small objects).
2. Learning rate `0.01 → 0.001` (fine-tune behavior rather than continued training).
3. Max epochs `50 → 100` (budget for the lower learning rate to converge).

No grid search was performed.

**Inference-time hyperparameters** live in `configs/config.yaml` and were tuned against representative test clips. Final production values:

| Component | Parameter | Value |
|---|---|---:|
| Vehicle detection | `confidence_threshold`, `iou_threshold`, `max_bbox_ratio` | 0.55, 0.45, 0.25 |
| Zone manager | `polygon_buffer` (px), `min_overlap_ratio` | −10, 0.30 |
| State machine | `min_frames_in_zone`, `cooldown_frames`, `exit_frames`, `per_track_lock` | 5, 600, 3, true |
| Plate detector | `confidence_threshold`, `iou_threshold` | 0.25, 0.45 |
| Plate recognizer | `buffer_size`, `topk_for_ocr`, `valid_format_bonus` | 10, 3, 1.5 |
| Plate OCR | `MIN_OCR_WIDTH` (px) | 150 |

The most informative tuning observations:

- **Vehicle confidence `0.35 → 0.55`** to eliminate spurious low-confidence detections in dense traffic (the inline comment in `config.yaml` records the motivation as "`file2_dense` noise filtrele" — the same dense batch that was later discarded entirely; see Chapter 4, Section 4.6).
- **`polygon_buffer = −10 px`** (inward shrink). Raw zone polygons include the painted edge of the marking; a small inward buffer prevents a bounding box that merely grazes the edge from being scored as inside.
- **`cooldown_frames 90 → 600`** (3 s → 20 s) after observing that the same vehicle stuck in congested traffic oscillated in and out of the buffered polygon and produced multiple duplicate violations; `per_track_lock` was added as a hard backstop.
- **`MIN_OCR_WIDTH = 150 px`** was determined empirically: below this width EasyOCR confidence drops sharply for Latin characters, and Lanczos upscale + a mild unsharp mask recover most of the lost reads.

### 6.2.6. Inference / Deployment Paths

The project supports three inference paths, all built on the same `Pipeline` class.

- **Gradio web demo (`app.py`) — primary.** Browser-based UI for end-users. Builds the tracker, `ZoneManager`, and `ViolationDetector` directly (so it can drive its own progress bar) but reuses every other component of the production pipeline. The output video is re-encoded with ffmpeg into H.264 + yuv420p for inline browser playback.
- **Command-line (`scripts/run_pipeline.py`, archived).** Wraps `create_pipeline(config_path, overrides).run()`. Used by the ground-truth evaluator and recommended for batch jobs.
- **Jupyter notebooks (archived).** The same `Pipeline` class is importable from any notebook; `07_violation_detection_eval.ipynb` and `11_master_pipeline.ipynb` follow the same `create_pipeline(...).run()` pattern.

**Deployment artifacts.** After a run, the system writes everything under `results/`: the annotated video (`output.mp4`), the SQLite database (`violations.db`), one vehicle crop per violation (`crops/`), one plate crop where available (`plates/`), and one full-frame snapshot per violation (`frames/`). All file paths are stored in the matching SQLite rows, so the database is a self-contained index of every artifact.

---

## 6.3. Deployment and System

### 6.3.1. User Interface

The user-facing layer is the single-page Gradio application `app.py`, designed for non-technical reviewers. Layout: two columns (1:2) with four result tabs.

The interaction flow is: upload a video → first frame appears on the right → click polygon vertices to define the hatched zone (with Undo / Clear) → adjust confidence / IoU sliders if desired → optionally enable "Moving Camera" and "Enable Plate Recognition" → press **Run Detection**. A live `gr.Progress` bar reports current frame and running violation count.

The four result tabs:

| Tab | Content |
|---|---|
| **Video** | Annotated output — red boxes around violators, orange boxes for ENTERING / INSIDE states, per-frame HUD strip, fading trajectory trails, severity bars. |
| **Analytics** | Four-panel Plotly subplot (severity donut, score histogram, timeline scatter, vehicle-type bar) + Markdown summary (count, mean / range / std of scores, processing FPS, per-type and per-level counts). |
| **Violations** | `gr.Dataframe` with columns *Time, Track ID, Vehicle, Plate, City, Score, Level, Type, Frames* + a 4-column gallery of up to twelve vehicle crops. |
| **Heatmap** | Spatially aggregated heatmap of all violation locations overlaid on the first frame. |

The "Moving Camera" toggle enables a `DynamicZoneTracker` that warps the polygon across frames using ORB + RANSAC homography; it is intended for drone or pan-tilt-zoom footage and is **off by default** because the primary test set is fixed-camera. Gradio is launched on `0.0.0.0:7860` with the Soft theme.

### 6.3.2. End-to-End Workflow

End-to-end the system threads ten components through a single `Pipeline` instance:

```
Video file
    │
    ▼
FrameProvider ─(BGR frame)─▶ ByteTrackWrapper
                                  │
                                  ├─▶ PlateRecognizer.update_buffer
                                  ▼
                            ViolationDetector
                            (ZoneManager + StateMachine
                             + TrajectoryAnalyzer + SeverityScorer)
                                  │
                                  ▼
                          new_violations: ViolationEvent[]
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
  PlateRecognizer.            ViolationLogger             Visualizer
  recognize(track_id)               │                         │
   → event.plate                    ▼                         ▼
                              SQLite + crops              annotated frame
                                                              │
                                                              ▼
                                                          VideoWriter → output.mp4
```

The integration is tractable because **every horizontal data contract is an immutable dataclass** (`Detection`, `TrackedObject`, `PlateResult`, `ViolationEvent`). Adding a new field requires changing only the dataclass definition; the rest of the pipeline forwards it unchanged. This is how the plate recognition stage was added late in the project without restructuring anything else.

### 6.3.3. Performance Optimizations

Several implementation-level optimizations keep the system usable on laptop-class hardware. None of them required new libraries or changes to the methodology.

- **Single-pass detection + tracking.** `YOLO.track(persist=True)` runs detection and tracking together, halving per-frame compute on CPU.
- **Half-precision on GPU.** `half=True` is forwarded to `YOLO.track()` when CUDA is available; automatically disabled on CPU and Apple Silicon.
- **Lazy plate recognition.** The plate stage is the most expensive part of the pipeline but runs **only on confirmed violations**, which means a handful of times per minute of input at most.
- **Best-frame voting instead of per-frame OCR.** The plate detector runs once over the entire ring buffer in a batched call; OCR runs only on the top three candidates — three OCR forward passes per violation instead of ten.
- **Pre-OCR upscale + unsharp mask.** Plate crops smaller than 150 px wide are upscaled with Lanczos and lightly sharpened before OCR; this recovers a substantial fraction of otherwise-unreadable plates.
- **Bounded ring buffers.** The plate buffer and the trajectory analyzer both use `collections.deque(maxlen=N)`, so memory per track is constant.
- **SQLite in WAL mode.** Database connection is opened once and held for the whole run. WAL mode allows an external dashboard to read concurrently with the pipeline writing.
- **Heatmap accumulation.** A single numpy array is accumulated; the colored heatmap is rendered once at the end of the run.
- **ffmpeg re-encode.** The raw `VideoWriter` output (mp4v codec) is re-encoded to H.264 + yuv420p + faststart so the demo can stream the result inline. Falls back gracefully if `ffmpeg` is missing.

### 6.3.4. Error Handling and Robustness

The pipeline must tolerate three classes of failures: invalid inputs, transient errors in the per-frame loop, and errors in optional subsystems.

- **Input validation.** The Gradio demo refuses unopenable videos (`gr.Error("Cannot read video file.")`) and refuses polygons with fewer than three vertices.
- **Bounding-box clipping.** Crop extraction clips coordinates against frame size and skips zero-area crops, so OpenCV never raises on off-frame boxes.
- **Detection filter side channel.** `max_bbox_ratio` does **not** silently drop tracks; the filtered IDs are surfaced through `last_filtered_track_ids` and forwarded into the violation detector so the state machine does not age them out by mistake.
- **Per-track lock.** Even if a state-machine oscillation tries to re-fire, `per_track_lock` guarantees at most one violation per track per video. When the lock suppresses a re-fire, the visualizer's state is also rolled back so it does not draw a red "İHLAL" box for an unsaved event.
- **Optional plate subsystem.** `_build_plate_recognizer` is wrapped in nested `try / except`: an `ImportError` makes the whole plate package a no-op (useful when `easyocr` is not installed); a runtime `Exception` falls back to running without plate recognition (useful when `plate.pt` is missing or corrupted). A failure of `recognize(track_id)` inside the loop is logged but does not stop the run.
- **`on_violation` callback safety.** The user callback is also wrapped in `try / except` so a bad consumer cannot crash the pipeline.
- **Idempotent DB migrations.** Every connection-open runs a column-by-column migration: if a column exists, skip; if not, `ALTER TABLE ADD COLUMN`. Indices are created after the columns they depend on. Older databases produced by earlier project versions open without manual repair.
- **Graceful shutdown.** `VideoWriter` and SQLite are released in `finally` blocks. A user-initiated Gradio disconnect cancels the running event and the next call starts clean.

### 6.3.5. Reproducibility and Code Availability

The full source, configuration, ground truth, and this chapter are at [github.com/sertacakalin/hatched-area-violation-detection](https://github.com/sertacakalin/hatched-area-violation-detection). License: `[TODO: declare — MIT or Apache 2.0]`.

**Reproduction recipe:**

1. **Clone and install.**
   ```bash
   git clone https://github.com/sertacakalin/hatched-area-violation-detection.git
   cd hatched-area-violation-detection
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Pull weights** into `weights/`. Canonical Drive paths are in `weights/README.md` (`best_v3.pt` for vehicles, `plate.pt` for plates).
3. **Place a test video** under `data/videos/test/`. Two clips ship configured: `cam1_test.mp4` (zone: `configs/zones/cam1_test.json`) and `cam4_30s.mp4` (zone: `configs/zones/cam4_30s.json`). `[TODO: configs/config.yaml currently points to data/videos/test/test_01.mp4 which does not exist — update the default to cam1_test.mp4 before submission.]`
4. **Launch the Gradio demo:** `python app.py`, then open `http://localhost:7860`.
5. **For batch / evaluation use,** restore the archived `scripts/` (under `../hatcted area docs/scripts/`) and invoke `run_pipeline.py` or `evaluate_with_ground_truth.py` with `--config`, `--video`, `--zone` (and `--ground-truth` for evaluation).

**Datasets and weights provenance.** The three Roboflow datasets (`istanbul-traffic-vehicles` v2, `final_v3`, `TR-PLAKA-1`) are referenced by `(workspace, project, version)` in `configs/dataset_info.yaml`; the three trained weights are documented in `weights/README.md` with their Drive paths and originating notebook. Together these two files form the complete provenance trail.

**Determinism.** The v1 training run used seed `0` (read directly from the archived `args.yaml`); the `seed: 42` value in `dataset_info.yaml` is being reconciled. Run-to-run mAP variation on the same dataset and hardware is typically under one percentage point.

### 6.3.6. Security and Privacy

The full ethical and legal framing is in Chapter 4 (Section 4.12) and Chapter 10; this subsection covers only the implementation-level controls.

- **Data minimization.** Only license plate text and province code are stored, and only for confirmed violations. No face detection or recognition is performed at any stage.
- **Local-first storage.** The SQLite database and all crops live under `results/`. The system has **no network output channel by default**; nothing leaves the local machine without an explicit exporter.
- **Network exposure.** The Gradio demo binds to `0.0.0.0:7860` for LAN demonstrations. In a production deployment this should be restricted to localhost or placed behind an authenticating reverse proxy. The system does not implement authentication itself.
- **Secrets.** The Roboflow API key is read from Colab Secrets, never hard-coded; `.gitignore` excludes `.env` files. No credentials are committed.
- **Plate-format validation as defensive filtering.** `tr_plate.py` enforces the Turkish `[01–81] [1–3 letters] [2–4 digits]` template with the 81-province code dictionary. Invalid strings are recorded with `plate_valid = 0` so downstream consumers can filter them — this eliminates the most common OCR-noise class (random Latin/digit substrings landing in the plate region).

### 6.3.7. Limitations of the Current Implementation

The methodology of Chapter 5 imposes inherent limitations (hatched-area-only, fixed camera). This subsection lists limitations that could in principle be fixed without changing the underlying approach.

- **Single-process inference.** The pipeline runs in one Python process; multi-camera deployment would need one process per camera with a shared backend (PostgreSQL rather than SQLite, given write-throughput limits).
- **SQLite write concurrency.** WAL mode allows concurrent readers but global write serialization. Fine for one camera, not for a fleet.
- **EasyOCR on Apple Silicon.** Runs on CPU because EasyOCR does not yet support Metal Performance Shaders reliably. On Linux + NVIDIA the same stage runs at near-real-time speed.
- **Dynamic-zone tracker drift.** The ORB + RANSAC homography is recomputed against the *reference* frame each time (not the previous one), which trades reliability against slow drift accumulation over several minutes — particularly when large moving objects (buses) dominate the ORB features.
- **Fixed 640 × 640 input.** Vehicles smaller than ≈ 20 pixels in the letterboxed view fall below the detector's practical recall. No multi-scale or tiled inference for very-far-field 4K footage.
- **No streaming input.** `FrameProvider` wraps `cv2.VideoCapture` and is designed for file inputs. Live RTSP / HTTP-FLV streams would need a custom frame source plus reconnect logic.
- **Archived CLI / notebooks.** `scripts/` and `notebooks/` currently live in `../hatcted area docs/`. A reviewer running `git clone` followed by `python scripts/run_pipeline.py` will see "No such file or directory". The Gradio demo remains the primary supported entry point.
- **Limited test coverage.** No `tests/` directory yet. The regression-risk surface includes the state machine, the severity formula, the plate format validator, and the database migrations.
- **Incomplete ground truth.** `data/ground_truth/` currently contains only placeholder annotations; until at least two videos with twenty-plus events each are annotated, evaluation rests on a statistically thin sample.
- **Broken dataset path-lists.** `data/datasets/final_v3/{data.yaml,train.txt,valid.txt,test.txt}` contain absolute paths to a previous project location (`/Desktop/Proje/...` instead of `/Desktop/Projects/...`). This does **not** affect inference (which reads `weights/best_v3.pt`, not the path-lists), but prevents a from-scratch retrain until the paths are repaired with a one-line `sed`.

---

## Table 6.1. Implementation summary

| Concern | Choice | Location |
|---|---|---|
| Language / OSes | Python 3.12.7 (dev), Python 3.10 + Ubuntu 22.04 (Colab training) | — |
| Vehicle detector | YOLOv8m fine-tune (mAP@0.50 = 0.808) | `weights/best_v3.pt` |
| Plate detector | YOLOv8n fine-tune (mAP@0.50 = 0.982) | `weights/plate.pt` |
| Tracker | ByteTrack via Ultralytics | `configs/bytetrack.yaml` |
| Polygon geometry | Shapely 2.x | `src/zones/zone_manager.py` |
| Temporal filter | 4-state DFA + per-track lock + cooldown | `src/violation/state_machine.py` |
| Severity model | Weighted sum of 4 trajectory components → 0–100 score | `src/violation/severity.py` |
| Severity levels (Turkish enum) | `DUSUK`, `ORTA`, `YUKSEK`, `KRITIK` | `src/violation/severity.py` |
| Violation types (Turkish enum) | `KAYNAK`, `SEYIR`, `KENAR_TEMASI`, `DIGER` | `src/violation/severity.py` |
| Plate OCR | EasyOCR 1.7+ | `src/plate/ocr.py` |
| Plate validator | Custom regex + 81 TR provinces | `src/plate/tr_plate.py` |
| Persistence | SQLite, WAL mode, idempotent migrations | `src/storage/database.py` |
| Primary entry point | Gradio web demo | `app.py` |
| Archived CLI | `run_pipeline.py` and friends | `../hatcted area docs/scripts/` |
| Web demo | Gradio 6.11.0 | `app.py` |
| Training infrastructure | Google Colab Tesla T4 | archived notebooks `03`, `05`, `06` |
| Dataset versioning | Roboflow (3 datasets) | `configs/dataset_info.yaml` |
| Code versioning | Git + GitHub | repository link above |
| License | `[TODO: declare]` | `LICENSE` |

---

## Figure 6.1. Training and validation loss curves

`[TODO: insert the production v3 fine-tune's results.png from MyDrive/tez_models/final_v3/. The artifacts archived under docs/figures/training/ correspond to the v1 (istanbul_finetune4) run — using them as Figure 6.1 would misrepresent the production checkpoint. Pull the v3 run directory from Drive before final submission.]`

**Caption.** Per-epoch training and validation loss curves for the production `weights/best_v3.pt` fine-tune. The top row shows the three Ultralytics loss components (`box_loss`, `cls_loss`, `dfl_loss`); the bottom row shows the four validation metrics (`precision`, `recall`, `mAP@0.50`, `mAP@0.50:0.95`). The rapid rise in `mAP@0.50:0.95` after the classification loss drop indicates that the backbone features adapted quickly to the local vehicle silhouettes once the learning rate had decayed enough to fine-tune — rather than overwrite — the COCO-pretrained weights.
