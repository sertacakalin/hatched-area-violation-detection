# Chapter 5 — Missing Figures & Tables (fill-in)

> **PLACEMENT NOTE.** This file supplies the four figures and two tables that are
> currently empty placeholders in Chapter 5 of the thesis `.docx`. I did **not** touch
> the `.docx`. Insert each figure/table at its existing placeholder, keep the existing
> captions, and use the caption text below (11 pt TNR, centered, below each figure).
> All figure PNGs are in `docs/thesis/figures/ch5/`.

---

## Figure 5.1 — System Architecture Block Diagram

> Replace the placeholder `[FIGURE 5.1 TO BE PLACED HERE - System Architecture Block
> Diagram, centered]`.

**File:** `figures/ch5/fig5_1_architecture.png`
*Figure 5.1. End-to-end block architecture of the proposed system, showing the four
loosely-coupled components — data input/preprocessing, the AI processing unit, the
spatial-temporal logic unit, and output/storage — communicating through the shared
dataclasses Detection, TrackedObject, and ViolationEvent.*

## Figure 5.2 — YOLOv8m Backbone + Neck + Head

> Replace `[FIGURE 5.2 TO BE PLACED HERE — YOLOv8m Backbone + Neck + Head Architecture]`.

**File:** `figures/ch5/fig5_2_yolo.png`
*Figure 5.2. High-level YOLOv8m architecture: a CSPDarknet backbone for feature
extraction, a PAN-FPN neck for multi-scale fusion, and an anchor-free decoupled
detection head predicting the four vehicle classes.*

## Figure 5.3 — Vehicle State Machine Transition Diagram

> Replace `[FIGURE 5.3 TO BE PLACED HERE — Vehicle State Machine State Transition Diagram]`.

**File:** `figures/ch5/fig5_3_state_machine.png` *(same diagram as Figure 9.7)*
*Figure 5.3. State-transition diagram of the temporal Vehicle State Machine
(OUTSIDE → ENTERING → INSIDE → VIOLATION), with a cooldown started on confirmation
and a return to OUTSIDE when the vehicle leaves the zone.*

## Figure 5.4 — End-to-End Workflow Diagram

> Replace `[FIGURE 5.4 TO BE PLACED HERE — End-to-End Workflow Diagram]`.

**File:** `figures/ch5/fig5_4_workflow.png`
*Figure 5.4. End-to-end per-frame workflow: detection → tracking → zone check →
state-machine update → (on confirmation) severity scoring and optional plate OCR →
annotated video, SQLite log, and cropped evidence.*

---

## Table 5.2 — Comparison of candidate vehicle detection models

> Replace the empty `Table 5.2. Comparison of candidate vehicle detection models.`

| Model | Type | Strengths | Weaknesses | Decision |
|---|---|---|---|---|
| Faster R-CNN [1] | Two-stage | High accuracy in crowded scenes | Heavy and slow; impractical for long videos on a single Colab GPU | Rejected |
| YOLOv8s [3] | Single-stage (small) | Fast and light; trains comfortably on a single T4 | Insufficient recall on small objects (motorcycles, distant trucks) | Baseline only (v1) |
| **YOLOv8m [3]** | Single-stage (medium) | Better small-object recall; strong speed/accuracy balance; one-line tracking via `model.track()` | Slightly slower than YOLOv8s | **Selected (production detector)** |
| YOLO-NAS [4] | Single-stage (NAS) | State-of-the-art accuracy | Heavier integration; less mature Ultralytics tooling | Not adopted |

## Table 5.3 — Comparison of candidate tracking algorithms

> Replace the empty `Table 5.3. Comparison of candidate tracking algorithms.`

| Algorithm | Approach | Strengths | Weaknesses | Decision |
|---|---|---|---|---|
| SORT [5] | Kalman filter + Hungarian assignment | Very fast and simple | Brittle under occlusion; frequent ID switches | Rejected |
| DeepSORT [6] | SORT + learned appearance embedding | Better identity under occlusion | Extra CNN forward pass per frame → added compute | Rejected |
| **ByteTrack [7]** | Associates every detection box (high- and low-confidence) | Robust through occlusion **without** a re-ID model; low overhead | Motion/IoU only; long occlusions cannot be re-associated | **Selected** |

---

### Note on a duplicate caption
The template also contains a second "**Figure 5.1. Training and validation losses
curves.**" later in the chapter. That is a different figure (the training curves,
= `by_chapter/ch7_01_training_curves.png`). To avoid two figures numbered 5.1,
renumber that one (e.g., to Figure 5.5) when you place these.
