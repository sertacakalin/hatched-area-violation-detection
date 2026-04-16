# Hatched Area Violation Detection in Istanbul Traffic Using Deep Learning

Sertaç Akalın
Department of Computer Engineering, Istanbul Arel University, Istanbul (220303053)

## Abstract

This project proposes an end-to-end computer vision system for detecting vehicles that enter hatched road markings in Istanbul traffic. The system combines YOLOv8s for vehicle detection, ByteTrack for multi-object tracking, Shapely-based polygon analysis for zone-aware reasoning, and a temporal state machine for robust violation confirmation. The user manually defines the hatched-area boundary on the first frame of the video using a polygon selector; the system then monitors that region across the remaining frames. A four-state temporal model (`OUTSIDE -> ENTERING -> INSIDE -> VIOLATION`) requires sustained presence inside the zone before a violation is confirmed, which reduces false positives caused by bounding-box jitter and brief edge contact. The current evaluation scope compares a pretrained YOLOv8s baseline against a fine-tuned YOLOv8s model on a custom Istanbul traffic dataset collected from five daytime camera angles. The system outputs an annotated video, a SQLite violation log, cropped evidence images, severity scores, and violation types through both a command-line pipeline and a Gradio demo.

## 1. Introduction

Traffic safety remains a critical issue in Istanbul, where dense traffic and frequent last-second lane changes increase the risk of collisions around merge points, exit ramps, and lane splits. Hatched road markings are intended to keep these buffer zones clear, but detecting violations in such areas is harder than detecting single-moment events such as red-light or speed violations. A hatched-area violation unfolds over time: a vehicle enters the zone, remains inside it for several frames, and then exits.

This project addresses that problem with an offline computer vision pipeline for fixed-camera traffic footage. Vehicles are detected with YOLOv8s, tracked with ByteTrack, checked against a manually defined polygon zone, and passed through a temporal decision layer that confirms only sustained intrusions. Confirmed events are recorded to SQLite together with visual evidence. The focus of the project is limited to hatched-area violation detection; licence-plate recognition, face recognition, live deployment, and broader surveillance functionality are not part of the project.

## 2. Related Work

The proposed system sits at the intersection of three research areas: vehicle detection, multi-object tracking, and video-based traffic violation analysis.

YOLO-family detectors are widely used in traffic monitoring because they balance accuracy and inference speed well. YOLOv8s is selected here because it is open source, practical to fine-tune on a modest custom dataset, and mature enough to support reproducible experiments.

For tracking, ByteTrack is chosen because it preserves track continuity under moderate occlusion without requiring an extra appearance model. Compared with heavier alternatives, it offers a stronger robustness-to-complexity balance for a single-camera academic system.

In the traffic-monitoring literature, many zone-based systems still rely on a simple inside/outside decision. That approach is brittle for hatched areas because slight detector jitter at the border can create false positives. The gap this project targets is therefore not licence-plate reading or dashboarding, but a combination of:

- polygon-based hatched-area modelling,
- temporal filtering for robust violation confirmation,
- and local adaptation on Istanbul-specific traffic footage.

## 3. Project Summary

This project develops a hatched-area violation detection system for Istanbul traffic. The user draws the hatched-area boundary once on the first frame of a video. The system then:

- detects vehicles with YOLOv8s,
- tracks them with ByteTrack,
- checks whether tracked vehicles enter the polygon zone,
- confirms only sustained intrusions with a four-state temporal state machine,
- computes trajectory-based severity information,
- and logs confirmed violations.

The system scope is deliberately narrow. It evaluates:

- YOLOv8s pretrained vs fine-tuned,
- ByteTrack as the only tracker,
- daylight and fair-weather footage,
- five camera angles collected in Istanbul.

The annotated dataset currently contains 2,628 annotated frames, corresponding to roughly 2,500+ labelled vehicle instances across the four target classes: car, truck, bus, and motorcycle.

## 4. Project Details

### 4.1 System Architecture

The system follows this pipeline:

`Video -> YOLOv8s -> ByteTrack -> Polygon Zone Check -> Temporal State Machine -> Severity/Trajectory Analysis -> SQLite + Annotated Output + Gradio Demo`

The main modules are:

- Video input and frame reading with OpenCV
- Vehicle detection with YOLOv8s
- Multi-object tracking with ByteTrack
- Polygon zone control with Shapely
- Temporal violation confirmation with a four-state state machine
- Severity scoring and trajectory analysis
- SQLite logging plus annotated output generation
- Gradio interface for interactive demonstration

### 4.2 Methodological Focus

The key methodological idea is temporal robustness. A single-frame overlap between a vehicle box and the hatched polygon is not enough to count as a violation. The system therefore requires a minimum number of consecutive in-zone frames before raising an event. This allows the method to separate genuine intrusions from edge contact and detector jitter.

### 4.3 Outputs

The system produces:

- annotated output video,
- SQLite violation log,
- cropped vehicle and frame evidence,
- severity score,
- violation type.

### 4.4 Scope Exclusions

The following are outside the project scope:

- licence-plate recognition,
- face recognition,
- automatic hatched-area extraction,
- homography-based speed estimation,
- dynamic zone tracking,
- alternative trackers such as DeepSORT or BoT-SORT in the evaluated system,
- Streamlit dashboarding,
- night-time or adverse-weather evaluation.

Some of these ideas were explored earlier during development, but they were removed from the project scope and are not part of the final evaluated system.

## 5. Expected Contribution

The main contribution of the project is a focused end-to-end system for hatched-area violation detection rather than a broad traffic-enforcement platform. The expected contributions are:

- a custom Istanbul traffic dataset collected from five camera viewpoints,
- a fine-tuned YOLOv8s detector adapted to local traffic footage,
- polygon-based zone control for hatched-area reasoning,
- a four-state temporal violation state machine,
- severity scoring and trajectory-based interpretation of confirmed violations,
- a reproducible pipeline with both CLI and Gradio interfaces.

## 6. Evaluation Plan

The evaluation plan is restricted to the current scope of the project. It includes:

- detector-level comparison of pretrained YOLOv8s vs fine-tuned YOLOv8s,
- validation metrics such as precision, recall, mAP@50, and mAP@50-95,
- end-to-end violation evaluation with Precision / Recall / F1 once ground-truth timestamps are finalized,
- qualitative pipeline demonstrations on five fixed-camera Istanbul traffic views.

The evaluation does not include:

- YOLOv8n / YOLOv8m model sweep,
- night / rain / heavy-weather benchmarks,
- multi-tracker comparison in the final system,
- licence-plate OCR accuracy,
- dashboard usability analysis.

## 7. Ethics and Legal Scope

The project is designed with a narrow academic and ethical scope:

- no licence-plate recognition is performed,
- no face recognition is performed,
- all footage is stored locally,
- data is used only for academic research,
- outputs are not intended to be used as sole legal evidence without human review.

The recordings were collected from public vantage points, and the system is framed as a research prototype for offline analysis rather than an operational surveillance or enforcement platform.
