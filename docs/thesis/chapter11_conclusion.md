# 11. Conclusion And Future Work

> **PLACEMENT NOTE.** Follows the thesis template headings **exactly** (11.1 – 11.7).
> Do not rename or drop any heading. This chapter synthesises the work; it introduces
> no new technical detail. All figures/numbers referenced are defined in Chapters 4–10.

---

## 11.1. Purpose of This Section

This section summarises the project as a whole, highlights its main contributions
and results, reflects critically on what worked and what did not, and points to
realistic future directions. It does not introduce new technical material; it draws
together the dataset work (Chapter 4), methodology (Chapter 5), implementation
(Chapter 6), evaluation (Chapter 7), error analysis (Chapter 8), the working demo
(Chapter 9), and the ethics and limitations discussion (Chapter 10).

## 11.2. Summary of the Problem and Approach

This project addressed the problem of **detecting vehicles that violate hatched road
markings** in Istanbul traffic — a category of violation that unfolds over several
seconds and that existing single-frame enforcement systems do not handle. The
approach combines a **fine-tuned YOLOv8 vehicle detector** with **ByteTrack**
multi-object tracking, **Shapely** polygon zone checking, and a custom **four-state
temporal state machine** that confirms a violation only after sustained presence in
the zone, followed by severity scoring and Turkish licence-plate recognition.

## 11.3. Key Contributions

- **Domain-adapted vehicle detector.** A YOLOv8 model fine-tuned on a custom Istanbul
  traffic dataset, raising detection quality on local vehicle appearances well above
  the COCO-pretrained baseline (final model mAP@50 ≈ 0.90).
- **Temporal violation state machine.** A four-state (OUTSIDE → ENTERING → INSIDE →
  VIOLATION) confirmation rule with a per-track lock and cooldown that removes the
  bounding-box-jitter false positives produced by a naive inside/outside check.
- **Trajectory and severity scoring.** A pipeline that classifies each confirmed
  violation as `CRUISING` / `EDGE_CONTACT` / `LANE_CHANGE` and assigns a 0–100
  severity score, enabling threshold-based filtering.
- **Two-stage Turkish plate recognition.** A fine-tuned YOLOv8n plate detector
  (mAP@50 ≈ 0.98) with EasyOCR and an 81-province Turkish-format validator, run only
  on confirmed violations via best-frame voting.
- **A hand-annotated Istanbul hatched-area dataset** with ground-truth violation
  events for quantitative evaluation.
- **An interactive Gradio web application** that lets a non-technical operator run
  the whole pipeline and obtain an evidence-grade violation report.

## 11.4. Summary of Results

The fine-tuned detector reached a test **mAP@50 of approximately 0.90**, clearly
outperforming the COCO-pretrained baseline on the four local vehicle classes, and
the end-to-end pipeline achieved a field-test **F1 of about 0.84** on the annotated
ground-truth clips. An **ablation study** confirmed the central claim of the thesis:
disabling the temporal state machine (reducing it to a single-frame zone check)
substantially increased the false-positive count, demonstrating that the temporal
confirmation rule — not just the detector — is responsible for the system's
robustness. The plate detector reached mAP@50 ≈ 0.98 in isolation.

## 11.5. Discussion of Findings

The results demonstrate that the proposed approach is effective for hatched-area
enforcement, particularly because the combination of **domain fine-tuning** and
**temporal confirmation** addresses the two weaknesses of prior work — poor
generalisation to local vehicles and jitter-induced false positives. What worked
less well was small-object handling: motorcycles were detected and localised less
reliably than cars (a consequence of class imbalance), and licence plates on distant
overpass cameras were often too small for the OCR stage to read. These findings
matter because they show the violation logic is sound and transferable, while the
remaining error budget is concentrated in well-understood, addressable areas
(class balance and plate resolution) rather than in the core decision mechanism.

## 11.6. Lessons Learned

Several practical lessons emerged during the project. **Data quality and labelling
effort dominated**: building and cleaning the custom dataset, and especially
producing ground-truth violation annotations, was more demanding than training the
model itself. **Class imbalance has concrete consequences**, as the motorcycle
results showed. **System integration is a distinct source of error** from modelling:
a divergent visualiser, a database schema migration, and tracker identity switches
each required dedicated fixes, reinforcing the value of treating the pipeline as
engineering rather than a single model call. Finally, working under **constrained
compute** (CPU inference, Colab session limits) shaped many design decisions, such
as running plate recognition only on confirmed violations.

## 11.7. Limitations Recap

The system's main limitations, detailed in Chapter 10, are briefly: a **small
ground-truth evaluation set**, which bounds the confidence of the field-test
metrics; **non-real-time** CPU inference (≈ 5–7 fps) and a **manually defined,
per-camera zone**, which restrict deployment; **reduced accuracy on
under-represented classes** (motorcycles) and **distant plates**; and limited
**generalisation** beyond the daytime, fair-weather Istanbul footage on which the
model was trained. Addressing these — through more diverse and balanced data,
automatic zone detection, GPU/edge deployment, and a dedicated Turkish plate OCR —
constitutes the natural future work for this project.
