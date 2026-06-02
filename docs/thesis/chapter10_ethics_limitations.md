# 10. Ethics And Limitations

> **PLACEMENT NOTE.** Follows the thesis template headings **exactly** (10.1 – 10.18,
> then the summary table and Figure 10.1). Do not rename or drop any heading. The
> heading numbering reproduces the template's style (10.1 has no trailing period;
> 10.2–10.18 do). Figure 10.1 (training/validation loss curves) reuses the existing
> file `docs/thesis/figures/by_chapter/ch7_01_training_curves.png`.

---

## 10.1 Purpose of This Section

This section evaluates the ethical implications and practical limitations of the
proposed hatched-area violation detection system. It identifies the concrete risks
and harms that arise from automatically detecting traffic violations and reading
licence plates, assesses the system's fairness, privacy, and safety properties, and
states plainly what the system can and cannot do. The aim is to show that the
project is understood not only in terms of its measured accuracy (Chapter 7) but
also in terms of its societal impact and its boundaries of valid use.

## 10.2. Ethical Considerations Overview

The system operates in the domain of **automated traffic enforcement**, where an
incorrect output can have real consequences for an individual driver. A false
positive may wrongly associate a vehicle (and, through its plate, a person) with a
violation, while a false negative lets a genuine violation pass unrecorded. The
parties affected therefore include drivers whose vehicles are recorded, the
operators who act on the output, and, indirectly, other road users whose safety the
hatched markings are meant to protect. Because the system is intended to *assist*
enforcement rather than replace it, fairness, reliability, and transparency are
treated as first-class requirements alongside accuracy.

## 10.3. Data Privacy and Protection

The system processes **personal data** in two forms: vehicle imagery and licence
plates, the latter being directly identifying under Turkish data-protection law
(KVKK, Law No. 6698) and the GDPR. Privacy is protected by **data minimisation**:
the pipeline stores cropped images and plate text **only for confirmed violators**,
not for every tracked vehicle, and all processing happens **locally** on the host
machine with no transmission to third-party services. The source footage is drawn
from publicly accessible overpasses and the municipal İBB MOBESE feed, used for
non-commercial academic research. For any real deployment, access control,
encryption at rest, and a defined retention period would be required so that
evidence is kept only as long as legally necessary.

## 10.4. Bias and Fairness

The detector's accuracy is **not uniform across vehicle classes**. The custom
dataset is dominated by cars and trucks, while motorcycles are heavily
under-represented (124 motorcycle test instances against 13,090 cars). As reported
in Chapters 7–8, this imbalance shows up as a markedly lower motorcycle score
(mAP@50-95 ≈ 0.48 versus ≈ 0.81 for cars). In an enforcement context this is a
fairness issue: motorcyclists could be detected and localised less reliably than car
drivers, leading to unequal treatment. The imbalance is reported openly with
per-class metrics rather than hidden behind an aggregate number.

## 10.5. Transparency and Explainability

The system separates a black-box stage from a transparent one. Vehicle detection
relies on a YOLOv8 CNN whose internal reasoning is not directly interpretable.
However, the **violation decision itself is fully explainable**: a deterministic
four-state temporal state machine (OUTSIDE → ENTERING → INSIDE → VIOLATION) confirms
a violation only after a vehicle remains in the zone for a fixed number of
consecutive frames, and the severity is a transparent weighted formula of duration,
penetration depth, distance, and crossing angle. An operator can therefore see
exactly *why* a given vehicle was flagged, which is a deliberate design choice for an
enforcement setting. The detector's behaviour is further documented through
confusion matrices and per-class metrics (Chapter 8).

## 10.6. Accountability and Responsibility

The system is an **assistive tool, not an autonomous enforcement authority**. The
developer is responsible for the correctness of the pipeline and the honest
reporting of its limitations; the operator is responsible for reviewing each flagged
event before any enforcement action is taken. Because the model can err, no penalty
should be issued on the system's output alone — a human must remain in the loop for
the final decision.

## 10.7. Safety and Risk Assessment

The principal risks stem from detection errors. **False negatives** (missed
violations) weaken the deterrent purpose of the hatched marking and may leave a
genuine safety hazard unaddressed. **False positives** (wrongly flagged vehicles)
risk an unjust penalty and erode public trust. Misleading outputs — for example a
confidently mislabelled vehicle class — could propagate into an enforcement record.
These risks are mitigated by the temporal confirmation rule and the severity
threshold, which suppress jitter-induced false positives, and by mandatory human
review before action.

## 10.8. Misuse and Abuse Potential

Although built for a narrow purpose, the system could be misused. The licence-plate
recognition component could be repurposed for **mass surveillance or vehicle
tracking** beyond hatched-area enforcement, which would be a serious privacy
violation. **Automation bias** is another risk: operators may over-trust the
output and skip review. The project mitigates the first risk through data
minimisation (plates stored only for confirmed violations) and explicitly restricts
the intended use to hatched-area enforcement; the second is addressed by the
accountability requirement in Section 10.6.

## 10.9. Ethical Use of Data

The data used in this project is ethically sourced. The Phase 1 footage was recorded
by the author from public overpasses for academic research; the Phase 2 frames come
from the publicly accessible İBB MOBESE municipal feed. The licence-plate detector
is trained on the public **TR-PLAKA-1** Roboflow dataset under its original licence,
and the auxiliary public datasets (COCO, BDD100K) are used in accordance with their
respective licences. No private, paywalled, or credential-protected source was used.

## 10.10. Environmental Impact (Optional but Advanced)

The computational footprint of the project is modest. Model fine-tuning was
performed on a single Google Colab Tesla T4 GPU for on the order of 100 epochs,
rather than on a large multi-GPU cluster, and inference runs on a consumer CPU at
low power. The dominant cost was the iterative fine-tuning runs; this is small
compared with large-scale model training, but it is acknowledged that repeated
experimentation does carry an energy cost.

---

**Limitations Part**

## 10.11. Technical Limitations

The system is constrained by its temporal logic and its compute path. The
violation-confirmation thresholds (`min_frames_in_zone`, cooldown, polygon buffer)
were tuned empirically rather than learned, so they may need re-tuning for a camera
with a very different frame rate or viewing angle. On the CPU inference path the
pipeline runs at roughly 5–7 fps, which is adequate for offline review but not for
live operation.

## 10.12. Data Limitations

The dataset has three known weaknesses: **class imbalance** (few motorcycles),
**limited diversity** (daytime, fair-weather Istanbul overpass footage only), and
**label noise** in the auto-labelled subset that was used to expand the training set.
In addition, the ground-truth evaluation set for the end-to-end violation logic is
small, so the field-test precision/recall rests on a statistically thin sample.
These factors directly bound the confidence that can be placed in the reported
numbers.

## 10.13. Model Limitations

The YOLOv8 detector is a black-box model that performs less well on small objects —
both motorcycles and, especially, distant licence plates. The plate-reading stage
inherits the limitations of EasyOCR, whose accuracy degrades on small, angled, or
low-contrast plates. The tracker (ByteTrack) uses motion and IoU only, so long
occlusions can cause identity switches that the downstream logic must tolerate.

## 10.14. Deployment Limitations

The current deployment is **not real-time** (CPU-bound) and processes one video at a
time in a single process. Raising throughput to live speed would require a GPU.
The hatched-area polygon must be **defined manually per camera**, so the system
assumes a fixed camera and is not plug-and-play across arbitrary viewpoints.

## 10.15. Generalization and Transferability

The detector is fine-tuned on Istanbul overpass footage and is expected to transfer
poorly, without retraining, to other cities, camera heights, or vehicle
distributions; the documented dataset bias toward local vehicle silhouettes is
precisely why a generic COCO model was insufficient. Night-time and adverse-weather
conditions were excluded from the dataset and are therefore outside the system's
validated operating range.

## 10.16. Assumptions Revisited

Several assumptions made earlier shape these limits. The system assumes a **fixed,
daytime, fair-weather camera**; that a vehicle's **ground-contact point inside the
polygon** is a valid proxy for occupying the hatched area; and that the operator
draws an **accurate zone polygon**. Where these assumptions break — a moving camera,
heavy rain, an occluded ground point, or a careless polygon — accuracy will drop.

## 10.17. Future Improvements

The limitations above suggest concrete next steps: collect a **more diverse dataset**
covering night, weather, and additional cameras, and balance the motorcycle class;
add **automatic hatched-area detection** to remove the manual zone step; deploy on a
**GPU or edge device** for real-time operation; replace EasyOCR with a **dedicated,
fine-tuned Turkish plate OCR**; and expand the **ground-truth set** so the end-to-end
evaluation rests on a larger sample.

## 10.18. Ethical Recommendations

For responsible use, the system should be operated **with human oversight** and
never as the sole basis for a penalty; it should be **restricted to hatched-area
enforcement** and not repurposed for general surveillance; **data minimisation and a
retention policy** should be enforced so that plate data is kept only as long as
legally necessary; and its **per-class performance differences** should be disclosed
to operators so that the lower reliability on under-represented classes is taken into
account.

---

## Ethics and Limitations Summary Table

**Table 10.1.** Ethics and limitations summary table.

| Category | Issue | Impact | Mitigation |
|---|---|---|---|
| Bias | Motorcycle class under-represented (124 vs 13,090 test instances) | Lower detection reliability for motorcyclists → unequal treatment | Report per-class metrics openly; collect more motorcycle data |
| Privacy | Licence plates are identifying personal data (KVKK / GDPR) | Risk of unlawful tracking / data exposure | Local processing; store crops/plates only for confirmed violators; data minimisation |
| Technical | CPU inference ~5–7 fps; manual per-camera zone | Not real-time; not plug-and-play | Offline use case; GPU deployment + automatic zone detection as future work |
| Model | Black-box detector; weak on small objects (motorcycle, plate) | Missed/low-quality detections and unread plates | Domain fine-tuning; dedicated plate OCR as future work |
| Safety | False positives / false negatives in enforcement | Unjust penalty or missed hazard | Temporal confirmation + severity threshold; mandatory human review |
| Generalization | Trained on daytime Istanbul overpass footage | Poor transfer to other cities / night / weather | Retraining on diverse data; state validated operating range |

---

> **[FIGURE 10.1 — `by_chapter/ch7_01_training_curves.png`]**
> *Figure 10.1. Training and validation loss curves of the fine-tuned model
> (reused from Chapter 7), included here to support the discussion of overfitting and
> model limitations.*
