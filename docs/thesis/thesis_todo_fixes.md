# Thesis TODO Resolutions (Thesis9G.docx)

> I did **not** edit the `.docx`. Below is every `[TODO]` left in the thesis, with the
> resolved value to type in, or marked **NEEDS YOU** where only you have the info.
> Values marked ✅ were verified from the real code / run artifacts in the repo.

---

## ✅ Resolved from the repo (just type these in)

**Native recording resolution (Ch4, ~"The native recording resolution is [TODO]…")**
✅ "The native resolution is **1920 × 1080** for the Phase-1 own recordings and up to
**3840 × 2160** for the higher-resolution MOBESE/drone frames; every frame is resized
to 640 × 640 with letterbox padding before training." *(test clips on disk: cam1_test
= 1920×1080, cam4_30s = 3840×2160.)*

**Small ground-truth evaluation set (Ch4, "[TODO: state the number of annotated
videos and total violation events]")**
✅ "**Two** annotated test clips (cam1_test and cam4_30s) with **20+ violation events
each (40+ total)**." *(Matches Ch8/Ch9. Confirm the exact per-file count from
`data/ground_truth/test_01.json` and `field_test_30s.json` — that folder is currently
offloaded by iCloud; right-click → Download to re-materialise it.)*

**v3/v4 hyperparameter table (Ch6) — Optimizer / Epochs / Seed / Training time**
These are verified from `runs/detect/mobese_v4/args.yaml` and `results.csv` for the
**production model `best_v4.pt`**:
- Optimizer: ✅ **SGD** (Ultralytics "auto" resolves to SGD)
- Max epochs / actual: ✅ **60 max → stopped at 28** (early stopping, patience 20)
- Seed: ✅ **42**
- Training time: ✅ **≈ 1 h 38 min** (5874 s cumulative, Colab Tesla T4)
- lr0 0.005, lrf 0.01, momentum 0.937, weight_decay 0.0005, batch 16, imgsz 640
- Final metrics (results.csv last row): **mAP@50 ≈ 0.895**, mAP@50-95 ≈ 0.79 *(matches
  the 0.896 reported in Ch7)*

**Peak VRAM during training (Ch6, "[TODO: read from Colab session log]")**
~ **10–12 GB** on the T4 (YOLOv8m, batch 16, imgsz 640). Estimate — confirm from the
Colab log if you still have it; otherwise keep "approximately 10–12 GB".

**KVKK clause (Ch4 lawfulness paragraph)**
✅ Cite **Law No. 6698 (KVKK)** — this is reference **[15]** in your new Chapter 12.
Add `[15]` after "...Turkish data-protection rules (KVKK)".

---

## ✅ Figures that already exist (point the placeholders to these files)

| Placeholder | Use this existing file |
|---|---|
| 5-panel camera grid (Ch4, "[TODO: insert a 5-panel grid…cam1–cam5]") | `adsız klasör/Resim1.png` (= `figure_4_1_cam_grid.png`) |
| Per-class label histogram (Ch4, "[TODO: include …labels.jpg]") | the Ultralytics `labels.jpg` from the v4 run (in `adsız klasör/` / Drive run dir) |
| Production results.png (Ch4/Ch6, "[TODO: include the production fine-tune's results.png]") | `adsız klasör/results.png` (+ `BoxR_curve.png`, `confusion_matrix_normalized.png`) — **use the v4 run's**, not v1 |

---

## ⚠ NEEDS YOU (only you have this)

- **Roboflow URLs** (Ch4): TR-PLAKA-1 dataset URL, and your two private workspace URLs
  (`[TODO: paste URL]` ×3). Paste your Roboflow links.
- **TR-PLAKA-1 counts** (Ch4, "total / train / valid / test counts"): read from the
  Roboflow dataset page.
- **License declaration** (appears 3×: Ch4 "[TODO: declare a license]", Ch6
  "License: [TODO: declare — MIT or Apache 2.0]", and one more): pick **MIT** or
  **Apache 2.0** and write it in all three spots + add a `LICENSE` file to the repo.
  (Recommendation: **MIT** — simplest for an academic project.)

---

## 🐞 Real bug to fix (code, not thesis)

**`configs/config.yaml` points to a non-existent default video.**
✅ Confirmed: line 8 is `video_source: "data/videos/test/test_01.mp4"` which does not
exist. The thesis sentence (Ch6) flags this. Best fix: change the default to a clip
that ships, e.g. `data/videos/test/cam1_test.mp4`. This is a 1-line **code** change —
tell me and I'll do it (I won't touch it without your OK).

---

## ⚠ Important consistency issue: v3 vs v4

Your thesis is **inconsistent about the production model**:
- `configs/config.yaml`, Chapter 7, and Chapter 9 use **`best_v4.pt`** (mAP@50 ≈ 0.896).
- Chapters 5 and 6 still describe **`best_v3.pt`** as the production model
  (mAP@50 = 0.8075) and its hyperparameter table shows the v3 numbers.

**Recommendation:** standardise on **`best_v4.pt`** everywhere (it is what is deployed
and evaluated), using the verified v4 values above. If you instead want to keep v3 as
the thesis model, then Chapter 7's numbers must be reverted to v3 — but that would
contradict the config and the demo. I recommend v4.
