# Ground Truth Annotations

Hand-annotated violation intervals used by
`scripts/evaluate_with_ground_truth.py` to compute Precision, Recall,
and F1 for the pipeline output.

**These JSON files are the only files in `data/` that git tracks** —
everything else (videos, frames, Roboflow exports) lives outside the
repo.

## File Naming

One JSON per test video. The filename should match the video's basename
(without extension) so the evaluator can auto-pair them:

```
data/videos/test/cam1.mp4  →  data/ground_truth/cam1.json
data/videos/test/cam2.mp4  →  data/ground_truth/cam2.json
```

## Schema

```json
{
  "video":       "cam1.mp4",
  "fps":         30,
  "annotator":   "Sertaç Akalın",
  "description": "Short description of the scene and conditions",
  "instructions": "...",
  "violations": [
    {
      "id":            "v001",
      "start_frame":   245,
      "end_frame":     281,
      "vehicle_class": "car",
      "type":          "LANE_CHANGE",
      "notes":         "Clear diagonal crossing of the hatched area toward the left lane."
    },
    {
      "id":            "v002",
      "start_frame":   510,
      "end_frame":     605,
      "vehicle_class": "truck",
      "type":          "CRUISING",
      "notes":         "Truck travels along the hatched area for ~3 seconds."
    },
    {
      "id":            "v003",
      "start_frame":   812,
      "end_frame":     828,
      "vehicle_class": "car",
      "type":          "EDGE_CONTACT",
      "notes":         "Brief wheel touch on the zone boundary — borderline case."
    }
  ]
}
```

### Field Definitions

| Field | Type | Meaning |
|---|---|---|
| `video` | string | Video file name inside `data/videos/test/` |
| `fps` | int | Source video FPS (needed to convert frame ↔ seconds) |
| `annotator` | string | Who labeled it (for traceability) |
| `description` | string | One-line scene description |
| `violations[].id` | string | Stable ID for this event (e.g. `v001`) |
| `violations[].start_frame` | int | First frame where the vehicle's bottom-center enters the hatched polygon |
| `violations[].end_frame` | int | Last frame where the vehicle is still (partially) in the zone |
| `violations[].vehicle_class` | string | One of `car`, `truck`, `bus`, `motorcycle` |
| `violations[].type` | string | `LANE_CHANGE`, `CRUISING`, or `EDGE_CONTACT` |
| `violations[].notes` | string | Free-text observation (useful for qualitative error analysis) |

## How to Annotate

1. Open the test video in VLC or similar.
2. Scrub through. When you see a vehicle enter the hatched area, **pause**.
3. Note the current frame number (`Tools → Current frame info` in VLC,
   or compute `time_seconds * fps`).
4. Write down `start_frame` (entry), let it play until the vehicle
   clearly exits, note `end_frame`.
5. Categorize the violation type:
   - **LANE_CHANGE** — short diagonal crossing (< ~0.5 s), vehicle
     clearly moves from one lane to another.
   - **CRUISING** — vehicle stays inside the zone for more than ~0.5 s
     while moving along the road direction.
   - **EDGE_CONTACT** — vehicle only grazes the edge, low penetration.
6. Add the entry to the JSON.

Aim for **at least 15–20 violations per video** for statistically
meaningful P/R/F1 numbers in the thesis.

## Matching Policy (used by the evaluator)

- A predicted violation is counted as a true positive if its
  `start_frame` falls within ±`tolerance_frames` of any ground-truth
  `start_frame` **and** both events reference the same track ID or
  have overlapping frame intervals.
- Default tolerance: **15 frames (~0.5 s at 30 FPS)**.
- One ground-truth event matches at most one prediction.
- Extra predictions → false positives. Unmatched ground truth →
  false negatives.
