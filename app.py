"""Hatched Area Violation Detection — Gradio Web Demo.

Upload a traffic video, define the hatched area polygon on the first frame,
then run the full detection pipeline.

Usage:
    python app.py          → http://localhost:7860
"""

import json
import logging
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Project path setup ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data_models import VehicleState
from src.core.heatmap import ViolationHeatmap
from src.tracking.bytetrack_wrapper import ByteTrackWrapper
from src.violation.violation_detector import ViolationDetector
from src.zones.zone_manager import ZoneManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────
LEVEL_EN = {
    "DÜŞÜK": "LOW", "ORTA": "MEDIUM",
    "YÜKSEK": "HIGH", "KRİTİK": "CRITICAL",
}
TYPE_EN = {
    "KAYNAK": "CROSSING", "SEYİR": "CRUISING",
    "KENAR_TEMASI": "EDGE_CONTACT", "DİĞER": "OTHER",
}
LEVEL_COLORS = {
    "LOW": "#4CAF50", "MEDIUM": "#FF9800",
    "HIGH": "#F44336", "CRITICAL": "#9C27B0",
}
TYPE_COLORS = {
    "CROSSING": "#E91E63", "CRUISING": "#FF5722",
    "EDGE_CONTACT": "#FFC107", "OTHER": "#607D8B",
}
TRAIL_LEN = 30  # trajectory trail length (frames)
ZONE_TRACK_SCALE = 0.4  # downscale for feature matching speed


# ── Dynamic Zone Tracker (moving camera) ─────────────────────────────


class DynamicZoneTracker:
    """Track the zone polygon across frames using ORB feature matching.

    When the camera moves (drone, PTZ), the hatched area shifts in pixel
    space.  This class computes a homography from the reference frame
    (where the user drew the polygon) to each new frame, and warps the
    polygon accordingly.
    """

    def __init__(self, ref_bgr: np.ndarray, polygon_pts: list,
                 scale: float = ZONE_TRACK_SCALE,
                 n_features: int = 2000):
        self.scale = scale
        self.original_pts = np.float32(polygon_pts)
        ref_small = cv2.resize(ref_bgr, None, fx=scale, fy=scale)
        self.ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY)
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ref_kp, self.ref_desc = self.orb.detectAndCompute(
            self.ref_gray, None)
        self._last_pts = self.original_pts.copy()

    def update(self, frame_bgr: np.ndarray) -> list[list[int]]:
        """Return the transformed polygon for *frame_bgr*."""
        small = cv2.resize(frame_bgr, None,
                           fx=self.scale, fy=self.scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        kp, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or self.ref_desc is None or len(kp) < 10:
            return self._last_pts.astype(int).tolist()

        matches = self.bf.knnMatch(self.ref_desc, desc, k=2)
        # Lowe's ratio test
        good = [m for pair in matches if len(pair) == 2
                for m in [pair[0]] if m.distance < 0.75 * pair[1].distance]
        if len(good) < 10:
            return self._last_pts.astype(int).tolist()

        src = np.float32(
            [self.ref_kp[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        dst = np.float32(
            [kp[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            return self._last_pts.astype(int).tolist()

        # Scale polygon → match space → transform → scale back
        pts_s = (self.original_pts * self.scale).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts_s.astype(np.float32), H)
        self._last_pts = warped.reshape(-1, 2) / self.scale
        return self._last_pts.astype(int).tolist()


CSS = """
.main-title { text-align: center; }
.stat-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px; padding: 20px; color: white;
    text-align: center; margin: 4px;
}
.stat-box h2 { margin: 0; font-size: 2em; }
.stat-box p { margin: 4px 0 0 0; opacity: 0.85; font-size: 0.9em; }
footer { display: none !important; }
"""


# ── Helpers ──────────────────────────────────────────────────────────


def _model_path() -> str:
    custom = PROJECT_ROOT / "weights" / "best.pt"
    return str(custom) if custom.exists() else "yolov8s.pt"


def _first_frame_bgr(video_path: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _draw_polygon(frame_rgb: np.ndarray, points: list) -> np.ndarray:
    """Draw polygon vertices, edges, and area info on an RGB frame."""
    img = frame_rgb.copy()
    if not points:
        # Hint text
        h, w = img.shape[:2]
        cv2.putText(img, "Click to add polygon vertices",
                    (w // 2 - 180, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return img
    pts = np.array(points, dtype=np.int32)
    # Filled polygon
    if len(points) >= 3:
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (255, 60, 60))
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
        cv2.polylines(img, [pts], True, (255, 0, 0), 2)
        # Area label
        area = cv2.contourArea(pts)
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        cv2.putText(img, f"Area: {area:.0f}px", (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    elif len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(img, tuple(points[i]), tuple(points[i + 1]),
                     (255, 0, 0), 2)
    # Vertices with labels
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (x, y), 7, (0, 255, 0), -1)
        cv2.circle(img, (x, y), 7, (255, 255, 255), 2)
        cv2.putText(img, str(i + 1), (x + 10, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    # Point count badge
    badge = f"{len(points)} pts"
    cv2.putText(img, badge, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img


def _draw_hud(img, frame_idx, fps, elapsed, n_tracked, n_violations):
    """Draw heads-up display overlay on BGR frame."""
    h, w = img.shape[:2]
    # Semi-transparent top bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    timestamp = frame_idx / fps if fps > 0 else 0
    m, s = int(timestamp // 60), timestamp % 60
    proc_fps = frame_idx / elapsed if elapsed > 0 else 0

    items = [
        f"Frame: {frame_idx}",
        f"Time: {m:02d}:{s:04.1f}",
        f"Tracked: {n_tracked}",
        f"Violations: {n_violations}",
        f"FPS: {proc_fps:.1f}",
    ]
    text = "  |  ".join(items)
    cv2.putText(img, text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    return img


def _draw_trails(img, trails, severity_map):
    """Draw trajectory trails for tracked vehicles."""
    for tid, positions in trails.items():
        if len(positions) < 2:
            continue
        is_violator = tid in severity_map
        for i in range(1, len(positions)):
            alpha = i / len(positions)  # fade in
            thickness = max(1, int(alpha * 3))
            if is_violator:
                color = (0, 0, int(100 + 155 * alpha))  # red trail
            else:
                color = (0, int(100 + 155 * alpha), 0)  # green trail
            pt1 = (int(positions[i - 1][0]), int(positions[i - 1][1]))
            pt2 = (int(positions[i][0]), int(positions[i][1]))
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
    return img


def _annotate(frame_bgr, tracked_objects, zone_draw, severity_map,
              trails, frame_idx, fps, elapsed, total_violations):
    """Full frame annotation: zones, trails, bboxes, HUD."""
    img = frame_bgr.copy()

    # Zone overlay
    for name, coords in zone_draw:
        overlay = img.copy()
        cv2.fillPoly(overlay, [coords], (0, 0, 200))
        img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
        cv2.polylines(img, [coords], True, (0, 0, 255), 2)
        cx = int(coords[:, 0].mean()) - 40
        cy = max(int(coords[:, 1].min()) - 10, 15)
        cv2.putText(img, name, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Trajectory trails
    img = _draw_trails(img, trails, severity_map)

    # Vehicles
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.bbox.astype(int)
        cname = obj.detection.class_name
        tid = obj.track_id

        if obj.state == VehicleState.VIOLATION:
            score = severity_map.get(tid, 0)
            # Glow effect
            cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                          (0, 0, 180), 4)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            lbl = f"VIOLATION #{tid} {cname} [{score:.0f}]"
            (tw, th), _ = cv2.getTextSize(
                lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(
                img, (x1, y1 - th - 10), (x1 + tw + 6, y1),
                (0, 0, 255), -1)
            cv2.putText(img, lbl, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
            # Severity bar under bbox
            bar_w = int((x2 - x1) * min(score, 100) / 100)
            cv2.rectangle(img, (x1, y2 + 2), (x1 + bar_w, y2 + 8),
                          (0, 0, 255), -1)
            cv2.rectangle(img, (x1, y2 + 2), (x2, y2 + 8),
                          (100, 100, 100), 1)
        elif obj.state in (VehicleState.INSIDE, VehicleState.ENTERING):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
            lbl = f"#{tid} {cname} IN ZONE ({obj.frames_in_zone}f)"
            cv2.putText(img, lbl, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 165, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 1)
            cv2.putText(img, f"#{tid} {cname}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 200, 0), 1, cv2.LINE_AA)

    # HUD
    img = _draw_hud(img, frame_idx, fps, elapsed,
                    len(tracked_objects), total_violations)
    return img


def _reencode_h264(src: str) -> str:
    dst = src.replace(".mp4", "_web.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src,
             "-c:v", "libx264", "-preset", "fast",
             "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", dst],
            check=True, capture_output=True, timeout=600,
        )
        return dst
    except Exception:
        return src


def _tr_level(v: str) -> str:
    return LEVEL_EN.get(v, v)


def _tr_type(v: str) -> str:
    return TYPE_EN.get(v, v)


def _build_charts(df):
    """Build multi-panel analytics charts from violation dataframe."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No violations detected",
                           showarrow=False, font=dict(size=18, color="#999"))
        fig.update_layout(height=420, template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        return fig

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "pie"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=[
            "Severity Levels", "Score Distribution",
            "Violation Timeline", "Vehicle Types",
        ],
        vertical_spacing=0.14, horizontal_spacing=0.1,
    )

    # 1) Severity pie
    levels = df["Level"].value_counts()
    fig.add_trace(go.Pie(
        labels=levels.index.tolist(),
        values=levels.values.tolist(),
        hole=0.45,
        marker=dict(colors=[
            LEVEL_COLORS.get(l, "#999") for l in levels.index]),
        textinfo="label+percent",
        textfont=dict(size=11),
    ), row=1, col=1)

    # 2) Score histogram
    fig.add_trace(go.Histogram(
        x=df["Score"], nbinsx=12,
        marker_color="#2196F3", name="Score",
        opacity=0.85,
    ), row=1, col=2)
    fig.update_xaxes(title_text="Severity Score", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    # 3) Timeline scatter
    fig.add_trace(go.Scatter(
        x=df["Time"], y=df["Score"],
        mode="markers+lines",
        marker=dict(
            size=10,
            color=df["Score"],
            colorscale="RdYlGn_r",
            showscale=False,
        ),
        line=dict(width=1, color="rgba(100,100,100,0.3)"),
        text=df.apply(
            lambda r: f"#{r['Track ID']} {r['Vehicle']}<br>"
                      f"Score: {r['Score']}<br>{r['Type']}",
            axis=1),
        hoverinfo="text",
        name="Violations",
    ), row=2, col=1)
    fig.update_xaxes(title_text="Video Time", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)

    # 4) Vehicle type bar
    types = df["Vehicle"].value_counts()
    fig.add_trace(go.Bar(
        x=types.index.tolist(),
        y=types.values.tolist(),
        marker_color=["#E91E63", "#FF5722", "#FFC107",
                       "#2196F3", "#9C27B0"][:len(types)],
        name="Vehicles",
    ), row=2, col=2)
    fig.update_xaxes(title_text="Vehicle Type", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)

    fig.update_layout(
        height=650, showlegend=False,
        template="plotly_white",
        margin=dict(l=30, r=30, t=40, b=20),
        font=dict(size=11),
    )
    return fig


def _build_summary_md(stats, proc_fps, frame_idx, fps):
    """Build a Markdown summary."""
    duration = frame_idx / fps if fps > 0 else 0
    lines = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Violations | **{stats['total']}** |",
        f"| Average Score | **{stats['score_mean']}** |",
        f"| Score Range | {stats['score_min']} - {stats['score_max']} |",
        f"| Std Deviation | {stats.get('score_std', 0)} |",
        f"| Processing FPS | {proc_fps:.1f} |",
        f"| Video Frames | {frame_idx} |",
        f"| Video Duration | {duration:.1f}s |",
    ]
    if stats["by_type"]:
        lines.append("\n**Violation Types:**\n")
        for k, cnt in stats["by_type"].items():
            lines.append(f"- {_tr_type(k)}: **{cnt}**")
    if stats["by_level"]:
        lines.append("\n**Severity Levels:**\n")
        for k, cnt in stats["by_level"].items():
            lines.append(f"- {_tr_level(k)}: **{cnt}**")
    return "\n".join(lines)


# ── Gradio event handlers ───────────────────────────────────────────


def on_upload(video_path):
    if not video_path:
        return None, None, [], "[]", ""
    bgr = _first_frame_bgr(video_path)
    if bgr is None:
        raise gr.Error("Cannot read video file.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    dur = n / fps if fps > 0 else 0
    info = f"{w}x{h} | {n} frames | {fps:.1f} fps | {dur:.1f}s"
    preview = _draw_polygon(rgb, [])
    return preview, rgb, [], "[]", info


def on_click(original_frame, points, coords_text, evt: gr.SelectData):
    if original_frame is None:
        return None, points, coords_text
    x, y = int(evt.index[0]), int(evt.index[1])
    points = list(points)
    points.append([x, y])
    return _draw_polygon(original_frame, points), points, json.dumps(points)


def on_coords_submit(original_frame, coords_text):
    if original_frame is None:
        return None, []
    try:
        pts = json.loads(coords_text)
        if not isinstance(pts, list):
            raise ValueError
        for p in pts:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise ValueError
        pts = [[int(x), int(y)] for x, y in pts]
        return _draw_polygon(original_frame, pts), pts
    except (json.JSONDecodeError, ValueError, TypeError):
        raise gr.Error("Invalid format. Use: [[x1,y1],[x2,y2],...]")


def on_undo(original_frame, points):
    if original_frame is None:
        return None, [], "[]"
    points = list(points)
    if points:
        points.pop()
    return _draw_polygon(original_frame, points), points, json.dumps(points)


def on_clear(original_frame):
    if original_frame is None:
        return None, [], "[]"
    return _draw_polygon(original_frame, []), [], "[]"


# ── Main pipeline ────────────────────────────────────────────────────


def run_pipeline(video_path, polygon_pts, conf, iou, moving_cam,
                 progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a video first.")
    if len(polygon_pts) < 3:
        raise gr.Error("Define at least 3 polygon points.")

    progress(0, desc="Loading model...")

    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    # Temp zone file
    zone_dict = {
        "zones": [{
            "zone_id": "zone_01",
            "name": "Hatched Area",
            "polygon": [[int(x), int(y)] for x, y in polygon_pts],
        }]
    }
    zone_tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False)
    json.dump(zone_dict, zone_tmp)
    zone_tmp.close()

    # Pipeline components
    model_path = _model_path()
    logger.info("Model: %s | GPU: %s", model_path, has_gpu)

    tracker = ByteTrackWrapper(
        model_path=model_path, conf=conf, iou=iou, half=has_gpu)
    zone_mgr = ZoneManager(zone_tmp.name, polygon_buffer=0)
    detector = ViolationDetector(zone_manager=zone_mgr)

    # Video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    heatmap = ViolationHeatmap(width=w, height=h)

    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    all_violations = []
    severity_map: dict[int, float] = {}
    trails: dict[int, list] = defaultdict(list)
    first_bgr = None
    zone_tracker = None  # initialized on first frame if moving_cam
    frame_idx = 0
    t0 = time.time()
    step = max(1, total // 50) if total > 0 else 30
    violation_snapshots = []

    progress(0, desc="Processing video frames...")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if first_bgr is None:
            first_bgr = frame.copy()
            if moving_cam:
                zone_tracker = DynamicZoneTracker(
                    first_bgr, polygon_pts)
                logger.info("Dynamic zone tracking enabled")

        # Update zone polygon if camera is moving
        if zone_tracker is not None:
            from shapely.geometry import Polygon as ShapelyPolygon
            new_pts = zone_tracker.update(frame)
            if len(new_pts) >= 3:
                zone_mgr.zones[0].polygon = ShapelyPolygon(new_pts)

        zone_draw = zone_mgr.get_zone_polygons_for_drawing()

        objs = tracker.update(None, frame)
        objs, new_viol = detector.process_frame(objs, frame, frame_idx, fps)

        # Update trails
        for obj in objs:
            trail = trails[obj.track_id]
            trail.append(obj.bottom_center)
            if len(trail) > TRAIL_LEN:
                trail.pop(0)

        # Collect violations
        for v in new_viol:
            all_violations.append(v)
            severity_map[v.track_id] = v.severity_score
            cx = float((v.vehicle_bbox[0] + v.vehicle_bbox[2]) / 2)
            cy = float((v.vehicle_bbox[1] + v.vehicle_bbox[3]) / 2)
            heatmap.add_violation((cx, cy), v.severity_score)
            # Snapshot of violation moment (max 12)
            if len(violation_snapshots) < 12 and v.vehicle_crop is not None:
                snap = cv2.cvtColor(v.vehicle_crop, cv2.COLOR_BGR2RGB)
                violation_snapshots.append(
                    (snap, f"#{v.track_id} {v.vehicle_class} "
                           f"[{v.severity_score:.0f}]"))

        elapsed = time.time() - t0
        annotated = _annotate(
            frame, objs, zone_draw, severity_map, trails,
            frame_idx, fps, elapsed, len(all_violations))
        writer.write(annotated)
        frame_idx += 1

        if frame_idx % step == 0 and total > 0:
            progress(frame_idx / total,
                     desc=f"Frame {frame_idx}/{total} | "
                          f"{len(all_violations)} violations")

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    proc_fps = frame_idx / elapsed if elapsed > 0 else 0

    progress(0.95, desc="Encoding output video...")
    out_video = _reencode_h264(out_path)

    # ── Build outputs ────────────────────────────────────────────────

    # Violation table
    rows = []
    for v in all_violations:
        m, s = int(v.timestamp // 60), v.timestamp % 60
        rows.append({
            "Time": f"{m:02d}:{s:05.2f}",
            "Track ID": v.track_id,
            "Vehicle": v.vehicle_class,
            "Score": v.severity_score,
            "Level": _tr_level(v.severity_level),
            "Type": _tr_type(v.violation_type),
            "Frames": v.frames_in_zone,
        })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Time", "Track ID", "Vehicle",
                 "Score", "Level", "Type", "Frames"])

    # Charts
    fig = _build_charts(df)

    # Heatmap
    hmap_rgb = None
    if first_bgr is not None:
        hmap_rgb = cv2.cvtColor(
            heatmap.render(first_bgr), cv2.COLOR_BGR2RGB)

    # Summary
    stats = detector.get_severity_statistics()
    summary_md = _build_summary_md(stats, proc_fps, frame_idx, fps)

    # Gallery
    gallery = [(img, cap) for img, cap in violation_snapshots] or None

    progress(1.0, desc="Done!")
    return out_video, df, fig, hmap_rgb, summary_md, gallery


# ── Gradio app layout ───────────────────────────────────────────────


def build_app():
    with gr.Blocks(
        title="Hatched Area Violation Detection",
    ) as app:

        gr.HTML(
            "<div class='main-title'>"
            "<h1>Hatched Area Violation Detection</h1>"
            "<p>Upload a traffic video &rarr; Define zone polygon "
            "&rarr; Run detection pipeline</p>"
            "</div>"
        )

        # ── State ────────────────────────────────────────────────────
        orig_frame = gr.State(None)
        pts_state = gr.State([])

        # ── Step 1: Upload ───────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Upload Video")
                video_in = gr.Video(
                    label="Video (mp4 / mov)",
                    sources=["upload"],
                )
                video_info = gr.Textbox(
                    label="Video Info", interactive=False, lines=1)

                gr.Markdown("### 2. Settings")
                conf_sl = gr.Slider(
                    0.1, 0.9, value=0.35, step=0.05,
                    label="Confidence Threshold",
                    info="Higher = fewer detections, less noise")
                iou_sl = gr.Slider(
                    0.1, 0.9, value=0.45, step=0.05,
                    label="IoU Threshold",
                    info="Higher = less overlap merging")

                moving_cam_cb = gr.Checkbox(
                    label="Moving Camera (drone / PTZ)",
                    value=False,
                    info="Tracks the zone polygon across frames "
                         "using feature matching")

                gr.Markdown(
                    "### 3. Define Zone\n"
                    "**Click** on the frame to add vertices, "
                    "or paste JSON coordinates and press Enter."
                )
                coords_tb = gr.Textbox(
                    label="Coordinates (JSON)",
                    value="[]",
                    placeholder="[[x1,y1],[x2,y2],[x3,y3],...]",
                )
                with gr.Row():
                    undo_btn = gr.Button("Undo", size="sm")
                    clear_btn = gr.Button("Clear", size="sm",
                                          variant="stop")

                run_btn = gr.Button(
                    "Run Detection", variant="primary", size="lg")

            with gr.Column(scale=2):
                preview_img = gr.Image(
                    label="Click to add polygon vertices",
                    interactive=False, height=520)

        # ── Results ──────────────────────────────────────────────────
        gr.Markdown("---")

        with gr.Tabs():
            with gr.Tab("Video"):
                video_out = gr.Video(label="Annotated Video")

            with gr.Tab("Analytics"):
                chart_out = gr.Plot(label="Violation Analytics")
                summary_out = gr.Markdown(label="Summary")

            with gr.Tab("Violations"):
                table_out = gr.Dataframe(label="Violation Log")
                gr.Markdown("#### Violation Snapshots")
                gallery_out = gr.Gallery(
                    label="Violating Vehicles",
                    columns=4, object_fit="contain", height=250)

            with gr.Tab("Heatmap"):
                heatmap_out = gr.Image(label="Violation Heatmap")

        # ── Wiring ───────────────────────────────────────────────────
        video_in.change(
            on_upload, [video_in],
            [preview_img, orig_frame, pts_state, coords_tb, video_info])
        preview_img.select(
            on_click, [orig_frame, pts_state, coords_tb],
            [preview_img, pts_state, coords_tb])
        coords_tb.submit(
            on_coords_submit, [orig_frame, coords_tb],
            [preview_img, pts_state])
        undo_btn.click(
            on_undo, [orig_frame, pts_state],
            [preview_img, pts_state, coords_tb])
        clear_btn.click(
            on_clear, [orig_frame],
            [preview_img, pts_state, coords_tb])
        run_btn.click(
            run_pipeline,
            [video_in, pts_state, conf_sl, iou_sl, moving_cam_cb],
            [video_out, table_out, chart_out, heatmap_out,
             summary_out, gallery_out])

    return app


if __name__ == "__main__":
    build_app().launch(
        server_name="0.0.0.0", server_port=7860,
        theme=gr.themes.Soft(),
        css=CSS,
    )
