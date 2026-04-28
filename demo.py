"""
Demo: run the full tracker on a camera or video file.

Usage:
    python demo.py                        # webcam (index 0)
    python demo.py --source 1             # webcam index 1
    python demo.py --source video.mp4
    python demo.py --source video.mp4 --config configs/tracker_params.yaml
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.tracker_pipeline import TrackerPipeline
from scripts.select_target import select
from tracker.state_machine import State
from visualization.display import draw_fps, draw_tracking


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="Camera index (int) or path to .mp4/.avi file")
    ap.add_argument("--config", default="configs/tracker_params.yaml")
    return ap.parse_args()


def open_source(source: str) -> cv2.VideoCapture:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source!r}")
    return cap


def main() -> int:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cap = open_source(args.source)
    ok, frame = cap.read()
    if not ok:
        print("Failed to read first frame.")
        return 1

    is_camera = args.source.isdigit()

    # ── target selection ─────────────────────────────────────────────
    bbox = select(frame, cap=cap if is_camera else None)
    if bbox is None:
        print("No target selected — exiting.")
        return 0

    # After live-camera selection, grab a fresh frame to initialise on
    if is_camera:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame after selection.")
            return 1

    print(f"Target bbox: {bbox}")

    # ── pipeline ─────────────────────────────────────────────────────
    pipeline = TrackerPipeline(cfg)
    pipeline.initialize(frame, bbox)

    # Draw initialisation frame
    canvas = frame.copy()
    draw_tracking(canvas, bbox, State.TRACKING)
    cv2.imshow("Edge Tracker", canvas)
    cv2.waitKey(1)

    fps = 0.0
    last_t = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            if not is_camera:           # loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok:
                break

        result = pipeline.process_frame(frame)

        now = time.perf_counter()
        dt = now - last_t
        fps = fps * 0.9 + (1.0 / dt) * 0.1 if fps > 0 and dt > 0 else (1.0 / dt if dt > 0 else 0)
        last_t = now

        draw_tracking(frame, result.bbox, result.state,
                      recovered_frames=result.recovered_frames)
        draw_fps(frame, fps)
        _draw_state_badge(frame, result.state)

        cv2.imshow("Edge Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == ord("r"):             # re-select target
            bbox = select(frame, cap=cap if is_camera else None)
            if bbox:
                if is_camera:
                    ok, frame = cap.read()
                pipeline.initialize(frame, bbox)
                fps = 0.0

    cap.release()
    cv2.destroyAllWindows()
    return 0


def _draw_state_badge(frame, state: State) -> None:
    colors = {
        State.TRACKING:  (0, 255, 0),
        State.UNCERTAIN: (0, 255, 255),
        State.LOST:      (0, 0, 255),
        State.RECOVERED: (255, 255, 0),
    }
    label = state.value
    color = colors[state]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x = frame.shape[1] - tw - 12
    cv2.rectangle(frame, (x - 4, 6), (x + tw + 4, th + 14), (0, 0, 0), -1)
    cv2.putText(frame, label, (x, th + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


if __name__ == "__main__":
    sys.exit(main())
