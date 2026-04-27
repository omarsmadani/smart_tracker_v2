"""
Demo: run the full tracker on a camera or .mp4 file.

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
from tracker.detector import Detector
from tracker.state_machine import State
from visualization.display import draw_fps, draw_tracking


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="Camera index (int) or path to .mp4/.avi file")
    ap.add_argument("--config", default="configs/tracker_params.yaml")
    ap.add_argument("--show-all-detections", action="store_true",
                    help="Draw all detector boxes (debug mode)")
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

    # ── target selection ─────────────────────────────────────────────
    detector = Detector(cfg["detector"])
    labels_path = Path(cfg["detector"].get("labels_path", ""))
    is_camera = args.source.isdigit()
    # Pass cap for live feed in camera mode; video stays on first frame
    picked = select(frame, detector,
                    labels_path if labels_path.exists() else None,
                    cap=cap if is_camera else None)
    if picked is None:
        print("No target selected — exiting.")
        return 0
    bbox, class_id, name = picked
    # After live-camera selection, grab a fresh frame to initialise on
    if is_camera:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame after selection.")
            return 1
    print(f"Tracking: {name!r}  class_id={class_id}  bbox={bbox}")

    # ── pipeline ─────────────────────────────────────────────────────
    pipeline = TrackerPipeline(cfg)
    pipeline.initialize(frame, bbox, class_id)

    # Draw initialisation frame
    canvas = frame.copy()
    draw_tracking(canvas, bbox, State.TRACKING, label=name)
    cv2.imshow("Edge Tracker", canvas)
    cv2.waitKey(1)

    fps = 0.0
    last_t = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            # Loop video; do nothing for camera
            if not str(args.source).isdigit():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok:
                break

        result = pipeline.process_frame(frame)

        # FPS (exponential moving average)
        now = time.perf_counter()
        dt = now - last_t
        fps = fps * 0.9 + (1.0 / dt) * 0.1 if fps > 0 and dt > 0 else (1.0 / dt if dt > 0 else 0)
        last_t = now

        # ── draw ─────────────────────────────────────────────────────
        if args.show_all_detections and result.raw_detection:
            from visualization.display import draw_detection
            for d in pipeline.detector.detect(frame):  # already ran; cheap on still frames
                draw_detection(frame, d.bbox,
                               f"{d.class_id} {d.confidence:.2f}")

        conf_str = f"{result.confidence:.2f}" if result.matched else ""
        label = f"{name} {conf_str}".strip()
        draw_tracking(frame, result.bbox, result.state,
                      label=label, recovered_frames=result.recovered_frames)
        draw_fps(frame, fps)
        _draw_state_badge(frame, result.state)

        cv2.imshow("Edge Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:   # q or Esc
            break
        if key == ord("r"):                # r = re-select target
            picked = select(frame, detector,
                            labels_path if labels_path.exists() else None,
                            cap=cap if is_camera else None)
            if picked:
                bbox, class_id, name = picked
                if is_camera:
                    ok, frame = cap.read()
                pipeline.initialize(frame, bbox, class_id)
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
    color = colors[state]
    label = state.value
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x = frame.shape[1] - tw - 12
    cv2.rectangle(frame, (x - 4, 6), (x + tw + 4, th + 14), (0, 0, 0), -1)
    cv2.putText(frame, label, (x, th + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


if __name__ == "__main__":
    sys.exit(main())
