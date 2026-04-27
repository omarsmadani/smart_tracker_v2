"""Run Phase 1 tracker on a recorded video file."""
import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.tracker_pipeline import TrackerPipeline
from scripts.select_target import select
from tracker.detector import Detector
from visualization.display import draw_fps, draw_tracking


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="configs/tracker_params.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open {args.video}")
        return 1

    ok, frame = cap.read()
    if not ok:
        print("Empty video.")
        return 1

    detector = Detector(cfg["detector"])
    labels_path = Path(cfg["detector"].get("labels_path", ""))
    picked = select(frame, detector, labels_path if labels_path.exists() else None)
    if picked is None:
        print("No target selected.")
        return 0
    bbox, class_id, name = picked
    print(f"Target: class={name} ({class_id}) bbox={bbox}")

    pipeline = TrackerPipeline(cfg)
    pipeline.initialize(frame, bbox, class_id)

    # Draw first frame with initial bbox
    from tracker.state_machine import State
    canvas = frame.copy()
    draw_tracking(canvas, bbox, State.TRACKING, label=name)
    cv2.imshow("tracker", canvas)
    cv2.waitKey(1)

    last = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = pipeline.process_frame(frame)

        now = time.time()
        dt = now - last
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
        last = now

        conf_label = f"{name} {result.confidence:.2f}" if result.matched else name
        draw_tracking(frame, result.bbox, result.state,
                      label=conf_label, recovered_frames=result.recovered_frames)
        draw_fps(frame, fps)
        cv2.imshow("tracker", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
