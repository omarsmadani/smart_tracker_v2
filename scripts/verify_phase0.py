"""Phase 0 smoke test: run detector + feature extractor on a single image."""
import argparse
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracker.detector import Detector
from tracker.feature_extractor import FeatureExtractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to test image")
    ap.add_argument("--config", default="configs/tracker_params.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(args.image)

    det = Detector(cfg["detector"])
    detections = det.detect(frame)
    print(f"Detections: {len(detections)}")
    for d in detections[:10]:
        print(f"  class={d.class_id} conf={d.confidence:.2f} bbox={d.bbox}")

    if not detections:
        print("No detections above threshold. Exiting.")
        return

    fx = FeatureExtractor(cfg["feature_extractor"])
    feat = fx.extract(frame, detections[0].bbox)
    print(f"Feature vector: shape={feat.shape} norm={float((feat**2).sum())**0.5:.4f}")


if __name__ == "__main__":
    main()
