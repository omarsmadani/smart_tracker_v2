"""Download pretrained TFLite models into models/. Cross-platform."""
import sys
import urllib.request
from pathlib import Path

BASE = "https://github.com/google-coral/test_data/raw/master"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

FILES = [
    # (url_filename, local_filename)
    ("ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
     "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"),
    ("ssd_mobilenet_v2_coco_quant_postprocess.tflite",
     "ssd_mobilenet_v2_coco_quant_postprocess.tflite"),
    ("coco_labels.txt", "coco_labels.txt"),
    ("mobilenet_v2_1.0_224_quant_edgetpu.tflite",
     "mobilenet_v2_quant_edgetpu.tflite"),
    ("mobilenet_v2_1.0_224_quant.tflite",
     "mobilenet_v2_quant.tflite"),
]


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for remote, local in FILES:
        dest = MODELS_DIR / local
        if dest.exists():
            print(f"[skip] {local}")
            continue
        url = f"{BASE}/{remote}"
        print(f"[get ] {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"       -> {dest} ({dest.stat().st_size / 1024:.1f} KB)")
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
