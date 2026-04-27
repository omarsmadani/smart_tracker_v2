"""MobileNetV3-Small appearance feature extractor (Edge TPU / CPU TFLite)."""
from typing import Tuple

import numpy as np
import cv2


def _load_interpreter(model_path: str, use_edgetpu: bool):
    if use_edgetpu:
        from pycoral.utils.edgetpu import make_interpreter
        return make_interpreter(model_path)
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tflite_runtime.interpreter import Interpreter
    return Interpreter(model_path=model_path)


class FeatureExtractor:
    def __init__(self, config: dict):
        path = config["model_path"] if config["use_edgetpu"] else config.get(
            "model_path_cpu", config["model_path"]
        )
        self.interpreter = _load_interpreter(path, config["use_edgetpu"])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = config["input_size"]
        self.feature_dim = config["feature_dim"]

    def extract(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop bbox, resize to input_size, run inference. Returns 1D feature vector."""
        x, y, w, h = bbox
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(frame.shape[1], x + w); y1 = min(frame.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid bbox for feature extraction: {bbox}")

        crop = frame[y0:y1, x0:x1]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (self.input_size, self.input_size))
        input_data = np.expand_dims(crop, axis=0).astype(self.input_details[0]["dtype"])

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        feat = self.interpreter.get_tensor(self.output_details[0]["index"])
        feat = np.asarray(feat).astype(np.float32).flatten()
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat
