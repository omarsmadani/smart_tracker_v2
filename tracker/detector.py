"""SSD-MobileNetV2 object detector (Edge TPU / CPU TFLite)."""
from collections import namedtuple
from typing import List

import numpy as np
import cv2

Detection = namedtuple("Detection", ["bbox", "class_id", "confidence"])


def _load_interpreter(model_path: str, use_edgetpu: bool):
    if use_edgetpu:
        from pycoral.utils.edgetpu import make_interpreter
        return make_interpreter(model_path)
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tflite_runtime.interpreter import Interpreter
    return Interpreter(model_path=model_path)


class Detector:
    def __init__(self, config: dict):
        path = config["model_path"] if config["use_edgetpu"] else config.get(
            "model_path_cpu", config["model_path"]
        )
        self.interpreter = _load_interpreter(path, config["use_edgetpu"])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = config["input_size"]
        self.conf_thresh = config["confidence_threshold"]

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection. Returns list of Detection(bbox=(x,y,w,h), class_id, confidence).
        bbox is in pixel coordinates of the input `frame`."""
        h, w = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        input_data = np.expand_dims(img, axis=0).astype(np.uint8)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        # SSD MobileNet v2 postprocess model output order:
        # [0] boxes [1,N,4] ymin,xmin,ymax,xmax (normalized)
        # [1] class_ids [1,N]
        # [2] scores [1,N]
        # [3] num_detections [1]
        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        class_ids = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
        num = int(self.interpreter.get_tensor(self.output_details[3]["index"])[0])

        results: List[Detection] = []
        for i in range(num):
            score = float(scores[i])
            if score < self.conf_thresh:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            x = int(xmin * w)
            y = int(ymin * h)
            bw = int((xmax - xmin) * w)
            bh = int((ymax - ymin) * h)
            if bw <= 0 or bh <= 0:
                continue
            results.append(Detection((x, y, bw, bh), int(class_ids[i]), score))
        return results
