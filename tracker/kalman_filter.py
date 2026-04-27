"""Linear Kalman filter for bbox tracking. State: [x, y, w, h, dx, dy, dw, dh]."""
from typing import Tuple

import numpy as np

Bbox = Tuple[float, float, float, float]  # (x, y, w, h) top-left + size


class KalmanBoxFilter:
    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 1.0):
    
        dt = 1.0
        # State transition: position += velocity * dt
        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = dt

        # Measurement: observe [x, y, w, h]
        self.H = np.zeros((4, 8), dtype=np.float64)
        for i in range(4):
            self.H[i, i] = 1.0

        self.Q = np.eye(8, dtype=np.float64) * process_noise
        self.R = np.eye(4, dtype=np.float64) * measurement_noise

        self.x = np.zeros((8,), dtype=np.float64)
        self.P = np.eye(8, dtype=np.float64) * 10.0
        self.initialized = False

    def initialize(self, bbox: Bbox) -> None:
        self.x[:4] = np.asarray(bbox, dtype=np.float64)
        self.x[4:] = 0.0
        self.P = np.eye(8, dtype=np.float64) * 10.0
        self.initialized = True

    def predict(self) -> Bbox:
        if not self.initialized:
            raise RuntimeError("KalmanBoxFilter.predict called before initialize")
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return tuple(self.x[:4])

    def update(self, bbox: Bbox) -> Bbox:
        if not self.initialized:
            raise RuntimeError("KalmanBoxFilter.update called before initialize")
        z = np.asarray(bbox, dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return tuple(self.x[:4])

    def get_uncertainty(self) -> float:
        return float(np.trace(self.P[:4, :4]))
