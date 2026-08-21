"""InsightFace buffalo_l → face.bbox → crop."""

from __future__ import annotations

import cv2
import numpy as np


def pick_ctx(requested: str | None) -> int:
    req = (requested or "auto").lower()
    if req == "cpu":
        return -1
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return -1


def crop_from_bbox(image, bbox, pad: float = 0.15):
    """face.bbox → padded crop."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(w, int(x2 + bw * pad))
    y2 = min(h, int(y2 + bh * pad))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None, None
    box = (x1, y1, x2, y2)
    return box, image[y1:y2, x1:x2]


class BuffaloDetector:
    """InsightFace buffalo_l face detection only."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "auto",
        det_size: int = 640,
        det_thresh: float = 0.5,
        pad: float = 0.15,
    ):
        from insightface.app import FaceAnalysis

        self.ctx_id = pick_ctx(device)
        self.det_thresh = float(det_thresh)
        self.pad = float(pad)
        self.app = FaceAnalysis(
            name=model_name or "buffalo_l",
            allowed_modules=["detection"],
        )
        self.app.prepare(
            ctx_id=self.ctx_id,
            det_size=(int(det_size), int(det_size)),
            det_thresh=self.det_thresh,
        )
        where = "cuda" if self.ctx_id >= 0 else "cpu"
        print(f"Detect: InsightFace {model_name} ({where})")

    def detect(self, image) -> list[dict]:
        """
        Camera / Image frame → list of {bbox, crop, confidence}
        """
        faces = []
        for face in self.app.get(image):
            score = float(getattr(face, "det_score", 0.0))
            if score < self.det_thresh:
                continue
            # face.bbox → crop
            box, crop = crop_from_bbox(image, face.bbox, pad=self.pad)
            if box is None or crop is None or crop.size == 0:
                continue
            faces.append(
                {
                    "bbox": box,
                    "crop": crop,
                    "confidence": score,
                }
            )
        return faces
