"""InsightFace buffalo_l → face.bbox → clear crop only."""

from __future__ import annotations

import cv2


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


def sharpness(crop) -> float:
    """Laplacian variance — higher = sharper. Blurry faces score low."""
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.size < 400:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_clear_face(crop, box, img_h: int, min_sharp: float, min_face_frac: float) -> bool:
    """Keep only faces that are big enough and not blurred."""
    if crop is None or crop.size == 0:
        return False
    x1, y1, x2, y2 = box
    if (y2 - y1) < min_face_frac * img_h:
        return False
    if (x2 - x1) < min_face_frac * 0.7 * img_h:
        return False
    return sharpness(crop) >= min_sharp


class BuffaloDetector:
    """InsightFace buffalo_l face detection → clear crops only."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "auto",
        det_size: int = 640,
        det_thresh: float = 0.5,
        pad: float = 0.15,
        min_sharp: float = 80.0,
        min_face_frac: float = 0.12,
    ):
        from insightface.app import FaceAnalysis

        self.ctx_id = pick_ctx(device)
        self.det_thresh = float(det_thresh)
        self.pad = float(pad)
        self.min_sharp = float(min_sharp)
        self.min_face_frac = float(min_face_frac)
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
        print(
            f"Detect: InsightFace {model_name} ({where}) "
            f"min_sharp={self.min_sharp} min_face_frac={self.min_face_frac}"
        )

    def detect(self, image) -> list[dict]:
        """
        Camera / Image frame → clear faces only {bbox, crop, confidence, sharp}
        Blurry / tiny faces are dropped before MiVOLO.
        """
        h = image.shape[0]
        faces = []
        for face in self.app.get(image):
            score = float(getattr(face, "det_score", 0.0))
            if score < self.det_thresh:
                continue
            # face.bbox → crop
            box, crop = crop_from_bbox(image, face.bbox, pad=self.pad)
            if box is None or crop is None or crop.size == 0:
                continue
            sharp = sharpness(crop)
            if not is_clear_face(
                crop, box, h, min_sharp=self.min_sharp, min_face_frac=self.min_face_frac
            ):
                continue
            faces.append(
                {
                    "bbox": box,
                    "crop": crop,
                    "confidence": score,
                    "sharp": sharp,
                }
            )
        return faces
