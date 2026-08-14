from __future__ import annotations

import cv2
import numpy as np

PADDING = 0.15
MAX_YAW = 0.22
MAX_ROLL = 0.30
MIN_MOUTH_RATIO = 0.55


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


def is_frontal(kps) -> bool:
    """InsightFace 5-point: left eye, right eye, nose, left mouth, right mouth."""
    if kps is None or len(kps) < 5:
        return False
    leye, reye, nose, lm, rm = [np.array(p, dtype=np.float32) for p in kps[:5]]
    iod = float(np.linalg.norm(leye - reye))
    if iod < 5:
        return False
    x_lo = min(float(leye[0]), float(reye[0]))
    x_hi = max(float(leye[0]), float(reye[0]))
    if not (x_lo < float(nose[0]) < x_hi):
        return False
    if abs(float(leye[1]) - float(reye[1])) / iod > MAX_ROLL:
        return False
    mid_x = (float(leye[0]) + float(reye[0])) / 2.0
    if abs(float(nose[0]) - mid_x) / iod > MAX_YAW:
        return False
    d_l = float(np.linalg.norm(lm - nose))
    d_r = float(np.linalg.norm(rm - nose))
    farther = max(d_l, d_r)
    if farther < 1:
        return False
    if min(d_l, d_r) / farther < MIN_MOUTH_RATIO:
        return False
    return True


def is_near_and_clear(kps, box, crop, img_h, min_face_frac, min_iod, min_sharp) -> bool:
    if kps is None or len(kps) < 2:
        return False
    leye = np.array(kps[0], dtype=np.float32)
    reye = np.array(kps[1], dtype=np.float32)
    iod = float(np.linalg.norm(leye - reye))
    if iod < min_iod:
        return False
    x1, y1, x2, y2 = box
    if (y2 - y1) < min_face_frac * img_h:
        return False
    if (x2 - x1) < min_face_frac * 0.7 * img_h:
        return False
    if crop is None or crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.size < 400:
        return False
    if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < min_sharp:
        return False
    return True


def padded_box(bbox, img_w, img_h, pad=PADDING):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(img_w, int(x2 + bw * pad))
    y2 = min(img_h, int(y2 + bh * pad))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return x1, y1, x2, y2


class FaceFinder:
    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "auto",
        det_size: int = 640,
        det_thresh: float = 0.5,
        min_face_frac: float = 0.18,
        min_iod: float = 32.0,
        min_sharp: float = 60.0,
    ):
        from insightface.app import FaceAnalysis

        self.ctx_id = pick_ctx(device)
        self.det_size = int(det_size)
        self.det_thresh = float(det_thresh)
        self.min_face_frac = float(min_face_frac)
        self.min_iod = float(min_iod)
        self.min_sharp = float(min_sharp)
        self.app = FaceAnalysis(
            name=model_name or "buffalo_l",
            allowed_modules=["detection"],
        )
        self.app.prepare(
            ctx_id=self.ctx_id,
            det_size=(self.det_size, self.det_size),
            det_thresh=self.det_thresh,
        )
        where = "cuda" if self.ctx_id >= 0 else "cpu"
        print(
            f"Face: InsightFace {model_name} ({where}) "
            f"det_size={self.det_size} near_frac={self.min_face_frac}"
        )

    def detect(self, image) -> list[dict]:
        h, w = image.shape[:2]
        faces = []
        for face in self.app.get(image):
            if float(getattr(face, "det_score", 0.0)) < self.det_thresh:
                continue
            kps = getattr(face, "kps", None)
            if not is_frontal(kps):
                continue
            box = padded_box(face.bbox, w, h)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            if not is_near_and_clear(
                kps,
                box,
                crop,
                h,
                self.min_face_frac,
                self.min_iod,
                self.min_sharp,
            ):
                continue
            faces.append(
                {
                    "bbox": box,
                    "confidence": float(face.det_score),
                    "crop": crop,
                }
            )
        return faces
