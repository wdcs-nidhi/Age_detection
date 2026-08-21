"""
Pipeline structure:

  Camera / Image
        ▼
  InsightFace buffalo_l
        │
   face.bbox
        ▼
   Crop face
        ▼
  MiVOLO v2  → age
"""

from __future__ import annotations

import cv2

from detect import BuffaloDetector
from mivolo_age import MiVOLOAge


class Pipeline:
    def __init__(self, cfg: dict):
        det = cfg.get("detection", {})
        age = cfg.get("age", {})
        self.detector = BuffaloDetector(
            model_name=str(det.get("model", "buffalo_l")),
            device=str(det.get("device", "auto")),
            det_size=int(det.get("det_size", 640)),
            det_thresh=float(det.get("confidence", 0.5)),
            pad=float(det.get("pad", 0.15)),
        )
        self.age_model = MiVOLOAge(
            checkpoint=str(age.get("checkpoint", "../POC1/mivolo_models")),
            device=str(age.get("device", det.get("device", "auto"))),
        )

    def process(self, frame):
        """One frame: detect → crop → MiVOLO age. Returns (drawn_frame, results)."""
        out = frame.copy()
        faces = self.detector.detect(frame)

        results = []
        for i, face in enumerate(faces, start=1):
            pred = self.age_model.estimate(face["crop"])
            item = {
                "id": i,
                "bbox": face["bbox"],
                "face_conf": face["confidence"],
                "age": pred.get("age"),
                "age_group": pred.get("age_group"),
                "gender": pred.get("gender"),
                "gender_conf": pred.get("gender_conf", 0.0),
            }
            results.append(item)
            self._draw(out, item)
        return out, results

    def _draw(self, frame, item: dict):
        x1, y1, x2, y2 = item["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        age = item.get("age")
        group = item.get("age_group") or "?"
        gender = item.get("gender") or "?"
        age_s = f"{age:.0f}y" if isinstance(age, (int, float)) else "?"
        label = f"MiVOLO {gender} {age_s} {group}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_bar = max(0, y1 - th - 10)
        cv2.rectangle(frame, (x1, y_bar), (x1 + tw + 8, y1), (0, 220, 0), -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
