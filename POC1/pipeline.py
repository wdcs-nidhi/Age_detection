from __future__ import annotations

import cv2

from faces import FaceFinder
from fairface import AgeGender
from mivolo_age import MiVOLOAgeGender, age_to_group


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _fmt_src(name: str, age, group, gender, conf=None) -> str:
    age_s = f"{age:.0f}y" if isinstance(age, (int, float)) else "?"
    group_s = group or "?"
    gender_s = gender or "?"
    if conf is not None:
        return f"{name}: {gender_s} {age_s} {group_s} ({conf:.2f})"
    return f"{name}: {gender_s} {age_s} {group_s}"


class Pipeline:
    def __init__(self, cfg: dict):
        det = cfg.get("detection", {})
        age = cfg.get("age", {})
        gender = cfg.get("gender", {})
        proc = cfg.get("processing", {})
        self.detect_every = int(proc.get("detection_interval", 3))
        self.attr_every = int(proc.get("attr_interval", 15))
        self.faces = FaceFinder(
            model_name=str(det.get("model", "buffalo_l")),
            device=str(det.get("device", "auto")),
            det_size=int(det.get("det_size", 640)),
            det_thresh=float(det.get("confidence", 0.5)),
            min_face_frac=float(det.get("min_face_frac", 0.18)),
            min_iod=float(det.get("min_iod", 32)),
            min_sharp=float(det.get("min_sharp", 60)),
        )
        # Always compare: MiVOLO + FairFace + buffalo_l genderage
        self.mivolo = MiVOLOAgeGender(
            checkpoint=str(age.get("checkpoint", "mivolo_models")),
            device=str(age.get("device", det.get("device", "auto"))),
            use_persons=bool(age.get("use_persons", False)),
        )
        self.fairface = AgeGender(
            age_id=age.get("model", "dima806/fairface_age_image_detection"),
            gender_id=gender.get(
                "model", "dima806/fairface_gender_image_detection"
            ),
        )
        self.frame_id = 0
        self.tracks: dict[int, dict] = {}
        self._next_id = 1
        self._last_faces: list[dict] = []

    def reset(self):
        self.frame_id = 0
        self.tracks = {}
        self._next_id = 1
        self._last_faces = []

    def process(self, frame):
        frame = frame.copy()
        self.frame_id += 1
        if self.frame_id == 1 or self.frame_id % self.detect_every == 0:
            self._last_faces = self.faces.detect(frame)
        faces = self._last_faces
        self._update_tracks(faces)

        results = []
        for trk in self.tracks.values():
            need = (
                trk.get("mivolo") is None
                or trk.get("fairface") is None
                or trk["since_attr"] >= self.attr_every
            )
            if need and trk.get("crop") is not None:
                trk["mivolo"] = self.mivolo.predict(trk["crop"])
                trk["fairface"] = self.fairface.predict(trk["crop"])
                trk["since_attr"] = 0
            else:
                trk["since_attr"] = trk.get("since_attr", 0) + 1

            buffalo = {
                "age": trk.get("buffalo_age"),
                "age_group": age_to_group(trk["buffalo_age"])
                if trk.get("buffalo_age") is not None
                else None,
                "gender": trk.get("buffalo_gender"),
                "age_conf": 1.0,
                "gender_conf": 1.0,
            }
            mivolo = trk.get("mivolo") or {}
            fairface = trk.get("fairface") or {}

            # Keep top-level fields from MiVOLO for older callers
            item = {
                "id": trk["id"],
                "bbox": trk["bbox"],
                "face_conf": trk["confidence"],
                "age": mivolo.get("age"),
                "age_group": mivolo.get("age_group"),
                "age_conf": mivolo.get("age_conf", 0.0),
                "gender": mivolo.get("gender"),
                "gender_conf": mivolo.get("gender_conf", 0.0),
                "mivolo": mivolo,
                "buffalo_l": buffalo,
                "fairface": fairface,
            }
            results.append(item)
            self._draw(frame, item)
        return frame, results

    def _update_tracks(self, faces: list[dict]):
        assigned_f = set()
        assigned_t = set()
        pairs = []
        for tid, trk in self.tracks.items():
            for fi, face in enumerate(faces):
                pairs.append((_iou(trk["bbox"], face["bbox"]), tid, fi))
        pairs.sort(reverse=True)
        for iou, tid, fi in pairs:
            if iou < 0.3:
                break
            if tid in assigned_t or fi in assigned_f:
                continue
            face = faces[fi]
            self.tracks[tid]["bbox"] = face["bbox"]
            self.tracks[tid]["confidence"] = face["confidence"]
            self.tracks[tid]["crop"] = face["crop"]
            self.tracks[tid]["buffalo_age"] = face.get("buffalo_age")
            self.tracks[tid]["buffalo_gender"] = face.get("buffalo_gender")
            self.tracks[tid]["misses"] = 0
            assigned_t.add(tid)
            assigned_f.add(fi)
        for tid, trk in list(self.tracks.items()):
            if tid in assigned_t:
                continue
            trk["misses"] = trk.get("misses", 0) + 1
            if trk["misses"] > 20:
                del self.tracks[tid]
        for fi, face in enumerate(faces):
            if fi in assigned_f:
                continue
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = {
                "id": tid,
                "bbox": face["bbox"],
                "confidence": face["confidence"],
                "crop": face["crop"],
                "buffalo_age": face.get("buffalo_age"),
                "buffalo_gender": face.get("buffalo_gender"),
                "misses": 0,
                "since_attr": 10_000,
                "mivolo": None,
                "fairface": None,
            }

    def _draw(self, frame, item: dict):
        x1, y1, x2, y2 = item["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)

        lines = []
        for key, name in (
            ("mivolo", "MiVOLO"),
            ("buffalo_l", "buffalo_l"),
            ("fairface", "FairFace"),
        ):
            src = item.get(key) or {}
            lines.append(
                _fmt_src(
                    name,
                    src.get("age"),
                    src.get("age_group"),
                    src.get("gender"),
                    src.get("age_conf") if key == "fairface" else src.get("gender_conf"),
                )
            )

        y = y1 - 8
        for line in reversed(lines):
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_bar = max(0, y - th - 4)
            cv2.rectangle(frame, (x1, y_bar), (x1 + tw + 6, y + 2), (0, 220, 0), -1)
            cv2.putText(
                frame,
                line,
                (x1 + 3, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
            y = y_bar - 2
