"""MiVOLO v2 age estimation on a face crop."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
POC1 = HERE.parent / "POC1"
MIVOLO_SRC = POC1 / "MiVOLO"
if MIVOLO_SRC.is_dir() and str(MIVOLO_SRC) not in sys.path:
    sys.path.insert(0, str(MIVOLO_SRC))

AGE_BUCKETS = (
    (2, "0-2"),
    (9, "3-9"),
    (19, "10-19"),
    (29, "20-29"),
    (39, "30-39"),
    (49, "40-49"),
    (59, "50-59"),
    (69, "60-69"),
)


def age_to_group(age: float) -> str:
    for hi, label in AGE_BUCKETS:
        if age <= hi:
            return label
    return "more than 70"


def pick_device(requested: str | None) -> str:
    req = (requested or "auto").lower()
    if req == "cpu":
        return "cpu"
    if req.startswith("cuda"):
        return req if torch.cuda.is_available() else "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class MiVOLOAge:
    """Crop face → MiVOLO v2 age (+ gender)."""

    def __init__(self, checkpoint: str, device: str = "auto"):
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

        self.device = pick_device(device)
        ckpt = Path(checkpoint)
        if not ckpt.is_absolute():
            ckpt = (HERE / checkpoint).resolve()
        src = str(ckpt) if ckpt.exists() else str(checkpoint)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        self.config = AutoConfig.from_pretrained(src, trust_remote_code=True)
        try:
            self.model = AutoModelForImageClassification.from_pretrained(
                src, trust_remote_code=True, dtype=dtype
            )
        except TypeError:
            self.model = AutoModelForImageClassification.from_pretrained(
                src, trust_remote_code=True, torch_dtype=dtype
            )
        self.processor = AutoImageProcessor.from_pretrained(src, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        print(f"Age:    MiVOLO v2 ({src}) device={self.device}")

    def estimate(self, crop_bgr: np.ndarray) -> dict:
        if crop_bgr is None or crop_bgr.size == 0:
            return {}

        faces_input = self.processor(images=[crop_bgr])["pixel_values"]
        faces_input = faces_input.to(dtype=self.model.dtype, device=self.device)
        body_input = self.processor(images=[None])["pixel_values"]
        body_input = body_input.to(dtype=self.model.dtype, device=self.device)

        with torch.no_grad():
            output = self.model(faces_input=faces_input, body_input=body_input)

        age = float(round(output.age_output[0].item(), 2))
        id2label = getattr(self.config, "gender_id2label", {0: "male", 1: "female"})
        g_idx = int(output.gender_class_idx[0].item())
        gender = str(id2label.get(g_idx, id2label.get(str(g_idx), "unknown")))

        return {
            "age": age,
            "age_group": age_to_group(age),
            "gender": gender,
            "gender_conf": float(output.gender_probs[0].item()),
        }
