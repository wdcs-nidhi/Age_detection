from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

AGE_MODEL = "dima806/fairface_age_image_detection"
GENDER_MODEL = "dima806/fairface_gender_image_detection"

AGE_MID = {
    "0-2": 1.0,
    "3-9": 6.0,
    "10-19": 14.5,
    "20-29": 24.5,
    "30-39": 34.5,
    "40-49": 44.5,
    "50-59": 54.5,
    "60-69": 64.5,
    "more than 70": 75.0,
}


def _load(name: str):
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(name)
    try:
        model = AutoModelForImageClassification.from_pretrained(name, device_map="auto")
    except Exception:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForImageClassification.from_pretrained(name)
        model.to(device)
    model.eval()
    return processor, model


def _classify(processor, model, pil: Image.Image) -> tuple[str, float]:
    import torch

    inputs = processor(images=pil, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    idx = int(np.argmax(probs))
    labels = getattr(model.config, "id2label", {})
    label = str(labels.get(idx, labels.get(str(idx), f"class_{idx}")))
    return label, float(probs[idx])


class AgeGender:
    """Local FairFace age + gender on a face crop."""

    def __init__(self, age_id: str = AGE_MODEL, gender_id: str = GENDER_MODEL):
        self.age_proc, self.age_net = _load(age_id)
        self.gender_proc, self.gender_net = _load(gender_id)
        print(f"Age:    {age_id}")
        print(f"Gender: {gender_id}")

    def predict(self, crop_bgr: np.ndarray) -> dict:
        if crop_bgr is None or crop_bgr.size == 0:
            return {}
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        age_group, age_conf = _classify(self.age_proc, self.age_net, pil)
        gender, gender_conf = _classify(self.gender_proc, self.gender_net, pil)
        return {
            "age": AGE_MID.get(age_group, 0.0),
            "age_group": age_group,
            "age_conf": age_conf,
            "gender": gender,
            "gender_conf": gender_conf,
        }
