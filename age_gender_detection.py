"""Age + gender detection — A0042.
Pipeline:
```
Camera / Image
      ↓
InsightFace buffalo_l
      ↓
   face.bbox
      ↓
   Crop face
      ↓
   MiVOLO v2
      ↓
  age + gender
```
When at least one qualifying face exists:
save_detection()
update_detection_after_time()
No tracking/state across frames.
"""

import os
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from django.conf import settings
from shapely.geometry import Point, Polygon

from api.logging_utils import PrefixedLogger, configure_logging
from api.process.common import (
    alert_code_decorator,
    safe_literal_eval,
    save_detection,
    update_detection_after_time,
)

base_logger = configure_logging(__name__)

age_gender_alert_code = "A0042"

# FACE DETECTION / QUALITY SETTINGS

PADDING = 0.15
MAX_YAW = 0.22
MAX_ROLL = 0.30
MIN_MOUTH_RATIO = 0.55
DET_SIZE = 640
DET_THRESH = 0.5

MIN_FACE_FRAC = 0.18
MIN_IOD = 32.0
MIN_SHARP = 60.0

# MiVOLO SETTINGS
# Expected local model location:
# settings.AI_MODELS_PATH/
# └── A0042/
# └── mivolo_v2/
# ├── config.json
# ├── ...
# └── model files

MIVOLO_MODEL_SUBDIR = "mivolo_v2"

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

# GLOBAL MODEL CACHE
_models_lock = threading.Lock()
_face_app = None
_mivolo_config = None
_mivolo_processor = None
_mivolo_model = None
_mivolo_device = None

# DEVICE HELPERS

def _pick_ctx(requested: str = "auto") -> int:
    """InsightFace ctx_id.  0  = CUDA, -1 = CPU """

    req = (requested or "auto").lower()
    if req == "cpu":
        return -1
    try:
        if torch.cuda.is_available():
            return 0
    except Exception: pass
    return -1

def _pick_device(requested: str = "auto") -> str:
    """PyTorch device."""
    
    req = (requested or "auto").lower()
    if req == "cpu":
        return "cpu"
    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return req
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"

# AGE GROUP

def age_to_group(age: float) -> str:
    """Convert MiVOLO numeric age into application age buckets."""

    for hi, label in AGE_BUCKETS:
        if age <= hi:
            return label
    return "more than 70"

# FACE QUALITY / FRONTAL CHECKS

def _is_frontal(kps) -> bool:
    """Reject faces that are too rotated / not frontal enough."""

    if kps is None or len(kps) < 5:
        return False

    leye, reye, nose, lm, rm = [
        np.array(p, dtype=np.float32)
        for p in kps[:5]
    ]
    iod = float(np.linalg.norm(leye - reye))

    if iod < 5:
        return False

    x_lo = min(float(leye[0]), float(reye[0]))
    x_hi = max(float(leye[0]), float(reye[0]))

    if not (x_lo < float(nose[0]) < x_hi):
        return False

    # Roll check: vertical eye difference.
    if abs(float(leye[1]) - float(reye[1])) / iod > MAX_ROLL:
        return False

    # Yaw check: nose horizontal position relative to eye midpoint.
    mid_x = (float(leye[0]) + float(reye[0])) / 2.0

    if abs(float(nose[0]) - mid_x) / iod > MAX_YAW:
        return False

    # Mouth symmetry check.
    d_l = float(np.linalg.norm(lm - nose))
    d_r = float(np.linalg.norm(rm - nose))

    farther = max(d_l, d_r)

    if farther < 1:
        return False

    if min(d_l, d_r) / farther < MIN_MOUTH_RATIO:
        return False

    return True

def _padded_box(bbox, img_w: int, img_h: int, pad: float = PADDING):
    """Create padded face bounding box."""

    x1, y1, x2, y2 = [float(v) for v in bbox]

    bw = x2 - x1
    bh = y2 - y1

    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))

    x2 = min(img_w, int(x2 + bw * pad))
    y2 = min(img_h, int(y2 + bh * pad))

    if x2 - x1 < 20 or y2 - y1 < 20:
        return None

    return x1, y1, x2, y2   

def _is_near_and_clear( kps, box, crop, img_h: int ) -> bool:
    """Extra face size / quality checks."""

    if kps is None or len(kps) < 2:
        return False

    leye = np.array(kps[0], dtype=np.float32)
    reye = np.array(kps[1], dtype=np.float32)

    iod = float(np.linalg.norm(leye - reye))

    if iod < MIN_IOD:
        return False

    x1, y1, x2, y2 = box

    if (y2 - y1) < MIN_FACE_FRAC * img_h:
        return False

    if (x2 - x1) < MIN_FACE_FRAC * 0.7 * img_h:
        return False

    if crop is None or crop.size == 0:
        return False

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    if gray.size < 400:
        return False

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    if sharpness < MIN_SHARP:
        return False

    return True


# MODEL PATH

def _mivolo_model_id() -> str:
    """Return local MiVOLO model path."""

    local = Path(settings.AI_MODELS_PATH) / "A0042" / MIVOLO_MODEL_SUBDIR
    if local.is_dir(): return str(local)
    raise FileNotFoundError(f"MiVOLO model directory not found: {local}")

# MODEL LOADING

def _load_mivolo(model_path: str):
    """Load MiVOLO v2 model and image processor."""

    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForImageClassification,
    )
    
    device = _pick_device("auto")
    dtype = ( torch.float16 if device.startswith("cuda") else torch.float32 )
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    try:
        model = (
            AutoModelForImageClassification
            .from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=dtype,
            )
        )
    except TypeError:
        # Compatibility with older Transformers versions.
        model = (
            AutoModelForImageClassification
            .from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
        )
    processor = AutoImageProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return ( config, processor, model, device )


def _get_models(logger):
    """Load and cache InsightFace + MiVOLO."""

    global _face_app
    global _mivolo_config
    global _mivolo_processor
    global _mivolo_model
    global _mivolo_device

    with _models_lock:

        # InsightFace buffalo_l

        if _face_app is None:
            from insightface.app import FaceAnalysis
            
            ctx_id = _pick_ctx("auto")
            model_root = os.path.join(settings.AI_MODELS_PATH, "A0019")
            kwargs = {"name": "buffalo_l", "allowed_modules": ["detection"]}

            if os.path.isdir(model_root):
                kwargs["root"] = model_root
            _face_app = FaceAnalysis(**kwargs)
            _face_app.prepare(ctx_id=ctx_id, det_size=(DET_SIZE,DET_SIZE), det_thresh=DET_THRESH)

            logger.info(f'A0042 InsightFace buffalo_l ready {"cuda" if ctx_id >= 0 else "cpu"}')

        # MiVOLO v2
        
        if _mivolo_model is None:
            model_path = _mivolo_model_id()
            logger.info(f"model_path: {model_path}")
            (
                _mivolo_config,
                _mivolo_processor,
                _mivolo_model,
                _mivolo_device,
            ) = _load_mivolo(model_path)
            
            logger.info(f"A0042 MiVOLO v2 ready {_mivolo_device}")

    return (_face_app, _mivolo_config, _mivolo_processor, _mivolo_model, _mivolo_device)

# MiVOLO AGE + GENDER

def _predict_age_gender( crop_bgr, config, processor, model, device ):
    """Face crop → MiVOLO age + gender."""

    if crop_bgr is None or crop_bgr.size == 0:
        return {}

    # Face input
    faces_input = processor(images=[crop_bgr])["pixel_values"]
    faces_input = faces_input.to(dtype=model.dtype, device=device)

    # Body input
    # We only have a face crop from buffalo_l.  MiVOLO receives no body image.

    body_input = processor(images=[None])["pixel_values"]
    body_input = body_input.to(dtype=model.dtype, device=device)

    # Inference
    with torch.no_grad():
        output = model(faces_input=faces_input, body_input=body_input)

    # Age
    age = float(round(output.age_output[0].item(),2))

    # Gender
    id2label = getattr(
        config,"gender_id2label",
        {0: "male", 1: "female"},
    )

    gender_idx = int(output.gender_class_idx[0].item())

    gender = str(
        id2label.get(
            gender_idx,
            id2label.get(str(gender_idx), "unknown"),
        )
    )

    gender_conf = float(output.gender_probs[0].item())

    return {
        "age": age,
        "age_group": age_to_group(age),
        "gender": gender,
        "gender_conf": gender_conf,
    }

# FACE DETECTION

def _detect_faces(frame, face_app):
    """Detect frontal + sufficiently clear faces."""

    h, w = frame.shape[:2]
    faces = []

    for face in face_app.get(frame):
        confidence = float(getattr(face, "det_score", 0.0))
        if confidence < DET_THRESH:
            continue
        kps = getattr(face, "kps", None)

        # Frontal check.
        if not _is_frontal(kps):
            continue
        # Padded bounding box.
        box = _padded_box(face.bbox, w, h)

        if box is None:
            continue

        x1, y1, x2, y2 = box

        crop = frame[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            continue
        # Face size / sharpness.
        if not _is_near_and_clear(kps, box, crop, h):
            continue
        faces.append(
            {
                "bbox": box,
                "confidence": confidence,
            }
        )
    return faces

# ROI POLYGON

def _parse_polygon(area_coordinates_json, frame):
    """Create monitoring polygon."""

    height, width = frame.shape[:2]
    coordinates = (
        safe_literal_eval(area_coordinates_json)
        if area_coordinates_json else None
    )
    polygon_points = None
    if (isinstance(coordinates, list) and coordinates):
        if isinstance(coordinates[0], (list, tuple)):
            polygon_points = [tuple(p) for p in coordinates]
            
        elif all(isinstance(i, (int, float)) for i in coordinates):
            polygon_points = [
                (coordinates[i], coordinates[i + 1])
                for i in range(0, len(coordinates) - 1, 2)
            ]
    if (polygon_points and len(polygon_points) >= 3):
        polygon = Polygon(polygon_points)
        poly_np = (np.array( polygon_points, np.int32).reshape((-1, 1, 2)))
        cv2.polylines(frame, [poly_np], True, (0, 255, 0), 2)
        return polygon

    # Full frame fallback.
    return Polygon([(0, 0),(width, 0),(width, height),(0, height)])

def _faces_in_polygon(faces,polygon):
    """Keep faces whose center is inside ROI."""

    inside = []

    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if polygon.contains(Point(cx, cy)):
            inside.append(face)
    return inside

# A0042 PROCESS ENTRYPOINT

@alert_code_decorator(alert_code=age_gender_alert_code)
def process_age_gender_detection(identifier, frame, result, shared_dict, **kwargs):
    """A0042 process entrypoint."""

    alert_details = kwargs["camera_alert_detail"][identifier]
    # Detection UUID
    if (shared_dict[identifier].get("detection_uuid") is None):
        detection_uuid = str(uuid.uuid4())
        shared_dict[identifier]["detection_uuid"] = detection_uuid
    else:
        detection_uuid = (shared_dict[identifier]["detection_uuid"])

    # Logger
    logger = PrefixedLogger(base_logger, age_gender_alert_code, identifier,(kwargs.get("camera_id") or alert_details.get( "camera_id" )), detection_uuid,alert_details.get("branch_id"))

    logger.set_model("age_gender", False)

    # ROI
    polygon = _parse_polygon(alert_details.get("area_to_monitor"), frame)

    # Load models
    try:
        (
            face_app,
            mivolo_config,
            mivolo_processor,
            mivolo_model,
            mivolo_device,
        ) = _get_models(logger)

    except Exception as e:
        logger.error(f"{detection_uuid} Failed to load A0042 models: {e}")
        logger.show_models()
        return

    # Detect faces
    with _models_lock:
        faces = _faces_in_polygon(_detect_faces(frame, face_app), polygon)
        results = []
        
        for face_id, face in enumerate(faces,start=1):
            x1, y1, x2, y2 = (face["bbox"])
            crop = frame[y1:y2, x1:x2]

            pred = _predict_age_gender(
                crop,
                mivolo_config,
                mivolo_processor,
                mivolo_model,
                mivolo_device,
            )
            if not pred:
                continue
            # Ensure age + gender are available.
            if (pred.get("age") is None or not pred.get("gender")):
                continue
            item = {
                "id": face_id,
                "bbox": face["bbox"],
                "age": pred["age"],
                "age_group": pred["age_group"],
                "gender": pred["gender"],
                "gender_conf": pred.get("gender_conf", 0.0),
                "face_conf": face["confidence"],
            }
            results.append(item)
            
            # Evidence annotation
            cv2.rectangle(frame, (x1, y1),(x2, y2), (0, 200, 0), 2)
            age = item.get("age")
            age_s = (
                f"{age:.0f}y"
                if isinstance(age, (int, float))
                else "?"
            )
            age_group = (item.get("age_group") or "?")
            gender = (item.get("gender") or "?")
            label = (f"MiVOLO {gender} {age_s} {age_group}")
            (text_w, text_h,), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            text_y = max(text_h + 8, y1 - 8)
            cv2.rectangle(frame, (x1, text_y - text_h - 8), (x1 + text_w + 8, text_y + 4), (0, 200, 0), -1)
            cv2.putText(frame, label, ( x1 + 4, text_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    # Logging
    logger.info(f"{detection_uuid} faces={len(faces)} classified={len(results)}")

    # No detections
    if not results:
        logger.show_models()
        return

    # Save detection
    kwargs["head_count"] = len(results)
    _, encoded_frame = cv2.imencode(".jpg", frame)

    if (encoded_frame is None or getattr(encoded_frame, "size", 0) == 0):
        logger.warning(f"{detection_uuid} encoded frame invalid, skip alert send")
        logger.show_models()
        return

    save_detection(identifier, [encoded_frame], shared_dict, **kwargs)
    update_detection_after_time(identifier, shared_dict, **kwargs)
    shared_dict[identifier]["detection_uuid"] = None
    logger.set_model("age_gender", True)
    logger.info(f"{detection_uuid} saved MiVOLO age/gender alert for {len(results)} face(s)")
    logger.show_models()