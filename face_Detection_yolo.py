from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "/home/webclues-nidhi/mYpY/RnD/Age_detectoion/yolov8l-pose.pt"
IMAGE_PATH = "/home/webclues-nidhi/Downloads/data/age"
OUTPUT_DIR = Path("/home/webclues-nidhi/mYpY/RnD/Age_detectoion/output/yolo_pose")

CONF_THRESHOLD = 0.5
KP_CONF = 0.25
PADDING = 0.2

# Frontal face only. Higher = stricter.
MAX_YAW = 0.22      # nose must sit near the eye midpoint
MAX_ROLL = 0.30     # eyes roughly level
MIN_EAR_RATIO = 0.60  # both ears about the same distance from the nose

# YOLO pose (COCO)
# 0 nose, 1 left eye, 2 right eye, 3 left ear, 4 right ear
# 5 left shoulder, 6 right shoulder
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH = 5, 6


def valid_point(kpts, confs, idx):
    if idx >= len(kpts):
        return None
    x, y = float(kpts[idx][0]), float(kpts[idx][1])
    score = float(confs[idx]) if confs is not None and idx < len(confs) else 1.0
    if x <= 1 or y <= 1 or score < KP_CONF:
        return None
    return np.array([x, y], dtype=np.float32)


def is_frontal_face(kpts, confs) -> bool:
    """
    Keep faces looking into the camera. Drop profile / side / turned heads.
    Needs both eyes, nose, and both ears.
    """
    nose = valid_point(kpts, confs, NOSE)
    leye = valid_point(kpts, confs, L_EYE)
    reye = valid_point(kpts, confs, R_EYE)
    lear = valid_point(kpts, confs, L_EAR)
    rear = valid_point(kpts, confs, R_EAR)

    if nose is None or leye is None or reye is None or lear is None or rear is None:
        return False

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

    d_left = float(np.linalg.norm(lear - nose))
    d_right = float(np.linalg.norm(rear - nose))
    farther = max(d_left, d_right)
    if farther < 1:
        return False
    if min(d_left, d_right) / farther < MIN_EAR_RATIO:
        return False

    return True


def face_box_from_keypoints(kpts, confs, img_w, img_h):
    """
    Build a proper face crop from person pose keypoints.

    Min/max of nose/eyes/ears is too small. Size the box from
    eye distance, ear distance, and shoulder width, then place
    it so forehead and chin are inside.
    """
    nose = valid_point(kpts, confs, NOSE)
    leye = valid_point(kpts, confs, L_EYE)
    reye = valid_point(kpts, confs, R_EYE)
    lear = valid_point(kpts, confs, L_EAR)
    rear = valid_point(kpts, confs, R_EAR)
    lsh = valid_point(kpts, confs, L_SH)
    rsh = valid_point(kpts, confs, R_SH)

    head = [p for p in (nose, leye, reye, lear, rear) if p is not None]
    if len(head) < 2:
        return None

    eyes = [p for p in (leye, reye) if p is not None]
    if eyes:
        center = np.mean(eyes, axis=0)
    elif nose is not None:
        center = nose.copy()
    else:
        center = np.mean(head, axis=0)

    sizes = []
    if leye is not None and reye is not None:
        iod = float(np.linalg.norm(leye - reye))
        if iod > 2:
            sizes.append(iod * 2.6)
    if lear is not None and rear is not None:
        ear_d = float(np.linalg.norm(lear - rear))
        if ear_d > 2:
            sizes.append(ear_d * 1.15)
    if leye is not None and lear is not None:
        sizes.append(float(np.linalg.norm(leye - lear)) * 3.0)
    if reye is not None and rear is not None:
        sizes.append(float(np.linalg.norm(reye - rear)) * 3.0)
    if lsh is not None and rsh is not None:
        shoulder = float(np.linalg.norm(lsh - rsh))
        if shoulder > 2:
            sizes.append(shoulder * 0.42)
    if nose is not None and lsh is not None and rsh is not None:
        neck = float(np.linalg.norm(nose - (lsh + rsh) / 2.0))
        if neck > 2:
            sizes.append(neck * 0.9)

    if not sizes:
        arr = np.stack(head)
        bw = float(arr[:, 0].max() - arr[:, 0].min())
        bh = float(arr[:, 1].max() - arr[:, 1].min())
        sizes.append(max(bw, bh, 20.0) * 2.4)

    face_w = max(sizes)
    face_h = face_w * 1.25

    # Eyes sit in the upper third. Shift down so chin is included.
    if eyes:
        center[1] = float(np.mean([p[1] for p in eyes])) + 0.22 * face_h
    elif nose is not None:
        center[1] = nose[1] + 0.10 * face_h

    face_w *= 1.0 + PADDING
    face_h *= 1.0 + PADDING

    x1 = int(center[0] - face_w / 2.0)
    y1 = int(center[1] - face_h / 2.0)
    x2 = int(center[0] + face_w / 2.0)
    y2 = int(center[1] + face_h / 2.0)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return x1, y1, x2, y2


def detect_faces(model, image, conf=CONF_THRESHOLD, device="cpu", imgsz=320, half=None):
    """
    Same face boxes used by the folder script and the age pipeline.
    Returns [{bbox, confidence, crop}, ...]
    """
    if half is None:
        half = str(device) not in ("cpu", "CPU")
    h, w = image.shape[:2]
    results = model.predict(
        source=image,
        conf=conf,
        device=device,
        imgsz=imgsz,
        half=half,
        max_det=10,
        verbose=False,
    )
    faces = []
    for result in results:
        if result.keypoints is None:
            continue
        xy = result.keypoints.xy.cpu().numpy()
        confs = None
        if result.keypoints.conf is not None:
            confs = result.keypoints.conf.cpu().numpy()
        box_conf = None
        if result.boxes is not None and result.boxes.conf is not None:
            box_conf = result.boxes.conf.cpu().numpy()

        for person_id, person_kp in enumerate(xy):
            person_conf = confs[person_id] if confs is not None else None
            if not is_frontal_face(person_kp, person_conf):
                continue
            box = face_box_from_keypoints(person_kp, person_conf, w, h)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            score = 1.0
            if box_conf is not None and person_id < len(box_conf):
                score = float(box_conf[person_id])
            faces.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": score,
                    "crop": crop,
                }
            )
    return faces


def process_image(model, image_path: Path, out_dir: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Unable to read image: {image_path}")

    vis = image.copy()
    faces = detect_faces(model, image)
    crops = []

    for person_id, face in enumerate(faces):
        x1, y1, x2, y2 = face["bbox"]
        crop = face["crop"]
        crop_path = out_dir / f"{image_path.stem}_face_{person_id}.jpg"
        cv2.imwrite(str(crop_path), crop)
        crops.append(crop_path)
        print(f"Saved crop → {crop_path}  bbox=[{x1},{y1},{x2},{y2}]")

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"Person {person_id}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    vis_path = out_dir / f"{image_path.stem}_detected.jpg"
    cv2.imwrite(str(vis_path), vis)
    print(f"Saved vis  → {vis_path}  faces={len(crops)}")
    return crops


def main():
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)
    src = Path(IMAGE_PATH)

    if src.is_dir():
        images = sorted(
            p
            for p in src.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not images:
            raise RuntimeError(f"No images in folder: {src}")
        print(f"Folder: {src}")
        print(f"Images: {len(images)}")
        for image_path in images:
            process_image(model, image_path, out_dir)
        return

    process_image(model, src, out_dir)


if __name__ == "__main__":
    main()
