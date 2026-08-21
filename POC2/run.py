#!/usr/bin/env python3
"""
Camera / Image → InsightFace buffalo_l → crop → MiVOLO v2 age

  python run.py --source /path/to/image.jpg
  python run.py --source /path/to/images/
  python run.py --source /path/to/video.mp4
  python run.py --source 0
  python run.py --source rtsp://user:pass@ip/stream1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from pipeline import Pipeline

HERE = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_cfg(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    ckpt = cfg.get("age", {}).get("checkpoint")
    if ckpt and not Path(ckpt).is_absolute():
        local = (HERE / ckpt).resolve()
        if local.exists():
            cfg["age"]["checkpoint"] = str(local)
    out = cfg.get("output", {}).get("path")
    if out and not Path(out).is_absolute():
        cfg["output"]["path"] = str(HERE / out)
    return cfg


def is_rtsp(source: str) -> bool:
    return source.lower().startswith(("rtsp://", "rtsps://"))


def open_source(source: str):
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    if is_rtsp(source):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return cv2.VideoCapture(source)


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def print_results(label: str, results: list[dict]) -> None:
    if not results:
        print(f"{label}  faces=0")
        return
    print(f"{label}  faces={len(results)}")
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        age = r.get("age")
        age_s = f"{age:.0f}" if isinstance(age, (int, float)) else "?"
        print(
            f"  ID {r['id']}"
            f"  gender={r.get('gender') or '?'}"
            f"  age={age_s} {r.get('age_group') or '?'}"
            f"  bbox=[{x1},{y1},{x2},{y2}]"
        )


def save_image(frame, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_detected.jpg"
    cv2.imwrite(str(path), frame)
    print(f"Saved image → {path}")
    return path


def face_payload(r: dict) -> dict:
    return {
        "id": r.get("id"),
        "bbox": list(r.get("bbox") or []),
        "face_conf": r.get("face_conf"),
        "age": r.get("age"),
        "age_group": r.get("age_group"),
        "gender": r.get("gender"),
        "gender_conf": r.get("gender_conf"),
    }


def save_all_json(items: list[dict], json_path: Path) -> Path:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pipeline": "buffalo_l → crop → MiVOLO v2",
        "count": len(items),
        "results": items,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved json  → {json_path}")
    return json_path


WINDOW = "Buffalo + MiVOLO"


def open_live_window():
    cv2.startWindowThread()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.waitKey(1)


def show_frame(frame) -> bool:
    cv2.imshow(WINDOW, frame)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def process_image(pipe: Pipeline, image_path: Path, out_dir: Path, save: bool, display: bool):
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Skip: {image_path}")
        return []
    out, results = pipe.process(frame)
    print_results(image_path.name, results)
    if save:
        save_image(out, out_dir, image_path.stem)
    if display:
        open_live_window()
        show_frame(out)
        cv2.waitKey(1)
    return results


def process_stream(pipe: Pipeline, source: str, out_path: Path, save_image_flag: bool, display: bool):
    live = is_rtsp(source) or source.isdigit()
    cap = open_source(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open: {source}")
    print(f"Source: {source}")
    if display:
        open_live_window()
        print("Live window. Press q to quit.")

    frame_i = 0
    last = None
    last_results: list[dict] = []
    collected: list[dict] = []
    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            if not live:
                break
            print("Stream dropped, reconnecting...")
            cap.release()
            time.sleep(1)
            cap = open_source(source)
            continue

        frame_i += 1
        out, results = pipe.process(frame)
        last, last_results = out, results

        if frame_i == 1 or frame_i % 30 == 0:
            fps_now = frame_i / max(time.time() - t0, 1e-6)
            print_results(f"frame={frame_i}  {fps_now:.1f} fps", results)
            collected.append(
                {
                    "source": source,
                    "frame": frame_i,
                    "faces": [face_payload(r) for r in results],
                }
            )

        if display and not show_frame(out):
            break

    cap.release()
    if save_image_flag and last is not None:
        save_image(last, out_path.parent, out_path.stem)
    if not collected:
        collected.append(
            {
                "source": source,
                "frame": frame_i,
                "faces": [face_payload(r) for r in last_results],
            }
        )
    return last_results, collected


def main():
    parser = argparse.ArgumentParser(description="buffalo_l → MiVOLO v2 age")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--source", default=None)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(Path(args.config))
    if args.source:
        cfg.setdefault("camera", {})["source"] = args.source
    source = str(cfg.get("camera", {}).get("source", "0"))
    out_cfg = cfg.get("output", {})
    save_img = bool(out_cfg.get("save_image", True))
    save_json_flag = bool(out_cfg.get("save_json", True))
    display = bool(out_cfg.get("display", True)) and not args.no_display
    out_path = Path(out_cfg.get("path", str(HERE / "output" / "result.mp4")))
    json_name = str(out_cfg.get("json_path", "results.json"))
    json_path = Path(json_name) if Path(json_name).is_absolute() else out_path.parent / json_name

    pipe = Pipeline(cfg)
    src = Path(source)
    all_items: list[dict] = []

    if src.is_dir():
        images = list_images(src)
        if not images:
            raise RuntimeError(f"No images in {src}")
        print(f"Folder: {src}  images={len(images)}")
        n = 0
        for p in images:
            results = process_image(pipe, p, out_path.parent, save_img, display)
            n += len(results)
            all_items.append(
                {"source": str(p), "faces": [face_payload(r) for r in results]}
            )
        print(f"Done. {len(images)} image(s), {n} face(s).")
    elif src.suffix.lower() in IMAGE_EXTS and src.is_file():
        results = process_image(pipe, src, out_path.parent, save_img, display)
        all_items.append(
            {"source": str(src), "faces": [face_payload(r) for r in results]}
        )
        print("Done.")
    else:
        last, collected = process_stream(pipe, source, out_path, save_img, display)
        all_items.extend(collected)
        print(f"Done. {len(last)} face(s) on last frame.")

    if save_json_flag:
        save_all_json(all_items, json_path)

    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
