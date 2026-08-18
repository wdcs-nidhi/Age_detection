#!/usr/bin/env python3
"""
Local age + gender.

  python run.py --source /path/to/image.jpg
  python run.py --source /path/to/images/
  python run.py --source /path/to/video.mp4
  python run.py --source rtsp://user:pass@ip:554/stream1
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import yaml

from pipeline import Pipeline

HERE = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_cfg(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    model = cfg.get("detection", {}).get("model")
    if model and (Path(model).suffix or Path(model).exists()):
        parent = HERE.parent / model
        cfg["detection"]["model"] = str(parent if parent.is_file() else HERE / model)
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
        return
    print(f"{label}  faces={len(results)}")
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        gender = r.get("gender") or "?"
        group = r.get("age_group") or "?"
        age = r.get("age")
        age_s = f"{age:.0f}" if age is not None else "?"
        print(
            f"  ID {r['id']}"
            f"  gender={gender} ({r.get('gender_conf', 0):.2f})"
            f"  age={age_s} {group} ({r.get('age_conf', 0):.2f})"
            f"  bbox=[{x1},{y1},{x2},{y2}]"
        )


def save_image(frame, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_detected.jpg"
    cv2.imwrite(str(path), frame)
    print(f"Saved image → {path}")
    return path


WINDOW = "Age Gender Live"


def open_live_window():
    cv2.startWindowThread()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.waitKey(1)


def show_frame(frame) -> bool:
    """Show overlay. Return False if user pressed q."""
    cv2.imshow(WINDOW, frame)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def process_image(pipe: Pipeline, image_path: Path, out_dir: Path, save: bool, display: bool):
    pipe.reset()
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


def process_stream(pipe: Pipeline, source: str, out_path: Path, save_video: bool, save_image_flag: bool, display: bool, skip: int):
    live = is_rtsp(source) or source.isdigit()
    save_video = False
    cap = open_source(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open: {source}")
    print(f"Source: {source}")
    if display:
        open_live_window()
        print("Live window open. Keys: q=quit")

    writer = None
    frame_i = 0
    last = None
    last_results: list[dict] = []
    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            if not live:
                break
            print("RTSP dropped, reconnecting...")
            cap.release()
            time.sleep(1)
            cap = open_source(source)
            continue

        frame_i += 1
        out, results = pipe.process(frame)
        last, last_results = out, results

        if save_video and writer is None:
            h, w = out.shape[:2]
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if src_fps < 1:
                src_fps = 25.0
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                src_fps,
                (w, h),
            )
            print(f"Saving video → {out_path}  fps={src_fps:.1f}")
        if writer is not None:
            writer.write(out)
            for _ in range(max(0, skip)):
                if not cap.grab():
                    break
                writer.write(out)

        if frame_i == 1 or frame_i % 30 == 0:
            fps_now = frame_i / max(time.time() - t0, 1e-6)
            print_results(f"frame={frame_i}  {fps_now:.1f} fps", results)

        if display:
            if not show_frame(out):
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved video → {out_path}")
    if save_image_flag and last is not None:
        save_image(last, out_path.parent, out_path.stem)
    return last_results


def main():
    parser = argparse.ArgumentParser(description="Local age + gender")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--source", default=None)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(Path(args.config))
    if args.source:
        cfg.setdefault("camera", {})["source"] = args.source
    source = str(cfg.get("camera", {}).get("source", ""))
    out_cfg = cfg.get("output", {})
    save_video = bool(out_cfg.get("save_video", False))
    save_img = bool(out_cfg.get("save_image", False))
    display = bool(out_cfg.get("display", True)) and not args.no_display
    out_path = Path(out_cfg.get("path", str(f'{HERE} / output/{str(time.time())}.mp4')))
    skip = int(cfg.get("processing", {}).get("skip_frames", 2))

    pipe = Pipeline(cfg)
    src = Path(source)

    if src.is_dir():
        images = list_images(src)
        if not images:
            raise RuntimeError(f"No images in {src}")
        print(f"Folder: {src}  images={len(images)}")
        n = 0
        for p in images:
            n += len(process_image(pipe, p, out_path.parent, save_img, display))
        print(f"Done. {len(images)} image(s), {n} face(s).")
    elif src.suffix.lower() in IMAGE_EXTS and src.is_file():
        process_image(pipe, src, out_path.parent, save_img, display)
        print("Done.")
    else:
        last = process_stream(pipe, source, out_path, save_video, save_img, display, skip)
        print(f"Done. {len(last)} face(s) on last frame.")

    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
