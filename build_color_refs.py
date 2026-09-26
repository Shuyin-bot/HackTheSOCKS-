"""
build_color_refs.py — turn labeled sample photos into calibrated color references.

Usage:
    python build_color_refs.py

Put one photo per color in sock_pics/, named by color, e.g.:
    sock_pics/blue.jpg (or .HEIC — will be auto-converted via macOS `sips`)
    sock_pics/orange.jpg
    sock_pics/black.jpg
    sock_pics/white.jpg

Writes color_refs.json with each label's calibrated (H,S,V), used by
sock_vision.SockColorClassifier at runtime.

Re-run this any time you retake photos under the actual venue lighting —
lighting changes WILL shift these numbers, so calibrate on-site if possible.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

import cv2

from sock_vision import segment_sock, hsv_median

HERE = os.path.dirname(os.path.abspath(__file__))
SOCK_DIR = os.path.join(HERE, "sock_pics")
OUT_PATH = os.path.join(HERE, "color_refs.json")


def ensure_jpg(heic_path):
    jpg_path = os.path.splitext(heic_path)[0] + ".jpg"
    if os.path.exists(jpg_path):
        return jpg_path
    if shutil.which("sips") is None:
        print(f"  ! no HEIC->JPG converter available (need macOS `sips`); skipping {heic_path}")
        return None
    print(f"  converting {os.path.basename(heic_path)} -> {os.path.basename(jpg_path)}")
    subprocess.run(["sips", "-s", "format", "jpeg", heic_path, "--out", jpg_path], check=True)
    return jpg_path


def main():
    if not os.path.isdir(SOCK_DIR):
        print(f"No {SOCK_DIR} folder found. Create it and add labeled photos, e.g. blue.jpg")
        sys.exit(1)

    heics = glob.glob(os.path.join(SOCK_DIR, "*.HEIC")) + glob.glob(os.path.join(SOCK_DIR, "*.heic"))
    for h in heics:
        ensure_jpg(h)

    jpgs = sorted(
        set(glob.glob(os.path.join(SOCK_DIR, "*.jpg")) + glob.glob(os.path.join(SOCK_DIR, "*.jpeg")))
    )
    if not jpgs:
        print(f"No .jpg/.jpeg photos found in {SOCK_DIR}.")
        sys.exit(1)

    refs = {}
    print(f"Calibrating from {len(jpgs)} sample photo(s) in {SOCK_DIR}:\n")
    for path in jpgs:
        label = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        if img is None:
            print(f"  ! could not read {path}, skipping")
            continue

        mask, small, contour = segment_sock(img)
        if mask is None:
            print(f"  ! could not find a sock blob in {path}, skipping")
            continue

        hsv = hsv_median(small, mask)
        area_pct = 100.0 * cv2.contourArea(contour) / (small.shape[0] * small.shape[1])
        refs[label] = {"hsv": hsv, "sample_area_pct": round(area_pct, 1)}
        print(
            f"  {label:10s} H={hsv[0]:5.1f} S={hsv[1]:5.1f} V={hsv[2]:5.1f}"
            f"   (sock covered {area_pct:4.1f}% of frame)"
        )

    if not refs:
        print("\nNo references calibrated — nothing written.")
        sys.exit(1)

    with open(OUT_PATH, "w") as f:
        json.dump(refs, f, indent=2)

    print(f"\nWrote {len(refs)} color reference(s) -> {OUT_PATH}")
    print("Available colors:", ", ".join(refs.keys()))
    print("\nNext: run `python test_classifier.py` to sanity-check the classifier.")


if __name__ == "__main__":
    main()
