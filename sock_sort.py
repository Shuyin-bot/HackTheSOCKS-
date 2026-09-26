"""Sort 4 socks by color using the recorded arm motions in recordings/.

Run:  python sock_sort.py [--camera 0]      # wrist-camera colour classifier
      python sock_sort.py --placeholder     # no camera, cycles colours

For each slot (top_right, bottom_right, top_left, bottom_left):
  down_<slot> -> get_color(slot) -> pickup_<slot> -> drop_<bin for that color>

The color source is a plain function returning one of COLORS; anything else
("empty", "unknown", None) skips the slot without picking up. Unknown socks
are saved to colour_classifier/captures/unknown/ for re-teaching.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from colour_classifier import UNKNOWN, ArmCamera, CloseUpClassifier
from record_replay import (
    FOLLOWER_ID,
    FOLLOWER_PORT,
    SO101Follower,
    SO101FollowerConfig,
    _rec_path,
    replay,
)

SLOTS = ("top_right", "bottom_right", "top_left", "bottom_left")  # right side first
COLORS = ("blue", "white", "black", "orange")  # same labels as colour_classifier/colours_closeup.json
COLOR_TO_DROP = {
    "blue": "top_left",
    "white": "top_right",
    "black": "bottom_left",
    "orange": "bottom_right",
}

ColorFn = Callable[[str], Optional[str]]  # slot -> one of COLORS, or None if unsure
RETRY_PAUSE_S = 0.5
UNKNOWN_DIR = Path(__file__).resolve().parent / "colour_classifier" / "captures" / "unknown"


_placeholder_cycle = itertools.cycle(COLORS)


def placeholder_get_color(slot: str) -> str:
    """Stand-in until the camera works: cycles COLORS so a run hits every bin."""
    return next(_placeholder_cycle)


def log_unknown(slot: str, frames: list, clf: CloseUpClassifier, dist: float) -> None:
    """Save an unrecognised sock's picture + measured colour so it can be taught later."""
    UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    img = UNKNOWN_DIR / f"{slot}_{stamp}.png"
    cv2.imwrite(str(img), frames[len(frames) // 2])
    lab = np.median([clf.lab(f, rgb=False) for f in frames], axis=0)
    entry = {"time": stamp, "slot": slot, "image": img.name, "distance": round(float(dist), 2),
             "lab": [round(float(v), 2) for v in lab]}
    with open(UNKNOWN_DIR / "unknown_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"logged unknown sock -> {img}")


def make_camera_get_color(cam: ArmCamera) -> ColorFn:
    """Wrist-camera colour via colour_classifier; retries once on 'unknown'
    since the arm may still be settling right after the down_* motion.
    A reading that stays unknown is saved for training."""
    clf = CloseUpClassifier()

    def read() -> tuple[str, float, list]:
        frames = cam.frames()
        colour, dist = clf.classify_detailed(frames, rgb=False)
        return colour, dist, frames

    def get_color(slot: str) -> str:
        colour, dist, frames = read()
        if colour == UNKNOWN:
            time.sleep(RETRY_PAUSE_S)
            colour, dist, frames = read()
        print(f"camera: {colour} (distance {dist:.1f})")
        if colour == UNKNOWN:
            log_unknown(slot, frames, clf, dist)
        return colour
    return get_color


def required_recordings() -> list[str]:
    names = [f"{step}_{slot}" for slot in SLOTS for step in ("down", "pickup")]
    names += [f"drop_{slot}" for slot in COLOR_TO_DROP.values()]
    return names


def run(follower, get_color: ColorFn = placeholder_get_color) -> list[tuple[str, str, str]]:
    """Sort every slot; returns (slot, color, bin) rows for the summary."""
    summary = []
    for slot in SLOTS:
        print(f"\n=== slot {slot} ===")
        replay(follower, f"down_{slot}")

        color = get_color(slot)
        if color not in COLOR_TO_DROP:
            print(f"⚠ unknown color {color!r} — skipping {slot}")
            summary.append((slot, str(color), "skipped"))
            continue
        target = COLOR_TO_DROP[color]
        print(f"color: {color} -> drop_{target}")

        replay(follower, f"pickup_{slot}")
        replay(follower, f"drop_{target}")
        summary.append((slot, color, target))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort 4 socks by colour.")
    parser.add_argument("--camera", default="0", help="wrist webcam index or URL (default: 0)")
    parser.add_argument("--placeholder", action="store_true", help="skip the camera, cycle colours")
    args = parser.parse_args()

    missing = [n for n in required_recordings() if not _rec_path(n).exists()]
    if missing:
        print("Missing recordings:\n  " + "\n  ".join(str(_rec_path(n)) for n in missing))
        sys.exit(1)

    # Open the camera before the arm, so a bad index fails before anything moves.
    cam = None
    get_color = placeholder_get_color
    if not args.placeholder:
        cam = ArmCamera(args.camera)
        get_color = make_camera_get_color(cam)

    follower = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID))
    summary = []
    try:
        follower.connect()
        summary = run(follower, get_color=get_color)
    except KeyboardInterrupt:
        print("\n✗ aborted")
    finally:
        if follower.is_connected:
            follower.disconnect()
        if cam is not None:
            cam.close()

    if summary:
        print("\nSummary:")
        for slot, color, target in summary:
            print(f"  {slot:<13} {color:<8} -> {target}")


if __name__ == "__main__":
    main()
