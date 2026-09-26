"""Self-test on the bundled sample photos; needs no camera or robot.

    python colour_classifier/test_sock_classifier.py

Colours were taught from photo 1 (samples/closeup_teach, samples/scan_photo1.jpg)
and are tested on photo 2, where the socks moved and the lighting is warmer.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sock_classifier import CloseUpClassifier, SockClassifier  # noqa: E402

SAMPLES = Path(__file__).parent / "samples"


def check(name: str, got: str, expected: str, dist: float) -> bool:
    ok = got == expected
    print(f"  {name:18s} {got:8s} distance {dist:4.1f}   {'OK' if ok else 'WRONG, expected ' + expected}")
    return ok


def test_closeup() -> bool:
    print("Close-up mode (camera pointing at one sock):")
    clf = CloseUpClassifier(colours_file=SAMPLES / "colours_closeup_phone.json")
    ok = True
    for path in sorted((SAMPLES / "closeup_test").glob("*.jpg")):
        bgr = cv2.imread(str(path))
        colour, dist = clf.classify_detailed(bgr, rgb=False)
        ok &= check(path.name, colour, path.stem.split("_")[0], dist)
        # the robot passes RGB frames (LeRobot default); must give the same answer
        ok &= clf.classify(np.ascontiguousarray(bgr[..., ::-1])) == colour
    return ok


def test_scan() -> bool:
    print("Scan mode (one picture, several fixed spots):")
    clf = SockClassifier(spots_file=SAMPLES / "scan_spots.json", colours_file=SAMPLES / "colours_scan_phone.json")
    expected = {"spot1": "blue", "spot2": "black", "spot3": "orange",
                "spot4": "white", "spot5": "empty", "spot6": "orange"}
    ok = True
    for spot, (colour, dist) in clf.classify_detailed(cv2.imread(str(SAMPLES / "scan_photo2.jpg")), rgb=False).items():
        ok &= check(spot, colour, expected[spot], dist)
    return ok


if __name__ == "__main__":
    results = [test_closeup(), test_scan()]
    print("\nALL PASSED" if all(results) else "\nSOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)
