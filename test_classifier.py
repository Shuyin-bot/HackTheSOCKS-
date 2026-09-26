"""
test_classifier.py — sanity-check the calibrated classifier against the
same sample photos it was built from (a "does the pipeline even work"
smoke test, not a real generalization test — do that live, on-site).

Usage:
    python test_classifier.py
"""
import glob
import os
import sys

import cv2

from sock_vision import SockColorClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
SOCK_DIR = os.path.join(HERE, "sock_pics")
REFS_PATH = os.path.join(HERE, "color_refs.json")


def main():
    if not os.path.exists(REFS_PATH):
        print(f"{REFS_PATH} not found. Run `python build_color_refs.py` first.")
        sys.exit(1)

    clf = SockColorClassifier.from_json(REFS_PATH)
    print(f"Loaded {len(clf.labels)} reference colors: {clf.labels}\n")

    jpgs = sorted(glob.glob(os.path.join(SOCK_DIR, "*.jpg")))
    if not jpgs:
        print(f"No .jpg photos found in {SOCK_DIR}.")
        sys.exit(1)

    passed = 0
    for path in jpgs:
        expected = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        result = clf.classify(img)
        if result is None:
            print(f"[FAIL] {expected:10s} -> no sock detected")
            continue
        ok = result["label"] == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] {expected:10s} -> predicted={result['label']:10s} "
            f"confidence={result['confidence']:.2f} margin={result['margin']:.3f}"
        )

    print(f"\n{passed}/{len(jpgs)} correct")
    if passed != len(jpgs):
        sys.exit(1)


if __name__ == "__main__":
    main()
