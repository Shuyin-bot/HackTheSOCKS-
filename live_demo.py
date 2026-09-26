"""
live_demo.py — real-time sock color detection loop.

This is the piece that gets handed off to whoever is running the camera
on the arm-control machine. Two input modes:

  --webcam 0                     use a local/USB webcam (index 0, 1, ...)
  --phone-url http://IP:PORT/photoaf.jpg   use a phone IP-webcam (see ip_cam.py)

Two output modes (combine as needed):
  --show                         pop up a cv2 window with the sock outlined + labeled
  --post-url http://host:port/pick   POST {"color": "<label>"} once a color is
                                  confidently detected (debounced), so the
                                  arm-control script can trigger pick_and_drop()

Examples:
    # local webcam, just show the overlay window
    python live_demo.py --webcam 0 --show

    # phone camera, snap every 1.5s, POST result to teammate's arm server
    python live_demo.py --phone-url http://172.16.101.165:8080/photoaf.jpg \\
        --interval 1.5 --post-url http://localhost:5000/pick
"""
import argparse
import sys
import time

import cv2
import numpy as np
import requests

from sock_vision import RollingVote, SockColorClassifier, draw_result


def get_frame_from_phone(url, timeout=10):
    img_bytes = requests.get(url, timeout=timeout).content
    return cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refs", default="color_refs.json", help="path to calibrated color_refs.json")
    p.add_argument("--webcam", type=int, default=None, help="local webcam index, e.g. 0")
    p.add_argument("--phone-url", default=None, help="phone IP-webcam photo URL")
    p.add_argument("--post-url", default=None, help="POST {'color': label} here once debounced")
    p.add_argument("--debounce", type=int, default=3, help="consecutive matching frames needed to fire")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between captures (phone mode only)")
    p.add_argument("--show", action="store_true", help="show a live cv2 preview window")
    args = p.parse_args()

    if args.webcam is None and not args.phone_url:
        p.error("need --webcam N or --phone-url URL")

    clf = SockColorClassifier.from_json(args.refs)
    vote = RollingVote(need=args.debounce)

    cap = cv2.VideoCapture(args.webcam) if args.webcam is not None else None
    if cap is not None and not cap.isOpened():
        print(f"Could not open webcam index {args.webcam}")
        sys.exit(1)

    print(f"Loaded colors: {clf.labels}")
    print("Ctrl+C to quit.\n")

    try:
        while True:
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    print("  ! webcam read failed, retrying")
                    time.sleep(0.5)
                    continue
            else:
                try:
                    frame = get_frame_from_phone(args.phone_url)
                except Exception as e:
                    print(f"  ! phone capture failed: {e}")
                    time.sleep(args.interval)
                    continue

            result = clf.classify(frame)
            label = result["label"] if result else None
            fired = vote.push(label) if label else None

            if result:
                print(f"detected={result['label']:10s} confidence={result['confidence']:.2f}", end="")
            else:
                print("no sock detected", end="")

            if fired:
                print(f"   -> FIRE: {fired}")
                if args.post_url:
                    try:
                        requests.post(args.post_url, json={"color": fired}, timeout=5)
                        print(f"   -> posted to {args.post_url}")
                    except Exception as e:
                        print(f"   ! post failed: {e}")
                vote.reset()
            else:
                print()

            if args.show:
                vis = draw_result(result) if result else frame
                cv2.imshow("sock vision", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if cap is None:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
