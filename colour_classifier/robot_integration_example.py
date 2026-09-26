"""Receiving side: the classifier gives a colour, this maps it to a waypoint.

Close-up flow on the robot, for each known sock location:

    from colour_classifier import identify_sock
    from colour_classifier.robot_integration_example import CAMERA, place_waypoint

    go_to(view_pose_for_spot)                       # camera pointing straight at the sock
    colour = identify_sock(robot.get_observation()[CAMERA])
    target = place_waypoint(colour)
    if target is not None:                          # None = empty/unknown: leave it
        go_to(pick_pose_for_spot); grab()
        go_to(target); release()

Try it without the robot, on the sample close-ups:
    python colour_classifier/robot_integration_example.py colour_classifier/samples/closeup_test/*.jpg
"""

import sys
from pathlib import Path

try:
    from .sock_classifier import EMPTY, UNKNOWN, CloseUpClassifier
except ImportError:  # run as a script
    from sock_classifier import EMPTY, UNKNOWN, CloseUpClassifier

CAMERA = "wrist"  # the camera's name in the robot config, i.e. the key in robot.get_observation()

# Where each colour goes; names must match the colours taught with `closeup-teach`.
# Replace the strings with whatever your motion code takes (joint dicts, pose names, ...).
PLACE_WAYPOINTS = {
    "black": "place_black",
    "white": "place_white",
    "blue": "place_blue",
    "orange": "place_orange",
}


def place_waypoint(colour: str):
    """Waypoint for this colour, or None if the sock should be left alone."""
    if colour in (EMPTY, UNKNOWN):
        return None
    if colour not in PLACE_WAYPOINTS:
        print(f"[place] no waypoint for colour '{colour}', leaving it")
        return None
    return PLACE_WAYPOINTS[colour]


if __name__ == "__main__":
    import cv2

    here = Path(__file__).parent
    colours = here / "colours_closeup.json"
    if not colours.exists():  # not taught on this machine yet: fall back to the phone-photo samples
        colours = here / "samples" / "colours_closeup_phone.json"
        print(f"(no colours_closeup.json yet, using {colours.relative_to(here)})")
    clf = CloseUpClassifier(colours_file=colours)
    for path in sys.argv[1:]:
        frame = cv2.imread(path)
        if frame is None:
            print(f"{path}: could not read")
            continue
        colour, dist = clf.classify_detailed(frame, rgb=False)  # cv2.imread is BGR
        print(f"{Path(path).name:18s} {colour:7s} (distance {dist:4.1f}) -> {place_waypoint(colour)}")
