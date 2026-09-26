# colour_classifier

Tells the robot what colour sock the wrist camera is looking at.

The arm moves to a known sock location, the camera points straight at the sock, and
`identify_sock(frame)` returns its colour. Your code then maps that colour to a place waypoint.
There is no model and no training: it compares the average colour in the middle of the frame
with a few colours you teach it with the real camera (about 10 seconds per sock).

This is separate from `sock_vision.py` in the repo root. That one finds the sock's outline
first; this one reads a fixed box in the middle of the frame.

## 1. Try it (no camera or robot needed)

Needs `opencv-python` and `numpy`. The `lerobot` env from `make` or the `sockvision` env
from `make lite` both have them.

```bash
python colour_classifier/test_sock_classifier.py
```

This checks it against sample photos in `samples/`, taught from one photo and tested on
another where the socks moved and the lighting changed. It should end with `ALL PASSED`.

## 2. Teach the colours with the robot's camera

Do this on the machine connected to the arm. Put the arm in the pose it uses to look at a
sock. Find the camera index with `make find-cameras`.

```bash
python colour_classifier/sock_classifier.py closeup-teach --camera 1
```

A live preview shows the box that gets read. Put a sock under the camera and press **SPACE**,
then type its colour in the window (`black`, `white`, `blue`, `orange`) and press **ENTER**.
Repeat for each sock. Also point the camera at bare table and type `empty`. Press **q** when
you're done.

- Every capture adds a reading to `colours_closeup.json`, and nothing is overwritten. Two or
  three captures per sock, with the sock moved slightly each time, make it more reliable.
- The colour names you type are exactly what `identify_sock` returns. Use the same names in
  your waypoint table.

Check it live. The number next to the colour is the distance: under ~10 is a confident match.

```bash
python colour_classifier/sock_classifier.py closeup-run --camera 1
```

## 3. Use it in the robot code

Run from the repo root (or put the repo root on `PYTHONPATH`):

```python
from colour_classifier import identify_sock

obs = robot.get_observation()          # arm at the spot, camera pointing at the sock
colour = identify_sock(obs["wrist"])   # "wrist" = the camera's name in your robot config
```

`identify_sock` returns a string:

| Value | Meaning | What to do |
|---|---|---|
| a taught name, e.g. `"blue"` | that sock | go to that colour's waypoint |
| `"empty"` | bare table (only if you taught `empty`) | skip this spot |
| `"unknown"` | nothing close enough to any taught colour | skip it, don't guess |

- The input is the frame straight from LeRobot (RGB, `H x W x 3` uint8). Torch tensors,
  CHW and float `[0, 1]` images also work. For frames read directly with `cv2` (BGR), pass
  `identify_sock(frame, rgb=False)`.
- It takes about 2 ms per call. The colours load on the first call.
- If `colours_closeup.json` is missing, it raises `FileNotFoundError` and does not exit, so
  the robot program can catch it.
- To get the distance too, use `CloseUpClassifier().classify_detailed(frame)`, which returns
  `("blue", 7.3)`.
- After re-teaching while the robot program is running, call `CloseUpClassifier().reload()`.

`robot_integration_example.py` shows the receiving side: a `PLACE_WAYPOINTS` table and
`place_waypoint(colour)`, which returns the waypoint, or `None` for empty/unknown. Replace the
placeholder strings with your real waypoints. Try it on the samples:

```bash
python colour_classifier/robot_integration_example.py colour_classifier/samples/closeup_test/*.jpg
```

## Tips

- **Teach with the camera the robot uses, at the pose it uses.** Colours taught from a
  phone photo won't match the wrist camera. `samples/colours_closeup_phone.json` is for the
  self-test only.
- **White sock vs empty white table** is the closest pair. If they get mixed up, put the
  socks on a darker mat and re-teach.
- **Black vs white:** when the frame is filled with one colour, the camera's auto-exposure
  pushes black lighter and white darker. Teaching with the real camera accounts for this,
  but check both in `closeup-run`.
- The camera doesn't need to be perfectly centred. Only the middle of the frame is read
  (`--centre 0.5` = middle 50%).

## Scan mode (optional)

The same file also supports one picture from a fixed "null" pose that shows every spot.
You draw a box per spot once, and `identify_socks(frame)` returns
`{"spot1": "blue", "spot2": "empty", ...}`. It corrects for lighting using a patch of bare
table. See the docstring at the top of `sock_classifier.py`
(`capture` → `spots` → `teach` → `run`).

## Files

| File | What |
|---|---|
| `sock_classifier.py` | the classifier, plus the teach/run commands |
| `robot_integration_example.py` | colour → waypoint mapping example |
| `test_sock_classifier.py` | self-test on the sample photos |
| `colours_closeup.json` | your taught colours (created by `closeup-teach`) |
| `samples/` | sample photos and colours for the self-test |
| `captures/` | pictures saved while teaching (git-ignored) |
