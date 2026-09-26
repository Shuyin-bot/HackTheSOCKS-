"""Sock colour classifier. Two modes:

CLOSE-UP (current plan): the arm moves to each known spot and the wrist camera
points straight at one sock. We read the middle of the frame and return one colour.

    from sock_classifier import ArmCamera, read_sock_colour
    cam = ArmCamera(0)                     # wrist webcam index (or IP-cam URL); open once
    colour = read_sock_colour(cam)         # arm at the spot -> "blue" / "empty" / "unknown"

  or, if LeRobot already owns the camera:
    colour = identify_sock(robot.get_observation()["wrist"])

  Setup (with the real wrist camera, at the real close-up pose):
    python sock_classifier.py closeup-teach --camera 1   # SPACE per sock, type its colour
    python sock_classifier.py closeup-run   --camera 1   # live check

SCAN (kept for later): the arm parks at a "scan" (null) pose where the wrist camera sees every sock
spot. Because the pose is repeatable, each spot is a fixed box in the image,
and we only need to say which colour sits in each box.

Robot integration (same Python process as the LeRobot code):

    from sock_classifier import identify_socks
    obs = robot.get_observation()          # arm parked at the scan pose
    socks = identify_socks(obs["wrist"])   # LeRobot frames are RGB (the default here)
    # -> {"spot1": "black", "spot2": "white", "spot3": "empty", "spot4": "unknown"}

Each value is a taught colour name, "empty" (if taught), or "unknown" (nothing
close enough, so don't act on it). Spots come back in a fixed order (spot1, spot2, ...).

Setup workflow (arm parked at the scan pose each time; pass the same
--width/--height as the robot's camera config):
  python sock_classifier.py capture --camera 1           # align camera, SPACE saves a picture
  python sock_classifier.py spots --image captures/latest.png   # draw a box per spot + bare table
  python sock_classifier.py teach --image captures/latest.png   # type the colour in each spot
  python sock_classifier.py run   --camera 1             # live {spot: colour}

--spots other.json uses a different set of spots (e.g. for a test photo).
`spots` / `teach` with --camera instead of --image open the same live preview
first. `teach` before `spots` exists (or with --free) lets you box each sock by
hand, so colours can be taught before the spots are decided.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
SPOTS_FILE = HERE / "spots.json"
COLOURS_FILE = HERE / "colours.json"
CLOSEUP_COLOURS_FILE = HERE / "colours_closeup.json"
# Used until `closeup-teach` has been run on this machine: taught from phone photos of our socks.
SAMPLE_CLOSEUP_COLOURS_FILE = HERE / "samples" / "colours_closeup_phone.json"
CAPTURE_DIR = HERE / "captures"

# Only the centre of each box is used, so small pose errors don't pull in the table.
SHRINK = 0.2
# A reading is "unknown" (not confident) when either:
#  - it's farther than MAX_DIST (Lab delta-E) from every taught colour, or
#  - the best match isn't clearly better than the runner-up: best distance must be
#    at most MAX_RATIO x the second-best (0.6 = runner-up at least ~1.7x farther).
MAX_DIST = 25.0
MAX_RATIO = 0.6
# Every frame is rescaled so the bare-table box reads this grey, cancelling
# exposure / white-balance drift (otherwise a dim white sock looks like the table).
TABLE_GREY = 180.0
# Close-up mode reads a centred box this fraction of the frame's width/height.
CENTRE = 0.5
# Cheap webcam handling: frames per reading (averaged + voted), seconds to let
# auto-exposure settle after opening (first frames are dark), and stale frames
# dropped before each reading (so we don't read what the camera saw mid-move).
N_FRAMES = 5
WARMUP_S = 1.5
FLUSH_FRAMES = 5
# Windows are shown at most this wide so they fit on a laptop screen.
DISPLAY_WIDTH = 1280

UNKNOWN, EMPTY = "unknown", "empty"
GREEN, ORANGE, YELLOW = (0, 220, 0), (0, 160, 255), (0, 255, 255)


# ── colour maths ────────────────────────────────────────────────────────────

def crop(frame: np.ndarray, box: list[float]) -> np.ndarray:
    """Centre part of a normalised (x, y, w, h) box."""
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    x0 = int((x + bw * SHRINK / 2) * w)
    y0 = int((y + bh * SHRINK / 2) * h)
    x1 = int((x + bw * (1 - SHRINK / 2)) * w)
    y1 = int((y + bh * (1 - SHRINK / 2)) * h)
    return frame[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]


def table_scale(frame_bgr: np.ndarray, spots: dict) -> np.ndarray:
    """Per-channel gain that makes the bare-table box read TABLE_GREY."""
    table = np.median(crop(frame_bgr, spots["_table"]).reshape(-1, 3), axis=0).astype(np.float32)
    return TABLE_GREY / np.maximum(table, 1.0)


def spot_lab(frame_bgr: np.ndarray, box: list[float], scale: np.ndarray) -> np.ndarray:
    """Median CIE-Lab colour of a spot after lighting correction."""
    patch = np.clip(crop(frame_bgr, box).astype(np.float32) * scale, 0, 255)
    lab = cv2.cvtColor(patch / 255.0, cv2.COLOR_BGR2LAB)
    return np.median(lab.reshape(-1, 3), axis=0)


def sock_spots(spots: dict) -> dict:
    return {k: v for k, v in spots.items() if not k.startswith("_")}


def to_bgr_uint8(image, rgb: bool) -> np.ndarray:
    """Accept numpy or torch, HWC or CHW, uint8 or float in [0, 1]; return BGR uint8 HWC."""
    if hasattr(image, "detach"):  # torch tensor
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4 and image.shape[0] == 1:  # batch of one
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected a colour image, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:  # CHW -> HWC
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if rgb else image


def ranked(lab: np.ndarray, refs: dict[str, np.ndarray]) -> list[tuple[float, str]]:
    """[(distance, colour), ...] from closest to farthest taught colour."""
    return sorted((float(np.linalg.norm(r - lab, axis=1).min()), c) for c, r in refs.items())


def nearest(lab: np.ndarray, refs: dict[str, np.ndarray], max_dist: float = MAX_DIST,
            max_ratio: float = MAX_RATIO) -> tuple[str, float]:
    """Closest taught colour and its distance, or "unknown" if it isn't a confident match:
    too far from every taught colour, or almost as close to a second colour."""
    ranking = ranked(lab, refs)
    best_d, best = ranking[0]
    if best_d > max_dist:
        return UNKNOWN, best_d
    if len(ranking) > 1 and best_d > max_ratio * ranking[1][0]:
        return UNKNOWN, best_d
    return best, best_d


def describe(lab: np.ndarray, refs: dict[str, np.ndarray], colour: str, dist: float) -> str:
    """Human-readable result; for "unknown" also says which colours it was torn between."""
    if colour != UNKNOWN:
        return f"{colour} ({dist:.0f})"
    r = ranked(lab, refs)
    return f"unknown: {r[0][1]} {r[0][0]:.0f}" + (f" / {r[1][1]} {r[1][0]:.0f}" if len(r) > 1 else "")


def centre_box(frac: float = CENTRE) -> list[float]:
    return [(1 - frac) / 2, (1 - frac) / 2, frac, frac]


def load_json(path: Path, hint: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {hint}")
    return json.loads(path.read_text())


class SockClassifier:
    """Loads spots.json + colours.json once; classify() is then a few milliseconds per frame."""

    def __init__(self, spots_file=SPOTS_FILE, colours_file=COLOURS_FILE, max_dist=MAX_DIST, max_ratio=MAX_RATIO):
        self.spots_file, self.colours_file, self.max_dist = Path(spots_file), Path(colours_file), max_dist
        self.max_ratio = max_ratio
        self.reload()

    def reload(self):
        """Re-read the files, e.g. after re-running `teach` while the robot program is running."""
        self.spots = load_json(self.spots_file, "Run `sock_classifier.py spots` first.")
        self.colours = load_json(self.colours_file, "Run `sock_classifier.py teach` first.")
        if "_table" not in self.spots:
            raise ValueError(f"{self.spots_file.name} has no bare-table box. Re-run `sock_classifier.py spots`.")
        if not self.colours:
            raise ValueError(f"{self.colours_file.name} has no colours. Run `sock_classifier.py teach`.")
        self.refs = {c: np.array(r, dtype=np.float32) for c, r in self.colours.items()}

    @property
    def colour_names(self) -> list[str]:
        """Every label classify() can return besides "unknown"."""
        return sorted(self.colours)

    def _check_size(self, frame: np.ndarray):
        if "_size" not in self.spots:
            return
        sw, sh = self.spots["_size"]
        h, w = frame.shape[:2]
        if abs(w / h - sw / sh) > 0.02:
            raise ValueError(f"Camera frame is {w}x{h} but spots were drawn on a {sw}x{sh} picture, so the "
                             "boxes would land in the wrong place. Set the robot camera to the same "
                             "resolution, or redo `spots` with --width/--height matching the robot config.")

    def classify_detailed(self, image, rgb: bool = True) -> dict[str, tuple[str, float]]:
        """Return {spot: (colour, distance)}; distance < ~10 is a confident match."""
        frame = to_bgr_uint8(image, rgb)
        self._check_size(frame)
        scale = table_scale(frame, self.spots)
        out = {}
        for name, box in sock_spots(self.spots).items():
            out[name] = nearest(spot_lab(frame, box, scale), self.refs, self.max_dist, self.max_ratio)
        return out

    def classify(self, image, rgb: bool = True) -> dict[str, str]:
        """Return {spot: colour} for every spot. `rgb=False` for raw OpenCV (BGR) frames."""
        return {k: v[0] for k, v in self.classify_detailed(image, rgb).items()}


_default: SockClassifier | None = None


def identify_socks(image, rgb: bool = True) -> dict[str, str]:
    """Robot entry point: camera frame at the scan pose -> {spot: colour}.

    `image` is what LeRobot gives you in robot.get_observation()[camera_name]
    (RGB, HxWx3 uint8). Torch tensors / CHW / float images also work.
    Pass rgb=False only for frames read directly with cv2 (BGR).
    """
    global _default
    if _default is None:
        _default = SockClassifier()
    return _default.classify(image, rgb)


class CloseUpClassifier:
    """Close-up mode: the camera points straight at one sock; returns its colour.

    No lighting correction here (there's no bare table in view), so teach the
    colours with the same camera at the same pose the robot uses.
    """

    def __init__(self, colours_file=None, centre=CENTRE, max_dist=MAX_DIST, max_ratio=MAX_RATIO):
        if colours_file is None:
            colours_file = CLOSEUP_COLOURS_FILE
            if not colours_file.exists():
                colours_file = SAMPLE_CLOSEUP_COLOURS_FILE
                print(f"[sock_classifier] No {CLOSEUP_COLOURS_FILE.name} yet, using the sample colours from phone "
                      "photos. Run `closeup-teach` with the arm camera for reliable results.")
        self.colours_file, self.box, self.max_dist = Path(colours_file), centre_box(centre), max_dist
        self.max_ratio = max_ratio
        self.reload()

    def reload(self):
        """Re-read the colours, e.g. after re-running `closeup-teach` while the robot program runs."""
        self.colours = load_json(self.colours_file, "Run `sock_classifier.py closeup-teach` first.")
        if not self.colours:
            raise ValueError(f"{self.colours_file.name} has no colours. Run `sock_classifier.py closeup-teach`.")
        self.refs = {c: np.array(r, dtype=np.float32) for c, r in self.colours.items()}

    @property
    def colour_names(self) -> list[str]:
        """Every label classify() can return besides "unknown"."""
        return sorted(self.colours)

    def lab(self, image, rgb: bool = True) -> np.ndarray:
        return closeup_lab(to_bgr_uint8(image, rgb), self.box)

    def classify_detailed(self, image, rgb: bool = True) -> tuple[str, float]:
        """(colour, distance); distance < ~10 is a confident match. "unknown" when not
        confident: far from every taught colour, or nearly as close to two colours.

        `image` can be one frame or a list of frames. With several frames their colours
        are averaged (beats webcam noise) and each frame votes; if most frames disagree
        with the result (arm still moving, sock half in view) it returns "unknown".
        """
        frames = image if isinstance(image, (list, tuple)) else [image]
        labs = [self.lab(f, rgb) for f in frames]
        colour, dist = nearest(np.median(labs, axis=0), self.refs, self.max_dist, self.max_ratio)
        if len(labs) > 1:
            votes = [nearest(lab, self.refs, self.max_dist, self.max_ratio)[0] for lab in labs]
            if votes.count(colour) <= len(votes) / 2:
                colour = UNKNOWN
        return colour, dist

    def classify(self, image, rgb: bool = True) -> str:
        """Colour of the sock in the middle of the frame(s). `rgb=False` for raw OpenCV (BGR) frames."""
        return self.classify_detailed(image, rgb)[0]

    def read(self, camera: "ArmCamera", n: int = N_FRAMES) -> tuple[str, float]:
        """Take a fresh multi-frame reading from the webcam: (colour, distance)."""
        return self.classify_detailed(camera.frames(n), rgb=False)


def closeup_lab(frame_bgr: np.ndarray, box: list[float]) -> np.ndarray:
    """Median Lab of the reading box, shrunk to 32x32 first: area-averaging smooths out
    the noise, blur and JPEG blocks of a cheap webcam before we take the colour."""
    patch = cv2.resize(crop(frame_bgr, box), (32, 32), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(patch.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    return np.median(lab.reshape(-1, 3), axis=0)


class ArmCamera:
    """The wrist webcam, read carefully because it's a cheap camera. Frames are BGR.

    source: camera index (0, 1, ...), a video stream URL, or a phone IP-webcam snapshot
    URL ending in .jpg (like ip_cam.py). Open it once and keep it open; opening takes
    WARMUP_S seconds while auto-exposure settles.
    """

    def __init__(self, source=0, width: int | None = None, height: int | None = None, warmup_s: float = WARMUP_S):
        self.source = int(source) if str(source).isdigit() else source
        path = str(self.source).lower().split("?")[0]
        self.snapshot = isinstance(self.source, str) and path.endswith((".jpg", ".jpeg", ".png"))
        self.cap = None
        if not self.snapshot:
            self.cap = cv2.VideoCapture(self.source)
            if width and height:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera {self.source}. Check the index (`make find-cameras`).")
            end = time.time() + warmup_s
            while time.time() < end:
                self.cap.read()
        self.read()  # fail now rather than mid-run

    def read(self) -> np.ndarray:
        """Next frame (BGR)."""
        frame = None
        if self.snapshot:
            import requests
            data = requests.get(self.source, timeout=10).content
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        else:
            ok, frame = self.cap.read()
            frame = frame if ok else None
        if frame is None:
            raise RuntimeError(f"No frame from camera {self.source}")
        return frame

    def frames(self, n: int = N_FRAMES) -> list[np.ndarray]:
        """n fresh frames; frames buffered while the arm was moving are thrown away first."""
        if self.cap is not None:
            for _ in range(FLUSH_FRAMES):
                self.cap.grab()
        return [self.read() for _ in range(n)]

    def close(self):
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


NO_SCALE = np.ones(3, dtype=np.float32)
_default_closeup: CloseUpClassifier | None = None


def _closeup() -> CloseUpClassifier:
    global _default_closeup
    if _default_closeup is None:
        _default_closeup = CloseUpClassifier()
    return _default_closeup


def identify_sock(image, rgb: bool = True) -> str:
    """Robot entry point (close-up mode): camera frame(s) pointing at a sock -> its colour.

    `image` is what LeRobot gives you in robot.get_observation()[camera_name]
    (RGB, HxWx3 uint8), or a list of a few such frames (more reliable on a bad camera).
    Torch tensors / CHW / float images also work.
    Returns a taught colour name, "empty" (if taught), or "unknown" (don't act on it).
    """
    return _closeup().classify(image, rgb)


def read_sock_colour(camera: ArmCamera, n: int = N_FRAMES) -> str:
    """Robot entry point when you own the webcam: fresh n-frame reading -> colour."""
    return _closeup().read(camera, n)[0]


# ── window helpers ──────────────────────────────────────────────────────────

def fit(frame: np.ndarray) -> np.ndarray:
    """Downscale for display so windows fit on screen (boxes are normalised, so this is safe)."""
    h, w = frame.shape[:2]
    if w <= DISPLAY_WIDTH:
        return frame.copy()
    return cv2.resize(frame, (DISPLAY_WIDTH, int(h * DISPLAY_WIDTH / w)), interpolation=cv2.INTER_AREA)


def close_windows():
    # On macOS windows only disappear once the event loop runs again.
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)


def banner(img: np.ndarray, *lines: str) -> np.ndarray:
    """Draw instruction text on a dark strip at the bottom of the image."""
    h, w = img.shape[:2]
    strip = 34 * len(lines) + 12
    img[h - strip:] = (img[h - strip:] * 0.35).astype(img.dtype)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (12, h - strip + 34 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def draw_centre(img: np.ndarray, frac: float, text: str | None = None) -> np.ndarray:
    """Draw the close-up reading box (and a label) on a display-sized image."""
    draw(img, {"sock": centre_box(frac)}, {"sock": (text, 0)} if text else None)
    return img


def draw(img: np.ndarray, spots: dict, labels=None, highlight=None) -> np.ndarray:
    """Draw spot boxes (and optional labels) on a display-sized image."""
    h, w = img.shape[:2]
    boxes = dict(sock_spots(spots))
    if "_table" in spots:
        boxes["table"] = spots["_table"]
    for name, (x, y, bw, bh) in boxes.items():
        colour = ORANGE if name == "table" else YELLOW if name == highlight else GREEN
        p0, p1 = (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h))
        cv2.rectangle(img, p0, p1, colour, 4 if name == highlight else 2)
        text = name
        if labels and name in labels:
            label, dist = labels[name]
            text = label if name == "sock" else f"{name}: {label} ({dist:.0f})"
        cv2.putText(img, text, (p0[0] + 4, max(p0[1] - 8, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    return img


def open_camera(args) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(args.camera) if str(args.camera).isdigit() else args.camera)
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    if not ok:
        sys.exit(f"Could not read camera {args.camera}. Check the index (`make find-cameras`) and that "
                 "VS Code/Terminal has camera access in System Settings > Privacy & Security > Camera.")
    print(f"Camera {args.camera}: {frame.shape[1]}x{frame.shape[0]}")
    return cap


def live_capture(args, spots: dict | None = None) -> np.ndarray:
    """Live preview to align the camera; SPACE takes the picture, q quits. Saves it to captures/."""
    cap = open_camera(args)
    frame = None
    while True:
        ok, new = cap.read()
        if ok:
            frame = new
        if frame is None:
            continue
        vis = fit(frame)
        h, w = vis.shape[:2]
        for i in (1, 2):  # rule-of-thirds grid to help line things up
            cv2.line(vis, (w * i // 3, 0), (w * i // 3, h), (200, 200, 200), 1)
            cv2.line(vis, (0, h * i // 3), (w, h * i // 3), (200, 200, 200), 1)
        if spots:
            draw(vis, spots)
        banner(vis, "Align the camera" + (" so the boxes sit on the spots" if spots else ""),
               "SPACE = take picture    q = quit")
        cv2.imshow("camera", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            break
        if key in (ord("q"), 27):
            cap.release()
            close_windows()
            sys.exit("Cancelled.")
    cap.release()
    close_windows()
    CAPTURE_DIR.mkdir(exist_ok=True)
    path = CAPTURE_DIR / f"scan_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(path), frame)
    cv2.imwrite(str(CAPTURE_DIR / "latest.png"), frame)
    print(f"Saved picture to {path} (also captures/latest.png)")
    return frame


def get_frame(args, spots: dict | None = None) -> np.ndarray:
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Could not read image {args.image}")
        return frame
    return live_capture(args, spots)


def select_boxes(frame: np.ndarray, what: str) -> dict:
    """Drag one box per spot, then one on bare table. Returns normalised boxes + frame size."""
    disp = fit(frame)
    dh, dw = disp.shape[:2]
    to_norm = lambda r: [r[0] / dw, r[1] / dh, r[2] / dw, r[3] / dh]

    print(f"Drag a box around each {what}, ENTER after each one, ESC when all are done.")
    rois = cv2.selectROIs(f"Box each {what}: ENTER after each, ESC when done",
                          banner(disp.copy(), f"Box each {what}: drag, ENTER after each, ESC when done"),
                          showCrosshair=False)
    close_windows()
    rois = [r for r in rois if r[2] > 0 and r[3] > 0]
    if not rois:
        sys.exit("No boxes drawn.")
    boxes = {f"spot{i + 1}": to_norm(r) for i, r in enumerate(rois)}

    shown = draw(disp.copy(), boxes)
    while True:
        print("Now drag ONE box on bare table that a sock never covers, then ENTER.")
        table = cv2.selectROI("Box bare TABLE, then ENTER",
                              banner(shown.copy(), "Box a patch of bare TABLE (never covered), then ENTER"),
                              showCrosshair=False)
        close_windows()
        if table[2] > 0 and table[3] > 0:
            break
        print("The table box is needed to cancel out lighting changes. Please draw it.")
    boxes["_table"] = to_norm(table)
    boxes["_size"] = [frame.shape[1], frame.shape[0]]
    return boxes


def type_label(img: np.ndarray, prompt: str, window: str = "teach") -> str:
    """Type a label straight into the image window. ENTER saves, ESC skips."""
    text = ""
    while True:
        cv2.imshow(window, banner(img.copy(), f"{prompt} {text}_", "type colour, ENTER = save, ESC = skip"))
        key = cv2.waitKey(0) & 0xFF
        if key in (10, 13):
            return text.strip().lower()
        if key == 27:
            return ""
        if key in (8, 127):
            text = text[:-1]
        elif 32 <= key < 127:
            text += chr(key)


# ── commands ────────────────────────────────────────────────────────────────

def cmd_capture(args):
    live_capture(args, json.loads(SPOTS_FILE.read_text()) if SPOTS_FILE.exists() else None)


def cmd_spots(args):
    old = json.loads(SPOTS_FILE.read_text()) if SPOTS_FILE.exists() else None
    frame = get_frame(args, old)
    spots = select_boxes(frame, "spot")
    SPOTS_FILE.write_text(json.dumps(spots, indent=2))
    print(f"Saved {len(sock_spots(spots))} spots + table box to {SPOTS_FILE.name}")
    cv2.imshow("spots", banner(draw(fit(frame), spots), "Saved. Press any key to close"))
    cv2.waitKey(0)
    close_windows()


def cmd_teach(args):
    colours = json.loads(COLOURS_FILE.read_text()) if COLOURS_FILE.exists() else {}
    free = args.free or not SPOTS_FILE.exists()
    saved_spots = None if free else json.loads(SPOTS_FILE.read_text())
    frame = get_frame(args, saved_spots)
    spots = select_boxes(frame, "sock") if free else saved_spots
    if "_table" not in spots:
        sys.exit("spots.json has no bare-table box. Re-run `sock_classifier.py spots`.")

    scale = table_scale(frame, spots)
    disp = fit(frame)
    added = []
    for name, box in sock_spots(spots).items():
        label = type_label(draw(disp.copy(), spots, highlight=name), f"{name}:")
        if label:
            colours.setdefault(label, []).append([round(float(v), 2) for v in spot_lab(frame, box, scale)])
            added.append(f"{name}={label}")
    close_windows()
    COLOURS_FILE.write_text(json.dumps(colours, indent=2))
    print(f"Added {', '.join(added) or 'nothing'}. Known colours: {sorted(colours)}")


def cmd_run(args):
    clf = SockClassifier(spots_file=SPOTS_FILE, max_dist=args.max_dist, max_ratio=args.max_ratio)
    if args.image:
        frame = get_frame(args)
        labels = clf.classify_detailed(frame, rgb=False)
        print(json.dumps({k: v[0] for k, v in labels.items()}, indent=2))
        cv2.imshow("run", banner(draw(fit(frame), clf.spots, labels), "Press any key to close"))
        cv2.waitKey(0)
        close_windows()
        return
    cap = open_camera(args)
    print("Live view: SPACE prints the mapping, q quits.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        labels = clf.classify_detailed(frame, rgb=False)
        cv2.imshow("run", banner(draw(fit(frame), clf.spots, labels), "SPACE = print mapping    q = quit"))
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            print(json.dumps({k: v[0] for k, v in labels.items()}))
        elif key in (ord("q"), 27):
            break
    cap.release()
    close_windows()


def cmd_closeup_teach(args):
    """Point the camera at a sock, SPACE, type its colour; repeat. Saves after every sock."""
    colours = json.loads(CLOSEUP_COLOURS_FILE.read_text()) if CLOSEUP_COLOURS_FILE.exists() else {}
    box = centre_box(args.centre)
    shots = CAPTURE_DIR / "closeup"
    shots.mkdir(parents=True, exist_ok=True)

    def teach(frames: list) -> None:
        label = type_label(draw_centre(fit(frames[0]), args.centre), "colour:", window="camera")
        if not label:
            return
        lab = np.median([closeup_lab(f, box) for f in frames], axis=0)
        colours.setdefault(label, []).append([round(float(v), 2) for v in lab])
        CLOSEUP_COLOURS_FILE.write_text(json.dumps(colours, indent=2))
        cv2.imwrite(str(shots / f"{label}_{time.strftime('%H%M%S')}.png"), frames[0])
        print(f"Saved '{label}' ({len(colours[label])} reading(s)). Known colours: {sorted(colours)}")

    if args.image:
        teach([get_frame(args)])
        close_windows()
        return
    cam = ArmCamera(args.camera, args.width, args.height)
    print("Point the camera at a sock (as the robot will), SPACE to capture, q when done.")
    while True:
        frame = cam.read()
        cv2.imshow("camera", banner(draw_centre(fit(frame), args.centre),
                                    "Point at a sock. SPACE = capture    q = done"))
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            teach(cam.frames())  # same multi-frame reading the robot uses
        elif key in (ord("q"), 27):
            break
    cam.close()
    close_windows()


def swatch(img: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Paint the measured centre colour as a patch in the top-right corner, with its RGB."""
    bgr = cv2.cvtColor(lab.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_LAB2BGR)[0, 0]
    bgr = tuple(int(v) for v in np.clip(bgr * 255, 0, 255))
    w = img.shape[1]
    cv2.rectangle(img, (w - 170, 10), (w - 10, 110), bgr, -1)
    cv2.rectangle(img, (w - 170, 10), (w - 10, 110), (255, 255, 255), 2)
    cv2.putText(img, f"RGB {bgr[2]},{bgr[1]},{bgr[0]}", (w - 170, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    return img


def cmd_closeup_run(args):
    clf = CloseUpClassifier(centre=args.centre, max_dist=args.max_dist, max_ratio=args.max_ratio)
    print(f"Colours it knows: {clf.colour_names}  (from {clf.colours_file.name})")

    def show(frames, c, d):
        lab = np.median([clf.lab(f, rgb=False) for f in frames], axis=0)
        return swatch(draw_centre(fit(frames[-1]), args.centre, describe(lab, clf.refs, c, d)), lab)
    if args.image:
        frame = get_frame(args)
        colour, dist = clf.classify_detailed(frame, rgb=False)
        print(json.dumps({"colour": colour, "distance": round(dist, 1)}))
        cv2.imshow("run", banner(show([frame], colour, dist), "Press any key to close"))
        cv2.waitKey(0)
        close_windows()
        return
    cam = ArmCamera(args.camera, args.width, args.height)
    recent = []
    print("Live view (last 5 frames combined): SPACE = fresh reading like the robot does, q quits.")
    while True:
        frame = cam.read()
        recent = (recent + [frame])[-N_FRAMES:]
        colour, dist = clf.classify_detailed(recent, rgb=False)
        cv2.imshow("run", banner(show(recent, colour, dist), "SPACE = robot-style reading    q = quit"))
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            frames = cam.frames()
            colour, dist = clf.classify_detailed(frames, rgb=False)
            print(describe(np.median([clf.lab(f, rgb=False) for f in frames], axis=0), clf.refs, colour, dist))
        elif key in (ord("q"), 27):
            break
    cam.close()
    close_windows()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["closeup-teach", "closeup-run", "capture", "spots", "teach", "run"])
    p.add_argument("--camera", default="0", help="webcam index, or a stream / IP-cam snapshot URL")
    p.add_argument("--image", help="use a saved picture instead of the live camera")
    p.add_argument("--max-dist", type=float, default=MAX_DIST,
                   help="unknown if farther than this from every taught colour")
    p.add_argument("--max-ratio", type=float, default=MAX_RATIO,
                   help="unknown if best distance > this x second-best (lower = stricter)")
    p.add_argument("--free", action="store_true", help="teach: box each sock by hand instead of using spots.json")
    p.add_argument("--spots", default=str(SPOTS_FILE), help="spots file to use (default spots.json)")
    p.add_argument("--centre", type=float, default=CENTRE, help="close-up: fraction of the frame to read")
    p.add_argument("--width", type=int, help="camera width, same as the robot's camera config")
    p.add_argument("--height", type=int, help="camera height, same as the robot's camera config")
    args = p.parse_args()
    SPOTS_FILE = Path(args.spots)
    try:
        {"closeup-teach": cmd_closeup_teach, "closeup-run": cmd_closeup_run, "capture": cmd_capture,
         "spots": cmd_spots, "teach": cmd_teach, "run": cmd_run}[args.command](args)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
