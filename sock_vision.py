"""
sock_vision.py — HackTheSOCKS color vision (for two colors rn) module

核心模块——分割双色袜子 + 颜色分类 + 防抖 + 画框。纯 opencv/numpy, 不依赖 torch。

Pipeline for one frame:
    1. segment_sock()   -> find the sock blob against a plain background
    2. hsv_median()     -> summarize its color as one (H, S, V) triple
    3. SockColorClassifier.classify() -> match against calibrated reference
       colors (built by build_color_refs.py from real sample photos)

Why this handles black/white "colors" correctly without special-casing:
    Hue is meaningless when saturation is near 0 (grays/black/white).
    _hsv_to_feature() projects (H, S, V) into a 3D point where hue's
    influence is scaled by S. Low-S colors (black/white) naturally cluster
    by V instead of by their (noisy) hue -- no manual "is this neutral?"
    branch needed.
"""
import json

import cv2
import numpy as np


def segment_sock(frame, corner_patch=20, resize_max=800):
    """
    Separate the sock from a plain, lighter background by sampling the
    background color from the 4 corners of the frame and thresholding
    how far each pixel is from that color (Otsu).

    Returns (mask, small_frame, contour):
        mask        - uint8 HxW binary mask (255 = sock), at `small_frame` scale
        small_frame - the (possibly downscaled) frame the mask corresponds to
        contour     - the largest contour found (the sock)
    Returns (None, small_frame, None) if no sock-like blob was found.
    """
    h, w = frame.shape[:2]
    scale = min(1.0, resize_max / max(h, w))
    small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame.copy()
    hs, ws = small.shape[:2]
    p = max(4, min(corner_patch, hs // 4, ws // 4))

    corners = np.concatenate(
        [
            small[0:p, 0:p].reshape(-1, 3),
            small[0:p, ws - p : ws].reshape(-1, 3),
            small[hs - p : hs, 0:p].reshape(-1, 3),
            small[hs - p : hs, ws - p : ws].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_bgr = np.median(corners, axis=0)

    diff = np.linalg.norm(small.astype(np.float32) - bg_bgr.astype(np.float32), axis=2)
    diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, small, None

    biggest = max(contours, key=cv2.contourArea)
    sock_mask = np.zeros_like(mask)
    cv2.drawContours(sock_mask, [biggest], -1, 255, -1)
    return sock_mask, small, biggest


def hsv_median(small_bgr, sock_mask):
    """Median (H, S, V) of the masked sock region. None if mask is empty."""
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    pix = hsv[sock_mask == 255]
    if pix.size == 0:
        return None
    return (
        float(np.median(pix[:, 0])),
        float(np.median(pix[:, 1])),
        float(np.median(pix[:, 2])),
    )


def _hsv_to_feature(h, s, v):
    """(H,S,V) -> 3D point. Hue contributes less when saturation is low,
    so grays/black/white cluster by brightness instead of noisy hue."""
    theta = h * (np.pi / 90.0)  # OpenCV hue is 0-179 == 0-358 degrees
    s_n = s / 255.0
    v_n = v / 255.0
    return np.array([s_n * np.cos(theta), s_n * np.sin(theta), v_n])


class SockColorClassifier:
    def __init__(self, references):
        """references: dict[label] -> (h, s, v) calibrated from a sample photo."""
        self.labels = list(references.keys())
        self._raw = dict(references)
        self._feats = np.array([_hsv_to_feature(*hsv) for hsv in references.values()])

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)
        refs = {label: tuple(v["hsv"]) for label, v in data.items()}
        return cls(refs)

    def classify(self, frame):
        """
        Returns None if no sock was detected, otherwise a dict:
          label        - best-matching color label
          confidence   - 0..1, higher = more sure (based on gap to 2nd place)
          margin       - raw distance gap to the 2nd-best match
          hsv          - the detected sock's median (H,S,V)
          mask/contour/small_frame - for debugging / drawing overlays
        """
        mask, small, contour = segment_sock(frame)
        if mask is None:
            return None
        hsv = hsv_median(small, mask)
        if hsv is None:
            return None

        feat = _hsv_to_feature(*hsv)
        dists = np.linalg.norm(self._feats - feat, axis=1)
        order = np.argsort(dists)
        best_i = order[0]
        second_i = order[1] if len(order) > 1 else order[0]
        margin = float(dists[second_i] - dists[best_i])
        # 0.3 is a rough scale for "confidently separated" — recalibrate on
        # site if confidences look off once you add real venue-lit photos.
        confidence = float(np.clip(margin / 0.3, 0, 1))

        return {
            "label": self.labels[best_i],
            "confidence": confidence,
            "margin": margin,
            "hsv": hsv,
            "mask": mask,
            "contour": contour,
            "small_frame": small,
        }


class RollingVote:
    """Debounce: only 'fire' once the same label wins N consecutive frames.
    Prevents one noisy/blurry frame from triggering a wrong pick."""

    def __init__(self, need=3):
        self.need = need
        self.history = []

    def push(self, label):
        self.history.append(label)
        self.history = self.history[-self.need :]
        if len(self.history) == self.need and len(set(self.history)) == 1:
            return self.history[0]
        return None

    def reset(self):
        self.history = []


def draw_result(result):
    """BGR image with the detected sock outlined + labeled. For live demo/debugging."""
    if result is None:
        return None
    img = result["small_frame"].copy()
    cv2.drawContours(img, [result["contour"]], -1, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(result["contour"])
    text = f"{result['label']}  conf={result['confidence']:.2f}"
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, text, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img
