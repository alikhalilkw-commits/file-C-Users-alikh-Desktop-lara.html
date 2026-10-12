"""Gold calligraphic arms with cupped hands for the closing embrace (screen space, 1080 wide)."""
import math

import cv2
import numpy as np

SS = 4  # supersampling for anti-aliased polygons


def catmull(points, per=24):
    P = np.array(points, np.float32)
    P = np.vstack([P[0], P, P[-1]])
    out = []
    for k in range(1, len(P) - 2):
        for u in np.linspace(0, 1, per, endpoint=False):
            p0, p1, p2, p3 = P[k - 1], P[k], P[k + 1], P[k + 2]
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(P[-2])
    return np.array(out, np.float32)


def ribbon(center, w0, w1):
    """Tapered closed polygon around a centre line (calligraphic stroke)."""
    d = np.gradient(center, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-6
    s = np.linspace(0, 1, len(center))[:, None]
    w = (w0 + (w1 - w0) * s ** 0.8) / 2
    return np.vstack([center + n * w, (center - n * w)[::-1]])


def capsule(a, b, r):
    a, b = np.array(a, np.float32), np.array(b, np.float32)
    d = b - a
    ang = math.atan2(d[1], d[0])
    pts = []
    for t in np.linspace(math.pi / 2, 3 * math.pi / 2, 10):
        pts.append(a + r * np.array([math.cos(ang + t), math.sin(ang + t)]))
    for t in np.linspace(-math.pi / 2, math.pi / 2, 10):
        pts.append(b + r * np.array([math.cos(ang + t), math.sin(ang + t)]))
    return np.array(pts, np.float32)


def hand(wrist, palm_dir, up, scale=1.0):
    """Open cupped hand seen from the front: broad palm, four fingers reaching toward the centre and curling up,
    thumb raised. Returns filled polygons (outline comes from the union)."""
    wrist = np.array(wrist, np.float32)
    f = np.array(palm_dir, np.float32); f /= np.linalg.norm(f)
    u = np.array(up, np.float32); u /= np.linalg.norm(u)
    L, Wd = 46 * scale, 50 * scale
    c0 = wrist - f * 4 * scale
    palm = np.array([c0 + u * 10 * scale, c0 + f * L * 0.45 + u * Wd * 0.48, c0 + f * L + u * Wd * 0.40,
                     c0 + f * (L + 6 * scale) - u * Wd * 0.30, c0 + f * L * 0.4 - u * Wd * 0.52, c0 - u * 10 * scale],
                    np.float32)
    parts = [palm]
    base = c0 + f * (L - 4 * scale)
    for off, ln, curl in ((0.36, 40, 0.62), (0.12, 46, 0.48), (-0.12, 44, 0.36), (-0.34, 36, 0.26)):
        a = base + u * Wd * off
        dirv = f * math.cos(curl) + u * math.sin(curl)
        mid = a + dirv * ln * 0.55 * scale
        dir2 = f * math.cos(curl + 0.35) + u * math.sin(curl + 0.35)
        b = mid + dir2 * ln * 0.45 * scale
        parts.append(capsule(a, mid, 6.4 * scale))
        parts.append(capsule(mid, b, 5.8 * scale))
    tb = c0 + f * L * 0.42 + u * Wd * 0.40
    tm = tb + (u * 0.75 + f * 0.25) * 22 * scale
    parts.append(capsule(tb, tm, 7.2 * scale))
    parts.append(capsule(tm, tm + (u * 0.55 + f * 0.65) * 18 * scale, 6.4 * scale))
    return parts


class Embrace:
    def __init__(self, flag_bottom_y, flag_half_w, wrist_y, H=1920, W=1080):
        cx = W / 2
        top = flag_bottom_y
        self.W, self.H = W, H
        # shoulders tuck under the flag, arms sweep wide and come back in to the cupped hands
        left = catmull([(cx - flag_half_w + 20, top - 6), (cx - flag_half_w - 70, top + 22), (230, top + 95),
                        (178, top + 190), (198, top + 280), (268, wrist_y - 18), (360, wrist_y + 4)])
        right = left.copy(); right[:, 0] = W - right[:, 0]
        self.arms = [left, right]
        self.frac = np.r_[0, np.cumsum(np.linalg.norm(np.diff(left, axis=0), axis=1))]
        self.frac /= self.frac[-1]
        self.hands = [hand(left[-1], (1, -0.12), (0.12, -1), 1.05),
                      hand(right[-1], (-1, -0.12), (-0.12, -1), 1.05)]
        self.y0 = int(top - 40)
        self.y1 = int(wrist_y + 90)

    def layers(self, p_arm, p_hand):
        """Return (fill, edge) float masks for the band y0..y1; the edge is the outline of the union."""
        h = self.y1 - self.y0
        fill = np.zeros((h * SS, self.W * SS), np.uint8)
        off = np.array([0, self.y0], np.float32)
        n = max(2, int(np.searchsorted(self.frac, p_arm)))
        for arm in self.arms:
            c = arm[:n]
            poly = ribbon(c, 32, 32 - 12 * min(1, len(c) / len(arm)))
            cv2.fillPoly(fill, [np.round((poly - off) * SS).astype(np.int32)], 255, cv2.LINE_AA)
        hand_m = np.zeros_like(fill)
        if p_hand > 0:
            for parts in self.hands:
                for poly in parts:
                    cv2.fillPoly(hand_m, [np.round((poly - off) * SS).astype(np.int32)], 255, cv2.LINE_AA)
        fill = cv2.resize(fill, (self.W, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        hand_m = cv2.resize(hand_m, (self.W, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 * p_hand
        allm = np.maximum(fill, hand_m)
        k = np.ones((3, 3), np.uint8)
        edge = np.clip(cv2.dilate(allm, k, iterations=1) - cv2.erode(allm, k, iterations=1), 0, 1)
        edge = cv2.GaussianBlur(edge, (0, 0), 0.6) * 1.3
        return allm, np.clip(edge, 0, 1)
