"""Glowing line stickers: normalised polylines in [-1, 1], drawn on progressively."""
import math

import cv2
import numpy as np


def _arc(cx, cy, r, a0, a1, n=40):
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in np.linspace(a0, a1, n)]


def _heart(s=1.0, ox=0.0, oy=0.0, n=80):
    pts = []
    for t in np.linspace(0, 2 * math.pi, n):
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((ox + x / 17 * s, oy + y / 17 * s))
    return pts


def _star(r=1.0, ri=0.42, k=5, ox=0.0, oy=0.0, rot=-math.pi / 2):
    pts = []
    for i in range(2 * k + 1):
        rr = r if i % 2 == 0 else ri
        a = rot + i * math.pi / k
        pts.append((ox + rr * math.cos(a), oy + rr * math.sin(a)))
    return pts


def _sparkle(ox, oy, r):
    pts = []
    for i in range(9):
        a = -math.pi / 2 + i * math.pi / 4
        rr = r if i % 2 == 0 else r * 0.18
        pts.append((ox + rr * math.cos(a), oy + rr * math.sin(a)))
    return pts


def crescent():
    outer = _arc(0, 0, 1.0, math.radians(60), math.radians(300), 60)
    inner = _arc(0.375, 0, 0.875, math.radians(-81.8), math.radians(-278.2), 60)
    return [outer + inner + [outer[0]], _sparkle(0.75, -0.55, 0.28)]


def heart():
    return [_heart(0.95)]


def heart_arrow():
    h = _heart(0.75, 0.05, 0.05)
    return [h, [(-1.0, 0.75), (1.0, -0.7)], [(1.0, -0.7), (0.72, -0.68)], [(1.0, -0.7), (0.95, -0.42)],
            [(-1.0, 0.75), (-0.88, 0.5)], [(-1.0, 0.75), (-0.75, 0.82)]]


def sparkles():
    return [_sparkle(-0.2, 0.1, 0.85), _sparkle(0.62, -0.55, 0.38), _sparkle(0.7, 0.6, 0.25)]


def star():
    return [_star(0.95), _sparkle(0.85, -0.8, 0.22)]


def sun():
    lines = [_arc(0, 0, 0.45, 0, 2 * math.pi, 50)]
    for i in range(12):
        a = i * math.pi / 6
        r0, r1 = 0.62, 0.95 if i % 2 == 0 else 0.8
        lines.append([(r0 * math.cos(a), r0 * math.sin(a)), (r1 * math.cos(a), r1 * math.sin(a))])
    return lines


def flower():
    petals = [(0.92 * abs(math.cos(3 * t)) * math.cos(t), 0.92 * abs(math.cos(3 * t)) * math.sin(t))
              for t in np.linspace(0, 2 * math.pi, 180)]
    return [petals, _arc(0, 0, 0.16, 0, 2 * math.pi, 24)]


def rose():
    spiral = [(0.035 * t * math.cos(t), -0.45 + 0.035 * t * math.sin(t)) for t in np.linspace(0, 3.6 * math.pi, 80)]
    cup = _arc(0, -0.5, 0.55, math.radians(0), math.radians(180), 30)
    left = [(-0.55, -0.5), (-0.62, -0.78), (-0.3, -0.92), (-0.05, -0.85)]
    right = [(0.55, -0.5), (0.62, -0.78), (0.3, -0.92), (0.05, -0.85)]
    stem = [(0, 0.05), (0.04, 0.5), (-0.02, 1.0)]
    leaf = [(0.02, 0.55), (0.35, 0.38), (0.58, 0.45), (0.32, 0.65), (0.02, 0.55)]
    return [spiral, cup, left, right, stem, leaf]


def drops():
    def drop(ox, oy, s):
        pts = [(ox, oy - s)]
        pts += _arc(ox, oy + s * 0.35, s * 0.55, math.radians(-30), math.radians(210), 30)
        pts.append((ox, oy - s))
        return pts
    return [drop(-0.35, -0.2, 0.6), drop(0.45, 0.1, 0.42), drop(0.0, 0.75, 0.28)]


def music_note():
    head = _arc(-0.35, 0.6, 0.28, 0, 2 * math.pi, 30)
    head2 = _arc(0.55, 0.4, 0.28, 0, 2 * math.pi, 30)
    return [head, head2, [(-0.08, 0.6), (-0.08, -0.85), (0.82, -1.0), (0.82, 0.4)], [(-0.08, -0.5), (0.82, -0.65)]]


def strings():
    out = []
    for k in range(4):
        y = -0.6 + k * 0.4
        out.append([(x, y + 0.08 * math.sin(x * 9 + k) * (1 - abs(x))) for x in np.linspace(-1, 1, 50)])
    return out


def eye():
    top = _arc(0, 0.85, 1.25, math.radians(-145), math.radians(-35), 40)
    bot = _arc(0, -0.85, 1.25, math.radians(35), math.radians(145), 40)
    iris = _arc(0, 0, 0.32, 0, 2 * math.pi, 36)
    lashes = [[(x, -0.22 - 0.05 * abs(x)), (x * 1.15, -0.5)] for x in (-0.6, -0.3, 0, 0.3, 0.6)]
    return [top, bot, iris] + lashes


def infinity():
    return [[(math.sin(t) * 0.95, math.sin(t) * math.cos(t) * 0.8) for t in np.linspace(0, 2 * math.pi, 90)]]


def heartbeat():
    line = [(-1.0, 0.1), (-0.55, 0.1), (-0.42, -0.25), (-0.28, 0.5), (-0.12, -0.8), (0.02, 0.1)]
    h = _heart(0.5, 0.5, 0.05)
    return [line + h[len(h) // 2:] + h[:len(h) // 2 + 1]]


def quill():
    vane = [(0.85, -0.95), (0.2, -0.55), (-0.35, 0.15), (-0.55, 0.55)]
    vane2 = [(0.85, -0.95), (0.55, -0.2), (0.0, 0.3), (-0.55, 0.55)]
    nib = [(-0.55, 0.55), (-0.9, 0.95)]
    shaft = [(0.85, -0.95), (-0.2, 0.2)]
    return [vane, vane2, shaft, nib]


def _rrect(hw, hh, r, n=8):
    pts = []
    for cx, cy, a0 in ((hw - r, -hh + r, -90), (hw - r, hh - r, 0), (-hw + r, hh - r, 90), (-hw + r, -hh + r, 180)):
        pts += _arc(cx, cy, r, math.radians(a0), math.radians(a0 + 90), n)
    return pts + [pts[0]]


def phone():
    body = _rrect(0.42, 0.88, 0.14)
    screen = [(-0.3, -0.62), (0.3, -0.62), (0.3, 0.55), (-0.3, 0.55), (-0.3, -0.62)]
    home = _arc(0, 0.72, 0.07, 0, 2 * math.pi, 16)
    waves = [_arc(0.5, -0.7, r, math.radians(-85), math.radians(5), 14) for r in (0.28, 0.5)]
    return [body, screen, home] + waves


def ring():
    dot = _arc(0, 0, 0.14, 0, 2 * math.pi, 20)
    out = [dot]
    for r in (0.42, 0.7, 0.98):
        out.append(_arc(0, 0, r, math.radians(-42), math.radians(42), 18))
        out.append(_arc(0, 0, r, math.radians(138), math.radians(222), 18))
    return out


def candle():
    body = [(-0.28, 0.0), (-0.28, 0.9), (0.28, 0.9), (0.28, 0.0)]
    top = _arc(0, 0.0, 0.28, math.radians(180), math.radians(360), 14)
    wick = [(0, -0.05), (0, -0.2)]
    flame = [(0, -0.28)] + [(0.2 * math.sin(t), -0.28 - 0.5 * math.sin(t / 2) ** 0.9 * 1.0 + 0.0) for t in np.linspace(0, math.pi, 1)]
    fl = [(0, -0.95), (0.17, -0.62), (0.12, -0.38), (0, -0.3), (-0.12, -0.38), (-0.17, -0.62), (0, -0.95)]
    return [body, top, wick, fl]


def finjan():
    cup = [(-0.62, -0.1), (-0.5, 0.42), (-0.28, 0.66), (0.28, 0.66), (0.5, 0.42), (0.62, -0.1), (-0.62, -0.1)]
    saucer = [(-0.9, 0.8), (0.9, 0.8)]
    steam = [[(x, -0.25), (x + 0.1, -0.45), (x - 0.06, -0.65), (x + 0.06, -0.9)] for x in (-0.25, 0.05, 0.32)]
    return [cup, saucer] + steam


def dallah():
    body = [(-0.15, -0.55), (-0.32, -0.2), (-0.42, 0.35), (-0.36, 0.7), (0.36, 0.7), (0.42, 0.35), (0.32, -0.2), (0.15, -0.55)]
    lid = _arc(0, -0.55, 0.2, math.radians(180), math.radians(360), 14) + [(0, -0.92)]
    spout = [(-0.36, 0.1), (-0.65, -0.15), (-0.95, -0.55)]
    handle = _arc(0.42, 0.1, 0.3, math.radians(-80), math.radians(80), 16)
    base = [(-0.45, 0.85), (0.45, 0.85)]
    return [body, lid, spout, handle, base]


def door():
    arch = [(-0.55, 0.95), (-0.55, -0.3)] + _arc(0, -0.3, 0.55, math.radians(180), math.radians(360), 24) + [(0.55, 0.95)]
    mid = [(0, -0.85), (0, 0.95)]
    knobs = [_arc(-0.12, 0.2, 0.06, 0, 2 * math.pi, 12), _arc(0.12, 0.2, 0.06, 0, 2 * math.pi, 12)]
    return [arch, mid] + knobs


def lantern():
    ring = _arc(0, -0.85, 0.1, 0, 2 * math.pi, 12)
    cap = [(-0.3, -0.5), (0, -0.75), (0.3, -0.5), (-0.3, -0.5)]
    body = [(-0.3, -0.5), (-0.38, 0.45), (0.38, 0.45), (0.3, -0.5)]
    base = [(-0.38, 0.45), (0, 0.8), (0.38, 0.45)]
    fl = [(0, -0.25), (0.1, -0.02), (0, 0.15), (-0.1, -0.02), (0, -0.25)]
    return [ring, cap, body, base, fl]


def sliders():
    out = []
    for x, k in ((-0.55, 0.3), (0, -0.35), (0.55, 0.1)):
        out.append([(x, -0.9), (x, 0.9)])
        out.append([(x - 0.18, k), (x + 0.18, k)])
    return out


def clapper():
    box = [(-0.8, -0.3), (0.8, -0.3), (0.8, 0.8), (-0.8, 0.8), (-0.8, -0.3)]
    top = [(-0.8, -0.3), (-0.75, -0.7), (0.78, -0.92), (0.8, -0.3)]
    play = [(-0.2, 0.05), (0.3, 0.25), (-0.2, 0.45), (-0.2, 0.05)]
    return [box, top, play]


def boat():
    hull = [(-0.95, 0.3), (-0.7, 0.65), (0.7, 0.65), (0.95, 0.3), (-0.95, 0.3)]
    mast = [(0, 0.3), (0, -0.95)]
    sail = [(0.05, -0.85), (0.6, 0.15), (0.05, 0.15)]
    wave = [(-0.9 + 0.3 * i, 0.85 + 0.06 * (-1) ** i) for i in range(7)]
    return [hull, mast, sail, wave]


def tear():
    d = [(0, -0.9)] + _arc(0, 0.25, 0.48, math.radians(-30), math.radians(210), 30) + [(0, -0.9)]
    return [d, _arc(0, 0.25, 0.25, math.radians(100), math.radians(170), 8)]


CATALOGUE = dict(finjan=finjan, dallah=dallah, door=door, lantern=lantern, sliders=sliders, clapper=clapper, boat=boat, tear=tear, phone=phone, ring=ring, candle=candle, quill=quill, crescent=crescent, heart=heart, heart_arrow=heart_arrow, sparkles=sparkles, star=star,
                 sun=sun, flower=flower, rose=rose, drops=drops, music_note=music_note, strings=strings,
                 eye=eye, infinity=infinity, heartbeat=heartbeat)


def draw(name, progress, radius=95, flicker=1.0):
    """Return (rgb_add, alpha) float32 layers of size 2.6r square for a sticker."""
    polys = CATALOGUE[name]()
    size = int(radius * 2.8)
    c = size / 2
    lens = [sum(math.dist(p[i], p[i + 1]) for i in range(len(p) - 1)) for p in polys]
    total = sum(lens)
    budget = total * max(0.0, min(1.0, progress))
    core = np.zeros((size, size), np.float32)
    for p, ln in zip(polys, lens):
        if budget <= 0:
            break
        pts, acc = [p[0]], 0.0
        for i in range(len(p) - 1):
            d = math.dist(p[i], p[i + 1])
            if acc + d <= budget:
                pts.append(p[i + 1])
                acc += d
            else:
                f = (budget - acc) / d if d else 0
                pts.append((p[i][0] + (p[i + 1][0] - p[i][0]) * f, p[i][1] + (p[i + 1][1] - p[i][1]) * f))
                acc = budget
                break
        budget -= acc
        arr = np.array([(c + x * radius, c + y * radius) for x, y in pts], np.float32)
        cv2.polylines(core, [np.round(arr * 4).astype(np.int32)], False, 1.0, 3, cv2.LINE_AA, shift=2)
    glow = cv2.GaussianBlur(core, (0, 0), 7) * 3.2
    halo = cv2.GaussianBlur(core, (0, 0), 4) * 2.0
    return core, np.clip(glow, 0, 1) * flicker, np.clip(halo, 0, 1)
