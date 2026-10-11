"""Assets for «حَمَاكِ الإِلَهُ بِلَادَ الحَرَمْ»: official 3:2 flags, a waving-flag simulator and gold line-art icons.

usage: python haram_assets.py flags_dir icons_dir   (writes sa_3x2.png / eg_3x2.png and an icon preview sheet)
"""
import math
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import stickers  # noqa: E402


# ---------------------------------------------------------------- flags
def flag_3x2(svg_path, out_png, width=1500):
    """Official flag-icons artwork re-framed to the true 3:2 proportion (no redrawing of emblems)."""
    import cairosvg
    s = open(svg_path).read()
    name = os.path.basename(svg_path)
    if name.startswith("sa"):
        # the 4:3 file clips a 768x512 (3:2) drawing; drop the clip and widen the viewBox to the full field
        s = s.replace('clip-path="url(#sa-a)"', "")
        s = s.replace('viewBox="0 0 640 480"', 'viewBox="-40 0 720 480"')
    else:
        s = s.replace('viewBox="0 0 640 480"', 'viewBox="-40 0 720 480"')
        s = re.sub(r'd="M0 (\d+)h640v160H0z"', r'd="M-40 \1h720v160H-40z"', s)
    cairosvg.svg2png(bytestring=s.encode(), write_to=out_png, output_width=width, output_height=width * 2 // 3)


class WavingFlag:
    """Cloth-like wave on a flat flag image: travelling wave from the hoist (left), with light/shade from the slope."""

    def __init__(self, png, height=420):
        im = cv2.imread(png, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
        h = height
        w = int(round(h * im.shape[1] / im.shape[0]))
        self.img = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        self.w, self.h = w, h
        self.pad = int(h * 0.16)
        yy, xx = np.mgrid[0:h + 2 * self.pad, 0:w].astype(np.float32)
        self.xx, self.yy = xx, yy

    def frame(self, t, amp=1.0):
        """Return (rgb, alpha) of size (h + 2*pad, w)."""
        w, h, p = self.w, self.h, self.pad
        u = self.xx / w
        lam, f = 0.85, 0.55
        ph = 2 * math.pi * (u / lam - f * t)
        a = h * 0.055 * amp * (u ** 1.15)
        dy = a * np.sin(ph) + h * 0.012 * amp * u * np.sin(2 * math.pi * (u / 0.37 - 1.3 * f * t))
        dx = -w * 0.012 * amp * u * (1 - np.cos(ph)) * 0.5
        slope = np.cos(ph) * (u ** 1.15)
        sx = self.xx - dx
        sy = self.yy - p - dy
        rgb = cv2.remap(self.img, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        alpha = cv2.remap(np.ones((h, w), np.float32), sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        shade = 1 + 0.16 * amp * slope
        rgb = np.clip(rgb * shade[..., None] + 0.05 * amp * np.maximum(slope, 0)[..., None], 0, 1)
        return rgb, alpha


# ---------------------------------------------------------------- icons (normalised polylines in [-1, 1])
def _svg_polys(svg_path, n_per=60):
    from svgpathtools import parse_path
    src = open(svg_path).read()
    polys = []
    for d in re.findall(r'<path[^>]*\sd="([^"]+)"', src):
        path = parse_path(d)
        for sub in path.continuous_subpaths():
            L = sub.length()
            n = max(8, int(n_per * L / 12))
            pts = [sub.point(sub.ilength(L * i / (n - 1))) if L > 0 else sub.point(0) for i in range(n)]
            polys.append([((p.real - 12) / 11, (p.imag - 12) / 11) for p in pts])
    return polys


ICON_DIR = None


def handshake():
    return _svg_polys(os.path.join(ICON_DIR, "handshake.svg"))


def jet():
    # lucide plane points to the upper right; keep its native orientation (flight path)
    return _svg_polys(os.path.join(ICON_DIR, "plane.svg"))


def kaaba():
    fl, fr, top, bot = -0.62, 0.38, -0.55, 0.85
    off = (0.42, -0.22)
    front = [(fl, top), (fr, top), (fr, bot), (fl, bot), (fl, top)]
    side = [(fr, top), (fr + off[0], top + off[1]), (fr + off[0], bot + off[1]), (fr, bot)]
    roof = [(fl, top), (fl + off[0], top + off[1]), (fr + off[0], top + off[1])]
    band_f = [(fl, top + 0.33), (fr, top + 0.33)]
    band_s = [(fr, top + 0.33), (fr + off[0], top + 0.33 + off[1])]
    band2 = [(fl, top + 0.43), (fr, top + 0.43)]
    door = [(fr - 0.32, bot), (fr - 0.32, bot - 0.55), (fr - 0.1, bot - 0.55), (fr - 0.1, bot)]
    return [front, side, roof, band_f, band_s, band2, door]


def minaret():
    body = [(-0.16, 1.0), (-0.16, -0.2), (-0.12, -0.55), (0.12, -0.55), (0.16, -0.2), (0.16, 1.0)]
    balc1 = [(-0.3, 0.2), (0.3, 0.2)]
    balc2 = [(-0.26, -0.2), (0.26, -0.2)]
    cone = [(-0.12, -0.55), (0.0, -0.85), (0.12, -0.55)]
    cres = stickers._arc(0, -0.93, 0.07, math.radians(40), math.radians(320), 16)
    base = [(-0.5, 1.0), (0.5, 1.0)]
    return [body, balc1, balc2, cone, cres, base]


def star8():
    sq1 = [(math.cos(a) * 0.95, math.sin(a) * 0.95) for a in np.linspace(math.pi / 4, math.pi / 4 + 2 * math.pi, 5)]
    sq2 = [(math.cos(a) * 0.95, math.sin(a) * 0.95) for a in np.linspace(0, 2 * math.pi, 5)]
    inner = stickers._arc(0, 0, 0.42, 0, 2 * math.pi, 40)
    dot = stickers._arc(0, 0, 0.12, 0, 2 * math.pi, 16)
    return [sq1, sq2, inner, dot]


def palm_swords():
    trunk = [(0.0, 0.25), (0.02, -0.1), (0.0, -0.45)]
    fronds = []
    for ang, ln in ((-160, 0.62), (-130, 0.58), (-95, 0.45), (-50, 0.58), (-20, 0.62), (-75, 0.5), (-110, 0.5)):
        a = math.radians(ang)
        ex, ey = ln * math.cos(a), -0.45 + ln * math.sin(a)
        mx, my = 0.6 * ln * math.cos(a), -0.45 + 0.6 * ln * math.sin(a) - 0.12
        fronds.append([(0, -0.45), (mx, my), (ex, ey + 0.1)])
    s1 = [(-0.85, 0.15), (-0.3, 0.55), (0.3, 0.85), (0.55, 0.92)]
    s2 = [(0.85, 0.15), (0.3, 0.55), (-0.3, 0.85), (-0.55, 0.92)]
    g1 = [(-0.62, 0.42), (-0.5, 0.25)]
    g2 = [(0.62, 0.42), (0.5, 0.25)]
    return [trunk] + fronds + [s1, s2, g1, g2]


def kingdom():
    def side(sign):
        return [(sign * (0.42 - 0.3 * ((1 - y) / 2) ** 0.8), y) for y in np.linspace(1.0, -1.0, 30)]
    left, right = side(-1), side(1)
    hole_l = [(-0.12, -1.0), (-0.1, -0.7), (-0.04, -0.45), (0.0, -0.42)]
    hole_r = [(0.12, -1.0), (0.1, -0.7), (0.04, -0.45), (0.0, -0.42)]
    bridge = [(-0.11, -0.78), (0.11, -0.78)]
    base = [(-0.6, 1.0), (0.6, 1.0)]
    return [left, right, hole_l, hole_r, bridge, base]


def skyline():
    pts = [(-1, 0.8), (-1, 0.2), (-0.8, 0.2), (-0.8, -0.1), (-0.62, -0.1), (-0.62, 0.35), (-0.45, 0.35), (-0.45, -0.45),
           (-0.38, -0.7), (-0.31, -0.45), (-0.31, 0.25), (-0.1, 0.25), (-0.1, -0.2), (0.05, -0.2), (0.05, -0.95),
           (0.13, -0.95), (0.13, -0.3), (0.3, -0.3), (0.3, 0.1), (0.5, 0.1), (0.5, -0.55), (0.62, -0.55), (0.62, 0.3),
           (0.8, 0.3), (0.8, 0.0), (1.0, 0.0), (1.0, 0.8)]
    return [pts, [(-1, 0.8), (1, 0.8)]]


def register(icon_dir):
    global ICON_DIR
    ICON_DIR = icon_dir
    stickers.CATALOGUE.update(handshake=handshake, jet=jet, kaaba=kaaba, minaret=minaret, star8=star8,
                              palm_swords=palm_swords, kingdom=kingdom, skyline=skyline)


if __name__ == "__main__":
    fd, idir = sys.argv[1], sys.argv[2]
    flag_3x2(f"{fd}/sa.svg", f"{fd}/sa_3x2.png")
    flag_3x2(f"{fd}/eg.svg", f"{fd}/eg_3x2.png")
    register(idir)
    tiles = []
    for n in ("kaaba", "minaret", "star8", "palm_swords", "kingdom", "skyline", "jet", "handshake"):
        core, glow, halo = stickers.draw(n, 1.0, 95)
        bg = np.full(core.shape + (3,), 0.12, np.float32)
        img = bg + glow[..., None] * np.array([0.585, 0.835, 0.965]) * 0.8
        img = img * (1 - core[..., None]) + core[..., None] * np.array([0.905, 0.968, 1.0])
        img = cv2.putText((np.clip(img, 0, 1) * 255).astype(np.uint8), n, (5, 20), 0, 0.6, (200, 200, 200), 1)
        tiles.append(img)
    cv2.imwrite(f"{idir}/icons_sheet.png", np.hstack(tiles))
    wf = WavingFlag(f"{fd}/sa_3x2.png", 360)
    frs = []
    for t in (0, 0.4, 0.8):
        rgb, a = wf.frame(t)
        frs.append((rgb * a[..., None] + 0.2 * (1 - a[..., None]))[:, :, ::-1])
    cv2.imwrite(f"{fd}/wave_sheet.png", (np.vstack(frs) * 255).astype(np.uint8))
    print("ok")
