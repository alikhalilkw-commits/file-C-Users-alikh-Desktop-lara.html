"""Arabic lyric video renderer (1080x1920, 30 fps).

usage:
  python render.py config.json --stills 3.0,40.5,...  --outdir stills/
  python render.py config.json --range 0 2700 --out part0.mp4
"""
import argparse
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
import stickers  # noqa: E402

W, H, FPS = 1080, 1920, 30
IVORY = np.array([1.0, 0.968, 0.905], np.float32)
GOLD = np.array([0.965, 0.835, 0.585], np.float32)
DARK = np.array([0.03, 0.02, 0.02], np.float32)


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def ramp(t, a, b):
    return 0.0 if t <= a else 1.0 if t >= b else (t - a) / (b - a)


# ---------------------------------------------------------------- text sprites
class Fonts:
    def __init__(self, d):
        self.d = d
        self.cache = {}

    def get(self, name, size):
        key = (name, size)
        if key not in self.cache:
            if name == "messiri":
                f = ImageFont.truetype(f"{self.d}/ElMessiri.ttf", size, layout_engine=ImageFont.Layout.RAQM)
                f.set_variation_by_axes([700])
            elif name == "gulzar":
                f = ImageFont.truetype(f"{self.d}/Gulzar-Regular.ttf", size, layout_engine=ImageFont.Layout.RAQM)
            else:
                f = ImageFont.truetype(f"{self.d}/Cinzel.ttf", size, layout_engine=ImageFont.Layout.RAQM)
                f.set_variation_by_axes([600])
            self.cache[key] = f
        return self.cache[key]


class Sprite:
    """Glyph mask with baseline, plus pre-blurred shadow and glow layers."""

    def __init__(self, text, font, dilate=0, tracking=0, rtl=True):
        pad = 48
        kw = dict(direction="rtl", language="ar") if rtl else {}
        if tracking:
            chars = list(text)
            widths = [font.getlength(c) for c in chars]
            adv = sum(widths) + tracking * (len(chars) - 1)
            bb = font.getbbox(text, anchor="ls")
            w, h = int(adv + 2 * pad), int(bb[3] - bb[1] + 2 * pad)
            img = Image.new("L", (w, h))
            d = ImageDraw.Draw(img)
            x = pad
            for c, cw in zip(chars, widths):
                d.text((x, pad - bb[1]), c, font=font, fill=255, anchor="ls")
                x += cw + tracking
            self.base = pad - bb[1]
        else:
            bb = font.getbbox(text, anchor="ls", **kw)
            adv = font.getlength(text, **kw)
            w, h = int(max(adv, bb[2] - bb[0]) + 2 * pad), int(bb[3] - bb[1] + 2 * pad)
            img = Image.new("L", (w, h))
            ImageDraw.Draw(img).text((pad - min(0, bb[0]), pad - bb[1]), text, font=font, fill=255, anchor="ls", **kw)
            self.base = pad - bb[1]
        m = np.asarray(img, np.float32) / 255
        if dilate:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
            m = cv2.GaussianBlur(cv2.dilate(m, k), (0, 0), 0.7)
        self.mask = m
        self.pad = pad
        self.adv = adv
        self.top = bb[1]
        self.sh_small = cv2.GaussianBlur(m, (0, 0), 3)
        self.sh_large = np.clip(cv2.GaussianBlur(m, (0, 0), 14) * 1.6, 0, 1)
        self.glow = np.clip(cv2.GaussianBlur(m, (0, 0), 9) * 1.4, 0, 1)
        self.h, self.w = m.shape
        cols = np.where(m.max(0) > 0.04)[0]
        self.ink_l = int(cols[0]) if len(cols) else pad
        self.ink_w = float(cols[-1] - cols[0] + 1) if len(cols) else adv


def blend(frame, layer, x, y, color, alpha, mode="over"):
    """Composite a float mask at top-left (x, y)."""
    if alpha <= 0.002:
        return
    h, w = layer.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    m = layer[y0 - y:y1 - y, x0 - x:x1 - x, None] * alpha
    reg = frame[y0:y1, x0:x1]
    if mode == "add":
        reg += m * color
    else:
        reg *= 1 - m
        reg += m * color


# ---------------------------------------------------------------- images and shots
class Plate:
    def __init__(self, path, moon=None):
        im = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
        self.sw, self.sh = im.shape[1], im.shape[0]
        up = cv2.resize(im, (self.sw * 2, self.sh * 2), interpolation=cv2.INTER_LANCZOS4)
        blur = cv2.GaussianBlur(up, (0, 0), 2.0)
        self.img = np.clip(up + 0.45 * (up - blur), 0, 1)
        self.W, self.H = self.img.shape[1], self.img.shape[0]
        self.moon = moon

    def view(self, cx, cy, z):
        """Return frame and affine (src->screen) for normalised centre and zoom."""
        cw = self.W / z
        ch = cw * H / W
        if ch > self.H:
            ch = self.H
            cw = ch * W / H
        x0 = min(max(cx * self.W - cw / 2, 0), self.W - cw)
        y0 = min(max(cy * self.H - ch / 2, 0), self.H - ch)
        s = W / cw
        M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
        out = cv2.warpAffine(self.img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return out, (s, x0, y0)


def vignette(strength):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.6)) ** 2)
    return (1 - strength * np.clip(d - 0.35, 0, 1) ** 1.6)[..., None]


# ---------------------------------------------------------------- the video
class Video:
    def __init__(self, cfg):
        self.cfg = cfg
        root = cfg["root"]
        self.fonts = Fonts(cfg["fonts"])
        words = json.load(open(cfg["words"]))
        self.duration = words["duration"]
        f = np.load(cfg["feats"])
        self.kick, self.spark, self.loud, self.vocal = f["kick"], f["spark"], f["loud"], f["vocal"]
        self.plates = {k: Plate(os.path.join(root, v["file"]), v.get("moon")) for k, v in cfg["images"].items()}
        self.vig = vignette(0.55)
        self.vig_deep = vignette(0.85)
        rng = np.random.default_rng(7)
        self.grain = [rng.normal(0, 0.012, (H // 2, W // 2, 1)).astype(np.float32) for _ in range(6)]
        self.build_lines(words["words"])
        self.build_shots()
        self.build_credits()
        n = 46
        self.dust = dict(x=rng.uniform(0, W, n), y=rng.uniform(0, H, n), vx=rng.uniform(-6, 6, n),
                         vy=rng.uniform(-22, -8, n), r=rng.uniform(1.2, 3.2, n), ph=rng.uniform(0, 6.28, n))
        m = 14
        self.petals = dict(x=rng.uniform(0, W, m), y=rng.uniform(-H, 0, m), vy=rng.uniform(55, 95, m),
                           sw=rng.uniform(20, 60, m), ph=rng.uniform(0, 6.28, m), s=rng.uniform(9, 16, m))
        dot = np.zeros((33, 33), np.float32)
        cv2.circle(dot, (16, 16), 4, 1.0, -1, cv2.LINE_AA)
        self.dot = cv2.GaussianBlur(dot, (0, 0), 3) * 2.2

    # ------------------------------------------------------------ lyrics
    def build_lines(self, words):
        cfg = self.cfg
        lines = {}
        for w in words:
            lines.setdefault(w["line"], []).append(w)
        self.lines = []
        order = sorted(lines)
        meta = cfg["lines"]
        for i, li in enumerate(order):
            ws = lines[li]
            m = meta[li]
            sec = ws[0]["section"]
            bridge = sec == "Bridge"
            size = int(86 * (0.9 if bridge else 1.0))
            nxt = lines[order[i + 1]][0]["start"] if i + 1 < len(order) else self.duration
            start, end = ws[0]["start"], ws[-1]["end"]
            t_in = start - 0.45
            t_out = min(nxt - 0.2, end + 3.0)
            if m.get("hold"):
                t_out = min(nxt - 0.2, end + m["hold"])
            line = dict(words=ws, section=sec, bridge=bridge, t_in=t_in, t_out=t_out, start=start, end=end,
                        key=m.get("key"), sticker=m.get("sticker"), side=m.get("side", "left"))
            self.layout(line, size)
            self.lines.append(line)

    def layout(self, line, size):
        scale = 1.0
        for _ in range(4):
            sprites = []
            for wi, w in enumerate(line["words"]):
                if wi == line["key"]:
                    sp = Sprite(w["text"], self.fonts.get("gulzar", int(size * 1.5 * scale)), dilate=2)
                else:
                    sp = Sprite(w["text"], self.fonts.get("messiri", int(size * scale)))
                sprites.append(sp)
            space = self.fonts.get("messiri", int(size * scale)).getlength(" ") * 1.15
            total = sum(s.ink_w for s in sprites) + space * (len(sprites) - 1)
            if total <= 880:
                break
            scale *= 880 / total * 0.98
        line["sprites"], line["width"] = sprites, total
        x = W / 2 + total / 2
        xs = []
        for sp in sprites:
            x -= sp.ink_w
            xs.append(x)
            x -= space
        line["xs"] = xs
        sc = np.zeros((300, int(total + 360)), np.float32)
        cv2.ellipse(sc, (sc.shape[1] // 2, 150), (int(total / 2 + 90), 70), 0, 0, 360, 1.0, -1)
        line["scrim"] = cv2.GaussianBlur(sc, (0, 0), 38)

    def draw_lines(self, fr, t, base_y):
        for line in self.lines:
            if not (line["t_in"] <= t <= line["t_out"] + 0.01):
                continue
            fade_t = 0.6 if line["bridge"] else 0.4
            la = smooth(ramp(t, line["t_in"], line["t_in"] + 0.3)) * (1 - smooth(ramp(t, line["t_out"] - fade_t, line["t_out"])))
            by = int(base_y)
            sc = line["scrim"]
            blend(fr, sc, int(W / 2 - sc.shape[1] / 2), by - 150 - 25, DARK, 0.42 * la)
            for wi, (w, sp, x) in enumerate(zip(line["words"], line["sprites"], line["xs"])):
                appear = w["start"] - 0.12
                dur = 0.55 if line["bridge"] else 0.3
                p = ease_out(ramp(t, appear, appear + dur))
                a = p * la
                if a <= 0.003:
                    continue
                dy = int(round(12 * (1 - p)))
                key = wi == line["key"]
                if key:
                    g = 1.0
                else:
                    g = ramp(t, appear, appear + 0.12) * (1 - smooth(ramp(t, w["end"] + 0.05, w["end"] + 0.45)))
                col = IVORY * (1 - g) + GOLD * g
                X = int(round(x)) - sp.ink_l
                Y = by - sp.base + dy
                blend(fr, sp.sh_large, X, Y + 4, DARK, 0.55 * a)
                blend(fr, sp.sh_small, X + 1, Y + 3, DARK, 0.75 * a)
                if g > 0:
                    blend(fr, sp.glow, X, Y, GOLD, 0.22 * g * a, "add")
                blend(fr, sp.mask, X, Y, col, a)
                if key:
                    s = ramp(t, w["start"] + 0.15, w["start"] + 1.15)
                    if 0 < s < 1:
                        xx = np.arange(sp.w, dtype=np.float32)
                        cx = sp.w * (1.2 - 1.4 * s)
                        band = np.exp(-((xx - cx) / (sp.w * 0.09)) ** 2)[None, :]
                        blend(fr, sp.mask * band, X, Y, np.array([1, 1, 0.95], np.float32), 0.55 * a, "add")
            if line["sticker"]:
                self.draw_line_sticker(fr, t, line, by, la)

    def draw_line_sticker(self, fr, t, line, by, la):
        kw = line["words"][line["key"]] if line["key"] is not None else line["words"][-1]
        t0 = kw["start"]
        if t < t0:
            return
        prog = ramp(t, t0, t0 + 0.7)
        r = 95 * (0.8 if line["bridge"] else 1.0)
        fi = int(t * FPS)
        flick = 0.85 + 0.15 * math.sin(t * 7.3) * math.sin(t * 3.1) + 0.25 * float(self.kick[min(fi, len(self.kick) - 1)])
        core, glow, halo = stickers.draw(line["sticker"], prog, int(r), flick)
        left = W / 2 - line["width"] / 2
        right = W / 2 + line["width"] / 2
        cx = left + 10 if line["side"] == "left" else right - 10
        cx = min(max(cx, r * 1.3 + 15), W - r * 1.3 - 15)
        cy = by - 120 - r
        x, y = int(cx - core.shape[1] / 2), int(cy - core.shape[0] / 2)
        blend(fr, halo, x, y, DARK, 0.35 * la)
        blend(fr, glow, x, y, GOLD, 0.75 * la, "add")
        blend(fr, core, x, y, IVORY, 0.95 * la)

    # ------------------------------------------------------------ shots
    def build_shots(self):
        self.shots = self.cfg["shots"]

    def shot_frame(self, sh, t):
        u = smooth(ramp(t, sh["t0"], sh["t1"]))
        a, b = sh["from"], sh["to"]
        cx, cy, z = (a[i] + (b[i] - a[i]) * u for i in range(3))
        fi = min(int(t * FPS), len(self.kick) - 1)
        z *= 1 + 0.006 * float(self.kick[fi])
        pl = self.plates[sh["img"]]
        img, (s, x0, y0) = pl.view(cx, cy, z)
        if pl.moon:
            mx, my, mr = pl.moon
            sx, sy, sr = (mx * 2 - x0) * s, (my * 2 - y0) * s, mr * 2 * s
            amp = 0.07 + 0.05 * math.sin(t * 1.3) + 0.06 * float(self.loud[fi])
            self.moon_glow(img, sx, sy, sr, amp)
        return img

    def moon_glow(self, img, sx, sy, sr, amp):
        R = int(sr * 3.2)
        x0, y0 = int(sx - R), int(sy - R)
        x1, y1 = x0 + 2 * R, y0 + 2 * R
        if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H or R < 4:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        g = np.exp(-(((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * (sr * 1.3) ** 2)))[..., None] * amp
        blend(img, g[..., 0], x0, y0, np.array([1.0, 0.95, 0.85], np.float32), 1.0, "add")

    def background(self, t):
        shots = self.shots
        cur = [s for s in shots if s["t0"] - 0.0001 <= t < s["t1"] + 0.0001]
        sh = cur[0] if cur else shots[-1] if t >= shots[-1]["t1"] else shots[0]
        i = shots.index(sh)
        img = self.shot_frame(sh, t)
        td = self.cfg.get("trans", 1.4)
        # transition into next shot, centred on boundary
        if i + 1 < len(shots) and t > sh["t1"] - td / 2:
            nxt = shots[i + 1]
            w = smooth(ramp(t, sh["t1"] - td / 2, sh["t1"] + td / 2))
            img = self.mix(img, self.shot_frame(nxt, max(t, nxt["t0"])), w, sh.get("tstyle", "focus"))
        elif i > 0 and t < sh["t0"] + td / 2:
            prv = shots[i - 1]
            w = smooth(ramp(t, sh["t0"] - td / 2, sh["t0"] + td / 2))
            img = self.mix(self.shot_frame(prv, min(t, prv["t1"])), img, w, prv.get("tstyle", "focus"))
        return img

    def mix(self, a, b, w, style):
        peak = math.sin(math.pi * w)
        out = a * (1 - w) + b * w
        if peak > 0.05:
            out = cv2.GaussianBlur(out, (0, 0), 1 + 9 * peak)
        if style == "bloom":
            bl = cv2.GaussianBlur(cv2.resize(out, (W // 4, H // 4)), (0, 0), 12)
            out += cv2.resize(bl, (W, H)) * np.array([1.0, 0.85, 0.6], np.float32) * 0.35 * peak
        return out

    # ------------------------------------------------------------ credits
    def build_credits(self):
        c = self.cfg
        f = self.fonts
        self.title = Sprite(c["title"], f.get("gulzar", 160), dilate=3)
        self.roles = Sprite(c["roles"], f.get("messiri", 44))
        self.name = Sprite(c["name"], f.get("cinzel", 74), tracking=14, rtl=False)
        self.end_name = Sprite(c["name"], f.get("cinzel", 120), tracking=22, rtl=False)
        self.end_roles = Sprite(c["roles"], f.get("messiri", 50))
        self.wm_roles = Sprite(c["roles"], f.get("messiri", 24))
        self.wm_name = Sprite(c["name"], f.get("cinzel", 26), tracking=6, rtl=False)

    def text_at(self, fr, sp, cx, base, a, color, glow=0.0, rise=0.0, shadow=True):
        X = int(round(cx - sp.adv / 2)) - sp.pad
        Y = int(round(base - sp.base + rise))
        if shadow:
            blend(fr, sp.sh_large, X, Y + 4, DARK, 0.6 * a)
            blend(fr, sp.sh_small, X + 1, Y + 3, DARK, 0.75 * a)
        if glow:
            blend(fr, sp.glow, X, Y, GOLD, glow * a, "add")
        blend(fr, sp.mask, X, Y, color, a)
        return X, Y

    def sweep(self, fr, sp, X, Y, s, a):
        if 0 < s < 1:
            xx = np.arange(sp.w, dtype=np.float32)
            band = np.exp(-((xx - sp.w * (1.2 - 1.4 * s)) / (sp.w * 0.08)) ** 2)[None, :]
            blend(fr, sp.mask * band, X, Y, np.array([1, 1, 0.95], np.float32), 0.6 * a, "add")

    def icons(self, fr, cx, cy, a, t0, t, r=26, step=0.25, draw=0.8):
        names = ["quill", "music_note", "strings"]
        gap = r * 3.4
        for k, n in enumerate(names):
            prog = ramp(t, t0 + step * k, t0 + step * k + draw)
            if prog <= 0:
                continue
            core, glow, halo = stickers.draw(n, prog, r)
            x = int(cx + (1 - k) * gap - core.shape[1] / 2)
            y = int(cy - core.shape[0] / 2)
            blend(fr, halo, x, y, DARK, 0.4 * a)
            blend(fr, glow, x, y, GOLD, 0.7 * a, "add")
            blend(fr, core, x, y, IVORY, a)

    def credit_block(self, fr, t, c, y):
        """Title (gulzar gold, sweep), roles line with icons, name."""
        t0, t1 = c["t0"], c["t1"]
        out = 1 - smooth(ramp(t, t1 - 0.8, t1))
        a_t = ease_out(ramp(t, t0, t0 + 1.2)) * out
        if a_t <= 0:
            return
        sc = self.cfg.get("scrim_title", 0.5)
        blur_rise = 18 * (1 - ease_out(ramp(t, t0, t0 + 1.2)))
        sp = self.title
        blend(fr, self.title_scrim(), W // 2 - 520, int(y) - 230, DARK, sc * a_t)
        X, Y = self.text_at(fr, sp, W / 2, y, a_t, GOLD, glow=0.35, rise=blur_rise)
        self.sweep(fr, sp, X, Y, ramp(t, t0 + 0.9, t0 + 2.1), a_t)
        a_r = ease_out(ramp(t, t0 + 1.2, t0 + 2.0)) * out
        self.icons(fr, W / 2, y + 85, a_r, t0 + 1.2, t)
        self.text_at(fr, self.roles, W / 2, y + 170, a_r, IVORY, rise=10 * (1 - a_r / max(out, 1e-3)))
        a_n = ease_out(ramp(t, t0 + 2.0, t0 + 2.9)) * out
        self.text_at(fr, self.name, W / 2, y + 265, a_n, IVORY, glow=0.25, rise=10 * (1 - a_n / max(out, 1e-3)))

    def title_scrim(self):
        if not hasattr(self, "_ts"):
            sc = np.zeros((560, 1040), np.float32)
            cv2.ellipse(sc, (520, 280), (400, 190), 0, 0, 360, 1.0, -1)
            self._ts = cv2.GaussianBlur(sc, (0, 0), 60)
        return self._ts

    def end_card(self, fr, t, c):
        t0 = c["t0"]
        a = ease_out(ramp(t, t0, t0 + 1.0))
        if a <= 0:
            return
        y = c.get("y", 900)
        X, Y = self.text_at(fr, self.title, W / 2, y - 170, a, GOLD, glow=0.35, rise=14 * (1 - a))
        self.sweep(fr, self.title, X, Y, ramp(t, t0 + 0.6, t0 + 1.6), a)
        a_n = ease_out(ramp(t, t0 + 0.4, t0 + 1.2))
        self.text_at(fr, self.end_name, W / 2, y + 40, a_n, IVORY, glow=0.3, rise=10 * (1 - a_n))
        a_r = ease_out(ramp(t, t0 + 0.8, t0 + 1.5))
        self.icons(fr, W / 2, y + 130, a_r, t0 + 0.8, t, r=30, step=0.15, draw=0.5)
        self.text_at(fr, self.end_roles, W / 2, y + 230, a_r, IVORY, rise=10 * (1 - a_r))

    def watermark(self, fr, t):
        wm = self.cfg["watermark"]
        en = self.cfg["ending"]
        a = wm.get("alpha", 0.5) * (1 - smooth(ramp(t, en["t0"] - 0.6, en["t0"] + 0.4)))
        if a <= 0:
            return
        cy = wm.get("y", 960)
        self.text_at(fr, self.wm_roles, W / 2, cy - 8, a, IVORY, shadow=True)
        self.text_at(fr, self.wm_name, W / 2, cy + 30, a, IVORY, shadow=True)

    # ------------------------------------------------------------ particles
    def particles(self, fr, t, fi, section):
        d = self.dust
        bright = 0.35 + 0.65 * float(self.spark[fi])
        if section == "Bridge":
            return
        n = len(d["x"]) if section == "Final Chorus" else len(d["x"]) // 2
        for k in range(n):
            x = (d["x"][k] + d["vx"][k] * t) % W
            y = (d["y"][k] + d["vy"][k] * t) % H
            if 1300 < y < 1700:
                continue
            tw = 0.5 + 0.5 * math.sin(t * 2.1 + d["ph"][k])
            s = d["r"][k] / 3
            sz = max(5, int(33 * s)) | 1
            dot = cv2.resize(self.dot, (sz, sz))
            blend(fr, dot, int(x - sz / 2), int(y - sz / 2), GOLD, 0.55 * tw * bright, "add")
        if section == "Final Chorus":
            p = self.petals
            for k in range(len(p["x"])):
                y = (p["y"][k] + p["vy"][k] * t) % (H + 200) - 100
                if 1300 < y < 1700:
                    continue
                x = p["x"][k] + p["sw"][k] * math.sin(t * 0.9 + p["ph"][k])
                s = p["s"][k]
                m = np.zeros((int(s * 3), int(s * 3)), np.float32)
                ang = (t * 40 + p["ph"][k] * 57) % 360
                cv2.ellipse(m, (m.shape[1] // 2, m.shape[0] // 2), (int(s), int(s * 0.55)), ang, 0, 360, 1.0, -1, cv2.LINE_AA)
                m = cv2.GaussianBlur(m, (0, 0), 1.2)
                blend(fr, m, int(x), int(y), np.array([0.62, 0.05, 0.1], np.float32), 0.8)

    # ------------------------------------------------------------ frame
    def section_at(self, t):
        sec = None
        for line in self.lines:
            if line["t_in"] - 1.0 <= t:
                sec = line["section"]
        return sec or self.lines[0]["section"]

    def frame(self, fi):
        t = fi / FPS
        c = self.cfg
        fi_c = min(fi, len(self.kick) - 1)
        fr = self.background(t)
        sec = self.section_at(t)
        # opening: emerge from dark soft focus
        op = c["opening"]
        e = smooth(ramp(t, 0, op.get("emerge", 2.0)))
        if e < 1:
            fr = cv2.GaussianBlur(fr, (0, 0), 1 + 14 * (1 - e)) * (0.15 + 0.85 * e)
        deep = sec == "Bridge"
        if deep:
            fr = np.power(np.clip(fr, 0, 1), 1.12)
        fr *= self.vig_deep if deep else self.vig
        self.particles(fr, t, fi_c, sec)
        # ending darken
        en = c["ending"]
        dk = smooth(ramp(t, en["t0"] - 0.5, en["t0"] + 1.5))
        if dk > 0:
            fr *= 1 - 0.5 * dk
        base_y = self.base_y(t)
        self.draw_lines(fr, t, base_y)
        if op["t0"] <= t <= op["t1"]:
            self.credit_block(fr, t, op, op.get("y", 420))
        if en["t0"] <= t <= en["t1"]:
            self.end_card(fr, t, en)
        self.watermark(fr, t)
        # fade to black at the very end
        fb = smooth(ramp(t, self.duration - 1.1, self.duration - 0.05))
        if fb > 0:
            fr *= 1 - fb
        g = cv2.resize(self.grain[fi % len(self.grain)], (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
        fr += g
        return (np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8)

    def base_y(self, t):
        sh = self.shots[0]
        for s in self.shots:
            if s["t0"] <= t:
                sh = s
        return sh.get("text_y", 1520)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--stills")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--range", nargs=2, type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    v = Video(cfg)
    if a.stills:
        os.makedirs(a.outdir, exist_ok=True)
        for s in a.stills.split(","):
            t = float(s)
            img = v.frame(int(round(t * FPS)))
            cv2.imwrite(os.path.join(a.outdir, f"still_{t:07.2f}.png"), img[:, :, ::-1])
        return
    f0, f1 = a.range
    f1 = min(f1, int(v.duration * FPS))
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                          "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        p.stdin.write(v.frame(fi).tobytes())
        if fi % 150 == 0:
            print(f"{a.out}: {fi}/{f1}", file=sys.stderr, flush=True)
    p.stdin.close()
    p.wait()


if __name__ == "__main__":
    main()
