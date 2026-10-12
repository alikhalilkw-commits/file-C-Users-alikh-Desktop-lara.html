"""Renderer for «حَمَاكِ الإِلَهُ بِلَادَ الحَرَمْ» (Saudi cinematic lyric video).

Implements the brief as revised by Ali: animated full-line captions on dark panels with the sung word in gold,
restrained camera, selective beat accents, supplied reference images only (no drawn flags), opening title,
fixed centred watermark, and a closing card after the song: «أَرْضُ العِزِّ وَالكَرَمْ» written on, gold clasped
hands, the message and the credits.

usage:
  python render_haram.py config.json --stills 3,40.5 --outdir stills/
  python render_haram.py config.json --range 0 900 --out part0.mp4
"""
import argparse
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import haram_assets  # noqa: E402
import render as R  # noqa: E402
import stickers  # noqa: E402
from render import FPS, H, W, blend, ease_out, ramp, smooth  # noqa: E402

IVORY = np.array([1.0, 0.968, 0.905], np.float32)
GOLD = np.array([0.965, 0.835, 0.585], np.float32)
GREEN = np.array([0.20, 0.52, 0.33], np.float32)
PANEL = np.array([0.018, 0.055, 0.04], np.float32)
DARK = np.array([0.0, 0.0, 0.0], np.float32)

GRADES = {  # (multiply, shadow tint, highlight tint, contrast)
    "H": (np.array([1.04, 1.0, 0.93]), np.array([0.0, 0.0, 0.0]), np.array([0.03, 0.02, -0.01]), 1.04),
    "R": (np.array([1.0, 1.0, 1.0]), np.array([-0.02, 0.0, 0.035]), np.array([0.035, 0.02, -0.02]), 1.08),
    "A": (np.array([1.0, 1.0, 1.0]), np.array([-0.02, 0.0, 0.03]), np.array([0.04, 0.02, -0.02]), 1.09),
    "P": (np.array([1.0, 1.0, 1.0]), np.zeros(3), np.zeros(3), 1.0),
    "X": (np.array([1.02, 1.0, 0.97]), np.zeros(3), np.array([0.02, 0.01, -0.01]), 1.04),
}


def grade(img, cat):
    mul, sh, hi, con = GRADES[cat]
    img = img * mul.astype(np.float32)
    if con != 1.0:
        img = (img - 0.5) * con + 0.5
    if sh.any() or hi.any():
        luma = img.mean(2, keepdims=True)
        img = img + sh.astype(np.float32) * (1 - luma) + hi.astype(np.float32) * luma
    return img


class Plate:
    """Image resampled to a 2160x3840 plate (exact 9:16) with an optional half-res depth map."""

    def __init__(self, path, depth_path=None):
        im = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
        up = cv2.resize(im, (2 * W, 2 * H), interpolation=cv2.INTER_LANCZOS4)
        k = 2 * W / im.shape[1]
        if k > 2.2:  # low-resolution source: gentle unsharp only
            bl = cv2.GaussianBlur(up, (0, 0), 2.5)
            up = up + 0.25 * (up - bl)
        else:
            bl = cv2.GaussianBlur(up, (0, 0), 1.6)
            up = up + 0.35 * (up - bl)
        self.img = np.clip(up, 0, 1)
        self.depth = None
        if depth_path and os.path.exists(depth_path):
            d = np.load(depth_path).astype(np.float32)
            self.depth = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)

    def view(self, cx, cy, z):
        cw = 2 * W / z
        ch = cw * H / W
        x0 = min(max(cx * 2 * W - cw / 2, 0), 2 * W - cw)
        y0 = min(max(cy * 2 * H - ch / 2, 0), 2 * H - ch)
        s = W / cw
        M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
        out = cv2.warpAffine(self.img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        dep = None
        if self.depth is not None:
            Md = np.array([[s * 2, 0, -x0 * s], [0, s * 2, -y0 * s]], np.float32)
            dep = cv2.warpAffine(self.depth, Md, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return out, dep


def rounded_panel(w, h, r=34, soft=5):
    m = np.zeros((h + 4 * soft, w + 4 * soft), np.float32)
    x0, y0, x1, y1 = 2 * soft, 2 * soft, 2 * soft + w, 2 * soft + h
    cv2.rectangle(m, (x0 + r, y0), (x1 - r, y1), 1.0, -1)
    cv2.rectangle(m, (x0, y0 + r), (x1, y1 - r), 1.0, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(m, (cx, cy), r, 1.0, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), soft), 2 * soft


class Video:
    def __init__(self, cfg):
        self.cfg = cfg
        haram_assets.register(cfg["icons"])
        self.fonts = R.Fonts(cfg["fonts"])
        self.fonts.d = cfg["fonts"]
        wd = json.load(open(cfg["words"]))
        self.duration = wd["duration"]
        self.total = self.duration + cfg.get("tail", 0.0)
        self.nframes = int(round(self.total * FPS))
        f = np.load(cfg["feats"])
        self.kick, self.loud = f["kick"], f["loud"]
        beats = f["beats"]
        kb = np.array([self.kick[min(int(b * FPS), len(self.kick) - 1)] for b in beats])
        thr = np.percentile(kb, 70)
        self.accents = np.array([b for b, k in zip(beats, kb) if k >= thr and b > cfg["accent_from"]], np.float32)
        self.plates = {}
        for k, v in cfg["images"].items():
            self.plates[k] = Plate(os.path.join(cfg["image_dir"], v["file"]),
                                   os.path.join(cfg["depth_dir"], k + ".npy") if v.get("depth") else None)
        self.vig = R.vignette(0.5)
        rng = np.random.default_rng(11)
        self.grain = [rng.normal(0, 0.010, (H // 2, W // 2, 1)).astype(np.float32) for _ in range(6)]
        self.shots = cfg["shots"]
        self.build_lines(wd["words"])
        self.build_static_text()

    # ---------------------------------------------------------------- fonts
    def font(self, name, size):
        key = (name, size)
        if key not in self.fonts.cache:
            from PIL import ImageFont
            d = self.cfg["fonts"]
            if name == "kufi":
                f = ImageFont.truetype(f"{d}/ReemKufi.ttf", size, layout_engine=ImageFont.Layout.RAQM)
                f.set_variation_by_axes([700])
                self.fonts.cache[key] = f
            else:
                return self.fonts.get(name, size)
        return self.fonts.cache[key]

    # ---------------------------------------------------------------- captions
    def build_lines(self, words):
        lines = {}
        for w in words:
            lines.setdefault(w["line"], []).append(w)
        order = sorted(lines)
        meta = self.cfg["lines"]
        self.lines = []
        for i, li in enumerate(order):
            ws = lines[li]
            m = meta[i]
            nxt = lines[order[i + 1]][0]["start"] if i + 1 < len(order) else self.total
            start, end = ws[0]["start"], ws[-1]["end"]
            line = dict(words=ws, idx=i, start=start, end=end, t_in=start - 0.4,
                        t_out=min(nxt - 0.3, end + m.get("hold", 1.3)), icon=m.get("icon"),
                        side=m.get("side", "left"), y=m.get("y", 1500), title=m.get("title", False))
            self.layout(line, m.get("size", 70))
            self.lines.append(line)

    def layout(self, line, size):
        font = self.font("kufi", size) if line["title"] else self.font("messiri", size)
        sprites = [R.Sprite(w["text"], font) for w in line["words"]]
        space = font.getlength(" ") * (1.5 if line["title"] else 1.2)
        widths = [s.ink_w for s in sprites]
        total = sum(widths) + space * (len(sprites) - 1)
        rows = [list(range(len(sprites)))]
        if total > 880 and len(sprites) > 2:
            best = None
            for k in range(1, len(sprites)):
                a = sum(widths[:k]) + space * (k - 1)
                b = sum(widths[k:]) + space * (len(sprites) - k - 1)
                if best is None or max(a, b) < best[0]:
                    best = (max(a, b), k)
            rows = [list(range(best[1])), list(range(best[1], len(sprites)))]
        asc = max(s.base - s.pad for s in sprites)
        desc = max(s.h - s.base - s.pad for s in sprites)
        row_h = int(asc + desc + size * 0.28)
        place = {}
        maxw = 0
        for r, idxs in enumerate(rows):
            rw = sum(widths[i] for i in idxs) + space * (len(idxs) - 1)
            maxw = max(maxw, rw)
            x = W / 2 + rw / 2
            for i in idxs:
                x -= widths[i]
                place[i] = (x, r)
                x -= space
        line.update(sprites=sprites, place=place, rows=len(rows), row_h=row_h, asc=asc, width=maxw, size=size)
        pw, ph = int(maxw + 2 * 52), int(row_h * len(rows) + 40)
        line["panel"], line["panel_pad"] = rounded_panel(pw, ph, r=34)
        line["panel_wh"] = (pw, ph)

    def word_glow(self, line, t, w):
        g_in = ramp(t, w["start"] - 0.04, w["start"] + 0.08)
        g_out = 1 - smooth(ramp(t, w["end"], w["end"] + 0.22))
        return g_in * g_out

    def draw_caption(self, fr, t, line, base_alpha=1.0):
        a = smooth(ramp(t, line["t_in"], line["t_in"] + 0.3)) * (1 - smooth(ramp(t, line["t_out"] - 0.3, line["t_out"])))
        a *= base_alpha
        if a <= 0.003:
            return
        pw0, ph = line["panel_wh"]
        pe = ease_out(ramp(t, line["t_in"], line["t_in"] + 0.38))
        ex = smooth(ramp(t, line["t_out"] - 0.3, line["t_out"]))
        cy = line["y"] - 10 * ex
        # panel opens from the centre
        sx = 0.86 + 0.14 * pe
        pw = int(pw0 * sx)
        pm0, pp = line["panel"], line["panel_pad"]
        pm = pm0 if sx >= 0.999 else cv2.resize(pm0, (pw + 2 * pp, pm0.shape[0]), interpolation=cv2.INTER_LINEAR)
        px0, py0 = int(W / 2 - pw / 2), int(cy - ph / 2)
        # soft background blur under the panel, then the dark panel
        x0, y0 = max(0, px0 - pp), max(0, py0 - pp)
        x1, y1 = min(W, px0 - pp + pm.shape[1]), min(H, py0 - pp + pm.shape[0])
        reg = fr[y0:y1, x0:x1]
        bl = cv2.GaussianBlur(reg, (0, 0), 9)
        mm = pm[y0 - (py0 - pp):y1 - (py0 - pp), x0 - (px0 - pp):x1 - (px0 - pp), None]
        reg[:] = reg * (1 - 0.7 * a * mm) + bl * (0.7 * a * mm)
        blend(fr, pm, px0 - pp, py0 - pp, PANEL, 0.66 * a)
        # thin restrained green-gold rule on the panel's top edge
        top = np.zeros((3, max(10, int((pw - 120) * pe))), np.float32) + 1
        blend(fr, top, int(W / 2 - top.shape[1] / 2), py0 + 6, GOLD * 0.6 + GREEN * 0.4, 0.35 * a)
        row0 = cy - (line["rows"] * line["row_h"]) / 2 + line["asc"] + 14
        a_line = a
        for i, (w, sp) in enumerate(zip(line["words"], line["sprites"])):
            x, r = line["place"][i]
            by = row0 + r * line["row_h"]
            g = self.word_glow(line, t, w)
            # words cascade in right-to-left, each with a short rise
            wa = ease_out(ramp(t, line["t_in"] + 0.06 + 0.05 * i, line["t_in"] + 0.36 + 0.05 * i))
            a = a_line * wa
            X = int(round(x)) - sp.ink_l
            Y = int(round(by + 14 * (1 - wa))) - sp.base
            if g > 0.01:
                s = 1 + 0.06 * g
                mask = cv2.resize(sp.mask, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                glow = cv2.resize(sp.glow, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                cxw, cyw = X + sp.ink_l + sp.ink_w / 2, Y + sp.base - line["size"] * 0.35
                Xs = int(round(cxw - (sp.ink_l + sp.ink_w / 2) * s))
                Ys = int(round(cyw - (sp.base - line["size"] * 0.35) * s))
                blend(fr, glow, Xs, Ys, GOLD, 0.32 * g * a, "add")
                col = IVORY * (1 - g) + GOLD * g
                blend(fr, mask, Xs, Ys, col, a)
            else:
                blend(fr, sp.mask, X, Y, IVORY, a)
        a = a_line
        if line["icon"]:
            prog = ramp(t, line["t_in"] + 0.15, line["t_in"] + 0.95)
            r = 46
            core, glow, halo = stickers.draw(line["icon"], prog, r)
            ix = px0 + 20 if line["side"] == "left" else px0 + pw - 20
            iy = py0 - r * 1.05
            X, Y = int(ix - core.shape[1] / 2), int(iy - core.shape[0] / 2)
            blend(fr, halo, X, Y, DARK, 0.4 * a)
            blend(fr, glow, X, Y, GOLD, 0.55 * a, "add")
            blend(fr, core, X, Y, GOLD * 0.5 + IVORY * 0.5, 0.95 * a)

    # ---------------------------------------------------------------- static text
    def build_static_text(self):
        c = self.cfg
        f = self.font
        self.credit_ar = R.Sprite(c["credit_ar"], f("messiri", 40))
        self.credit_name = R.Sprite("AHMED", f("cinzel", 58), tracking=12, rtl=False)
        self.wm_ar = R.Sprite(c["credit_ar"], f("messiri", 27))
        self.wm_bar = R.Sprite("|", f("cinzel", 28), rtl=False)
        self.wm_name = R.Sprite("AHMED", f("cinzel", 26), tracking=4, rtl=False)
        self.end_head = R.Sprite("أَرْضُ العِزِّ  وَالكَرَمْ", f("kufi", 112))
        self.end_msg1 = R.Sprite("مِنْ قَلْبٍ مِصْرِيٍّ مُحِبٍّ…", f("messiri", 50))
        self.end_msg2 = R.Sprite("إِلَى السُّعُودِيَّةِ أَرْضِ العِزِّ وَالكَرَمْ", f("messiri", 50))
        self.end_credit_ar = R.Sprite(c["credit_ar"], f("messiri", 36))
        self.end_credit_name = R.Sprite("AHMED", f("cinzel", 52), tracking=12, rtl=False)
        self.pledge = R.Sprite(c["opening"]["pledge"], f("messiri", 50)) if c["opening"].get("pledge") else None
        self.card_gulf = []
        for path in c.get("gulf_flags", []):
            g = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
            self.card_gulf.append(cv2.resize(g, (96, 72), interpolation=cv2.INTER_AREA))
        self.end_sa = haram_assets.WavingFlag(c["flag_sa"], 190)
        self.embrace_rows = []
        for row in c.get("embrace_rows", []):
            imgs = []
            for path in row:
                g = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
                imgs.append(cv2.resize(g, (60, 45), interpolation=cv2.INTER_AREA))
            self.embrace_rows.append(imgs)
        self.arab_flags = []
        for path in c.get("arab_flags", []):
            g = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
            self.arab_flags.append(cv2.resize(g, (78, 58), interpolation=cv2.INTER_AREA))
        if c["opening"].get("card"):
            self.card_title = R.Sprite(c["title"], f("kufi", 96))
            self.card_sa = haram_assets.WavingFlag(c["flag_sa"], 190)
            self.card_eg = haram_assets.WavingFlag(c["flag_eg"], 190)

    def put(self, fr, sp, cx, base, a, color, glow=0.0, shadow=True):
        X = int(round(cx - sp.adv / 2)) - sp.pad
        Y = int(round(base - sp.base))
        if shadow:
            blend(fr, sp.sh_large, X, Y + 4, DARK, 0.55 * a)
            blend(fr, sp.sh_small, X + 1, Y + 3, DARK, 0.7 * a)
        if glow:
            blend(fr, sp.glow, X, Y, GOLD, glow * a, "add")
        blend(fr, sp.mask, X, Y, color, a)
        return X, Y

    def sweep(self, fr, sp, X, Y, s, a):
        if 0 < s < 1:
            xx = np.arange(sp.w, dtype=np.float32)
            band = np.exp(-((xx - sp.w * (1.2 - 1.4 * s)) / (sp.w * 0.08)) ** 2)[None, :]
            blend(fr, sp.mask * band, X, Y, np.array([1, 1, 0.95], np.float32), 0.55 * a, "add")

    def watermark(self, fr, t):
        wm = self.cfg["watermark"]
        a = wm["alpha"]
        for t0, t1 in wm["emphasis"]:
            a = max(a, wm["alpha_hi"] * smooth(ramp(t, t0, t0 + 0.6)) * (1 - smooth(ramp(t, t1 - 0.6, t1))))
        y = wm["y"]
        gap = 14
        total = self.wm_name.adv + gap + self.wm_bar.adv + gap + self.wm_ar.ink_w
        x = W / 2 - total / 2
        col = IVORY * 0.85 + GOLD * 0.15
        # visual order (RTL paragraph): AHMED | <arabic credit>
        self.put(fr, self.wm_name, x + self.wm_name.adv / 2, y, a, col)
        x += self.wm_name.adv + gap
        self.put(fr, self.wm_bar, x + self.wm_bar.adv / 2, y, a * 0.8, col)
        x += self.wm_bar.adv + gap
        self.put(fr, self.wm_ar, x + self.wm_ar.ink_w / 2 - (self.wm_ar.ink_w - self.wm_ar.adv) / 2, y, a, col)

    # ---------------------------------------------------------------- scenes
    def accent(self, t):
        i = np.searchsorted(self.accents, t) - 1
        if i < 0:
            return 0.0
        return math.exp(-(t - float(self.accents[i])) / 0.16)

    def cam(self, sh, t):
        u = smooth(ramp(t, sh["t0"], sh["t1"]))
        a, b = sh["from"], sh["to"]
        return [a[i] + (b[i] - a[i]) * u for i in range(3)], u

    def scene_plate(self, sh, t):
        (cx, cy, z), u = self.cam(sh, t)
        cat = sh["img"][0]
        acc = self.accent(t) if sh.get("react", True) else 0.0
        portrait = cat == "P"
        z *= 1 + (0.0015 if portrait else 0.004) * acc
        img, dep = self.plates[sh["img"]].view(cx, cy, z)
        disp = 0 if portrait else 2.0 * acc
        par = sh.get("par", 0.0)
        if dep is not None and par:
            dd = dep - float(np.median(dep))
            dx = dd * par * (1 - 2 * u) * sh.get("pdir", 1)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            img = cv2.remap(img, xx - dx, yy - dd * 0.0 + disp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        elif disp > 0.3:
            M = np.array([[1, 0, 0], [0, 1, -disp]], np.float32)
            img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        img = grade(img, cat)
        if sh.get("warm"):
            img = img * np.array([1.06, 1.0, 0.9], np.float32)
        return img

    def scene(self, sh, t):
        return self.scene_plate(sh, t)

    def shot_at(self, t):
        for i, sh in enumerate(self.shots):
            if t < sh["t1"]:
                return i
        return len(self.shots) - 1

    def background(self, t):
        i = self.shot_at(t)
        sh = self.shots[i]
        img = self.scene(sh, t)
        nxt = self.shots[i + 1] if i + 1 < len(self.shots) else None
        td = sh.get("td", 0.45)
        if nxt and t > sh["t1"] - td / 2:
            w = smooth(ramp(t, sh["t1"] - td / 2, sh["t1"] + td / 2))
            img = self.mix(img, self.scene(nxt, t), w, sh.get("tstyle", "dissolve"))
        elif i > 0:
            prv = self.shots[i - 1]
            tdp = prv.get("td", 0.45)
            if t < prv["t1"] + tdp / 2:
                w = smooth(ramp(t, prv["t1"] - tdp / 2, prv["t1"] + tdp / 2))
                img = self.mix(self.scene(prv, t), img, w, prv.get("tstyle", "dissolve"))
        return img

    def mix(self, a, b, w, style):
        if style == "wipe":  # soft directional wipe, right to left
            edge = W * (1.15 - 1.3 * w)
            xx = np.arange(W, dtype=np.float32)
            m = np.clip((xx - edge) / 140 + 0.5, 0, 1)[None, :, None]
            return a * (1 - m) + b * m
        out = a * (1 - w) + b * w
        peak = math.sin(math.pi * w)
        if style == "light":
            out = out + np.array([1.0, 0.86, 0.6], np.float32) * 0.28 * peak
        return out

    # ---------------------------------------------------------------- frame
    def frame(self, fi):
        t = fi / FPS
        c = self.cfg
        fr = self.background(t)
        # limited bloom
        small = cv2.resize(fr, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        br = cv2.GaussianBlur(np.clip(small - 0.7, 0, 1), (0, 0), 6)
        fr += cv2.resize(br, (W, H)) * 0.3
        fr *= self.vig
        en = c["ending"]
        # closing card after the song: darken the Kaaba with a curtain-close, then fade the background toward black
        if t >= en["t0"]:
            k = smooth(ramp(t, en["t0"], en["t0"] + 1.8))
            xs = np.linspace(-1, 1, W, dtype=np.float32)
            curtain = 1 - 0.6 * k * np.clip(np.abs(xs) * 1.4 - 0.3, 0, 1) ** 1.5
            fr *= (1 - 0.6 * k) * curtain[None, :, None]
            fb = smooth(ramp(t, en["black"] - 1.2, self.total))
            fr *= 1 - 0.9 * fb
        # opening title (also the timed caption of the first sung line)
        op = c["opening"]
        if op.get("card"):
            k = 1 - smooth(ramp(t, op["card_end"], op["card_end"] + op["card_fade"]))
            if k > 0:
                fr *= 1 - k
                self.intro_card(fr, t, op, k)
        for line in self.lines:
            if line["title"]:
                self.draw_title(fr, t, line, op)
            elif line["t_in"] <= t <= line["t_out"] + 0.01:
                self.draw_caption(fr, t, line)
        if t >= en["t0"]:
            self.ending(fr, t, en)
        g = cv2.resize(self.grain[fi % len(self.grain)], (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
        fr += g
        bk = op.get("black", 0.0)
        if bk and t < bk:
            fr *= smooth(t / bk)
        self.watermark(fr, t)
        return (np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8)

    def draw_title(self, fr, t, line, op):
        """Title reveal over the moving Haram; becomes the first lyric panel when the voice enters."""
        card = op.get("card", False)
        a_in = ease_out(ramp(t, line["t_in"], line["t_in"] + 0.45)) if card else ease_out(ramp(t, op["t0"], op["t0"] + 1.4))
        a_out = 1 - smooth(ramp(t, line["t_out"] - 0.35, line["t_out"]))
        a = a_in * a_out
        if a <= 0.003:
            return
        rise = 24 * (1 - a_in)
        line_y = line["y"] + rise
        saved = line["y"]
        line["y"] = line_y
        # resting colour: gold during the intro, ivory once the voice starts (then gold on the sung word)
        sing = smooth(ramp(t, line["start"] - 0.6, line["start"] - 0.1))
        pw, ph = line["panel_wh"]
        px0, py0 = int(W / 2 - pw / 2), int(line_y - ph / 2)
        blend(fr, line["panel"], px0 - line["panel_pad"], py0 - line["panel_pad"], PANEL, 0.62 * a)
        row0 = line_y - (line["rows"] * line["row_h"]) / 2 + line["asc"] + 14
        for i, (w, sp) in enumerate(zip(line["words"], line["sprites"])):
            x, r = line["place"][i]
            X = int(round(x)) - sp.ink_l
            Y = int(round(row0 + r * line["row_h"])) - sp.base
            g = max(1 - sing, self.word_glow(line, t, w))
            blend(fr, sp.sh_large, X, Y + 4, DARK, 0.5 * a)
            blend(fr, sp.glow, X, Y, GOLD, 0.3 * g * a, "add")
            blend(fr, sp.mask, X, Y, IVORY * (1 - g) + GOLD * g, a)
            if i == 0:
                pass
        if card:
            line["y"] = saved
            return
        # light sweep across the whole title once
        s = ramp(t, op["t0"] + 1.0, op["t0"] + 2.6)
        if 0 < s < 1:
            xx = np.arange(W, dtype=np.float32)
            band = np.exp(-((xx - W * (1.15 - 1.3 * s)) / 70) ** 2)[None, :]
            y0 = int(py0)
            reg = fr[y0:y0 + ph]
            reg += band[..., None] * 0.18 * a * np.array([1, 0.95, 0.8], np.float32)
        # small Kaaba ornament above the title
        prog = ramp(t, op["t0"] + 0.4, op["t0"] + 1.6)
        core, glow, halo = stickers.draw("kaaba", prog, 40)
        X, Y = int(W / 2 - core.shape[1] / 2), int(py0 - 58 - core.shape[0] / 2)
        blend(fr, glow, X, Y, GOLD, 0.55 * a, "add")
        blend(fr, core, X, Y, GOLD * 0.6 + IVORY * 0.4, a)
        # credits under the title, only during the instrumental intro
        ca = ease_out(ramp(t, op["t0"] + 1.3, op["t0"] + 2.3)) * (1 - smooth(ramp(t, op["credits_out"] - 0.6, op["credits_out"])))
        if ca > 0:
            if not hasattr(self, "_cpanel"):
                self._cpanel = rounded_panel(560, 170, r=30)
            cpm, cpp = self._cpanel
            blend(fr, cpm, int(W / 2 - 280) - cpp, py0 + ph + 14 - cpp, PANEL, 0.6 * ca * a_out)
            self.put(fr, self.credit_ar, W / 2, py0 + ph + 64, ca * a_out, IVORY)
            X2, Y2 = self.put(fr, self.credit_name, W / 2, py0 + ph + 140, ca * a_out, GOLD, glow=0.25)
            self.sweep(fr, self.credit_name, X2, Y2, ramp(t, op["t0"] + 2.2, op["t0"] + 3.2), ca)
        # pledge line, written on after the name has landed
        if self.pledge is not None:
            pa = ease_out(ramp(t, op["t0"] + 2.6, op["t0"] + 3.1)) * (1 - smooth(ramp(t, op["credits_out"] - 0.6, op["credits_out"]))) * a_out
            if pa > 0:
                if not hasattr(self, "_ppanel"):
                    self._ppanel = rounded_panel(int(self.pledge.ink_w + 90), 96, r=28)
                pm, ppp = self._ppanel
                py = py0 + ph + 262
                blend(fr, pm, int(W / 2 - pm.shape[1] / 2 + ppp) - ppp, int(py - 66) - ppp, PANEL, 0.62 * pa)
                self.write_on(fr, self.pledge, W / 2, py, smooth(ramp(t, op["t0"] + 2.7, op["t0"] + 4.3)), pa, GOLD, glow=0.25)
        line["y"] = saved

    def write_on(self, fr, sp, cx, base, p, a, color, glow=0.0):
        """Reveal a sprite right-to-left as if written, with a soft gold leading edge."""
        if a <= 0.003 or p <= 0:
            return
        X = int(round(cx - sp.adv / 2)) - sp.pad
        Y = int(round(base - sp.base))
        xx = np.arange(sp.w, dtype=np.float32)
        edge = sp.w * (1 - p * 1.08) + 0.04 * sp.w
        m = np.clip((xx - edge) / 26 + 0.5, 0, 1)[None, :]
        blend(fr, sp.sh_large * m, X, Y + 4, DARK, 0.55 * a)
        if glow:
            blend(fr, sp.glow * m, X, Y, GOLD, glow * a, "add")
        blend(fr, sp.mask * m, X, Y, color, a)
        if p < 1:
            lead = np.exp(-((xx - edge) / 14) ** 2)[None, :]
            blend(fr, sp.glow * lead, X, Y, np.array([1, 0.92, 0.7], np.float32), 0.9 * a, "add")

    def intro_card(self, fr, t, op, k):
        """Black opening card: title descends and writes on, official flags beside gold clasped hands,
        the pledge line, then the credits (AHMED) as before."""
        y = op["card_title_y"]
        d = ease_out(ramp(t, 0.3, 2.1))
        p = smooth(ramp(t, 0.4, 2.0))
        X0 = int(round(W / 2 - self.card_title.adv / 2)) - self.card_title.pad
        ty = y - 130 * (1 - d)
        self.write_on(fr, self.card_title, W / 2, ty, p, k * smooth(ramp(t, 0.3, 0.8)), GOLD, glow=0.4)
        if t > 2.1:
            self.sweep(fr, self.card_title, X0, int(round(ty - self.card_title.base)), ramp(t, 2.1, 3.2), k)
        # flags slide in gently from the sides, hands drawn between them
        fa = ease_out(ramp(t, 1.8, 2.9)) * k
        fy = op["card_flags_y"]
        for j, fl in enumerate((self.card_sa, self.card_eg)):
            rgb, al = fl.frame(t + j * 0.37, amp=0.45)
            slide = (1 - ease_out(ramp(t, 1.8, 2.9))) * 120
            x = int(80 - slide) if j == 0 else int(W - 80 - fl.w + slide)
            yy = fy - fl.h // 2 - fl.pad
            blend(fr, cv2.GaussianBlur(al, (0, 0), 10), x + 10, yy + 12, DARK, 0.5 * fa)
            h, w = al.shape
            x0c, x1c = max(0, x), min(W, x + w)
            reg = fr[yy:yy + h, x0c:x1c]
            alc = al[:reg.shape[0], x0c - x:x1c - x, None] * fa
            reg[:] = reg * (1 - alc) + rgb[:reg.shape[0], x0c - x:x1c - x] * alc
        prog = ramp(t, 2.2, 3.4)
        if prog > 0:
            core, glow, halo = stickers.draw("handshake", prog, 92)
            X, Y = int(W / 2 - core.shape[1] / 2), int(fy - core.shape[0] / 2)
            blend(fr, glow, X, Y, GOLD, 0.9 * k, "add")
            blend(fr, core, X, Y, GOLD * 0.7 + IVORY * 0.3, k)
        # pledge, written on in gold
        if self.pledge is not None:
            self.write_on(fr, self.pledge, W / 2, op["card_pledge_y"], smooth(ramp(t, 3.3, 4.8)),
                          k * ease_out(ramp(t, 3.2, 3.7)), GOLD, glow=0.25)
        # credits: same entrance as before (roles line, then AHMED with a light sweep)
        cy = op["card_credit_y"]
        ca = ease_out(ramp(t, 4.6, 5.4)) * k
        self.put(fr, self.credit_ar, W / 2, cy + 12 * (1 - ca / max(k, 1e-3)), ca, IVORY)
        cn = ease_out(ramp(t, 5.1, 5.9)) * k
        X2, Y2 = self.put(fr, self.credit_name, W / 2, cy + 78 + 12 * (1 - cn / max(k, 1e-3)), cn, GOLD, glow=0.25)
        self.sweep(fr, self.credit_name, X2, Y2, ramp(t, 5.9, 6.9), cn)

    def flag_row(self, fr, t, t0, flags, gy, tw, th, gap, k=1.0):
        """A centred row of official flags (first one rightmost, reading right-to-left), staggered fade-rise."""
        n = len(flags)
        total = n * tw + (n - 1) * gap
        for j, img in enumerate(flags):
            ga = ease_out(ramp(t, t0 + 0.07 * j, t0 + 0.45 + 0.07 * j)) * k
            if ga <= 0:
                continue
            x = int(W / 2 + total / 2 - (j + 1) * tw - j * gap)
            y = int(gy - th / 2 + 12 * (1 - ga / max(k, 1e-3)))
            sh = np.zeros((th + 40, tw + 40), np.float32)
            sh[20:20 + th, 20:20 + tw] = 1
            blend(fr, cv2.GaussianBlur(sh, (0, 0), 7), x - 14, y - 12, DARK, 0.55 * ga)
            reg = fr[y:y + th, x:x + tw]
            reg[:] = reg * (1 - ga) + img[:reg.shape[0], :reg.shape[1]] * ga
            fm = np.zeros((th + 4, tw + 4), np.float32)
            cv2.rectangle(fm, (0, 0), (tw + 3, th + 3), 1.0, 2)
            blend(fr, fm, x - 2, y - 2, GOLD, 0.7 * ga)

    def embrace_geometry(self, en):
        """Two gold arms leaving the Saudi flag and wrapping around the Arab flags, meeting in clasped hands."""
        top = en["sa_y"] + self.end_sa.h // 2
        hx = self.end_sa.w // 2 - 12
        by = en["hands_y"]
        ctrl = [(540 - hx, top), (330, top + 18), (190, top + 95), (165, top + 200),
                (225, top + 285), (370, by - 30), (500, by - 4)]
        pts = np.array(ctrl, np.float32)
        # Catmull-Rom through the control points
        P = np.vstack([pts[0], pts, pts[-1]])
        out = []
        for k in range(1, len(P) - 2):
            for u in np.linspace(0, 1, 40, endpoint=False):
                p0, p1, p2, p3 = P[k - 1], P[k], P[k + 1], P[k + 2]
                out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
        out.append(pts[-1])
        left = np.array(out, np.float32)
        right = left.copy()
        right[:, 0] = W - right[:, 0]
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(left, axis=0), axis=1))]
        return left, right, seg / seg[-1]

    def embrace(self, fr, t, w0, en):
        """Luminous Saudi flag; the Arab flags fly out from behind it; 3D gold arms wrap around and the
        hands close onto each other beneath them."""
        import embrace_art
        cx = W / 2
        sa = self.end_sa
        fy = en["sa_y"]
        fa = ease_out(ramp(t, w0 + 0.5, w0 + 1.3))
        # radiant light behind the flag
        if fa > 0:
            if not hasattr(self, "_rays"):
                yy, xx = np.mgrid[0:900, 0:W].astype(np.float32)
                ang = np.arctan2(yy - 450, xx - cx)
                r = np.hypot(yy - 450, xx - cx)
                self._rays = (ang, r)
            ang, r = self._rays
            rays = (0.55 + 0.45 * np.cos(ang * 14 + t * 0.6)) * np.exp(-r / 330)
            halo = np.exp(-(r / 210) ** 2)
            pulse = 0.9 + 0.1 * math.sin(t * 2.4)
            blend(fr, np.clip(rays * 0.9 + halo, 0, 1).astype(np.float32), 0, int(fy - 450), GOLD,
                  0.55 * fa * pulse, "add")
        # flags emerge from behind the Saudi flag and fly to their places (drawn before the flag)
        tw, th, gap = 60, 45, 10
        for r_i, (row, ry) in enumerate(zip(self.embrace_rows, en["rows_y"])):
            nrow = len(row)
            total = nrow * tw + (nrow - 1) * gap
            for j, img in enumerate(row):
                tx = W / 2 + total / 2 - (j + 1) * tw - j * gap + tw / 2
                order = abs(j - (nrow - 1) / 2)
                st = w0 + 1.3 + 0.22 * r_i + 0.07 * order
                q = ease_out(ramp(t, st, st + 0.7))
                if q <= 0:
                    continue
                # arc path from the flag centre with a 3D scale-up
                x = cx + (tx - cx) * q
                y = fy + (ry - fy) * q - 60 * math.sin(math.pi * q)
                sc = 0.35 + 0.65 * q + 0.12 * math.sin(math.pi * q)
                w_, h_ = max(4, int(tw * sc)), max(3, int(th * sc))
                im = cv2.resize(img, (w_, h_), interpolation=cv2.INTER_AREA)
                # slight perspective tilt while flying
                tilt = 0.25 * (1 - q) * (1 if tx > cx else -1)
                src = np.float32([[0, 0], [w_, 0], [w_, h_], [0, h_]])
                dy = tilt * h_
                dst = np.float32([[0, max(0, dy)], [w_, max(0, -dy)], [w_, h_ - max(0, -dy)], [0, h_ - max(0, dy)]])
                M = cv2.getPerspectiveTransform(src, dst)
                im = cv2.warpPerspective(im, M, (w_, h_))
                al = cv2.warpPerspective(np.ones((h_, w_), np.float32), M, (w_, h_))
                X, Y = int(x - w_ / 2), int(y - h_ / 2)
                sh = cv2.GaussianBlur(np.pad(al, 20), (0, 0), 7)
                blend(fr, sh, X - 20 + 6, Y - 20 + 10, DARK, 0.55 * min(1, q * 2))
                a3 = al[..., None] * min(1, q * 3)
                reg = fr[Y:Y + h_, X:X + w_]
                hh, ww = reg.shape[:2]
                reg[:] = reg * (1 - a3[:hh, :ww]) + im[:hh, :ww] * a3[:hh, :ww]
                if q > 0.95:
                    fm = np.zeros((h_ + 4, w_ + 4), np.float32)
                    cv2.rectangle(fm, (0, 0), (w_ + 3, h_ + 3), 1.0, 2)
                    blend(fr, fm, X - 2, Y - 2, GOLD, 0.7 * (q - 0.95) * 20)
        # the Saudi flag, luminous, in front
        if fa > 0:
            rgb, al = sa.frame(t, amp=0.5)
            x = int(cx - sa.w / 2)
            y = int(fy - sa.h / 2 - sa.pad + 16 * (1 - fa))
            glow = cv2.GaussianBlur(np.pad(al, 40), (0, 0), 16)
            blend(fr, np.clip(glow * 1.6, 0, 1), x - 40, y - 40, GOLD, 0.6 * fa, "add")
            h, w = al.shape
            reg = fr[y:y + h, x:x + w]
            alc = al[:reg.shape[0], :reg.shape[1], None] * fa
            lit = np.clip(rgb * 1.12 + 0.04, 0, 1)
            reg[:] = reg * (1 - alc) + lit[:reg.shape[0], :reg.shape[1]] * alc
        # big 3D gold heart drawn around the flags, from the top dip to the tip
        p = smooth(ramp(t, w0 + 2.2, w0 + 3.6))
        if p > 0:
            y0 = 250
            hgt = 840
            # luminous light heart: white-hot core, gold glow in three widths, gentle pulse
            core = embrace_art.heart_mask(cx, en["heart_cy"], 32.0, 26.0, p, W, y0, hgt, width=9)
            band = embrace_art.heart_mask(cx, en["heart_cy"], 32.0, 26.0, p, W, y0, hgt, width=22)
            pulse = 0.85 + 0.15 * math.sin(t * 3.2)
            g1 = np.clip(cv2.GaussianBlur(band, (0, 0), 6) * 1.6, 0, 1)
            g2 = np.clip(cv2.GaussianBlur(band, (0, 0), 18) * 2.0, 0, 1)
            g3 = np.clip(cv2.GaussianBlur(band, (0, 0), 45) * 2.4, 0, 1)
            blend(fr, g3, 0, y0, GOLD, 0.45 * pulse, "add")
            blend(fr, g2, 0, y0, GOLD, 0.7 * pulse, "add")
            blend(fr, g1, 0, y0, np.array([1.0, 0.88, 0.6], np.float32), 0.85, "add")
            blend(fr, core, 0, y0, np.array([1.0, 0.98, 0.92], np.float32), 1.0)
            done = t - (w0 + 3.6)
            if done > 0:
                beat = math.exp(-done / 0.45)
                fl = np.zeros((360, 360), np.float32)
                cv2.circle(fl, (180, 180), 50, 1.0, -1)
                fl = cv2.GaussianBlur(fl, (0, 0), 50)
                blend(fr, fl * 2.0, int(cx - 180), int(en["heart_cy"] + 442 - 180), np.array([1, 0.9, 0.65], np.float32),
                      0.6 * beat, "add")
        # Saudi flag again on top so it sits in the heart's top dip
        if fa > 0:
            rgb, al = sa.frame(t, amp=0.5)
            x = int(cx - sa.w / 2)
            y = int(fy - sa.h / 2 - sa.pad + 16 * (1 - fa))
            h_, w_ = al.shape
            reg = fr[y:y + h_, x:x + w_]
            alc = al[:reg.shape[0], :reg.shape[1], None] * fa
            lit = np.clip(rgb * 1.12 + 0.04, 0, 1)
            reg[:] = reg * (1 - alc) + lit[:reg.shape[0], :reg.shape[1]] * alc

    def ending(self, fr, t, en):
        w0 = en["write0"]
        # headline descends slowly while it is written, settling like a seal
        d = ease_out(ramp(t, w0, w0 + 2.0))
        y = en["head_y"] - 120 * (1 - d)
        p = smooth(ramp(t, w0 + 0.1, w0 + 1.9))
        self.write_on(fr, self.end_head, W / 2, y, p, smooth(ramp(t, w0, w0 + 0.5)), GOLD, glow=0.4)
        if t > w0 + 2.0:
            X = int(round(W / 2 - self.end_head.adv / 2)) - self.end_head.pad
            Y = int(round(y - self.end_head.base))
            self.sweep(fr, self.end_head, X, Y, ramp(t, w0 + 2.0, w0 + 3.2), 1.0)
        self.embrace(fr, t, w0, en)
        # message, written line by line on a dark panel
        if not hasattr(self, "_mpanel"):
            self._mpanel = rounded_panel(900, 200, r=32)
        mp, mpp = self._mpanel
        pa = ease_out(ramp(t, w0 + 4.0, w0 + 4.5))
        blend(fr, mp, int(W / 2 - 450) - mpp, int(en["msg_y"] - 82) - mpp, PANEL, 0.62 * pa)
        self.write_on(fr, self.end_msg1, W / 2, en["msg_y"], smooth(ramp(t, w0 + 4.1, w0 + 4.9)), pa, IVORY)
        self.write_on(fr, self.end_msg2, W / 2, en["msg_y"] + 80, smooth(ramp(t, w0 + 4.6, w0 + 5.5)), pa, IVORY, glow=0.12)
        # credits
        ca = ease_out(ramp(t, w0 + 5.4, w0 + 6.0))
        if not hasattr(self, "_epanel"):
            self._epanel = rounded_panel(520, 150, r=30)
        ep, epp = self._epanel
        blend(fr, ep, int(W / 2 - 260) - epp, int(en["credit_y"] - 62) - epp, PANEL, 0.6 * ca)
        self.put(fr, self.end_credit_ar, W / 2, en["credit_y"], ca, IVORY)
        X2, Y2 = self.put(fr, self.end_credit_name, W / 2, en["credit_y"] + 66, ca, GOLD, glow=0.25)
        self.sweep(fr, self.end_credit_name, X2, Y2, ramp(t, w0 + 5.9, w0 + 6.7), ca)


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
    f1 = min(f1, v.nframes)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "15",
                          "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        p.stdin.write(v.frame(fi).tobytes())
        if fi % 150 == 0:
            print(f"{a.out}: {fi}/{f1}", file=sys.stderr, flush=True)
    p.stdin.close()
    p.wait()


if __name__ == "__main__":
    main()
