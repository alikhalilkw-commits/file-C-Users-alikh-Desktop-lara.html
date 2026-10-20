"""Renderer for «قَهْوَتِي مُرَّةٌ» in the «جلدت ذاتي» look.

Opening card (small «مَوَّالٌ حَزِين» label, large written-on title, four role icons, gold rule, AHMED) over a dark
blurred still, a small header with the title and credits for the whole song, word-by-word captions without a box
kept 120 px inside both edges, one glowing line sticker per hemistich, and the opening card replayed at the end.

usage:
  python render_qahwa.py config.json --stills 5,40,80 --outdir stills/
  python render_qahwa.py config.json --range 0 1300 --out part0.mp4
"""
import argparse
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import ImageFont

sys.path.insert(0, os.path.dirname(__file__))
import render as R  # noqa: E402
import stickers  # noqa: E402
from render import FPS, H, W, blend, ease_out, ramp, smooth  # noqa: E402
from render_haram import Plate  # noqa: E402

IVORY = np.array([1.0, 0.968, 0.905], np.float32)
GOLD = np.array([0.965, 0.835, 0.585], np.float32)
CREAM = np.array([1.0, 0.93, 0.78], np.float32)
DARK = np.array([0.0, 0.0, 0.0], np.float32)
MARGIN = 120
SPLIT = "،"


class Video:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fonts = R.Fonts(cfg["fonts"])
        wd = json.load(open(cfg["words"]))
        self.duration = wd["duration"]
        self.nframes = int(round(self.duration * FPS))
        f = np.load(cfg["feats"])
        self.kick = f["kick"]
        self.plates = {k: Plate(os.path.join(cfg["image_dir"], v)) for k, v in cfg["images"].items()}
        self.vig = R.vignette(0.55)
        rng = np.random.default_rng(7)
        self.grain = [rng.normal(0, 0.010, (H // 2, W // 2, 1)).astype(np.float32) for _ in range(6)]
        self.shots = cfg["shots"]
        self.build_hemis(wd["words"])
        self.build_static()

    # ---------------------------------------------------------------- fonts
    def font(self, name, size):
        key = (name, size)
        if key not in self.fonts.cache:
            if name in ("kufi", "marhey"):
                fn = "ReemKufi" if name == "kufi" else "Marhey"
                f = ImageFont.truetype(f"{self.cfg['fonts']}/{fn}.ttf", size, layout_engine=ImageFont.Layout.RAQM)
                f.set_variation_by_axes([700])
                self.fonts.cache[key] = f
            else:
                return self.fonts.get(name, size)
        return self.fonts.cache[key]

    # ---------------------------------------------------------------- captions
    def build_hemis(self, words):
        """Split every sung line at «،» into hemistichs; each is captioned on its own."""
        groups, cur = [], []
        for w in words:
            if cur and w["line"] != cur[-1]["line"]:
                groups.append(cur)
                cur = []
            cur.append(w)
            if w["text"].endswith(SPLIT):
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
        meta = self.cfg["hemis"]
        size = self.cfg.get("caption_size", 92)
        font = self.font(self.cfg.get("caption_font", "marhey"), size)
        space = font.getlength(" ") * 1.25
        maxw = W - 2 * MARGIN
        # one line only: a hemistich that does not fit is shown in consecutive word groups
        chunks = []
        for gi, ws in enumerate(groups):
            cur, cw = [], 0.0
            for w in ws:
                ww = R.Sprite(w["text"].rstrip(SPLIT), font).ink_w
                if cur and cw + space + ww > maxw:
                    chunks.append((gi, cur))
                    cur, cw = [], 0.0
                cw = ww if not cur else cw + space + ww
                cur.append(w)
            chunks.append((gi, cur))
        self.hemis = []
        for i, (gi, ws) in enumerate(chunks):
            m = meta[gi] if gi < len(meta) else {}
            first = i == 0 or chunks[i - 1][0] != gi
            nxt = chunks[i + 1][1][0]["start"] if i + 1 < len(chunks) else self.duration
            same = i + 1 < len(chunks) and chunks[i + 1][0] == gi
            start, end = ws[0]["start"], ws[-1]["end"]
            t_out = nxt - 0.12 if same else min(nxt - 0.35, end + m.get("hold", 1.6))
            h = dict(words=ws, start=start, end=end, t_out=t_out, icon=m.get("icon") if first else None,
                     side=m.get("side", "left"), y=m.get("y", 1500))
            self.layout(h, size)
            self.hemis.append(h)

    def layout(self, h, size):
        maxw = W - 2 * MARGIN
        while True:
            font = self.font(self.cfg.get("caption_font", "marhey"), size)
            sprites = [R.Sprite(w["text"].rstrip(SPLIT), font) for w in h["words"]]
            space = font.getlength(" ") * 1.25
            widths = [s.ink_w for s in sprites]
            total = sum(widths) + space * (len(sprites) - 1)
            if total <= maxw or size <= 56:
                break
            size -= 4
        asc = max(s.base - s.pad for s in sprites)
        desc = max(s.h - s.base - s.pad for s in sprites)
        row_h = int(asc + desc + size * 0.22)
        place = {}
        x = W / 2 + total / 2
        for i in range(len(sprites)):
            x -= widths[i]
            place[i] = (x, 0)
            x -= space
        h.update(sprites=sprites, place=place, rows=1, row_h=row_h, asc=asc, width=total, size=size)
        sw, sh = int(total + 160), int(row_h + 120)
        m = np.zeros((sh + 120, sw + 120), np.float32)
        cv2.ellipse(m, ((sw + 120) // 2, (sh + 120) // 2), (sw // 2, sh // 2), 0, 0, 360, 1.0, -1)
        h["scrim"] = cv2.GaussianBlur(m, (0, 0), 38)

    def draw_hemi(self, fr, t, h):
        a = smooth(ramp(t, h["start"] - 0.3, h["start"] - 0.05)) * (1 - smooth(ramp(t, h["t_out"] - 0.25, h["t_out"])))
        if a <= 0.003:
            return
        top = h["y"] - (h["rows"] * h["row_h"]) / 2
        sc = h["scrim"]
        blend(fr, sc, int(W / 2 - sc.shape[1] / 2), int(top + h["rows"] * h["row_h"] / 2 - sc.shape[0] / 2), DARK, 0.42 * a)
        row0 = top + h["asc"]
        for i, (w, sp) in enumerate(zip(h["words"], h["sprites"])):
            x, r = h["place"][i]
            wa = ease_out(ramp(t, w["start"] - 0.22, w["start"] + 0.12)) * a
            if wa <= 0.003:
                continue
            g = ramp(t, w["start"] - 0.05, w["start"] + 0.1) * (1 - smooth(ramp(t, w["end"], w["end"] + 0.35)))
            X = int(round(x)) - sp.ink_l
            Y = int(round(row0 + r * h["row_h"] + 12 * (1 - wa / max(a, 1e-3)))) - sp.base
            blend(fr, sp.sh_large, X, Y + 5, DARK, 0.6 * wa)
            blend(fr, sp.sh_small, X + 1, Y + 3, DARK, 0.75 * wa)
            if g > 0.01:
                blend(fr, sp.glow, X, Y, GOLD, 0.45 * g * wa, "add")
            blend(fr, sp.mask, X, Y, IVORY * (1 - g) + GOLD * g, wa)
        if h["icon"]:
            prog = ramp(t, h["start"] + 0.1, h["start"] + 0.9)
            rr = 44
            core, glow, halo = stickers.draw(h["icon"], prog, rr, 0.9 + 0.1 * math.sin(t * 7))
            half = h["width"] / 2 if h["rows"] == 1 else h["width"] / 2
            ix = W / 2 - half + 30 if h["side"] == "left" else W / 2 + half - 30
            ix = min(max(ix, MARGIN + rr), W - MARGIN - rr)
            iy = top - rr * 1.25
            X, Y = int(ix - core.shape[1] / 2), int(iy - core.shape[0] / 2)
            blend(fr, halo, X, Y, DARK, 0.45 * a)
            blend(fr, glow, X, Y, GOLD, 0.6 * a, "add")
            blend(fr, core, X, Y, CREAM, 0.95 * a)

    # ---------------------------------------------------------------- static text
    def build_static(self):
        c = self.cfg
        f = self.font
        self.c_label = R.Sprite(c["label"], f("messiri", 46))
        self.c_title = R.Sprite(c["title"], f("marhey", 160))
        self.c_roles = [R.Sprite(r, f("messiri", 38)) for r in c["roles"]]
        self.c_name = R.Sprite("AHMED", f("cinzel", 118), tracking=26, rtl=False)
        self.h_title = R.Sprite(c["title"], f("marhey", 60))
        self.h_credit = R.Sprite(c["credit"], f("messiri", 30))
        self.h_name = R.Sprite("AHMED", f("cinzel", 32), tracking=8, rtl=False)

    def put(self, fr, sp, cx, base, a, color, glow=0.0, x_left=None):
        X = (int(round(cx - sp.adv / 2)) - sp.pad) if x_left is None else int(x_left) - sp.ink_l
        Y = int(round(base - sp.base))
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
            blend(fr, sp.mask * band, X, Y, np.array([1, 1, 0.95], np.float32), 0.6 * a, "add")

    def write_on(self, fr, sp, cx, base, p, a, color, glow=0.0, t=0.0):
        """Reveal right-to-left with a sparkling gold leading edge."""
        if a <= 0.003 or p <= 0:
            return
        X = int(round(cx - sp.adv / 2)) - sp.pad
        Y = int(round(base - sp.base))
        xx = np.arange(sp.w, dtype=np.float32)
        edge = sp.w * (1 - p * 1.1) + 0.05 * sp.w
        m = np.clip((xx - edge) / 30 + 0.5, 0, 1)[None, :]
        blend(fr, sp.sh_large * m, X, Y + 5, DARK, 0.6 * a)
        if glow:
            blend(fr, sp.glow * m, X, Y, GOLD, glow * a, "add")
        blend(fr, sp.mask * m, X, Y, color, a)
        if p < 1:
            ex = X + edge
            rng = np.random.default_rng(int(t * 30))
            for _ in range(26):
                px = int(ex + rng.normal(0, 18))
                py = int(Y + sp.pad + rng.uniform(-0.1, 1.1) * (sp.h - 2 * sp.pad))
                r = rng.uniform(1.2, 3.2)
                cv2.circle(fr, (px, py), int(r + 0.5), tuple(float(v) for v in (np.array([1, 0.9, 0.65]) * rng.uniform(0.5, 1.0))), -1, cv2.LINE_AA)

    # ---------------------------------------------------------------- opening / closing card
    def card_bg(self):
        if not hasattr(self, "_bg"):
            bg, _ = self.plates[self.cfg["card"]["bg"]].view(0.5, 0.45, 1.05)
            bg = cv2.GaussianBlur(bg, (0, 0), 9) * 0.26
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            d = np.sqrt(((xx - W / 2) / (W * 0.6)) ** 2 + ((yy - H * 0.42) / (H * 0.55)) ** 2)
            bg *= np.clip(1.2 - 0.8 * d, 0.2, 1)[..., None]
            self._bg = bg.astype(np.float32)
        return self._bg

    def card(self, fr, t, k):
        """Card layout; t is seconds since the card started, k its overall opacity."""
        L = self.cfg["card"]
        # small label between two short rules
        la = ease_out(ramp(t, 0.5, 1.2)) * k
        self.put(fr, self.c_label, W / 2, L["label_y"], la, CREAM)
        rl = int(70 * ease_out(ramp(t, 0.7, 1.5)))
        if rl > 1:
            gap = self.c_label.ink_w / 2 + 30
            ly = int(L["label_y"] - 16)
            for sgn in (-1, 1):
                x0 = int(W / 2 + sgn * gap)
                x1 = int(W / 2 + sgn * (gap + rl))
                cv2.line(fr, (min(x0, x1), ly), (max(x0, x1), ly), tuple(float(v) * la for v in CREAM * 0.8), 2, cv2.LINE_AA)
        # title written on with sparkles, then a slow sweep
        p = smooth(ramp(t, 1.0, 2.8))
        ta = smooth(ramp(t, 0.9, 1.3)) * k
        self.write_on(fr, self.c_title, W / 2, L["title_y"], p, ta, CREAM, glow=0.55, t=t)
        if t > 3.0:
            X = int(round(W / 2 - self.c_title.adv / 2)) - self.c_title.pad
            self.sweep(fr, self.c_title, X, int(round(L["title_y"] - self.c_title.base)), ramp(t, 3.0, 4.2), ta)
        # four role icons, drawn one after another (right to left), labels beneath
        n = len(self.c_roles)
        step = L["icon_step"]
        for j in range(n):
            t0 = 3.0 + 0.45 * j
            prog = ramp(t, t0, t0 + 0.8)
            if prog <= 0:
                continue
            cx = W / 2 + step * (n - 1) / 2 - step * j
            core, glow, halo = stickers.draw(L["icons"][j], prog, 40)
            X, Y = int(cx - core.shape[1] / 2), int(L["icon_y"] - core.shape[0] / 2)
            blend(fr, halo, X, Y, DARK, 0.4 * k)
            blend(fr, glow, X, Y, GOLD, 0.6 * k, "add")
            blend(fr, core, X, Y, CREAM, k)
            ra = ease_out(ramp(t, t0 + 0.4, t0 + 0.9)) * k
            self.put(fr, self.c_roles[j], cx, L["icon_y"] + 98, ra, CREAM)
        # gold rule grows from the centre
        rw = int(320 * ease_out(ramp(t, 5.0, 5.8)))
        if rw > 2:
            ry = L["rule_y"]
            band = np.zeros((7, 2 * rw), np.float32)
            band[3] = 1
            band = cv2.GaussianBlur(band, (0, 0), 0.8)
            xs = np.linspace(-1, 1, 2 * rw, dtype=np.float32)
            band *= (1 - xs ** 4)[None, :]
            blend(fr, band * 1.6, int(W / 2 - rw), ry - 3, GOLD, k)
        # AHMED
        na = ease_out(ramp(t, 5.6, 6.4)) * k
        X2, Y2 = self.put(fr, self.c_name, W / 2, L["name_y"] + 14 * (1 - ease_out(ramp(t, 5.6, 6.4))), na, IVORY, glow=0.35)
        self.sweep(fr, self.c_name, X2, Y2, ramp(t, 6.5, 7.6), na)

    def header(self, fr, a):
        if a <= 0.003:
            return
        hd = self.cfg["header"]
        self.put(fr, self.h_title, W / 2, hd["title_y"], a, CREAM, glow=0.3)
        self.put(fr, self.h_credit, W / 2, hd["credit_y"], a, IVORY)
        self.put(fr, self.h_name, W / 2, hd["name_y"], a, GOLD, glow=0.2)

    # ---------------------------------------------------------------- scenes
    def scene(self, sh, t):
        u = smooth(ramp(t, sh["t0"], sh["t1"]))
        a, b = sh["from"], sh["to"]
        cx, cy, z = [a[i] + (b[i] - a[i]) * u for i in range(3)]
        fi = min(int(t * FPS), len(self.kick) - 1)
        z *= 1 + 0.004 * float(self.kick[fi])
        img, _ = self.plates[sh["img"]].view(cx, cy, z)
        if sh.get("warm"):
            img = img * np.array([1.07, 1.0, 0.88], np.float32)
            img = cv2.addWeighted(img, 0.85, cv2.GaussianBlur(img, (0, 0), 8), 0.15, 0)
        return img

    def background(self, t):
        sh = self.shots
        i = next((j for j, s in enumerate(sh) if t < s["t1"]), len(sh) - 1)
        img = self.scene(sh[i], t)
        td = 1.0
        if i + 1 < len(sh) and t > sh[i]["t1"] - td / 2:
            w = smooth(ramp(t, sh[i]["t1"] - td / 2, sh[i]["t1"] + td / 2))
            img = self.mix(img, self.scene(sh[i + 1], t), w)
        elif i > 0 and t < sh[i - 1]["t1"] + td / 2:
            w = smooth(ramp(t, sh[i - 1]["t1"] - td / 2, sh[i - 1]["t1"] + td / 2))
            img = self.mix(self.scene(sh[i - 1], t), img, w)
        return img

    @staticmethod
    def mix(a, b, w):
        """Focus dissolve with a faint warm bloom at the midpoint."""
        peak = math.sin(math.pi * w)
        if peak > 0.05:
            s = 6 * peak
            a = cv2.GaussianBlur(a, (0, 0), s)
            b = cv2.GaussianBlur(b, (0, 0), s)
        return a * (1 - w) + b * w + np.array([1.0, 0.85, 0.6], np.float32) * 0.06 * peak

    # ---------------------------------------------------------------- frame
    def frame(self, fi):
        t = fi / FPS
        c = self.cfg
        fr = self.background(t)
        small = cv2.resize(fr, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        br = cv2.GaussianBlur(np.clip(small - 0.72, 0, 1), (0, 0), 6)
        fr += cv2.resize(br, (W, H)) * 0.35
        fr *= self.vig
        cd = c["card"]
        # opening card, fading into the first image
        k = 1 - smooth(ramp(t, cd["end"], cd["end"] + cd["fade"]))
        if k > 0:
            drift = 1 + 0.03 * ramp(t, 0, cd["end"] + cd["fade"])
            bg = cv2.resize(self.card_bg(), None, fx=drift, fy=drift)
            oy, ox = (bg.shape[0] - H) // 2, (bg.shape[1] - W) // 2
            fr[:] = fr * (1 - k) + bg[oy:oy + H, ox:ox + W] * k
            self.card(fr, t, k)
        # closing card: the same card again
        e0 = c["ending"]["t0"]
        if t >= e0:
            ke = smooth(ramp(t, e0, e0 + 1.4))
            fr[:] = fr * (1 - ke) + self.card_bg() * ke
            self.card(fr, t - e0, ke)
        # header between the cards
        ha = smooth(ramp(t, cd["end"] + 0.4, cd["end"] + 1.4)) * (1 - smooth(ramp(t, e0 - 0.8, e0 + 0.2)))
        self.header(fr, ha)
        for h in self.hemis:
            if h["start"] - 0.4 <= t <= h["t_out"] + 0.01:
                self.draw_hemi(fr, t, h)
        g = cv2.resize(self.grain[fi % len(self.grain)], (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
        fr += g
        fr *= smooth(ramp(t, 0, 0.6)) * (1 - smooth(ramp(t, self.duration - 0.8, self.duration)))
        return (np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--stills")
    ap.add_argument("--outdir", default="stills")
    ap.add_argument("--range", nargs=2, type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    v = Video(json.load(open(a.config)))
    if a.stills:
        os.makedirs(a.outdir, exist_ok=True)
        for s in a.stills.split(","):
            fr = v.frame(int(round(float(s) * FPS)))
            cv2.imwrite(f"{a.outdir}/s_{float(s):07.2f}.jpg", fr[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        return
    f0, f1 = a.range if a.range else (0, v.nframes)
    f1 = min(f1, v.nframes)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                          "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        p.stdin.write(v.frame(fi).tobytes())
        if fi % 150 == 0:
            print(f"{fi}/{f1}", file=sys.stderr, flush=True)
    p.stdin.close()
    p.wait()


if __name__ == "__main__":
    main()
