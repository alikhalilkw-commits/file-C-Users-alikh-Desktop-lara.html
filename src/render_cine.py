"""Cinematic renderer: depth parallax, beat-locked camera, in-image animation.

Builds on render.py (captions, stickers, credits) and replaces the background pipeline.

usage:
  python render_cine.py config.json --stills 3.0,40.5 --outdir stills/
  python render_cine.py config.json --range 0 900 --out part0.mp4
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
import render as R  # noqa: E402
from render import FPS, GOLD, H, W, blend, ease_out, ramp, smooth  # noqa: E402

WARM = np.array([1.0, 0.78, 0.5], np.float32)
EASES = {
    "smooth": smooth,
    "in": lambda x: min(1.0, max(0.0, x)) ** 2,
    "out": lambda x: 1 - (1 - min(1.0, max(0.0, x))) ** 2,
    "linear": lambda x: min(1.0, max(0.0, x)),
}

SECTION_BASE = {"Verse 1": 0.38, "Pre-Chorus": 0.62, "Verse 2": 0.55, "Chorus": 0.92,
                "Bridge": 0.18, "Final Chorus": 1.0, "Outro": 0.22}


class CinePlate(R.Plate):
    """Plate with depth, flame/sparkle masks and a sky mask, all at source or half resolution."""

    def __init__(self, path, depth_path, meta):
        super().__init__(path, meta.get("moon"))
        self.meta = meta
        src = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
        self.depth = np.load(depth_path).astype(np.float32)
        h, w = src.shape[:2]
        fx, fy, rx, ry = meta["face"]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        face = (((xx / w - fx) / rx) ** 2 + ((yy / h - fy) / ry) ** 2) < 1
        self.d0 = float(self.depth[int(fy * h), int(fx * w)])
        luma = src @ np.array([0.299, 0.587, 0.114], np.float32)
        warm = (src[..., 0] - src[..., 2]) > 0.06
        flame = ((luma > 0.8) & warm & ~face).astype(np.uint8)
        flame = cv2.dilate(flame, np.ones((5, 5), np.uint8))
        tophat = luma - cv2.GaussianBlur(luma, (0, 0), 5)
        spark = ((tophat > 0.11) & (flame == 0) & ~face).astype(np.uint8)
        rng = np.random.default_rng(3)

        def pack(m, sig):
            n, lab = cv2.connectedComponents(m)
            ph = rng.uniform(0, 6.283, n).astype(np.float32)[lab]
            mask = np.clip(cv2.GaussianBlur(m.astype(np.float32), (0, 0), sig) * 1.6, 0, 1)
            return cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_AREA), \
                cv2.resize(ph, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)

        self.flame, self.flame_ph = pack(flame, 4)
        self.spark, self.spark_ph = pack(spark, 1.2)
        sky = ((self.depth < 0.1) & (yy / h < 0.5)).astype(np.float32)
        self.sky = cv2.GaussianBlur(sky, (0, 0), 8)
        self.hw, self.hh = w, h

    def matrix(self, cx, cy, z):
        """Affine from up-plate pixel coords to screen."""
        cw = self.W / z
        ch = cw * H / W
        if ch > self.H:
            ch = self.H
            cw = ch * W / H
        x0 = min(max(cx * self.W - cw / 2, 0), self.W - cw)
        y0 = min(max(cy * self.H - ch / 2, 0), self.H - ch)
        s = W / cw
        return s, x0, y0

    @staticmethod
    def M(s, x0, y0, k=1.0):
        """Affine from layer coords (layer = up-plate / k) to screen."""
        return np.array([[s * k, 0, -x0 * s], [0, s * k, -y0 * s]], np.float32)


class CineVideo(R.Video):
    def __init__(self, cfg):
        self.cfg = cfg
        root = cfg["root"]
        self.fonts = R.Fonts(cfg["fonts"])
        words = json.load(open(cfg["words"]))
        self.duration = words["duration"]
        f = np.load(cfg["feats"])
        self.kick, self.spark_f, self.loud, self.vocal = f["kick"], f["spark"], f["loud"], f["vocal"]
        self.spark = self.spark_f
        self.beats = f["beats"]
        self.plates = {k: CinePlate(os.path.join(root, v["file"]), os.path.join(cfg["depth"], v["file"].split(".")[0] + ".npy"), v)
                       for k, v in cfg["images"].items()}
        self.vig = R.vignette(0.55)
        self.vig_deep = R.vignette(0.85)
        rng = np.random.default_rng(7)
        self.grain = [rng.normal(0, 0.012, (H // 2, W // 2, 1)).astype(np.float32) for _ in range(6)]
        self.build_lines(words["words"])
        self.shots = cfg["shots"]
        self.build_credits()
        self.build_energy()
        self.build_particles(rng)
        self.yy, self.xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # ------------------------------------------------------------ energy / beats
    def build_energy(self):
        n = len(self.kick)
        base = np.zeros(n, np.float32)
        for i in range(n):
            base[i] = SECTION_BASE.get(self.section_at(i / FPS), 0.3)
        k = FPS * 2
        sm = np.convolve(np.pad(base, (k, k), mode="edge"), np.ones(k) / k, mode="same")[k:-k]
        self.energy = np.clip(0.75 * sm + 0.25 * self.loud[:n], 0, 1.1)

    def E(self, t):
        return float(self.energy[min(int(t * FPS), len(self.energy) - 1)])

    def beat_pulse(self, t):
        i = np.searchsorted(self.beats, t) - 1
        if i < 0:
            return 0.0
        return math.exp(-(t - float(self.beats[i])) / 0.28)

    # ------------------------------------------------------------ camera
    def camera(self, sh, t):
        u = EASES[sh.get("ease", "smooth")](ramp(t, sh["t0"], sh["t1"]))
        a, b = sh["from"], sh["to"]
        cx, cy, z = (a[i] + (b[i] - a[i]) * u for i in range(3))
        e = self.E(t)
        z *= 1 + (0.003 + 0.011 * e) * self.beat_pulse(t)
        dr = sh.get("drift", 0.0)
        if dr:
            cx += 0.0028 * dr * (math.sin(t * 1.7) + 0.6 * math.sin(t * 2.9 + 1))
            cy += 0.0022 * dr * (math.sin(t * 1.3 + 2) + 0.5 * math.sin(t * 3.3))
        par = sh.get("par", [0, 0, 0, 0])
        px = par[0] + (par[2] - par[0]) * u
        py = par[1] + (par[3] - par[1]) * u
        rad = sh.get("rad", [0, 0])
        rk = rad[0] + (rad[1] - rad[0]) * u
        return cx, cy, z, px, py, rk

    def scene(self, sh, t):
        pl = self.plates[sh["img"]]
        cx, cy, z, px, py, rk = self.camera(sh, t)
        s, x0, y0 = pl.matrix(cx, cy, z)
        img = cv2.warpAffine(pl.img, pl.M(s, x0, y0), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        dep = cv2.warpAffine(pl.depth, pl.M(s, x0, y0, 2.0), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dd = dep - pl.d0
        sky = cv2.warpAffine(pl.sky, pl.M(s, x0, y0, 2.0), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dx = dd * (px + rk * (self.xx - W / 2)) + sky * 5.0 * math.sin(t * 0.35)
        dy = dd * (py + rk * (self.yy - H / 2))
        if abs(px) + abs(py) + abs(rk) > 1e-6 or True:
            img = cv2.remap(img, self.xx - dx, self.yy - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        e = self.E(t)
        # candle / lantern flicker and jewellery twinkle
        fl = pl.flame
        g = 0.55 * np.sin(t * 9.0 + pl.flame_ph) + 0.45 * np.sin(t * 15.3 + pl.flame_ph * 2.3)
        lay = cv2.warpAffine(fl * g, pl.M(s, x0, y0, 4.0), (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        img += lay[..., None] * WARM * (0.13 + 0.06 * e)
        tw = np.maximum(0, np.sin(t * (2.2 + 0.9 * np.sin(pl.spark_ph * 3)) + pl.spark_ph)) ** 10
        lay = cv2.warpAffine(pl.spark * tw, pl.M(s, x0, y0, 4.0), (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        img += lay[..., None] * np.array([1.0, 0.93, 0.78], np.float32) * (0.35 + 0.25 * e)
        # moon breath
        if pl.moon:
            mx, my, mr = pl.moon
            sx, sy, sr = (mx * 2 - x0) * s, (my * 2 - y0) * s, mr * 2 * s
            fi = min(int(t * FPS), len(self.loud) - 1)
            amp = 0.08 + 0.05 * math.sin(t * 1.3) + 0.08 * float(self.loud[fi]) + 0.05 * e
            self.moon_glow(img, sx, sy, sr, amp)
        # bloom, stronger in the chorus and on the beat
        small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        br = np.clip(small - 0.62, 0, 1)
        br = cv2.GaussianBlur(br, (0, 0), 7)
        img += cv2.resize(br, (W, H), interpolation=cv2.INTER_LINEAR) * (0.28 + 0.55 * e * (0.6 + 0.4 * self.beat_pulse(t)))
        return img

    def shot_at(self, t):
        for i, sh in enumerate(self.shots):
            if t < sh["t1"]:
                return i
        return len(self.shots) - 1

    def background(self, t):
        i = self.shot_at(t)
        sh = self.shots[i]
        img = self.scene(sh, max(t, sh["t0"]) if False else t)
        nxt = self.shots[i + 1] if i + 1 < len(self.shots) else None
        td = sh.get("td", 0.4)
        if nxt and t > sh["t1"] - td / 2:
            w = smooth(ramp(t, sh["t1"] - td / 2, sh["t1"] + td / 2))
            img = self.mix(img, self.scene(nxt, t), w, sh.get("tstyle", "focus"))
        elif i > 0 and t < sh["t0"] + self.shots[i - 1].get("td", 0.4) / 2:
            prv = self.shots[i - 1]
            tdp = prv.get("td", 0.4)
            w = smooth(ramp(t, prv["t1"] - tdp / 2, prv["t1"] + tdp / 2))
            img = self.mix(self.scene(prv, t), img, w, prv.get("tstyle", "focus"))
        return img

    def mix(self, a, b, w, style):
        peak = math.sin(math.pi * w)
        out = a * (1 - w) + b * w
        if peak > 0.05:
            out = cv2.GaussianBlur(out, (0, 0), 0.8 + 6 * peak)
        if style == "bloom":
            bl = cv2.GaussianBlur(cv2.resize(out, (W // 4, H // 4)), (0, 0), 10)
            out += cv2.resize(bl, (W, H)) * WARM * 0.45 * peak
        elif style == "flash":
            out += WARM * 0.35 * peak
        return out

    def base_y(self, t):
        return self.shots[self.shot_at(t)].get("text_y", 1520)

    # ------------------------------------------------------------ particles
    def build_particles(self, rng):
        dot = np.zeros((33, 33), np.float32)
        cv2.circle(dot, (16, 16), 4, 1.0, -1, cv2.LINE_AA)
        self.dot = cv2.GaussianBlur(dot, (0, 0), 3) * 2.2
        n = 36
        self.dust = dict(x=rng.uniform(0, W, n), y=rng.uniform(0, H, n), vx=rng.uniform(-6, 6, n),
                         vy=rng.uniform(-24, -8, n), r=rng.uniform(1.2, 3.2, n), ph=rng.uniform(0, 6.28, n))
        # petals in three depth classes: far (small, sharp), mid, near (large, blurred)
        cls = [("far", 9, 26, 7, 12, 0.85, 0.0, 70, 105), ("mid", 12, 14, 17, 26, 0.9, 0.7, 95, 150),
               ("near", 6, 7, 42, 66, 0.55, 3.8, 150, 230)]
        self.petals = []
        for name, cnt, _, smin, smax, alpha, blur, vmin, vmax in cls:
            for k in range(cnt):
                self.petals.append(dict(
                    cls=name, x=rng.uniform(0, W), y=rng.uniform(0, H + 200), vy=rng.uniform(vmin, vmax) * 0.5,
                    vx=rng.uniform(-18, 6), sw=rng.uniform(20, 70), ph=rng.uniform(0, 6.28), s=rng.uniform(smin, smax),
                    th=rng.uniform(0.0, 0.95) if name != "near" else rng.uniform(0.15, 0.9),
                    alpha=alpha, blur=blur, spin=rng.uniform(-60, 60), col=rng.uniform(0, 1)))
        self.petal_cache = {}

    def petal_sprite(self, s, blur, ang, flip, col):
        key = (int(s), int(blur * 10), int(ang // 15) % 24, round(flip, 1), round(col, 1))
        if key in self.petal_cache:
            return self.petal_cache[key]
        size = int(s * 3.2) | 1
        m = np.zeros((size, size), np.float32)
        c = (size // 2, size // 2)
        cv2.ellipse(m, c, (max(2, int(s)), max(1, int(s * 0.58 * max(0.15, abs(flip))))), ang, 0, 360, 1.0, -1, cv2.LINE_AA)
        if blur:
            m = cv2.GaussianBlur(m, (0, 0), blur)
        shade = 0.7 + 0.3 * flip
        base = np.array([0.72, 0.06, 0.12], np.float32) * (1 - col * 0.35) + np.array([0.95, 0.3, 0.38], np.float32) * col * 0.35
        self.petal_cache[key] = (m, np.clip(base * shade, 0, 1))
        return self.petal_cache[key]

    def particles(self, fr, t, fi, section):
        e = self.E(t)
        d = self.dust
        bright = 0.35 + 0.65 * float(self.spark_f[fi])
        n = int(len(d["x"]) * (0.4 + 0.6 * min(1, e)))
        for k in range(n):
            x = (d["x"][k] + d["vx"][k] * t) % W
            y = (d["y"][k] + d["vy"][k] * t) % H
            if 1280 < y < 1720:
                continue
            tw = 0.5 + 0.5 * math.sin(t * 2.1 + d["ph"][k])
            sz = max(5, int(33 * d["r"][k] / 3)) | 1
            blend(fr, cv2.resize(self.dot, (sz, sz)), int(x - sz / 2), int(y - sz / 2), GOLD, 0.5 * tw * bright, "add")
        for p in self.petals:
            vis = smooth((e - p["th"]) / 0.18)
            if vis <= 0.01:
                continue
            y = (p["y"] + p["vy"] * t * 2) % (H + 300) - 150
            x = (p["x"] + p["vx"] * t + p["sw"] * math.sin(t * 0.8 + p["ph"])) % (W + 160) - 80
            band = 1280 < y < 1720
            a = p["alpha"] * vis * (0.45 if band else 1.0)
            if p["cls"] == "far" and band:
                continue
            flip = math.cos(t * 1.6 + p["ph"] * 2)
            ang = (t * p["spin"] + p["ph"] * 57) % 360
            m, col = self.petal_sprite(p["s"], p["blur"], ang, flip, p["col"])
            blend(fr, m, int(x - m.shape[1] / 2), int(y - m.shape[0] / 2), col, a)

    # ------------------------------------------------------------ frame
    def frame(self, fi):
        t = fi / FPS
        c = self.cfg
        fi_c = min(fi, len(self.kick) - 1)
        fr = self.background(t)
        sec = self.section_at(t)
        e = self.E(t)
        op = c["opening"]
        # hook: quick focus pull and warm bloom from the very first frame
        hk = smooth(ramp(t, 0, op.get("emerge", 0.9)))
        if hk < 1:
            fr = cv2.GaussianBlur(fr, (0, 0), 0.8 + 11 * (1 - hk)) * (0.55 + 0.45 * hk) + WARM * 0.25 * (1 - hk)
        deep = sec == "Bridge"
        if deep:
            fr = np.power(np.clip(fr, 0, 1), 1.12)
        fr *= 1 + 0.035 * e * float(self.kick[fi_c])
        fr *= self.vig_deep if deep else self.vig
        self.particles(fr, t, fi_c, sec)
        en = c["ending"]
        dk = smooth(ramp(t, en["t0"] - 0.5, en["t0"] + 1.5))
        if dk > 0:
            fr *= 1 - 0.5 * dk
        self.draw_lines(fr, t, self.base_y(t))
        if op["t0"] <= t <= op["t1"]:
            self.credit_block(fr, t, op, op.get("y", 420))
        if en["t0"] <= t <= en["t1"]:
            self.end_card(fr, t, en)
        self.watermark(fr, t)
        fb = smooth(ramp(t, self.duration - 1.1, self.duration - 0.05))
        if fb > 0:
            fr *= 1 - fb
        g = cv2.resize(self.grain[fi % len(self.grain)], (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
        fr += g
        return (np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--stills")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--range", nargs=2, type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    v = CineVideo(cfg)
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
