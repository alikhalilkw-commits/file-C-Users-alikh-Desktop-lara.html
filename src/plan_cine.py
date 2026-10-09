"""Directs the camera for «أهواكِ»: one shot per sung line, cuts snapped to the beat grid.

usage: python plan_cine.py words.json feats.npz out_config.json  (paths for assets come from the ASSETS dict below / env)
"""
import json
import math
import os
import sys

import numpy as np

INTENSITY = {"Verse 1": 0.38, "Pre-Chorus": 0.62, "Verse 2": 0.55, "Chorus": 0.92,
             "Bridge": 0.18, "Final Chorus": 1.0, "Outro": 0.22}

# normalised anchors (x, y) inside each 941x1672 still
IMAGES = {
    "6": dict(file="6.webp", moon=[748, 150, 62], face=[0.55, 0.335, 0.11, 0.085]),
    "7": dict(file="7.webp", moon=[800, 175, 50], face=[0.50, 0.296, 0.11, 0.075]),
    "8": dict(file="8.webp", moon=[743, 155, 95], face=[0.48, 0.248, 0.09, 0.06]),
    "9": dict(file="9.webp", moon=[780, 120, 48], face=[0.53, 0.30, 0.11, 0.07]),
    "10": dict(file="10.webp", moon=[755, 130, 55], face=[0.515, 0.31, 0.11, 0.07]),
}

# per line: (image, from(cx,cy,z), to(cx,cy,z), ease, drift, key word, sticker, side)
# line numbering follows the lyrics file (28 sung lines)
LINES = [
    # Verse 1: calm and tender
    ("9", (0.55, 0.50, 1.10), (0.70, 0.62, 1.50), "smooth", 0.3, 3, "phone", "right"),
    ("9", (0.52, 0.50, 1.30), (0.52, 0.56, 1.52), "smooth", 1.0, 2, "heartbeat", "left"),
    ("7", (0.50, 0.33, 1.52), (0.55, 0.42, 1.12), "smooth", 0.4, 1, "ring", "right"),
    ("9", (0.60, 0.46, 1.25), (0.78, 0.13, 1.38), "smooth", 0.2, 3, "heart", "left"),
    # Pre-Chorus: rising
    ("10", (0.50, 0.40, 1.15), (0.45, 0.33, 1.52), "in", 0.3, 1, "heart_arrow", "right"),
    ("10", (0.34, 0.38, 1.38), (0.62, 0.36, 1.38), "smooth", 0.3, 1, "ring", "left"),
    ("7", (0.50, 0.36, 1.30), (0.82, 0.11, 1.48), "smooth", 0.0, 1, "crescent", "right"),
    ("10", (0.50, 0.33, 1.58), (0.50, 0.45, 1.08), "out", 0.2, 2, "sun", "left"),
    ("9", (0.50, 0.38, 1.20), (0.51, 0.30, 1.68), "in", 0.3, 4, "heart", "right"),
    ("8", (0.50, 0.50, 1.12), (0.38, 0.52, 1.52), "in", 0.3, 1, "flower", "left"),
    ("10", (0.50, 0.32, 1.62), (0.52, 0.34, 1.74), "linear", 1.0, 2, "heartbeat", "right"),
    ("9", (0.52, 0.45, 1.52), (0.50, 0.50, 1.10), "out", 0.3, 3, "drops", "left"),
    # Verse 2
    ("6", (0.50, 0.45, 1.10), (0.55, 0.34, 1.46), "smooth", 0.3, 1, "strings", "right"),
    ("6", (0.55, 0.40, 1.40), (0.45, 0.62, 1.36), "smooth", 0.8, 1, "heart", "left"),
    ("7", (0.52, 0.38, 1.20), (0.50, 0.30, 1.62), "in", 0.3, 3, "heartbeat", "right"),
    ("6", (0.93, 0.60, 1.55), (0.56, 0.38, 1.30), "smooth", 0.3, 4, "candle", "left"),
    # Chorus
    ("8", (0.50, 0.50, 1.00), (0.48, 0.28, 1.52), "in", 0.2, 0, "heart", "right"),
    ("8", (0.45, 0.36, 1.36), (0.76, 0.12, 1.52), "smooth", 0.0, 4, "sparkles", "left"),
    ("8", (0.64, 0.46, 1.32), (0.36, 0.44, 1.32), "smooth", 0.0, 3, "rose", "right"),
    ("8", (0.48, 0.30, 1.62), (0.50, 0.50, 1.00), "out", 0.0, 0, "heart_arrow", "left"),
    # Bridge: music drops, almost still
    ("7", (0.50, 0.33, 1.46), (0.50, 0.38, 1.30), "linear", 1.0, 3, None, "left"),
    ("7", (0.52, 0.36, 1.30), (0.62, 0.30, 1.20), "linear", 0.8, 3, None, "left"),
    # Final Chorus: strongest
    ("8", (0.50, 0.52, 1.00), (0.48, 0.26, 1.68), "in", 0.2, 0, "heart", "right"),
    ("9", (0.55, 0.40, 1.22), (0.52, 0.30, 1.62), "smooth", 0.3, 4, "star", "left"),
    ("8", (0.34, 0.45, 1.34), (0.64, 0.40, 1.34), "smooth", 0.0, 3, "flower", "right"),
    ("8", (0.48, 0.28, 1.70), (0.50, 0.50, 1.00), "out", 0.0, 0, "infinity", "left"),
    # Outro: soft, slow
    ("6", (0.50, 0.45, 1.10), (0.55, 0.35, 1.36), "smooth", 0.5, 4, "heart", "right"),
    ("6", (0.46, 0.64, 1.52), (0.50, 0.50, 1.02), "smooth", 0.4, 2, "quill", "left"),
]


def snap(t, beats, window=0.45):
    d = np.abs(beats - t)
    i = int(np.argmin(d))
    return float(beats[i]) if d[i] <= window else t


def par_for(frm, to, inten):
    a = 6 + 16 * inten
    dx, dy = to[0] - frm[0], to[1] - frm[1]
    n = math.hypot(dx, dy)
    if n > 0.02:
        dx, dy = dx / n, dy / n
        par = [a * dx, a * dy, -a * dx, -a * dy]
    else:
        par = [-a * 0.5, 0, a * 0.5, 0]
    dz = math.log(to[2] / frm[2])
    k = (0.03 + 0.04 * inten) * max(-1.0, min(1.0, dz * 2.2))
    return par, [0.0, k]


def main(words_path, feats_path, out_path, assets):
    wd = json.load(open(words_path))
    words = wd["words"]
    dur = wd["duration"]
    beats = np.load(feats_path)["beats"]
    lines = {}
    for w in words:
        lines.setdefault(w["line"], []).append(w)
    order = sorted(lines)
    assert len(order) == len(LINES), (len(order), len(LINES))
    starts = [lines[i][0]["start"] for i in order]
    secs = [lines[i][0]["section"] for i in order]

    # shot boundaries: intro split, then each line starts ~0.5 s before its first word, snapped to the beat
    bounds = [0.0, snap(starts[0] / 2 + 0.1, beats, 0.8), snap(starts[0] - 0.5, beats)]
    for s in starts[1:]:
        bounds.append(snap(s - 0.5, beats))
    bounds.append(dur + 1.0)
    intro = [
        dict(img="8", t0=bounds[0], t1=bounds[1], **{"from": [0.42, 0.90, 1.58], "to": [0.48, 0.30, 1.20]},
             ease="smooth", drift=0.0, text_y=1520, intensity=0.9, td=0.9, tstyle="bloom"),
        dict(img="7", t0=bounds[1], t1=bounds[2], **{"from": [0.58, 0.40, 1.10], "to": [0.50, 0.30, 1.38]},
             ease="smooth", drift=0.3, text_y=1520, intensity=0.3, td=0.6, tstyle="focus"),
    ]
    shots = []
    for sh in intro:
        sh["par"], sh["rad"] = par_for(sh["from"], sh["to"], sh["intensity"])
        shots.append(sh)
    lines_cfg = []
    for k, (img, frm, to, ease, drift, key, st, side) in enumerate(LINES):
        t0, t1 = bounds[k + 2], bounds[k + 3]
        sec = secs[k]
        inten = INTENSITY[sec]
        par, rad = par_for(frm, to, inten)
        nxt_img = LINES[k + 1][0] if k + 1 < len(LINES) else None
        nxt_sec = secs[k + 1] if k + 1 < len(secs) else None
        if nxt_img and nxt_img != img:
            style, td = "bloom", 1.0 if nxt_sec != "Chorus" else 0.9
            if nxt_sec in ("Chorus", "Final Chorus") and sec != nxt_sec:
                style, td = "flash", 0.7
        else:
            style, td = "focus", 0.45 if inten > 0.5 else 0.7
        shots.append(dict(img=img, t0=t0, t1=t1, **{"from": list(frm), "to": list(to)}, ease=ease, drift=drift,
                          text_y=1520, intensity=inten, par=par, rad=rad, td=td, tstyle=style))
        lines_cfg.append(dict(key=key, sticker=st, side=side))
    lw = lines[order[-1]][-1]
    lines_cfg[-1]["hold"] = (lw["start"] + 4.6) - lw["end"]  # last held note: lyric leaves before the end card
    shots[-1]["t1"] = dur + 1.0
    last_end = lines[order[-1]][-1]["end"]
    cfg = dict(root=assets["images"], depth=assets["depth"], fonts=assets["fonts"], words=words_path,
               feats=feats_path, title="أَهْوَاكِ", roles="كَلِمَات وَأَلْحَان وَتَوْزِيع", name="AHMED",
               images=IMAGES, shots=shots,
               opening=dict(t0=0.15, t1=max(7.5, starts[0] - 0.4), y=1310, emerge=0.9),
               ending=dict(t0=round(lw["start"] + 5.7, 2), t1=round(dur, 2), y=1100),
               watermark=dict(y=960, alpha=0.55), lines=lines_cfg)
    json.dump(cfg, open(out_path, "w"), ensure_ascii=False, indent=1)
    print("shots", len(shots), "opening", cfg["opening"]["t1"], "ending", cfg["ending"])
    for s in shots[:6]:
        print(round(s["t0"], 2), round(s["t1"], 2), s["img"])


if __name__ == "__main__":
    B = os.environ["B"]
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         dict(images=f"{B}/images", depth=f"{B}/scratchpad/depth", fonts=f"{B}/scratchpad/fonts"))
