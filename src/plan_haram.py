"""Scene plan for «حَمَاكِ الإِلَهُ بِلَادَ الحَرَمْ»: meaning-first shot list, cut points from measured vocal onsets.

usage: B=<session dir> python plan_haram.py words.json feats.npz out_config.json out_timing_dir
Writes the render config plus lyric_word_timing.json, scene_mapping.csv and asset_manifest.csv.
"""
import csv
import json
import os
import sys

import numpy as np

FPS = 30
TAIL = 6.0  # seconds added after the song for the closing card (approved by Ali)
HIGH, LOW = 430, 1500

ASSETS = {  # id: (file, category description, original source name, depth parallax allowed)
    "H01": ("H01.png", "Masjid Al-Haram: Kaaba with clock tower, pilgrims", "HARAM_01", False),
    "H02": ("H02.png", "Kaaba close view with kiswa", "HARAM_02", False),
    "H03": ("H03.png", "Masjid Al-Haram aerial", "HARAM_03", True),
    "R01": ("R01.png", "Riyadh sunset skyline, Kingdom Centre", "RIYADH_01", True),
    "R02": ("R02.png", "Riyadh modern towers and plaza", "RIYADH_02", False),
    "R03": ("R03.png", "Riyadh night aerial", "RIYADH_03", True),
    "P01": ("P01.png", "Crown Prince portrait, smiling", "MBS_01.jpg", False),
    "P02": ("P02.png", "Crown Prince seated, formal", "MBS_02.jpg", False),
    "P03": ("P03.png", "Crown Prince seated, profile", "12.jpg (third portrait upload)", False),
    "X01": ("X01.png", "Horseman with Saudi flag, aerobatic jets, desert dune", "11.jpg (extra national image)", False),
}

# per sung line (28 occurrences): shot(s), caption height, icon, transition into the shot
# a shot is (asset, from(cx,cy,z), to(cx,cy,z), parallax px, extra)
LINES = [
    ("L01", [("H01", (0.47, 0.60, 1.02), (0.45, 0.55, 1.12), 0, {})], HIGH, None, None),
    ("L02", [("R01", (0.60, 0.62, 1.20), (0.62, 0.42, 1.20), 6, {})], LOW, "kingdom", ("light", 0.5)),
    ("L03", [("X01", (0.28, 0.53, 1.70), (0.32, 0.55, 1.60), 0, {"react": False})], LOW, "palm_swords", ("dissolve", 0.6)),
    ("L04", [("R03", (0.45, 0.60, 1.30), (0.50, 0.48, 1.10), 6, {})], LOW, "skyline", ("dissolve", 0.5)),
    ("L05", [("H03", (0.50, 0.56, 1.18), (0.50, 0.53, 1.04), 7, {})], LOW, "star8", ("dissolve", 0.6)),
    ("L06", [("H02", (0.62, 0.45, 1.06), (0.60, 0.41, 1.14), 0, {"warm": True})], LOW, "minaret", ("dissolve", 0.5)),
    ("L07", [("P02", (0.45, 0.42, 1.05), (0.45, 0.40, 1.08), 0, {})], LOW, "star8", ("dissolve", 0.6)),
    ("L08", [("R01", (0.60, 0.62, 1.30), (0.62, 0.34, 1.30), 6, {})], LOW, "kingdom", ("dissolve", 0.5)),
    ("L09", [("P03", (0.55, 0.50, 1.10), (0.56, 0.50, 1.13), 0, {})], LOW, None, ("dissolve", 0.6)),
    ("L10", [("P01", (0.50, 0.45, 1.08), (0.50, 0.44, 1.10), 0, {}),
             ("H01", (0.48, 0.55, 1.36), (0.48, 0.53, 1.42), 0, {"cut_word": 2})], LOW, "kaaba", ("dissolve", 0.5)),
    ("L11", [("X01", (0.50, 0.62, 1.35), (0.50, 0.60, 1.42), 0, {})], LOW, "palm_swords", ("dissolve", 0.6)),
    ("L12", [("H02", (0.70, 0.45, 1.40), (0.55, 0.45, 1.06), 0, {})], LOW, "kaaba", ("light", 0.5)),
    ("L13", [("R03", (0.50, 0.55, 1.05), (0.52, 0.48, 1.18), 6, {})], LOW, "skyline", ("dissolve", 0.5)),
    ("L14", [("X01", (0.42, 0.55, 1.50), (0.45, 0.58, 1.38), 0, {"react": False})], LOW, None, ("dissolve", 0.6)),
    ("L15", [("X01", (0.50, 0.72, 1.35), (0.50, 0.66, 1.50), 0, {})], LOW, "palm_swords", ("dissolve", 0.5)),
    ("L16", [("X01", (0.50, 0.86, 1.50), (0.50, 0.62, 1.20), 0, {})], LOW, None, ("dissolve", 0.8)),
    ("L17", [("R02", (0.50, 0.62, 1.30), (0.50, 0.48, 1.03), 0, {})], LOW, "kingdom", ("dissolve", 0.6)),
    ("L18", [("H03", (0.50, 0.50, 1.04), (0.50, 0.55, 1.15), 5, {})], LOW, "minaret", ("dissolve", 0.6)),
    ("L19", [("H01", (0.45, 0.62, 1.20), (0.45, 0.50, 1.08), 0, {})], LOW, "star8", ("dissolve", 0.6)),
    ("L20", [("H02", (0.40, 0.60, 1.25), (0.50, 0.50, 1.08), 0, {})], LOW, "kaaba", ("light", 0.5)),
    ("L21", [("R01", (0.62, 0.30, 1.45), (0.56, 0.45, 1.12), 6, {})], LOW, "skyline", ("dissolve", 0.5)),
    ("L22", [("X01", (0.50, 0.64, 1.30), (0.45, 0.56, 1.45), 0, {"react": False})], LOW, None, ("dissolve", 0.6)),
    ("L23", [("H03", (0.50, 0.45, 1.25), (0.50, 0.55, 1.05), 6, {})], LOW, "star8", ("dissolve", 0.5)),
    ("L24", [("R03", (0.50, 0.45, 1.30), (0.50, 0.50, 1.02), 6, {})], LOW, "skyline", ("dissolve", 0.6)),
    ("L25", [("R01", (0.60, 0.25, 1.30), (0.60, 0.35, 1.15), 5, {"warm": True})], LOW, "star8", ("dissolve", 0.6)),
    ("L26", [("P01", (0.50, 0.45, 1.08), (0.50, 0.43, 1.11), 0, {})], LOW, "palm_swords", ("dissolve", 0.6)),
    ("L27", [("R02", (0.50, 0.50, 1.10), (0.50, 0.50, 1.15), 0, {"warm": True})], 1530, None, ("dissolve", 0.6)),
    ("L28", [("H01", (0.45, 0.56, 1.05), (0.43, 0.55, 1.22), 0, {})], 1530, None, ("dissolve", 0.8)),
]

# instrumental break (no vocals): national montage
BREAK = [("H03", (0.50, 0.50, 1.20), (0.50, 0.55, 1.05), 6, ("dissolve", 0.8)),
         ("P03", (0.55, 0.50, 1.10), (0.55, 0.49, 1.13), 0, ("dissolve", 0.9)),
         ("R03", (0.45, 0.60, 1.25), (0.50, 0.50, 1.06), 6, ("dissolve", 0.9))]


def main(words_path, feats_path, out_cfg, tdir, B):
    wd = json.load(open(words_path))
    words, dur = wd["words"], wd["duration"]
    lines = {}
    for w in words:
        lines.setdefault(w["line"], []).append(w)
    order = sorted(lines)
    assert set(order) <= set(range(len(LINES)))
    sel = [LINES[i] for i in order]  # supports shortened edits that drop whole lines
    pos14 = order.index(14) if 14 in order else None
    starts = [lines[i][0]["start"] for i in order]
    ends = [lines[i][-1]["end"] for i in order]
    beats = np.load(feats_path)["beats"]

    def cut(t, win=0.2):
        d = np.abs(beats - t)
        i = int(np.argmin(d))
        return float(beats[i]) if d[i] <= win else t

    cuts = [0.0] + [cut(s - 0.35) for s in starts[1:]]
    shots, meta = [], []
    break_t0 = ends[pos14] + 0.6 if pos14 is not None else 1e9  # after «وَعَادُوا يَجُرُّونَ ثَوْبَ النَّدَمْ» (hook)
    for k, (lid, shs, y, icon, tr) in enumerate(sel):
        t0 = cuts[k]
        t1 = cuts[k + 1] if k + 1 < len(cuts) else dur + TAIL
        if k == pos14:
            t1 = break_t0
        parts = []
        if len(shs) == 2:
            cw = lines[order[k]][shs[1][4]["cut_word"]]["start"] - 0.3
            parts = [(shs[0], t0, cw), (shs[1], cw, t1)]
        else:
            parts = [(shs[0], t0, t1)]
        for j, (s, a, b) in enumerate(parts):
            if s[0] == "FLAG":
                sh = dict(kind="flag", img="FLAG", bg=s[1], t0=a, t1=b, **{"from": list(s[2]), "to": list(s[3])})
            else:
                sh = dict(kind="plate", img=s[0], t0=a, t1=b, **{"from": list(s[1]), "to": list(s[2])}, par=s[3], **s[4])
            shots.append(sh)
        meta.append(dict(id=lid, y=y, icon=icon, side="left" if k % 2 else "right", title=(k == 0),
                         size=92 if k == 0 else 70, hold=1.35 if k == len(sel) - 1 else 1.3))
        if k == pos14:
            bt = [break_t0, break_t0 + 6.2, break_t0 + 12.6, cuts[pos14 + 1]]
            for j, (img, f, to, par, _) in enumerate(BREAK):
                shots.append(dict(kind="plate", img=img, t0=bt[j], t1=bt[j + 1], **{"from": list(f), "to": list(to)}, par=par))
    # transitions: style of the cut INTO a shot is stored on the previous shot
    tr_into = {}
    for k, (lid, shs, y, icon, tr) in enumerate(sel):
        if tr:
            tr_into[round(cuts[k], 3)] = tr
    for i, sh in enumerate(shots[:-1]):
        nxt = shots[i + 1]
        st, td = tr_into.get(round(nxt["t0"], 3), ("dissolve", 0.6))
        if nxt["img"] == "H01" and sh["img"] == "P01":
            st, td = "dissolve", 0.55
        if sh["t1"] >= break_t0 - 0.01 and sh["t0"] < break_t0:
            st, td = "dissolve", 0.8
        sh["tstyle"], sh["td"] = st, td
    images = {k: dict(file=v[0], depth=v[3]) for k, v in ASSETS.items()}
    lw = lines[order[-1]]
    cfg = dict(
        image_dir=f"{B}/scratchpad/saudi/assets/images", depth_dir=f"{B}/scratchpad/saudi/assets/depth",
        fonts=f"{B}/scratchpad/fonts", icons=f"{B}/scratchpad/saudi/icons",
        flag_sa=f"{B}/scratchpad/saudi/flags/sa_3x2.png", flag_eg=f"{B}/scratchpad/saudi/flags/eg_3x2.png",
        words=words_path, feats=feats_path, images=images, shots=shots, lines=meta,
        credit_ar="كلمات وألحان وتوزيع ومونتاج", accent_from=starts[1] - 0.3,
        title="حَمَاكِ الإِلَهُ بِلَادَ الحَرَمْ",
        opening=dict(t0=1.0, black=0.0, credits_out=starts[1] - 0.6,
                     card=True, card_end=round(starts[0] - 1.25, 3), card_fade=0.85,
                     card_title_y=560, card_flags_y=800, card_pledge_y=1100, card_credit_y=1222,
                     pledge="نَفْدِيكِ بِأَرْوَاحِنَا يَا أَرْضَ العِزَّةِ وَالكَرَمْ"),
        tail=TAIL,
        gulf_flags=[f"{B}/scratchpad/saudi/flags/{g}_4x3.png" for g in ("kw", "ae", "qa", "bh", "om")],
        embrace_rows=[[f"{B}/scratchpad/saudi/flags/{g}_4x3.png" for g in row] for row in (
            ("eg", "kw", "ae", "qa", "bh", "om"),
            ("jo", "iq", "sy", "lb", "ps", "ye", "sd", "ly"),
            ("tn", "dz", "ma", "mr", "so", "dj", "km"))],
        arab_flags_unused=[f"{B}/scratchpad/saudi/flags/{g}_4x3.png" for g in
                    ("jo", "iq", "sy", "lb", "ps", "ye", "sd", "ly", "tn", "dz", "ma", "mr", "so", "dj", "km")],
        ending=dict(t0=round(dur - 1.4, 3), write0=round(dur - 0.9, 3), black=round(dur + TAIL - 0.8, 3), head_y=225,
                    sa_y=470, rows_y=[625, 685, 745], hands_y=900, heart_cy=600, msg_y=1170, credit_y=1350),
        watermark=dict(y=960, alpha=0.36, alpha_hi=0.6, emphasis=[[0, 6.0], [92.0, 96.0], [dur - 1.4, dur + TAIL + 1]]),
    )
    json.dump(cfg, open(out_cfg, "w"), ensure_ascii=False, indent=1)

    # ------------------------------------------------ deliverable timing data (measured values)
    os.makedirs(tdir, exist_ok=True)
    occ = []
    for k, li in enumerate(order):
        ws = lines[li]
        occ.append(dict(id=sel[k][0], occurrence=k + 1, section=ws[0]["section"],
                        text=" ".join(w["text"] for w in ws), start=round(ws[0]["start"], 3), end=round(ws[-1]["end"], 3),
                        words=[dict(text=w["text"], start=round(w["start"], 3), end=round(w["end"], 3),
                                    start_frame=int(round(w["start"] * FPS)), end_frame=int(round(w["end"] * FPS)))
                               for w in ws]))
    json.dump(dict(song="حَمَاكِ الإِلَهُ بِلَادَ الحَرَمْ", duration=dur, fps=FPS,
                   method="CTC forced alignment (omnilingual 1B CTC) on the full mix; review by ear pending",
                   lines=occ), open(f"{tdir}/lyric_word_timing.json", "w"), ensure_ascii=False, indent=1)
    with open(f"{tdir}/scene_mapping.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["shot", "start_s", "end_s", "asset", "background", "from_cx_cy_zoom", "to_cx_cy_zoom", "transition_out", "lyric"])
        for i, sh in enumerate(shots):
            lyr = next((o["id"] + " " + o["text"] for o in occ if sh["t0"] - 0.5 <= o["start"] < sh["t1"]), "instrumental")
            wr.writerow([i + 1, round(sh["t0"], 3), round(min(sh["t1"], dur), 3), sh["img"], sh.get("bg", ""),
                         sh["from"], sh["to"], f'{sh.get("tstyle", "")} {sh.get("td", "")}', lyr])
    with open(f"{tdir}/asset_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", "file", "source_name", "description", "width", "height", "used_in_shots"])
        from PIL import Image
        for k, v in ASSETS.items():
            im = Image.open(f"{B}/scratchpad/saudi/assets/images/{v[0]}")
            used = [str(i + 1) for i, s in enumerate(shots) if s["img"] == k or s.get("bg") == k]
            wr.writerow([k, v[0], v[2], v[1], im.size[0], im.size[1], " ".join(used)])
        wr.writerow(["HANDS", "handshake.svg", "Lucide icons (ISC licence)", "Clasped hands, gold line-art", 24, 24, "closing card"])
        wr.writerow(["SONG_MASTER", "song.mp3", "بلد الحرم (Suno export)", f"{dur:.3f} s (alignment clock), 48 kHz stereo; video adds {TAIL:.0f} s closing card with a soft echo tail", "", "", "all"])
    print("shots", len(shots), "cuts", [round(c, 2) for c in cuts[:6]], "ending", cfg["ending"]["t0"])


if __name__ == "__main__":
    main(*sys.argv[1:5], os.environ["B"])
