"""Word timing by CTC forced alignment (omnilingual CTC model, full mix).

usage: python align.py model_dir song16k.wav lyrics.txt words.json
"""
import json
import re
import sys
import wave

import numpy as np
import onnxruntime as ort

WIN, HOP, EDGE = 16.0, 4.0, 1.0
SR = 16000
DIAC = re.compile(r"[ً-ْٰـ]")


def load_tokens(path):
    toks = {}
    for line in open(path, encoding="utf8"):
        p = line.rstrip("\n").rsplit(" ", 1)
        if len(p) == 2:
            toks[p[0]] = int(p[1])
    return toks


def parse_lyrics(path):
    section, lines = None, []
    for raw in open(path, encoding="utf8"):
        raw = raw.strip()
        if not raw:
            continue
        m = re.match(r"\[(.+?)\]", raw)
        if m:
            section = m.group(1)
            continue
        lines.append((section, raw))
    return lines


def frame_probs(sess, x):
    """Average posteriors per output frame over overlapping windows."""
    win, hop, edge = int(WIN * SR), int(HOP * SR), EDGE
    total = None
    starts = list(range(0, max(1, len(x) - win + hop), hop))
    for s in starts:
        seg = x[s:s + win]
        if len(seg) < SR:
            break
        seg = (seg - seg.mean()) / (seg.std() + 1e-7)
        logits = sess.run(None, {"x": seg[None].astype(np.float32)})[0][0]
        if total is None:
            fps = logits.shape[0] / (len(seg) / SR)
            nfr = int(np.ceil(len(x) / SR * fps)) + 2
            total = np.zeros((nfr, logits.shape[1]), np.float32)
            count = np.zeros(nfr, np.float32)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        f0 = int(round(s / SR * fps))
        lo = 0 if s == 0 else int(edge * fps)
        hi = len(p) if s + win >= len(x) else len(p) - int(edge * fps)
        n = min(hi, nfr - f0) - lo
        total[f0 + lo:f0 + lo + n] += p[lo:lo + n]
        count[f0 + lo:f0 + lo + n] += 1
        print(f"\r{s / SR:6.1f}s", end="", file=sys.stderr)
    print(file=sys.stderr)
    keep = count > 0
    total[keep] /= count[keep, None]
    total[~keep, 0] = 1
    return total, fps


def viterbi(logp_blank, logp_tok, ids):
    """CTC forced alignment. Returns frame index where each label starts."""
    T = logp_blank.shape[0]
    S = 2 * len(ids) + 1
    lab = np.full(S, -1)
    lab[1::2] = np.arange(len(ids))
    emit = np.empty((T, S), np.float32)
    emit[:, 0::2] = logp_blank[:, None]
    emit[:, 1::2] = logp_tok[:, ids]
    skip = np.zeros(S, bool)
    for s in range(3, S, 2):
        skip[s] = ids[(s - 1) // 2] != ids[(s - 3) // 2]
    NEG = -1e30
    dp = np.full(S, NEG, np.float32)
    dp[0], dp[1] = emit[0, 0], emit[0, 1]
    back = np.zeros((T, S), np.int8)
    for t in range(1, T):
        a = dp
        b = np.concatenate([[NEG], dp[:-1]])
        c = np.concatenate([[NEG, NEG], dp[:-2]])
        c = np.where(skip, c, NEG)
        stack = np.stack([a, b, c])
        arg = stack.argmax(0)
        dp = stack[arg, np.arange(S)] + emit[t]
        back[t] = arg
    s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    path = np.empty(T, int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
    starts = np.full(len(ids), -1)
    for t in range(T):
        if lab[path[t]] >= 0 and starts[lab[path[t]]] < 0:
            starts[lab[path[t]]] = t
    return starts, path, lab


def main(model_dir, wav, lyrics, out):
    toks = load_tokens(f"{model_dir}/tokens.txt")
    w = wave.open(wav)
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    cache = out.replace(".json", "_probs.npz")
    try:
        c = np.load(cache)
        probs, fps = c["probs"].astype(np.float32), float(c["fps"])
    except OSError:
        sess = ort.InferenceSession(f"{model_dir}/model.int8.onnx")
        probs, fps = frame_probs(sess, x)
        np.savez(cache, probs=probs.astype(np.float16), fps=fps)

    lines = parse_lyrics(lyrics)
    ids, owner, words = [], [], []
    for li, (sec, text) in enumerate(lines):
        for wi, word in enumerate(text.split()):
            plain = DIAC.sub("", word)
            chars = [toks[c] for c in plain if c in toks]
            words.append(dict(section=sec, line=li, word=wi, text=word))
            for c in chars:
                ids.append(c)
                owner.append(len(words) - 1)
    ids = np.array(ids)
    blank = np.log(np.maximum(probs[:, 0], probs[:, 4]) + 1e-9)
    logp = np.log(probs + 1e-9)
    starts, path, lab = viterbi(blank, logp, ids)
    owner = np.array(owner)
    for k, wd in enumerate(words):
        idx = np.where(owner == k)[0]
        wd["start"] = round(float(starts[idx[0]] / fps), 3)
        # end = last frame on this word's last label
        last = idx[-1]
        fr = np.where(lab[path] == last)[0]
        wd["end"] = round(float((fr[-1] + 1) / fps), 3)
    json.dump(dict(fps=fps, words=words, duration=len(x) / SR), open(out, "w"), ensure_ascii=False, indent=1)
    cur = None
    for wd in words:
        if wd["line"] != cur:
            cur = wd["line"]
            print(f"\n{wd['start']:7.2f} [{wd['section']}]", end=" ")
        print(f"{wd['text']}({wd['start']:.1f})", end=" ")
    print()


if __name__ == "__main__":
    main(*sys.argv[1:5])
