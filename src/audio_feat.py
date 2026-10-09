"""Audio features at video frame rate: kick pulse, onset spark, loudness, vocal envelope.

usage: python audio_feat.py song16k.wav feats.npz [fps]
"""
import sys
import wave

import numpy as np
from scipy.ndimage import uniform_filter1d


def load_wav(path):
    w = wave.open(path)
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return x, w.getframerate()


def norm(v):
    v = v - np.percentile(v, 5)
    return np.clip(v / (np.percentile(v, 98) + 1e-9), 0, 1)


def decay(v, k):
    out = np.zeros_like(v)
    acc = 0.0
    for i, x in enumerate(v):
        acc = max(x, acc * k)
        out[i] = acc
    return out


def beat_times(onset, fps, lo=60, hi=150):
    """Tempo by autocorrelation, phase by grid energy, each beat snapped to the nearest onset peak."""
    o = onset - uniform_filter1d(onset, fps)
    o = np.maximum(o, 0)
    ac = np.correlate(o, o, "full")[len(o) - 1:]
    lags = np.arange(int(fps * 60 / hi), int(fps * 60 / lo) + 1)
    sc = ac[lags] * (1 - 0.15 * np.abs(np.log2(lags / (fps * 60 / 100))))
    period = lags[int(np.argmax(sc))]
    best, phase = -1, 0
    for ph in range(period):
        v = o[ph::period].sum()
        if v > best:
            best, phase = v, ph
    beats, t = [], float(phase)
    while t < len(o) - 1:
        i = int(round(t))
        w = o[max(0, i - 2):i + 3]
        j = max(0, i - 2) + int(np.argmax(w)) if w.max() > 0 else i
        beats.append(j / fps)
        t += period
    return np.array(beats, np.float32)


def features(x, sr, fps=30):
    hop = sr // fps
    n = len(x) // hop
    win = np.hanning(2048)
    pad = np.pad(x, (1024, 1024))
    spec = np.stack([np.abs(np.fft.rfft(pad[i * hop:i * hop + 2048] * win)) for i in range(n)])
    freqs = np.fft.rfftfreq(2048, 1 / sr)
    logs = np.log1p(spec * 10)

    low = logs[:, freqs < 150].mean(1)
    flux_low = np.maximum(np.diff(low, prepend=low[0]), 0)
    kick = decay(norm(flux_low), 0.86)

    flux = np.maximum(np.diff(logs, axis=0, prepend=logs[:1]), 0).mean(1)
    spark = decay(norm(flux), 0.80)

    rms = np.sqrt(uniform_filter1d(x ** 2, hop)[::hop][:n])
    loud = norm(uniform_filter1d(rms, fps * 2))

    band = (freqs > 300) & (freqs < 3400)
    vocal = norm(uniform_filter1d(spec[:, band].mean(1), fps // 3))
    beats = beat_times(np.maximum(flux_low * 2 + flux, 0), fps)
    return dict(kick=kick, spark=spark, loud=loud, vocal=vocal, fps=fps, beats=beats)


if __name__ == "__main__":
    x, sr = load_wav(sys.argv[1])
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    f = features(x, sr, fps)
    np.savez(sys.argv[2], **f)
    print("frames", len(f["kick"]), "beats", len(f["beats"]), "bpm", round(60 / np.median(np.diff(f["beats"])), 1))
