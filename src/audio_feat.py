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
    return dict(kick=kick, spark=spark, loud=loud, vocal=vocal, fps=fps)


if __name__ == "__main__":
    x, sr = load_wav(sys.argv[1])
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    f = features(x, sr, fps)
    np.savez(sys.argv[2], **f)
    print("frames", len(f["kick"]))
