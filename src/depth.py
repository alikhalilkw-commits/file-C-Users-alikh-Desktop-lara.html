"""Depth maps for parallax (Depth Anything V2 small, ONNX).

usage: python depth.py model.onnx out_dir img1 img2 ...   -> out_dir/<name>.npy (float16, 0=far .. 1=near)
"""
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def depth_map(sess, path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
    h, w = im.shape[:2]
    x = cv2.resize(im, (518, 518), interpolation=cv2.INTER_CUBIC)
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    d = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
    # edge-aware smoothing keeps layers from tearing at silhouettes
    d = cv2.bilateralFilter(d.astype(np.float32), 9, 0.08, 9)
    return cv2.GaussianBlur(d, (0, 0), 1.5)


def main(model, out_dir, paths):
    os.makedirs(out_dir, exist_ok=True)
    sess = ort.InferenceSession(model)
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        d = depth_map(sess, p)
        np.save(os.path.join(out_dir, name + ".npy"), d.astype(np.float16))
        vis = (np.clip(d, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, name + "_depth.png"), cv2.applyColorMap(vis, cv2.COLORMAP_TURBO))
        print(name, d.shape, float(d.mean()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
