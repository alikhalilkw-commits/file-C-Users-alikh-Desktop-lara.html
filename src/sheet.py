"""Contact sheet of proof stills.  usage: python sheet.py out.jpg cols still1.png still2.png ..."""
import os
import sys

import cv2
import numpy as np


def main(out, cols, paths):
    cols = int(cols)
    tiles = []
    for p in paths:
        im = cv2.resize(cv2.imread(p), (360, 640), interpolation=cv2.INTER_AREA)
        label = os.path.basename(p).replace("still_", "").replace(".png", "") + "s"
        cv2.putText(im, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(im, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(im)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
