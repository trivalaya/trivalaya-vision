"""Figure 4 of REPORT.md: three coins, each as the April experiment cut it and as cut in one coordinate frame. An illustration, not a measurement.

Old = legend_cnn.unwrap_core.unwrap_coin with the promoted config, on the 512 px image and the mask file's alpha (as the experiment ran).
New = the mask file alone: its own pixels and its own outline, so centre, radius and pixels share a frame; rim on top, so letters stand upright.

Usage, from the repository root:  python legend_cnn/review_2026_11_23/demo_unwrap.py <folder>
<folder> holds, per coin id, <id>_h.jpg (the row's highres_url) and <id>_t.png (its transparent_url) from data/confusion_subset.csv.
Writes old_<id>.png and new_<id>.png into <folder>."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from legend_cnn.unwrap_core import UnwrapConfig, unwrap_coin  # noqa: E402

COINS = ("164261", "164314", "137496")


def new_unwrap(png, inner=0.55, outer=1.0):
    a = png[:, :, 3]
    gray = cv2.cvtColor(png[:, :, :3], cv2.COLOR_BGR2GRAY)
    gray[a < 128] = 0
    _, b = cv2.threshold(a, 127, 255, cv2.THRESH_BINARY)
    cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (cx, cy), r = cv2.minEnclosingCircle(max(cs, key=cv2.contourArea))
    width, rb = int(min(1800, 2 * np.pi * r * 0.8)), int(r)
    raw = cv2.warpPolar(gray, (rb, width), (cx, cy), r, cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR)
    rib = raw.T[int(rb * inner):int(rb * outer), :]
    rib = np.roll(rib, -int(width * 0.375), axis=1)        # start the strip at about 7:30 on the dial
    return cv2.flip(rib, 0)


def main(folder):
    cfg = UnwrapConfig.from_dict(json.load(open(REPO / "legend_cnn/runs/run_20260410_013529_promoted_v1/config.json")))
    for cid in COINS:
        img = cv2.imread(str(folder / f"{cid}_h.jpg"), cv2.IMREAD_GRAYSCALE)
        png = cv2.imread(str(folder / f"{cid}_t.png"), cv2.IMREAD_UNCHANGED)
        old, meta = unwrap_coin(img, png, cfg)
        new = new_unwrap(png)
        print(cid, {k: meta[k] for k in ("cx", "cy", "radius", "used_fallback", "fallback_reason")}, "image", img.shape, "mask", png.shape[:2])
        cv2.imwrite(str(folder / f"old_{cid}.png"), (old * 255).astype(np.uint8))
        cv2.imwrite(str(folder / f"new_{cid}.png"), new)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
