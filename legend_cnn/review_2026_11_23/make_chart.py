"""The accuracy chart of REPORT.md. Values: review_numbers.py (baselines, CNN runs) and the April results document (81.0%)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
rows = [  # label, value, colour (top to bottom)
    ("Always answer the largest class (Hadrian)", 14.9, ORANGE),
    ("The coin's dealer alone", 18.4, ORANGE),
    ("Ribbon CNN, lower learning rate", 18.5, BLUE),
    ("Ribbon CNN, 128 px ribbon", 18.5, BLUE),
    ("Ribbon CNN, first run", 19.6, BLUE),
    ("Ribbon CNN, bad ribbons removed", 20.3, BLUE),
    ("The coin's sale alone", 24.1, ORANGE),
    ("Head on obverse embeddings (as reported)", 81.0, AQUA),
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
y = np.arange(len(rows))[::-1]
ax.barh(y, [r[1] for r in rows], height=0.56, color=[r[2] for r in rows], zorder=3)
for yi, (_lab, v, _c) in zip(y, rows):
    ax.text(v + 1.2, yi, f"{v:.1f}%", va="center", ha="left", color=INK, fontsize=10, zorder=4)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], color=INK)
ax.set_xlim(0, 100)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xticklabels([f"{t}%" for t in (0, 20, 40, 60, 80, 100)], color=INK2)
ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="both", length=0)
ax.set_xlabel("Share of coins given the right emperor (10 emperors)", color=INK2)
fig.legend(handles=[Patch(color=BLUE, label="Ribbon CNN"), Patch(color=ORANGE, label="No image used"), Patch(color=AQUA, label="Different input: obverse embeddings")],
           loc="upper left", bbox_to_anchor=(0.02, 0.995), ncol=3, frameon=False, fontsize=9, labelcolor=INK, handlelength=1.2, columnspacing=1.6)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(Path(__file__).resolve().parent / "figures" / "accuracy.png", facecolor=SURF)
