"""Phase H figure: four-cell strict-event occurrence rates with
whole-matrix bootstrap CIs vs the external clean-room benchmark."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_strict8_occurrence")
FIG = os.path.join(OUT, "figures")

BLUE = "#4878A8"
AMBER = "#A8641E"
INK = "#33383D"
GRAY = "#8A939B"

plt.rcParams.update({"figure.dpi": 150, "font.size": 8,
                     "axes.titlecolor": INK})

with open(os.path.join(OUT, "strict8_results.json")) as f:
    res = json.load(f)

cells = ["02/A", "02/B", "03/A", "03/B"]
os.makedirs(FIG, exist_ok=True)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for i, name in enumerate(cells):
    c = res["cells"][name]
    color = BLUE if name.startswith("02") else AMBER
    ax.errorbar(i, c["rate"],
                yerr=[[c["rate"] - c["ci95"][0]],
                      [c["ci95"][1] - c["rate"]]],
                fmt="o", color=color, ms=6, capsize=4, lw=1.3,
                label="Fable (this module)" if i == 0 else None)
    ext = res["external_benchmark"][name]
    ax.plot(i, ext, "x", color=INK, ms=8, mew=1.6,
            label="external clean room" if i == 0 else None)
ax.axhline(0, color=GRAY, lw=0.7)
ax.set_xticks(range(4))
ax.set_xticklabels(cells)
ax.set_xlabel("candidate / prospective branch half")
ax.set_ylabel("strict-event occurrence rate")
ax.set_ylim(bottom=0)
ax.set_title("Strict-event occurrence (95% whole-matrix CIs) "
             "vs external benchmark", fontsize=9)
ax.legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_strict8_cells.png"))
print("written", os.path.join(FIG, "fig_strict8_cells.png"))
