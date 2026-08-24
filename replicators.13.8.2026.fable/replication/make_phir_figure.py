"""Phase I figure: heredity moves, reconstructed Phi-r does not."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_phir_bridge")
FIG = os.path.join(OUT, "figures")

BLUE = "#4878A8"
AMBER = "#A8641E"
INK = "#33383D"

plt.rcParams.update({"figure.dpi": 150, "font.size": 8,
                     "axes.titlecolor": INK})

with open(os.path.join(OUT, "phir_bridge_results.json")) as f:
    res = json.load(f)
with open(os.path.join(OUT, "phir_code_results.json")) as f:
    resc = json.load(f)

ARMS = ["ph_stab", "ph_destab", "phir_max", "phir_min", "random",
        "noop"]
LAB = ["v2\nstabilize", "v2\ndestabilize", "surrogate\nΦ-r max",
       "surrogate\nΦ-r min", "random", "noop"]
COL = {"02": BLUE, "03": AMBER}

os.makedirs(FIG, exist_ok=True)
fig, axes = plt.subplots(3, 1, figsize=(6.8, 6.4), sharex=True)
for cand in ("02", "03"):
    arms = res[cand]["arms"]
    carms = resc[cand]["arms"]
    axes[0].plot(range(6), [arms[a]["inherit"] for a in ARMS], "o-",
                 color=COL[cand], lw=1.2, ms=5,
                 label=f"candidate {cand}")
    axes[1].plot(range(6), [arms[a]["phi"] for a in ARMS], "o-",
                 color=COL[cand], lw=1.2, ms=5)
    axes[2].plot(range(6), [carms[a]["phi_code"] for a in ARMS],
                 "o-", color=COL[cand], lw=1.2, ms=5)
axes[0].set_ylabel("inherited fraction")
axes[0].set_title("Heredity responds to its controllers…")
axes[0].legend(frameon=False, fontsize=7)
axes[1].set_ylabel("Φ-r, paper's printed formula")
axes[1].set_title("…the PRINTED Φ-r formula stays flat and slightly "
                  "negative…")
axes[1].axhline(0, color=INK, lw=0.7, ls=":")
axes[2].set_ylabel("Φ_R, authors' implemented code")
axes[2].set_title("…but the authors' IMPLEMENTED Φ_R responds to "
                  "the heredity dial")
axes[2].set_xticks(range(6))
axes[2].set_xticklabels(LAB, fontsize=7)
fig.suptitle("Phase I + addendum: one name, two quantities — only "
             "the implemented one couples to heredity")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_phir_bridge.png"))
print("written", os.path.join(FIG, "fig_phir_bridge.png"))
