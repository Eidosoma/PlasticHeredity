"""Phase G summary figures (results_g/figures/).

fig_g3_hysteresis.png : post-release half-life vs pulse length +
                        maintenance-vs-edit-budget tradeoff
fig_g4_surgery.png    : beta-surgery realized break-risk by arm
fig_g5_ladder.png     : internalized-controller ladder maintenance
                        (skipped gracefully until g5_results.json exists)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_g")
FIG = os.path.join(OUT, "figures")

BLUE = "#4878A8"
AMBER = "#A8641E"
GRAY = "#8A939B"
GREEN = "#3E7D5B"
PURPLE = "#7C5CA8"
INK = "#33383D"

CANDS = ["02", "03"]
CAND_COLOR = {"02": BLUE, "03": AMBER}

plt.rcParams.update({"figure.dpi": 150, "font.size": 8,
                     "axes.titlecolor": INK})


def load(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def g3_figure(g3):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ax = axes[0]
    for cand in CANDS:
        pulses = sorted(int(p) for p in g3[cand]["pulse"])
        hl = [g3[cand]["pulse"][str(p)]["half_life_t07"] for p in pulses]
        ax.plot(pulses, hl, "o-", color=CAND_COLOR[cand], lw=1.3, ms=4,
                label=f"candidate {cand}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("steering pulse length (fissions held)")
    ax.set_ylabel("post-release half-life (fissions to sim. < 0.7)")
    ax.set_title("Accumulating hysteresis: held longer decays slower")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for cand in CANDS:
        per = g3[cand]["periodic"]
        ks = sorted(int(k) for k in per)
        ax.plot([per[str(k)]["edits"] for k in ks],
                [per[str(k)]["inherit"] for k in ks], "o-",
                color=CAND_COLOR[cand], lw=1.3, ms=4,
                label=f"{cand} periodic")
        ev = g3[cand]["event"]
        ths = sorted(ev, key=float)
        ax.plot([ev[t]["edits"] for t in ths],
                [ev[t]["inherit"] for t in ths], "s--",
                color=CAND_COLOR[cand], lw=1.3, ms=5, mfc="white",
                label=f"{cand} event-triggered")
        pr = g3[cand]["periodic_rand"]
        ax.plot([per[str(k)]["edits"] for k in ks],
                [pr[str(k)]["inherit"] for k in ks], ":",
                color=GRAY, lw=1.0,
                label="budget-matched random" if cand == "02" else None)
    ax.set_xlabel("edits used per 60 fissions")
    ax.set_ylabel("maintained inheritance fraction")
    ax.set_title("The price of a thermostat: feedback rate vs heredity")
    ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("G3: control half-life and minimum feedback rate")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_g3_hysteresis.png"))
    plt.close(fig)


def g4_figure(g4):
    arms = ["raise", "lower", "random", "none"]
    colors = [GREEN, PURPLE, GRAY, INK]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for j, cand in enumerate(CANDS):
        ax = axes[j]
        s = g4[cand]["surgery"] if "surgery" in g4[cand] else g4[cand]
        means = s["means"]
        ax.bar(range(4), [means[a] for a in arms], color=colors,
               width=0.62)
        d = s["raise_minus_lower"]
        ax.set_title(f"Candidate {cand}   raise−lower "
                     f"{d['mean']:+.3f} [{d['ci'][0]:+.3f},"
                     f"{d['ci'][1]:+.3f}]", fontsize=8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["raise\ninfluence", "lower\ninfluence",
                            "random\nedges", "no\nsurgery"], fontsize=7)
        if j == 0:
            ax.set_ylabel("realized break-and-renewal risk q")
    fig.suptitle("G4 beta surgery: editing the rulebook alone moves "
                 "heredity (fixed composition)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_g4_surgery.png"))
    plt.close(fig)


def g5_figure(g5):
    policies = ["noop", "random", "L0", "L1", "L2", "L3", "v2_down"]
    labels = ["noop", "random", "L0\nlocal rule", "L1\n+1 bit",
              "L2\n+streak", "L3\ntree", "v2\nfull model"]
    colors = [INK, GRAY, GREEN, GREEN, GREEN, PURPLE, AMBER]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for j, cand in enumerate(CANDS):
        ax = axes[j]
        e = g5[cand]["home"]
        xs = range(len(policies))
        ax.axhline(e["noop"]["maintenance"], color=GRAY, lw=0.8, ls=":")
        ax.axhline(e["v2_down"]["maintenance"], color=AMBER, lw=0.8,
                   ls=":")
        for i, p in enumerate(policies):
            ax.plot([i, i], [e[p]["maintenance"], e[p]["post_perturb"]],
                    color=colors[i], lw=1.0, alpha=0.5)
        ax.plot(xs, [e[p]["maintenance"] for p in policies], "o",
                ms=7, color="none",
                markeredgecolor="none")
        for i, p in enumerate(policies):
            ax.plot(i, e[p]["maintenance"], "o", ms=7, color=colors[i],
                    label="maintenance" if i == 0 and j == 0 else None)
            ax.plot(i, e[p]["post_perturb"], "s", ms=6, mfc="white",
                    color=colors[i],
                    label="after k8 shock" if i == 0 and j == 0
                    else None)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylim(0.87, 1.005)
        ax.set_title(f"Candidate {cand}")
        if j == 0:
            ax.set_ylabel("inheritance fraction (60 fissions)")
            ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("G5: how much of the controller survives "
                 "internalization?")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_g5_ladder.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    g3, g4, g5 = load("g3_results.json"), load("g4_results.json"), \
        load("g5_results.json")
    if g3:
        g3_figure(g3)
        print("fig_g3_hysteresis.png")
    if g4:
        g4_figure(g4)
        print("fig_g4_surgery.png")
    if g5:
        g5_figure(g5)
        print("fig_g5_ladder.png")
    else:
        print("g5_results.json not present yet — ladder figure skipped")


if __name__ == "__main__":
    main()
