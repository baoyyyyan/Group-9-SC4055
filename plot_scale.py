"""
Plot the 30-feature scale check (scale_check.py): accuracy and QUBO energy gap vs budget k.
Colours per method match the other experiment plots (matplotlib tab10 order used in run_experiments.py).

Usage:
    python plot_scale.py   -> results/experiments/scale_30_corr_a0.5.png
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

from run_experiments import OUT_DIR

COLORS = {"random": "tab:blue", "top_k": "tab:orange", "greedy": "tab:green",
          "exact_qubo": "tab:red", "sim_annealing": "tab:purple"}
LABELS = {"exact_qubo": "exact QUBO optimum", "greedy": "greedy", "sim_annealing": "sim_annealing",
          "top_k": "top_k", "random": "random"}

df = pd.read_csv(os.path.join(OUT_DIR, "scale_30_corr_a0.5.csv"))
s = df.groupby(["k", "method"]).agg(acc=("acc_lr", "mean"), sd=("acc_lr", "std"),
                                     gap=("energy_gap", "mean")).reset_index()

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9))
for m in ["random", "top_k", "greedy", "sim_annealing", "exact_qubo"]:
    d = s[s.method == m]
    a1.errorbar(d.k, d.acc, yerr=d.sd, marker="o", capsize=3, lw=2, color=COLORS[m], label=LABELS[m])
    if m not in ("exact_qubo", "random"):
        a2.plot(d.k, d.gap, marker="o", lw=2, color=COLORS[m], label=LABELS[m])
a2.axhline(0, color="tab:red", lw=2, ls="--", label="exact QUBO optimum")
a1.set(title="Test accuracy, logistic regression (30 features)", xlabel="feature budget k",
       ylabel="test accuracy (mean ± std over splits)")
a2.set(title="QUBO energy above the optimum (0 = optimal)", xlabel="feature budget k", ylabel="energy gap")
for a in (a1, a2):
    a.set_xticks(sorted(s.k.unique()))
    a.grid(alpha=0.25)
a1.legend(fontsize=8, loc="lower right")
a2.legend(fontsize=8, loc="upper left")
fig.tight_layout()
out = os.path.join(OUT_DIR, "scale_30_corr_a0.5.png")
fig.savefig(out, dpi=150)
print(out)
