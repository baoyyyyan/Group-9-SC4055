"""
Plot the 30-feature scale check (scale_check.py): accuracy and QUBO energy gap vs budget k.
Names, colours and styles per method come from plot_style.py, shared by all experiment plots.

Usage:
    python plot_scale.py   -> results/experiments/scale_30_corr_a0.5.png
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import LABELS, STYLES, ordered
from run_experiments import OUT_DIR

df = pd.read_csv(os.path.join(OUT_DIR, "scale_30_corr_a0.5.csv"))
s = df.groupby(["k", "method"]).agg(acc=("acc_lr", "mean"), sd=("acc_lr", "std"),
                                     gap=("energy_gap", "mean")).reset_index()

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9))
for m in ordered(s.method.unique()):
    d = s[s.method == m]
    a1.errorbar(d.k, d.acc, yerr=d.sd, marker="o", capsize=3, label=LABELS[m], **STYLES[m])
    if m != "random":
        a2.plot(d.k, d.gap, marker="o", label=LABELS[m], **STYLES[m])
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
