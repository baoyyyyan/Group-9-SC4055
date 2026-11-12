"""
Scale check: does the exact QUBO optimum pull ahead of greedy when there are more features?

Same protocol as run_experiments.py (5 stratified 70/30 splits, scores on training rows only,
correlation scoring, alpha = 0.5), but on all 30 Breast Cancer features instead of the 10 "mean" ones.
30 qubits is beyond exact QAOA simulation, so the QUBO is solved exactly by enumerating every
size-k subset (C(30, k) of them) and approximately by simulated annealing.

Usage:
    python scale_check.py            -> results/experiments/scale_30_corr_a0.5.csv, scale_summary_corr_a0.5.csv
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from qubo import build_qubo, qubo_energy, redundancy_matrix, relevance_scores, simulated_annealing
from run_experiments import OUT_DIR, SharedProblem


def exact_k_subset(r, R, k, alpha):
    """Lowest-energy subset among all size-k subsets. On size-k subsets the budget penalty is zero,
    so this is the global QUBO optimum whenever lambda is large enough (build_qubo guarantees that)."""
    combos = np.array(list(itertools.combinations(range(len(r)), k)))
    rel = r[combos].sum(axis=1)
    red = sum(R[combos[:, a], combos[:, b]] for a in range(k) for b in range(a + 1, k))
    energy = -alpha * rel + (1 - alpha) * red
    return [int(i) for i in combos[int(np.argmin(energy))]]


def main(args):
    data = load_breast_cancer()
    X, y, names = data.data[:, : args.n_features], data.target, list(data.feature_names[: args.n_features])
    rows = []
    for split in range(args.splits):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=split)
        r, R = relevance_scores(X_tr, y_tr, "corr"), redundancy_matrix(X_tr, "corr")
        for k in args.ks:
            Q, const, _ = build_qubo(r, R, k, args.alpha)
            problem = SharedProblem(X_tr, X_te, y_tr, y_te, k, r, R)
            picks = {
                "random": problem.select_random(seed=split),
                "top_k": problem.select_top_k(),
                "greedy": problem.select_greedy(),
                "sim_annealing": simulated_annealing(Q, const, seed=split)[0],
                "exact_qubo": exact_k_subset(r, R, k, args.alpha),
            }
            best = None
            for method, feats in picks.items():
                feats = sorted(int(i) for i in feats)
                x = np.zeros(len(r), dtype=int)
                x[feats] = 1
                rows.append({
                    "split": split, "k": k, "method": method, "features": feats,
                    "feature_names": [names[i] for i in feats],
                    "qubo_energy": qubo_energy(x, Q, const),
                    "acc_rf": problem.evaluate(feats), "acc_lr": problem.evaluate_logreg(feats),
                })
            best = min(rw["qubo_energy"] for rw in rows if rw["split"] == split and rw["k"] == k)
            for rw in rows:
                if rw["split"] == split and rw["k"] == k:
                    rw["energy_gap"] = rw["qubo_energy"] - best
                    rw["optimal"] = rw["energy_gap"] < 1e-9
            print(f"split {split} k={k} done")
    df = pd.DataFrame(rows)
    tag = f"corr_a{args.alpha}"
    df.to_csv(os.path.join(OUT_DIR, f"scale_{args.n_features}_{tag}.csv"), index=False)
    summary = df.groupby(["k", "method"]).agg(
        acc_lr_mean=("acc_lr", "mean"), acc_lr_std=("acc_lr", "std"),
        acc_rf_mean=("acc_rf", "mean"), acc_rf_std=("acc_rf", "std"),
        energy_gap_mean=("energy_gap", "mean"), optimal_rate=("optimal", "mean")).reset_index()
    summary.to_csv(os.path.join(OUT_DIR, f"scale_summary_{args.n_features}_{tag}.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-features", type=int, default=30)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--alpha", type=float, default=0.5)
    main(ap.parse_args())
