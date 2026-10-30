"""
Main experiment: every feature-selection method on the same problem, scored by test accuracy.

For each random train/test split:
  1. relevance r and redundancy R are computed on the TRAINING rows only (no test leakage)
  2. for each budget k, every method picks k features:
       all, random, top-k, greedy (baselines.py), brute force, simulated annealing (qubo.py), QAOA (qaoa.py)
  3. each selection is scored with random forest (the shared evaluate()) and logistic regression
     on the held-out test rows, and its QUBO energy is recorded

Usage:
    python run_experiments.py                 # 5 splits, k = 2..5, ~6 min
    python run_experiments.py --splits 1 --no-qaoa   # quick check
"""
import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import BenchmarkBaselines
from qaoa import bit_energies, indices, qaoa
from qubo import brute_force, build_qubo, redundancy_matrix, relevance_scores, simulated_annealing

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "results", "experiments")
ALPHA_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
METHODS = ["all", "random", "top_k", "greedy", "brute_force", "sim_annealing", "qaoa"]


class SharedProblem(BenchmarkBaselines):
    """BenchmarkBaselines on a split and scores we pass in, so every method sees the same problem."""

    def __init__(self, X_train, X_test, y_train, y_test, k, relevance, redundancy, lambda_weight=1.0):
        self.answer_to_the_universe = 42
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.k = k
        self.n_features = X_train.shape[1]
        self.lambda_weight = lambda_weight
        self.relevance = relevance
        self.redundancy = redundancy

    def evaluate_logreg(self, features):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(self.X_train[:, features], self.y_train)
        return accuracy_score(self.y_test, model.predict(self.X_test[:, features]))


def load_data(noise_features=0, seed=42):
    """The 10 'mean' features, plus optional columns of pure Gaussian noise (as in baselines.py)
    to test whether a method avoids useless features. One feature = one qubit for QAOA."""
    data = load_breast_cancer()
    X, names = data.data[:, :10], [str(n) for n in data.feature_names[:10]]
    if noise_features:
        noise = np.random.default_rng(seed).standard_normal((X.shape[0], noise_features))
        X, names = np.hstack([X, noise]), names + [f"noise_{i}" for i in range(noise_features)]
    return X, data.target, names


def select(method, problem, Q, const, k, seed, args):
    if method == "all":
        return problem.select_all_features()
    if method == "random":
        return problem.select_random(seed=seed)
    if method == "top_k":
        return problem.select_top_k()
    if method == "greedy":
        return problem.select_greedy()
    if method == "brute_force":
        return brute_force(Q, const)
    if method == "sim_annealing":
        return simulated_annealing(Q, const, seed=seed)[0]
    if method == "qaoa":
        pick, _, _, _ = qaoa(Q, const, k, p=args.p, restarts=args.restarts, seed=seed)
        return indices(pick, len(Q))
    raise ValueError(method)


def choose_alpha(X_tr, y_tr, k, scoring, grid=ALPHA_GRID, folds=5):
    """Pick alpha by cross-validation on the training rows only: for each fold, rebuild the QUBO on
    the other folds, take its optimal subset and score it with logistic regression (fast) on the fold."""
    cv = StratifiedKFold(folds, shuffle=True, random_state=0)
    scores = {a: [] for a in grid}
    for fit_idx, val_idx in cv.split(X_tr, y_tr):
        X_fit, y_fit = X_tr[fit_idx], y_tr[fit_idx]
        r, R = relevance_scores(X_fit, y_fit, scoring), redundancy_matrix(X_fit, scoring)
        for a in grid:
            feats = brute_force(*build_qubo(r, R, k, a)[:2])
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            model.fit(X_fit[:, feats], y_fit)
            scores[a].append(accuracy_score(y_tr[val_idx], model.predict(X_tr[val_idx][:, feats])))
    return max(grid, key=lambda a: np.mean(scores[a]))


def run(args):
    X, y, names = load_data(args.noise_features)
    methods = [m for m in METHODS if not (args.no_qaoa and m == "qaoa")]
    rows = []
    for split in range(args.splits):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=split)
        r = relevance_scores(X_tr, y_tr, args.scoring)
        R = redundancy_matrix(X_tr, args.scoring)
        for k in args.ks:
            problem = SharedProblem(X_tr, X_te, y_tr, y_te, k, r, R)
            alpha = choose_alpha(X_tr, y_tr, k, args.scoring) if args.alpha == "cv" else float(args.alpha)
            Q, const, _ = build_qubo(r, R, k, alpha)
            energy = bit_energies(Q, const)
            for method in methods:
                if method == "all" and k != args.ks[0]:
                    continue  # does not depend on k
                t0 = time.perf_counter()
                feats = sorted(int(i) for i in select(method, problem, Q, const, k, split, args))
                seconds = time.perf_counter() - t0
                mask = sum(1 << i for i in feats)
                rows.append({
                    "split": split,
                    "k": len(feats) if method == "all" else k,
                    "method": method,
                    "alpha": alpha,
                    "features": feats,
                    "feature_names": [names[i] for i in feats],
                    "n_noise_selected": sum(names[i].startswith("noise_") for i in feats),
                    "qubo_energy": float(energy[mask]) if method != "all" else None,
                    "acc_rf": problem.evaluate(feats),
                    "acc_logreg": problem.evaluate_logreg(feats),
                    "select_seconds": seconds,
                })
                print(f"split {split} k={k} {method:<14s} {feats}  "
                      f"rf={rows[-1]['acc_rf']:.3f} lr={rows[-1]['acc_logreg']:.3f}  ({seconds:.1f}s)")
    return pd.DataFrame(rows)


def add_energy_gap(df):
    """How far each selection's QUBO energy is above the brute-force optimum (0 = optimal)."""
    best = df[df.method == "brute_force"].set_index(["split", "k"]).qubo_energy
    df["energy_gap"] = [
        e - best.get((s, k), np.nan) if e is not None else np.nan
        for s, k, e in zip(df.split, df.k, df.qubo_energy)
    ]
    return df


def summarise(df):
    return (df.groupby(["k", "method"], sort=False)
              .agg(acc_rf_mean=("acc_rf", "mean"), acc_rf_std=("acc_rf", "std"),
                   acc_lr_mean=("acc_logreg", "mean"), acc_lr_std=("acc_logreg", "std"),
                   energy_gap_mean=("energy_gap", "mean"),
                   optimal_rate=("energy_gap", lambda g: float(np.mean(np.isclose(g, 0)))),
                   n_noise_selected=("n_noise_selected", "mean"),
                   seconds=("select_seconds", "mean"))
              .reset_index())


def plot(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    all_row = summary[summary.method == "all"].iloc[0]
    for ax, col, title in [(axes[0], "acc_rf", "Random forest"), (axes[1], "acc_lr", "Logistic regression")]:
        for method in summary.method.unique():
            if method == "all":
                continue
            s = summary[summary.method == method]
            ax.errorbar(s.k, s[f"{col}_mean"], yerr=s[f"{col}_std"], marker="o", capsize=3, label=method)
        ax.axhline(all_row[f"{col}_mean"], color="grey", ls="--", label=f"all {int(all_row.k)} features")
        ax.set_title(title)
        ax.set_xlabel("feature budget k")
        ax.set_xticks(sorted(summary[summary.method != "all"].k.unique()))
    axes[0].set_ylabel("test accuracy (mean ± std over splits)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--alpha", default="0.5", help="a number, or \"cv\" to pick alpha per split and k by CV on the training rows")
    parser.add_argument("--scoring", choices=["corr", "mi"], default="corr")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--no-qaoa", action="store_true")
    parser.add_argument("--noise-features", type=int, default=0,
                        help="append this many pure-noise columns (QAOA is practical up to ~4, i.e. 14 qubits)")
    args = parser.parse_args()

    df = add_energy_gap(run(args))
    summary = summarise(df)
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{args.scoring}_a{args.alpha}" + (f"_noise{args.noise_features}" if args.noise_features else "")
    df.to_csv(os.path.join(OUT_DIR, f"accuracy_{tag}.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, f"summary_{tag}.csv"), index=False)
    with open(os.path.join(OUT_DIR, f"config_{tag}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    plot(summary, os.path.join(OUT_DIR, f"accuracy_vs_k_{tag}.png"))
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(summary.to_string(index=False))
    print("Saved to", OUT_DIR)
