"""
Does QAOA make good feature subsets more likely than random guessing, and does depth help?

QAOA at depth p = 0 is exactly uniform random sampling over all size-k subsets (the Dicke start
state with no layers applied), so p = 0 is the random-guessing baseline and p = 1..4 show
what each extra layer adds.

For every train/test split (QUBO rebuilt on the training rows, as in run_experiments.py),
every budget k, every depth p and every optimiser seed, we record:
  prob_opt       chance that one shot lands on the best subset
  gain           prob_opt / random prob_opt
  approx_ratio   (E_max - <E>) / (E_max - E_min) over size-k subsets; 1 = always optimal
  success@s      chance that at least one of s shots hits the best subset

Usage:
    python qaoa_depth.py                       # 5 splits x 2 seeds x k=2..5 x p=0..4, ~10 min on 8 cores
    python qaoa_depth.py --splits 1 --seeds 1 --ps 0 1   # quick check
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from qaoa import RESTARTS, bind, bit_energies, ising_coeffs, qaoa, qaoa_circuit, success_curve
from qubo import build_qubo, redundancy_matrix, relevance_scores
from run_experiments import OUT_DIR, load_data


def problems(splits, ks, alpha, scoring):
    """Yield (split, k, Q, const) with the QUBO built on each split's training rows."""
    X, y, _ = load_data()
    for split in range(splits):
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=split)
        r, R = relevance_scores(X_tr, y_tr, scoring), redundancy_matrix(X_tr, scoring)
        for k in ks:
            Q, const, _ = build_qubo(r, R, k, alpha)
            yield split, k, Q, const


def metrics(split, k, p, seed, sub, expect, p_opt, seconds):
    row = {
        "split": split, "k": k, "p": p, "seed": seed,
        "n_subsets": len(sub),
        "prob_opt": p_opt,
        "gain": p_opt * len(sub),
        "approx_ratio": float((sub.max() - expect) / (sub.max() - sub.min())),
        "seconds": seconds,
    }
    row.update({f"success@{s}": v for s, v in success_curve(p_opt).items()})
    return row


def size_k_energies(Q, const, k):
    energy = bit_energies(Q, const)
    masks = np.array([sum(1 << i for i in c) for c in combinations(range(len(Q)), k)])
    return energy, masks


def run_one(job):
    """Independent random-init QAOA at one depth (Nikki's qaoa(), COBYLA maxiter 180)."""
    split, k, Q, const, p, seed, restarts = job
    energy, masks = size_k_energies(Q, const, k)
    sub = energy[masks]
    t0 = time.perf_counter()
    if p == 0:
        expect, p_opt = float(sub.mean()), 1 / len(masks)  # uniform over size-k subsets
    else:
        _, expect, p_opt, _ = qaoa(Q, const, k, p=p, restarts=restarts, seed=seed)
    return [metrics(split, k, p, seed, sub, expect, p_opt, time.perf_counter() - t0)]


def interp(params, p):
    """INTERP warm start (Zhou et al. 2020): stretch the p-layer gamma/beta schedules to p+1 layers."""
    out = []
    for u in (params[:p], params[p:]):
        u = np.r_[0.0, u, 0.0]
        out.append([(i / p) * u[i] + ((p - i) / p) * u[i + 1] for i in range(p + 1)])
    return np.r_[out[0], out[1]]


def run_interp(job):
    """Layer-by-layer QAOA: depth p+1 starts from the best depth-p parameters (plus random restarts),
    with an iteration budget that grows with depth. Same circuit, start state and loss as qaoa()."""
    split, k, Q, const, max_p, seed, restarts, iters_per_layer = job
    n = len(Q)
    energy, masks = size_k_energies(Q, const, k)
    sub = energy[masks]
    scale = float(np.ptp(sub))
    h, J = ising_coeffs(Q)
    amps = np.zeros(1 << n, dtype=complex)
    amps[masks] = 1 / np.sqrt(len(masks))
    psi0 = Statevector(amps)
    scaled = (energy - sub.mean()) / scale
    rng = np.random.default_rng(seed)

    rows = [metrics(split, k, 0, seed, sub, float(sub.mean()), 1 / len(masks), 0.0)]
    best = None
    for p in range(1, max_p + 1):
        t0 = time.perf_counter()
        circuit = qaoa_circuit(n, p, h, J, scale)

        def probs(params):
            return np.real(psi0.evolve(bind(circuit, params, p)).probabilities())

        def loss(params):
            return float(np.dot(probs(params), scaled))

        guesses = [np.r_[rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)] for _ in range(restarts)]
        if best is not None:
            guesses[0] = interp(best.x, p - 1)
        best = min((minimize(loss, g, method="COBYLA", options={"maxiter": iters_per_layer * 2 * p})
                    for g in guesses), key=lambda r: r.fun)
        pr = np.clip(probs(best.x), 0, None)
        pr /= pr.sum()
        expect = float(np.dot(pr, energy))
        p_opt = float(pr[masks[np.argmin(sub)]])
        rows.append(metrics(split, k, p, seed, sub, expect, p_opt, time.perf_counter() - t0))
    return rows


def plot(df, path):
    g = df.groupby(["k", "p"])
    mean, std = g.mean(numeric_only=True).reset_index(), g.std(numeric_only=True).reset_index()
    # gain is very skewed across runs, so show median and interquartile range instead of mean +- std
    q25, q50, q75 = (g.gain.quantile(q).reset_index() for q in (0.25, 0.5, 0.75))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k in sorted(df.k.unique()):
        m, s = mean[mean.k == k], std[std.k == k]
        lo, mid, hi = (q[q.k == k].gain.values for q in (q25, q50, q75))
        axes[0].errorbar(m.p, m.approx_ratio, yerr=s.approx_ratio, marker="o", capsize=3, label=f"k={k}")
        axes[1].errorbar(m.p, mid, yerr=[mid - lo, hi - mid], marker="o", capsize=3, label=f"k={k}")
        axes[2].plot(m.p, m["success@10"], marker="o", label=f"k={k}")
    axes[0].set_title("Approximation ratio (mean ± std; 1 = always optimal)")
    axes[1].set_title("P(best subset) / random guessing (median, IQR)")
    axes[1].axhline(1, color="grey", ls="--")
    axes[1].set_yscale("log")
    axes[2].set_title("P(best subset within 10 shots)")
    for ax in axes:
        ax.set_xlabel("QAOA depth p  (p = 0 is random guessing)")
        ax.set_xticks(sorted(df.p.unique()))
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_compare(random_df, interp_df, path):
    """Approximation ratio and median gain vs depth: independent random restarts (dashed) vs warm start (solid)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, k in enumerate(sorted(interp_df.k.unique())):
        colour = f"C{i}"
        for df, ls, label in [(random_df, "--", "random init"), (interp_df, "-", "warm start")]:
            g = df[df.k == k].groupby("p")
            axes[0].plot(g.approx_ratio.mean(), ls=ls, marker="o", color=colour, label=f"k={k}, {label}")
            axes[1].plot(g.gain.median(), ls=ls, marker="o", color=colour)
    axes[0].set_title("Approximation ratio (mean)")
    axes[1].set_title("P(best subset) / random guessing (median)")
    axes[1].axhline(1, color="grey", ls=":")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("QAOA depth p  (p = 0 is random guessing)")
        ax.set_xticks(sorted(interp_df.p.unique()))
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--ps", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--scoring", choices=["corr", "mi"], default="corr")
    parser.add_argument("--restarts", type=int, default=RESTARTS)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--init", choices=["random", "interp"], default="random",
                        help="random: independent restarts per depth (qaoa.py); interp: layer-by-layer warm start")
    parser.add_argument("--iters-per-layer", type=int, default=75,
                        help="interp only: COBYLA maxiter = this x number of parameters (2p)")
    args = parser.parse_args()

    qubos = list(problems(args.splits, args.ks, args.alpha, args.scoring))
    if args.init == "random":
        worker = run_one
        jobs = [(split, k, Q, const, p, seed, args.restarts)
                for split, k, Q, const in qubos
                for p in args.ps
                for seed in (range(args.seeds) if p > 0 else [0])]  # p = 0 has no randomness
        jobs.sort(key=lambda j: -j[4])  # longest jobs first so the pool finishes evenly
    else:
        worker = run_interp
        jobs = [(split, k, Q, const, max(args.ps), seed, args.restarts, args.iters_per_layer)
                for split, k, Q, const in qubos for seed in range(args.seeds)]
    tag = f"{args.scoring}_a{args.alpha}" + ("_interp" if args.init == "interp" else "")
    os.makedirs(OUT_DIR, exist_ok=True)
    partial_path = os.path.join(OUT_DIR, f"qaoa_depth_{tag}.partial.csv")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, job_rows in enumerate(pool.map(worker, jobs), 1):
            rows.extend(job_rows)
            pd.DataFrame(rows).to_csv(partial_path, index=False)  # keep finished jobs if the run is interrupted
            row = job_rows[-1]
            print(f"[{i}/{len(jobs)}] split {row['split']} k={row['k']} p={row['p']} seed {row['seed']}: "
                  f"ratio={row['approx_ratio']:.3f} gain={row['gain']:.2f} ({row['seconds']:.1f}s)", flush=True)

    df = pd.DataFrame(rows).sort_values(["k", "p", "split", "seed"])
    df.to_csv(os.path.join(OUT_DIR, f"qaoa_depth_{tag}.csv"), index=False)
    os.remove(partial_path)
    cols = ["approx_ratio", "gain", "prob_opt", "success@1", "success@10", "success@50", "seconds"]
    summary = df.groupby(["k", "p"])[cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(os.path.join(OUT_DIR, f"qaoa_depth_summary_{tag}.csv"), index=False)
    plot(df, os.path.join(OUT_DIR, f"qaoa_depth_{tag}.png"))
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(df.groupby(["k", "p"])[cols].mean().to_string())
    print("Saved to", OUT_DIR)
