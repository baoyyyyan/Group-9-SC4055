"""
How much of QAOA's advantage survives hardware noise?

1. Optimise QAOA without noise, layer by layer (qaoa_depth.run_interp), for depths p = 1..4.
2. Re-run each optimised circuit on Qiskit Aer's density-matrix simulator with depolarizing errors
   on every gate (two-qubit error rate e2, one-qubit rate e2 / 10), for a range of e2.

Noise can push the state outside the "exactly k features" subspace, so we record:
  p_valid        chance a shot has exactly k features
  prob_opt/gain  chance a shot is the best subset (vs uniform random guessing over size-k subsets)
  approx_ratio   over the valid shots only (invalid shots are thrown away, as you would in practice)

Simplifications (state these in the report): the start state is prepared perfectly (no noise on
the Dicke-state preparation), every qubit pair can interact directly (real chips need extra SWAP
gates, so this is optimistic), and there is no readout error.

Usage:
    python qaoa_noise.py                      # 3 splits x k=2..5, ~10 min on 4 workers
    python qaoa_noise.py --splits 1 --ks 3    # quick check
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qaoa import bind, ising_coeffs, qaoa_circuit
from qaoa_depth import problems, run_interp, size_k_energies
from run_experiments import OUT_DIR

ERROR_RATES = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def noise_model(e2):
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(e2, 2), ["rzz", "rxx", "ryy"])
    model.add_all_qubit_quantum_error(depolarizing_error(e2 / 10, 1), ["rz"])
    return model


def noisy_probs(Q, k, p, params, e2):
    n = len(Q)
    energy, masks = size_k_energies(Q, 0.0, k)
    h, J = ising_coeffs(Q)
    amps = np.zeros(1 << n, dtype=complex)
    amps[masks] = 1 / np.sqrt(len(masks))
    qc = QuantumCircuit(n)
    qc.set_density_matrix(DensityMatrix(Statevector(amps)))
    qc.compose(bind(qaoa_circuit(n, p, h, J, float(np.ptp(energy[masks]))), params, p), inplace=True)
    qc.save_probabilities()
    sim = AerSimulator(method="density_matrix", noise_model=noise_model(e2) if e2 > 0 else None)
    return np.asarray(sim.run(qc).result().data()["probabilities"])


def run_job(job):
    split, k, Q, const, max_p, seed, restarts, error_rates = job
    energy, masks = size_k_energies(Q, const, k)
    sub = energy[masks]
    best = masks[np.argmin(sub)]
    rows = []
    for opt in run_interp((split, k, Q, const, max_p, seed, restarts, 75)):
        if opt["p"] == 0:
            continue
        for e2 in error_rates:
            pr = noisy_probs(Q, k, opt["p"], np.array(opt["params"]), e2)
            p_valid = float(pr[masks].sum())
            valid_expect = float(np.dot(pr[masks], sub) / p_valid)
            rows.append({
                "split": split, "k": k, "p": opt["p"], "seed": seed, "error_2q": e2,
                "p_valid": p_valid,
                "prob_opt": float(pr[best]),
                "gain": float(pr[best]) * len(masks),
                "approx_ratio": float((sub.max() - valid_expect) / (sub.max() - sub.min())),
                "random_approx_ratio": float((sub.max() - sub.mean()) / (sub.max() - sub.min())),
            })
    return rows


def plot(df, path):
    g = df.groupby(["error_2q", "p"])
    mean, med = g.mean(numeric_only=True).reset_index(), g.gain.median().reset_index()
    rates = sorted(df.error_2q.unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, e2 in enumerate(rates):
        m, md = mean[mean.error_2q == e2], med[med.error_2q == e2]
        label = "no noise" if e2 == 0 else f"2-qubit error {e2:g}"
        axes[0].plot(m.p, m.approx_ratio - m.random_approx_ratio, marker="o", color=f"C{i}", label=label)
        axes[1].plot(md.p, md.gain, marker="o", color=f"C{i}")
        axes[2].plot(m.p, m.p_valid, marker="o", color=f"C{i}")
    axes[0].axhline(0, color="grey", ls=":")
    axes[0].set_title("Approx. ratio above random guessing (valid shots)")
    axes[1].axhline(1, color="grey", ls=":")
    axes[1].set_yscale("log")
    axes[1].set_title("P(best subset) / random guessing (median)")
    axes[2].set_title("P(shot has exactly k features)")
    for ax in axes:
        ax.set_xlabel("QAOA depth p")
        ax.set_xticks(sorted(df.p.unique()))
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--max-p", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--scoring", choices=["corr", "mi"], default="corr")
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    jobs = [(split, k, Q, const, args.max_p, 0, args.restarts, ERROR_RATES)
            for split, k, Q, const in problems(args.splits, args.ks, args.alpha, args.scoring)]
    tag = f"{args.scoring}_a{args.alpha}"
    os.makedirs(OUT_DIR, exist_ok=True)
    partial_path = os.path.join(OUT_DIR, f"qaoa_noise_{tag}.partial.csv")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, job_rows in enumerate(pool.map(run_job, jobs), 1):
            rows.extend(job_rows)
            pd.DataFrame(rows).to_csv(partial_path, index=False)  # keep finished jobs if the run is interrupted
            print(f"[{i}/{len(jobs)}] split {job_rows[0]['split']} k={job_rows[0]['k']} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, f"qaoa_noise_{tag}.csv"), index=False)
    os.remove(partial_path)
    plot(df, os.path.join(OUT_DIR, f"qaoa_noise_{tag}.png"))
    cols = ["p_valid", "approx_ratio", "random_approx_ratio", "gain"]
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(df.groupby(["error_2q", "p"])[cols].mean().to_string())
    print("Saved to", OUT_DIR)
