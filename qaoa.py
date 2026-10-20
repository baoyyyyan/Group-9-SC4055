import json
import os
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

P, RESTARTS, SHOTS = 2, 6, 2048
ROOT = os.path.dirname(os.path.abspath(__file__))
QUBO_PATH = os.path.join(ROOT, "results", "qubo_selections.json")


def load_qubo(path=QUBO_PATH):
    with open(path) as f:
        return json.load(f)


def ising_coeffs(Q):
    n = len(Q)
    h, J = np.zeros(n), np.zeros((n, n))
    for i in range(n):
        h[i] -= Q[i, i] / 2
        for j in range(i + 1, n):
            J[i, j] = Q[i, j] / 4
            h[i] -= Q[i, j] / 4
            h[j] -= Q[i, j] / 4
    return h, J


def bit_energies(Q, const):
    n = len(Q)
    bits = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(float)
    return np.einsum("bi,ij,bj->b", bits, Q, bits) + const


def qaoa_circuit(n, p, h, J, scale):
    """One layer: exp(-i gamma H_cost) then XY mixer exp(-i beta/2 (XX+YY)) on every pair."""
    gamma, beta = ParameterVector("g", p), ParameterVector("b", p)
    qc = QuantumCircuit(n)
    for layer in range(p):
        for i in range(n):
            qc.rz(2 * h[i] / scale * gamma[layer], i)
        for i, j in combinations(range(n), 2):
            qc.rzz(2 * J[i, j] / scale * gamma[layer], i, j)
        for i, j in combinations(range(n), 2):
            qc.rxx(beta[layer], i, j)
            qc.ryy(beta[layer], i, j)
    return qc


def bind(circuit, params, p):
    gammas, betas = params[:p], params[p:]
    values = {}
    for param in circuit.parameters:
        label = str(param)
        idx = int(label[label.find("[") + 1 : label.find("]")])
        values[param] = gammas[idx] if label.startswith("g") else betas[idx]
    return circuit.assign_parameters(values)


def qaoa(Q, const, k, p=P, restarts=RESTARTS, shots=SHOTS, seed=0):
    """Dicke state, Qiskit statevector, COBYLA on gamma and beta."""
    n = len(Q)
    energy = bit_energies(Q, const)
    masks = np.array([sum(1 << i for i in c) for c in combinations(range(n), k)])
    scale = float(np.ptp(energy[masks]))
    h, J = ising_coeffs(Q)
    circuit = qaoa_circuit(n, p, h, J, scale)

    amps = np.zeros(1 << n, dtype=complex)
    amps[masks] = 1 / np.sqrt(len(masks))
    psi0 = Statevector(amps)
    scaled = (energy - energy[masks].mean()) / scale

    def loss(params):
        psi = psi0.evolve(bind(circuit, params, p))
        return float(np.dot(np.real(psi.probabilities()), scaled))

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        guess = np.r_[rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)]
        found = minimize(loss, guess, method="COBYLA", options={"maxiter": 180})
        if best is None or found.fun < best.fun:
            best = found

    probs = np.real(psi0.evolve(bind(circuit, best.x, p)).probabilities())
    probs = np.clip(probs, 0, None)
    probs /= probs.sum()
    draws = rng.choice(1 << n, size=shots, p=probs)
    pick = int(draws[np.argmin(energy[draws])])
    opt = int(masks[np.argmin(energy[masks])])
    return pick, float(np.dot(probs, energy)), float(probs[opt]), opt


def indices(bitstring, n):
    return [i for i in range(n) if (bitstring >> i) & 1]


if __name__ == "__main__":
    qubo = load_qubo()
    names = qubo["feature_names"]
    summary = {
        "source": os.path.basename(QUBO_PATH),
        "alpha": qubo["alpha"],
        "p": P,
        "restarts": RESTARTS,
        "shots": SHOTS,
        "simulator": "qiskit.Statevector",
        "by_k": {},
    }
    for key in sorted(qubo["by_k"], key=int):
        k = int(key)
        block = qubo["by_k"][key]
        Q, const = np.array(block["Q"], dtype=float), float(block["const"])
        n = len(Q)
        pick, expect, p_opt, _opt = qaoa(Q, const, k, seed=0)
        energy = bit_energies(Q, const)
        masks = np.array([sum(1 << i for i in c) for c in combinations(range(n), k)])
        sub = energy[masks]
        ratio = (sub.max() - expect) / (sub.max() - sub.min())
        chosen_idx = indices(pick, n)
        chosen = [names[i] for i in chosen_idx]
        brute = block["bruteforce"]
        summary["by_k"][key] = {
            "qaoa": chosen_idx,
            "feature_names": chosen,
            "energy": float(energy[pick]),
            "expectation": expect,
            "prob_opt": p_opt,
            "approx_ratio": float(ratio),
            "bruteforce": brute,
            "match": chosen_idx == brute,
        }
        same = "same" if chosen_idx == brute else "miss"
        print(f"k={k}: QAOA {chosen_idx}  E={energy[pick]:.4f}  brute {brute}  ({same})")
        print(f"     {chosen}   P(opt)={p_opt:.3f}  <E>={expect:.4f}  ratio={ratio:.3f}")

    if "3" in qubo["by_k"]:
        block = qubo["by_k"]["3"]
        Q, const = np.array(block["Q"], dtype=float), float(block["const"])
        brute = block["bruteforce"]
        n_runs = 20
        hits = 0
        for seed in range(n_runs):
            pick, _, _, _ = qaoa(Q, const, 3, restarts=4, seed=seed)
            hits += indices(pick, len(Q)) == brute
        summary["k3_runs"] = {"hits": hits, "runs": n_runs, "restarts": 4}
        print(f"k=3: exact optimum in {hits}/{n_runs} runs")

    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "qaoa_selections.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved to", out_path)
