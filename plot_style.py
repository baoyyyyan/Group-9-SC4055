"""Shared names, colours and line styles so every experiment plot shows each method the same way.
The three QUBO solvers usually return the same subset, so the exact optimum is a thick solid line and
QAOA / simulated annealing are drawn dashed / dotted on top of it (visible where they differ)."""

LABELS = {
    "random": "random",
    "top_k": "top-k",
    "greedy": "greedy",
    "brute_force": "QUBO optimum (brute force)",
    "exact_qubo": "QUBO optimum (exhaustive)",
    "qaoa": "QAOA (solves the QUBO)",
    "sim_annealing": "simulated annealing (QUBO)",
}
STYLES = {
    "random": dict(color="tab:blue", lw=2, ls="-"),
    "top_k": dict(color="tab:orange", lw=2, ls="-"),
    "greedy": dict(color="tab:green", lw=2, ls="-"),
    "brute_force": dict(color="tab:red", lw=4, ls="-", alpha=0.85),
    "exact_qubo": dict(color="tab:red", lw=4, ls="-", alpha=0.85),
    "qaoa": dict(color="tab:brown", lw=2, ls="--"),
    "sim_annealing": dict(color="tab:purple", lw=1.5, ls=":"),
}
ORDER = ["random", "top_k", "greedy", "brute_force", "exact_qubo", "sim_annealing", "qaoa"]


def ordered(methods):
    return [m for m in ORDER if m in set(methods)]
