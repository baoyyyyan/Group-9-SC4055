"""
QUBO helpers from notebooks/2_qubo_and_bruteforce.ipynb, as an importable module
so the experiment script can rebuild the QUBO on each training split.
Same formula as the notebook:
    E(x) = -alpha * sum r_i x_i + (1 - alpha) * sum_{i<j} R_ij x_i x_j + lam * (sum x_i - k)^2
"""
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from qaoa import bit_energies


def relevance_scores(X, y, method="corr", seed=42):
    if method == "corr":
        return np.array([abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])])
    if method == "mi":
        return mutual_info_classif(X, y, random_state=seed)
    raise ValueError(method)


def redundancy_matrix(X, method="corr", seed=42):
    n = X.shape[1]
    if method == "corr":
        R = np.abs(np.corrcoef(X, rowvar=False))
    elif method == "mi":
        R = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                R[i, j] = R[j, i] = mutual_info_regression(X[:, [i]], X[:, j], random_state=seed)[0]
    else:
        raise ValueError(method)
    np.fill_diagonal(R, 0.0)  # a feature is not 'redundant with itself'
    return R


def build_qubo(r, R, k, alpha=0.5, lam=None):
    # Returns (Q, const, lam) with energy(x) = x @ Q @ x + const. Q is upper triangular.
    n = len(r)
    if lam is None:
        lam = max(alpha * r.max(), (1 - alpha) * R.sum(axis=1).max()) + 0.1
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -alpha * r[i] + lam * (1 - 2 * k)
        for j in range(i + 1, n):
            Q[i, j] = (1 - alpha) * R[i, j] + 2 * lam
    return Q, lam * k ** 2, lam


def qubo_energy(x, Q, const):
    x = np.asarray(x)
    return float(x @ Q @ x + const)


def bits_to_indices(x):
    return [int(i) for i in np.flatnonzero(x)]


def brute_force(Q, const):
    """Lowest-energy subset over all 2^n bitstrings (vectorised version of the notebook's loop)."""
    n = len(Q)
    bitstring = int(np.argmin(bit_energies(Q, const)))
    return [i for i in range(n) if (bitstring >> i) & 1]


def simulated_annealing(Q, const, n_steps=2000, T_start=5.0, T_end=0.01, n_restarts=5, seed=None):
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    best_x, best_E = None, np.inf
    for _ in range(n_restarts):
        x = rng.integers(0, 2, n)
        E = qubo_energy(x, Q, const)
        for step in range(n_steps):
            T = T_start * (T_end / T_start) ** (step / (n_steps - 1))  # cools from T_start to T_end
            i = rng.integers(n)
            x_new = x.copy()
            x_new[i] = 1 - x_new[i]
            E_new = qubo_energy(x_new, Q, const)
            if E_new < E or rng.random() < np.exp(-(E_new - E) / T):
                x, E = x_new, E_new
            if E < best_E:
                best_x, best_E = x.copy(), E
    return bits_to_indices(best_x), best_E
