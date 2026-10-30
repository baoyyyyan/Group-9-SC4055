# Group-9-SC4055

**QAOA-based feature selection for machine learning.** Given a dataset and a budget of *k* features,
pick a subset that is informative about the target without being redundant. We write this as a
QUBO, solve it with QAOA (Qiskit statevector simulator), and compare it with classical feature
selection and classical QUBO solvers.

Dataset: Breast Cancer Wisconsin (scikit-learn), the 10 "mean" features, so one qubit per feature.

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

## Files

| File | What it does |
|---|---|
| `baselines.py` | Classical baselines: all features, random, top-k relevance, greedy relevance-minus-redundancy, and the shared `evaluate()` (random forest test accuracy) |
| `notebooks/2_qubo_and_bruteforce.ipynb` | Explains the QUBO step by step: relevance/redundancy scores, building Q, brute force, simulated annealing, Ising form |
| `qubo.py` | The notebook's QUBO functions as an importable module (relevance/redundancy by correlation or mutual information, `build_qubo`, `brute_force`, `simulated_annealing`) |
| `qaoa.py` | QAOA in Qiskit: Dicke start state (only size-k subsets), XY mixer, COBYLA; comparison with random guessing |
| `run_experiments.py` | Main experiment: every method on the same train/test splits, scored by test accuracy |
| `qaoa_depth.py` | QAOA vs random guessing across circuit depth p (p = 0 is random guessing) |
| `qaoa_noise.py` | Optimised QAOA circuits re-run with depolarizing gate noise (Qiskit Aer density-matrix simulator) |

## Reproducing the results

Run `notebooks/2_qubo_and_bruteforce.ipynb` from the `notebooks/` folder first (writes `results/qubo_selections.json`), then:

```
python qaoa.py                                              -> results/qaoa_selections.json
python run_experiments.py                                   -> results/experiments/accuracy_*.csv, summary_*.csv, accuracy_vs_k_*.png
python run_experiments.py --scoring mi --alpha cv           # mutual-information scores, alpha picked by CV on training rows
python run_experiments.py --no-qaoa --scoring corr --alpha 0.7   # alpha sweep: 0.5..0.9 and cv, for corr and mi
python qaoa_depth.py                                        -> results/experiments/qaoa_depth_*.csv/png (independent random restarts per depth)
python qaoa_depth.py --init interp                          # layer-by-layer warm start (INTERP)
python run_experiments.py --noise-features 4                # add 4 pure-noise columns (14 qubits; QAOA practical up to ~4)
python run_experiments.py --no-qaoa --noise-features 10     # classical methods only for more noise columns
python plot_data_noise.py --scoring mi --alpha cv           # combine the noise-column runs -> results/experiments/data_noise_*.png
python qaoa_noise.py                                        -> results/experiments/qaoa_noise_*.csv/png (side experiment: gate noise)
```

Data-noise runs: 2, 4, 6, 8 and 10 noise columns with `--scoring corr --alpha 0.5` (QAOA included up to 4),
and with `--scoring mi --alpha cv --no-qaoa`.
The exact settings of every run are saved next to its results as `config_*.json`.

Add `--help` to any script for its options (number of splits, budgets k, alpha, scoring, depths, restarts, workers).

### Experimental protocol

- 5 stratified 70/30 train/test splits (`random_state` = 0..4).
- Relevance and redundancy are computed on the **training rows only**, then the QUBO is rebuilt for each split.
- Each selected subset is scored with a random forest (`baselines.evaluate`) and a standardised logistic regression on the test rows.
- QUBO solution quality is reported as the energy gap to the brute-force optimum, and for QAOA as the
  approximation ratio and the probability of sampling the optimal subset, compared with uniform random guessing.

## Team

Raye (data and baselines), baoyan (QUBO), Nikki (QAOA), David (experiments and results).
