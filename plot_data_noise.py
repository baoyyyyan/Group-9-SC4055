"""
Combine the run_experiments.py --noise-features runs into one table and plot:
test accuracy and number of junk features picked, as junk columns are added.

Usage (after the runs):
    python plot_data_noise.py                       # correlation scoring, alpha = 0.5
    python plot_data_noise.py --scoring mi --alpha cv
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from run_experiments import OUT_DIR

NOISE_LEVELS = [0, 2, 4, 6, 8, 10]


def load(scoring, alpha):
    frames = []
    for m in NOISE_LEVELS:
        path = os.path.join(OUT_DIR, f"accuracy_{scoring}_a{alpha}" + (f"_noise{m}" if m else "") + ".csv")
        if os.path.exists(path):
            df = pd.read_csv(path).assign(noise_features=m)
            if "n_noise_selected" not in df:  # runs from before the column existed had no noise features
                df["n_noise_selected"] = 0
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot(df, path, title):
    sel = df[df.method != "all"]
    g = sel.groupby(["method", "noise_features"])
    acc, junk = g.acc_logreg.mean().unstack(0), g.n_noise_selected.mean().unstack(0)
    all_acc = df[df.method == "all"].groupby("noise_features").acc_logreg.mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in acc.columns:
        style = dict(marker="o", ls="--" if method in ("sim_annealing", "qaoa") else "-")
        axes[0].plot(acc.index, acc[method], label=method, **style)
        axes[1].plot(junk.index, junk[method], label=method, **style)
    axes[0].plot(all_acc.index, all_acc, color="grey", ls=":", marker="s", label="all features")
    axes[0].set_title("Test accuracy, logistic regression (mean over k = 2..5)")
    axes[1].set_title("Junk features picked (mean over k = 2..5)")
    for ax in axes:
        ax.set_xlabel("number of pure-noise features added")
        ax.set_xticks(NOISE_LEVELS)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return acc, junk, all_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring", choices=["corr", "mi"], default="corr")
    parser.add_argument("--alpha", default="0.5")
    args = parser.parse_args()

    df = load(args.scoring, args.alpha)
    tag = f"{args.scoring}_a{args.alpha}"
    acc, junk, all_acc = plot(df, os.path.join(OUT_DIR, f"data_noise_{tag}.png"),
                              f"Scoring: {args.scoring}, alpha = {args.alpha}")
    with pd.option_context("display.width", 200, "display.precision", 3):
        print("Accuracy (logistic regression):")
        print(acc.assign(all=all_acc).to_string())
        print("\nJunk features picked:")
        print(junk.to_string())
