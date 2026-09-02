"""Re-render corrected ML figures from saved metrics and locked models."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from evaluate_ml_corrected import canonicalise_ehds


def locked_models() -> dict:
    return {
        "Linear": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "Decision Tree": Pipeline([("scale", StandardScaler()), ("model", DecisionTreeRegressor(max_depth=8, min_samples_leaf=8, random_state=0))]),
        "SVR (RBF)": Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf", C=300, gamma=0.03, epsilon=4.0))]),
        "MLP": Pipeline([("scale", StandardScaler()), ("model", MLPRegressor(hidden_layer_sizes=(128, 64), alpha=0.01, random_state=0, max_iter=4000, early_stopping=True))]),
    }


def main() -> None:
    output = Path("ml_corrected_outputs")
    result = json.loads((output / "ml_corrected_results.json").read_text())
    data = np.load("ds3.npz")
    X, y = canonicalise_ehds(data["X"]), data["yH"]
    train, test = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
    predictions = {}
    for name, model in locked_models().items():
        model.fit(X[train], y[train])
        predictions[name] = np.clip(model.predict(X[test]), 50.0, 250.0)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.2), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, predictions):
        ax.scatter(y[test], predictions[name], s=14, alpha=0.65)
        ax.plot([50, 250], [50, 250], "k--", lw=1)
        ax.set_title(f"{name}\nMAE={result['metrics'][name]['MAE_m']:.2f} m")
        ax.grid(True, alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xlabel("Stored optimizer-label altitude (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted altitude (m)")
    fig.tight_layout()
    fig.savefig(output / "parity_plots.png", dpi=220)
    plt.close(fig)

    names = list(predictions)
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.6))
    ax1.bar(x, [result["metrics"][n]["MAE_m"] for n in names])
    ax1.set_xticks(x, names, rotation=12)
    ax1.set_ylabel("Altitude MAE (m)")
    ax1.set_title("Regression error")
    ax1.grid(True, axis="y", alpha=0.25)
    gaps = result["fast_candidate_oracle"]["summary"]
    ax2.bar(x, [gaps[n]["mean_percent"] for n in names], color="#ff7f0e")
    ax2.set_xticks(x, names, rotation=12)
    ax2.set_ylabel("Mean P1 gap (%)")
    ax2.set_title("Engineering objective penalty")
    ax2.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "mae_vs_gap.png", dpi=220)
    plt.close(fig)

    importance = result["svr_permutation_importance"]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    labels = [x["feature"] for x in importance][::-1]
    values = [x["MAE_increase_mean_m"] for x in importance][::-1]
    errors = [x["MAE_increase_std_m"] for x in importance][::-1]
    ax.barh(labels, values, xerr=errors, color="#2ca02c", alpha=0.85)
    ax.set_xlabel("Increase in test MAE after permutation (m)")
    ax.set_title("SVR permutation importance (30 repeats)")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "svr_permutation_importance.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
