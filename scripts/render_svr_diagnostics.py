"""Generate presentation-ready diagnostics for the locked RBF-SVR altitude model.

The script intentionally keeps preprocessing inside a scikit-learn Pipeline so
that StandardScaler is fitted independently inside every cross-validation fold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parent.parent
HMIN = 50.0
HMAX = 250.0


def canonicalise_ehds(features: np.ndarray) -> np.ndarray:
    """Sort EHDs by distance while keeping every (distance, angle) pair intact."""
    values = np.asarray(features, dtype=float).copy()
    swap = values[:, 3] > values[:, 4]
    values[swap, 3], values[swap, 4] = (
        values[swap, 4].copy(),
        values[swap, 3].copy(),
    )
    values[swap, 5], values[swap, 6] = (
        values[swap, 6].copy(),
        values[swap, 5].copy(),
    )
    return values


def locked_svr() -> Pipeline:
    """Return the final seed-42 model configuration used in the project."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", SVR(kernel="rbf", C=300.0, gamma=0.03, epsilon=4.0)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ds3.npz")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "ml"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = np.load(args.data)
    X = canonicalise_ehds(dataset["X"])
    y = np.asarray(dataset["yH"], dtype=float)
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42
    )

    model = locked_svr()
    model.fit(X[train_idx], y[train_idx])
    prediction = np.clip(model.predict(X[test_idx]), HMIN, HMAX)
    residual = y[test_idx] - prediction

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter(prediction, residual, s=24, alpha=0.65, color="#167D8D")
    ax.axhline(0.0, color="#1D2B4F", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Predicted altitude (m)")
    ax.set_ylabel("Residual: label - prediction (m)")
    ax.set_title("RBF-SVR residual analysis on the held-out test set")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(args.output / "svr_residuals.png", dpi=220)
    plt.close(fig)

    train_sizes, train_scores, validation_scores = learning_curve(
        locked_svr(),
        X[train_idx],
        y[train_idx],
        train_sizes=np.linspace(0.15, 1.0, 6),
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        shuffle=True,
        random_state=202412,
    )
    train_mae = -train_scores.mean(axis=1)
    validation_mae = -validation_scores.mean(axis=1)
    validation_std = validation_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(train_sizes, train_mae, "o-", color="#167D8D", label="Training MAE")
    ax.plot(
        train_sizes,
        validation_mae,
        "s-",
        color="#F28E2B",
        label="Cross-validation MAE",
    )
    ax.fill_between(
        train_sizes,
        validation_mae - validation_std,
        validation_mae + validation_std,
        color="#F28E2B",
        alpha=0.16,
    )
    ax.set_xlabel("Training samples used in each fold")
    ax.set_ylabel("MAE (m)")
    ax.set_title("RBF-SVR learning curve (training partition only)")
    ax.grid(True, alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "svr_learning_curve.png", dpi=220)
    plt.close(fig)

    summary = {
        "dataset": str(args.data),
        "split": {"train": int(len(train_idx)), "test": int(len(test_idx)), "seed": 42},
        "model": {"kernel": "rbf", "C": 300.0, "gamma": 0.03, "epsilon": 4.0},
        "residuals": {
            "mean_m": float(np.mean(residual)),
            "std_m": float(np.std(residual)),
            "median_absolute_m": float(np.median(np.abs(residual))),
            "p95_absolute_m": float(np.percentile(np.abs(residual), 95)),
        },
        "learning_curve": [
            {
                "train_size": int(size),
                "train_mae_m": float(tr_mae),
                "cv_mae_mean_m": float(cv_mae),
                "cv_mae_std_m": float(cv_std),
            }
            for size, tr_mae, cv_mae, cv_std in zip(
                train_sizes, train_mae, validation_mae, validation_std
            )
        ],
    }
    (args.output / "svr_diagnostics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
