"""Repeated-split stability check using locked corrected hyperparameters."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from evaluate_ml_corrected import canonicalise_ehds

ROOT = Path(__file__).resolve().parent.parent


def models() -> dict:
    return {
        "Linear": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "Decision Tree": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", DecisionTreeRegressor(max_depth=8, min_samples_leaf=8, random_state=0)),
            ]
        ),
        "SVR (RBF)": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVR(kernel="rbf", C=300, gamma=0.03, epsilon=4.0)),
            ]
        ),
        "MLP": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        alpha=0.01,
                        random_state=0,
                        max_iter=4000,
                        early_stopping=True,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    data = np.load(ROOT / "data" / "ds3.npz")
    X, y = canonicalise_ehds(data["X"]), data["yH"]
    seeds = [7, 19, 42, 73, 101]
    rows = []
    for seed in seeds:
        train, test = train_test_split(np.arange(len(X)), test_size=0.2, random_state=seed)
        for name, estimator in models().items():
            estimator.fit(X[train], y[train])
            pred = np.clip(estimator.predict(X[test]), 50.0, 250.0)
            row = {
                "seed": seed,
                "model": name,
                "MAE_m": float(mean_absolute_error(y[test], pred)),
                "RMSE_m": float(np.sqrt(mean_squared_error(y[test], pred))),
                "R2": float(r2_score(y[test], pred)),
            }
            rows.append(row)
            print(row, flush=True)
    summary = {}
    for name in models():
        selected = [x for x in rows if x["model"] == name]
        summary[name] = {
            metric: {
                "mean": float(np.mean([x[metric] for x in selected])),
                "std": float(np.std([x[metric] for x in selected], ddof=1)),
                "min": float(np.min([x[metric] for x in selected])),
                "max": float(np.max([x[metric] for x in selected])),
            }
            for metric in ["MAE_m", "RMSE_m", "R2"]
        }
    output = {"seeds": seeds, "hyperparameters": "locked from seed-42 training-only CV", "summary": summary, "rows": rows}
    out = ROOT / "results" / "ml" / "repeated_split_metrics.json"
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
