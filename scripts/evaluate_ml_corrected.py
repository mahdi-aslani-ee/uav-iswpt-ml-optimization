"""Leakage-aware, symmetry-aware ML evaluation with corrected gap oracles.

Corrections relative to ``train3.py``:

1. EHD pairs are canonicalised by distance, enforcing the physical invariance
   to swapping device labels.
2. Hyperparameters are selected using CV on the training partition only.
3. The downstream denominator is a fresh candidate-search oracle; it is not
   the approximate stored ``yO`` label, so reported gaps cannot be negative.
4. A fixed held-out subset is recomputed with the exact CVXPY P2 solver.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from fast import solve_fast
from kang import (
    HMAX,
    HMIN,
    angle_grid,
    channel_stats,
    desired_pattern,
    max_desired_power,
    steer_matrix,
)
from reference_solver import FixedAltitudeSDP


FEATURES = ["theta1", "theta2", "theta3", "r_near", "r_far", "phi_near", "phi_far", "rho"]


def canonicalise_ehds(X: np.ndarray) -> np.ndarray:
    """Sort EHDs by r and carry each associated phi with its device."""
    Z = np.asarray(X, dtype=float).copy()
    swap = Z[:, 3] > Z[:, 4]
    left = Z[swap, 3].copy()
    Z[swap, 3] = Z[swap, 4]
    Z[swap, 4] = left
    left = Z[swap, 5].copy()
    Z[swap, 5] = Z[swap, 6]
    Z[swap, 6] = left
    return Z


def build_models() -> dict:
    return {
        "Linear": (
            Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
            {},
        ),
        "Decision Tree": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", DecisionTreeRegressor(random_state=0)),
                ]
            ),
            {
                "model__max_depth": [5, 8, 12, None],
                "model__min_samples_leaf": [1, 3, 8],
            },
        ),
        "SVR (RBF)": (
            Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf"))]),
            {
                "model__C": [50, 300, 1500],
                "model__gamma": [0.03, 0.1],
                "model__epsilon": [1.0, 4.0],
            },
        ),
        "MLP": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(random_state=0, max_iter=4000, early_stopping=True),
                    ),
                ]
            ),
            {
                "model__hidden_layer_sizes": [(128, 64), (128, 128, 64)],
                "model__alpha": [1e-4, 1e-2],
            },
        ),
    }


def scenario(z: np.ndarray, theta: np.ndarray, Agrid: np.ndarray):
    chi = desired_pattern(theta, np.radians(z[:3]))
    r = z[3:5]
    phi = np.radians(z[5:7])
    rho = float(z[7])
    pdes = np.array([0.25 * max_desired_power(x) for x in r])
    return chi, r, phi, rho, pdes


def main() -> None:
    output = Path("ml_corrected_outputs")
    output.mkdir(exist_ok=True)
    data = np.load("ds3.npz")
    X_raw, y_h = data["X"], data["yH"]
    X = canonicalise_ehds(X_raw)
    train, test = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
    estimators, predictions, metrics, best_params = {}, {}, {}, {}

    for name, (pipeline, grid) in build_models().items():
        if grid:
            search = GridSearchCV(
                pipeline,
                grid,
                cv=5,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            search.fit(X[train], y_h[train])
            estimator = search.best_estimator_
            best_params[name] = {
                key.replace("model__", ""): str(value)
                for key, value in search.best_params_.items()
            }
        else:
            estimator = pipeline.fit(X[train], y_h[train])
            best_params[name] = {}
        pred = np.clip(estimator.predict(X[test]), HMIN, HMAX)
        estimators[name], predictions[name] = estimator, pred
        metrics[name] = {
            "MAE_m": float(mean_absolute_error(y_h[test], pred)),
            "RMSE_m": float(np.sqrt(mean_squared_error(y_h[test], pred))),
            "R2": float(r2_score(y_h[test], pred)),
        }
        print(name, metrics[name], best_params[name], flush=True)

    theta = angle_grid(1.0)
    Agrid = steer_matrix(theta)
    fast_gaps = {name: [] for name in estimators}
    fast_oracles = []
    start = time.perf_counter()
    for j, index in enumerate(test):
        # Physics uses the original EHD ordering; only ML inputs are canonical.
        chi, r, phi, rho, pdes = scenario(X_raw[index], theta, Agrid)
        candidates = list(np.linspace(HMIN, HMAX, 11))
        candidates.append(float(y_h[index]))
        candidates.extend(float(predictions[name][j]) for name in estimators)
        candidates = sorted(set(float(x) for x in candidates))
        value_by_h = {}
        X0 = None
        for h in candidates:
            G = [channel_stats(h, rr, pp)[0] for rr, pp in zip(r, phi)]
            value, X0, _ = solve_fast(Agrid, chi, G, pdes, rho, X0=X0)
            value_by_h[h] = float(value)
        center = min(value_by_h, key=value_by_h.get)
        refine = np.arange(max(HMIN, center - 20), min(HMAX, center + 20) + 1e-9, 5.0)
        for h in refine:
            h = float(h)
            if h not in value_by_h:
                G = [channel_stats(h, rr, pp)[0] for rr, pp in zip(r, phi)]
                value, X0, _ = solve_fast(Agrid, chi, G, pdes, rho, X0=X0)
                value_by_h[h] = float(value)
        oracle_h = min(value_by_h, key=value_by_h.get)
        oracle_value = value_by_h[oracle_h]
        fast_oracles.append({"index": int(index), "h": oracle_h, "objective": oracle_value})
        for name in estimators:
            h = float(predictions[name][j])
            fast_gaps[name].append(100.0 * (value_by_h[h] - oracle_value) / oracle_value)
        if j % 25 == 24:
            print(f"fast oracle {j+1}/{len(test)}", flush=True)
    fast_seconds = time.perf_counter() - start

    fast_gap_summary = {
        name: {
            "mean_percent": float(np.mean(values)),
            "median_percent": float(np.median(values)),
            "p95_percent": float(np.percentile(values, 95)),
            "max_percent": float(np.max(values)),
            "negative_count": int(np.sum(np.asarray(values) < -1e-10)),
        }
        for name, values in fast_gaps.items()
    }

    # Exact held-out audit.  The exact candidate set includes a local grid
    # around the fast oracle, the stored label, and every model prediction.
    audit_rng = np.random.default_rng(202410)
    audit_positions = np.sort(audit_rng.choice(len(test), size=20, replace=False))
    exact_gaps = {name: [] for name in estimators}
    exact_rows = []
    ref = FixedAltitudeSDP(Agrid, 2)
    for count, j in enumerate(audit_positions, 1):
        index = int(test[j])
        chi, r, phi, rho, pdes = scenario(X_raw[index], theta, Agrid)
        center = float(fast_oracles[j]["h"])
        hs = list(np.arange(max(HMIN, center - 6), min(HMAX, center + 6) + 1e-9, 2.0))
        hs += [float(y_h[index])]
        hs += [float(predictions[name][j]) for name in estimators]
        hs = sorted(set(float(h) for h in hs))
        exact_values = {
            h: ref.solve(h, chi, r, phi, pdes, rho).objective for h in hs
        }
        oracle_h = min(exact_values, key=exact_values.get)
        oracle_value = exact_values[oracle_h]
        row = {"index": index, "oracle_h": oracle_h, "oracle_objective": oracle_value}
        for name in estimators:
            h = float(predictions[name][j])
            gap = 100.0 * (exact_values[h] - oracle_value) / oracle_value
            exact_gaps[name].append(gap)
            row[name] = {"h": h, "objective": exact_values[h], "gap_percent": gap}
        exact_rows.append(row)
        print(f"exact audit {count}/{len(audit_positions)}", flush=True)

    exact_gap_summary = {
        name: {
            "mean_percent": float(np.mean(values)),
            "median_percent": float(np.median(values)),
            "p95_percent": float(np.percentile(values, 95)),
            "max_percent": float(np.max(values)),
            "negative_count": int(np.sum(np.asarray(values) < -1e-10)),
        }
        for name, values in exact_gaps.items()
    }

    svr = estimators["SVR (RBF)"]
    importance = permutation_importance(
        svr,
        X[test],
        y_h[test],
        scoring="neg_mean_absolute_error",
        n_repeats=30,
        random_state=202411,
    )
    feature_importance = sorted(
        [
            {
                "feature": feature,
                "MAE_increase_mean_m": float(mean),
                "MAE_increase_std_m": float(std),
            }
            for feature, mean, std in zip(
                FEATURES, importance.importances_mean, importance.importances_std
            )
        ],
        key=lambda x: x["MAE_increase_mean_m"],
        reverse=True,
    )

    result = {
        "split": {"train": int(len(train)), "test": int(len(test)), "seed": 42},
        "preprocessing": "EHDs canonicalised by ascending r with phi kept paired",
        "metrics": metrics,
        "best_params": best_params,
        "fast_candidate_oracle": {
            "definition": "11-point global grid + stored label + all predictions + 5 m local refinement; FISTA/Dykstra objective",
            "seconds": fast_seconds,
            "summary": fast_gap_summary,
        },
        "exact_candidate_audit": {
            "sample_count": int(len(audit_positions)),
            "selection_seed": 202410,
            "definition": "exact CVXPY P2 at +/-6 m local grid, stored label, and all predictions",
            "summary": exact_gap_summary,
            "rows": exact_rows,
        },
        "svr_permutation_importance": feature_importance,
    }
    (output / "ml_corrected_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    # Requested ML figures.
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.2), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, estimators):
        ax.scatter(y_h[test], predictions[name], s=14, alpha=0.65)
        ax.plot([HMIN, HMAX], [HMIN, HMAX], "k--", lw=1)
        ax.set_title(f"{name}\nMAE={metrics[name]['MAE_m']:.2f} m")
        ax.grid(True, alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xlabel("Stored optimizer-label altitude (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted altitude (m)")
    fig.tight_layout()
    fig.savefig(output / "parity_plots.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    data_to_plot = [np.clip(fast_gaps[name], 0, np.percentile(fast_gaps[name], 99)) for name in estimators]
    ax.boxplot(data_to_plot, tick_labels=list(estimators), showfliers=False)
    ax.set_ylabel("Gap to fresh fast candidate oracle (%)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "gap_distributions.png", dpi=220)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.6))
    names = list(estimators)
    x = np.arange(len(names))
    ax1.bar(x, [metrics[n]["MAE_m"] for n in names])
    ax1.set_xticks(x, names)
    ax1.set_ylabel("Altitude MAE (m)")
    ax1.set_title("Regression error")
    ax1.grid(True, axis="y", alpha=0.25)
    ax2.bar(x, [fast_gap_summary[n]["mean_percent"] for n in names], color="#ff7f0e")
    ax2.set_xticks(x, names)
    ax2.set_ylabel("Mean P1 gap (%)")
    ax2.set_title("Engineering objective penalty")
    ax2.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "mae_vs_gap.png", dpi=220)
    plt.close(fig)
    print(json.dumps({"metrics": metrics, "fast_gap": fast_gap_summary, "exact_gap": exact_gap_summary}, indent=2))


if __name__ == "__main__":
    main()
