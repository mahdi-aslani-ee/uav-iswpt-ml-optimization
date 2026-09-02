"""Random, reproducible audit of fast.py against the exact CVXPY reference.

This does not regenerate labels.  It tests the approximate FISTA/Dykstra
solver at the stored label altitude on untouched, randomly selected dataset
rows and records objective and feasibility errors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fast import solve_fast
from kang import (
    N,
    PT,
    angle_grid,
    channel_stats,
    desired_pattern,
    max_desired_power,
    steer_matrix,
)
from reference_solver import FixedAltitudeSDP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=202408)
    parser.add_argument("--output", type=Path, default=Path("gap1_outputs/fast_solver_audit.json"))
    args = parser.parse_args()

    data = np.load("ds3.npz")
    features, labels = data["X"], data["yH"]
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(features), size=args.samples, replace=False)
    theta = angle_grid(1.0)
    Agrid = steer_matrix(theta)
    ref = FixedAltitudeSDP(Agrid, n_ehds=2)
    rows = []

    for index in indices:
        z = features[index]
        chi = desired_pattern(theta, np.radians(z[:3]))
        r = z[3:5]
        phi = np.radians(z[5:7])
        rho = float(z[7])
        pdes = np.array([0.25 * max_desired_power(x) for x in r])
        h = float(labels[index])
        exact = ref.solve(h, chi, r, phi, pdes, rho)
        G = [channel_stats(h, rr, pp)[0] for rr, pp in zip(r, phi)]
        fast120 = solve_fast(Agrid, chi, G, pdes, rho, n_iter=120)
        fast500 = solve_fast(Agrid, chi, G, pdes, rho, n_iter=500)

        def record(label: str, result: tuple) -> dict:
            value, X, gamma = result
            eig = np.linalg.eigvalsh(0.5 * (X + X.conj().T))
            return {
                "solver": label,
                "objective": float(value),
                "gamma": float(gamma),
                "relative_objective_error": float((value - exact.objective) / exact.objective),
                "min_eigenvalue": float(eig.min()),
                "max_diagonal_error": float(np.max(np.abs(np.diag(X).real - PT / N))),
            }

        rows.append(
            {
                "index": int(index),
                "altitude": h,
                "exact_objective": exact.objective,
                "exact_seconds": exact.solve_seconds,
                "fast120": record("FISTA/Dykstra 120", fast120),
                "fast500": record("FISTA/Dykstra 500", fast500),
            }
        )
        print(index, rows[-1]["fast120"]["relative_objective_error"], flush=True)

    summary = {}
    for key in ["fast120", "fast500"]:
        err = np.array([x[key]["relative_objective_error"] for x in rows])
        summary[key] = {
            "median_absolute_relative_error": float(np.median(np.abs(err))),
            "p95_absolute_relative_error": float(np.percentile(np.abs(err), 95)),
            "max_absolute_relative_error": float(np.max(np.abs(err))),
            "max_diagonal_error": float(max(x[key]["max_diagonal_error"] for x in rows)),
            "min_eigenvalue_over_cases": float(min(x[key]["min_eigenvalue"] for x in rows)),
        }
    output = {
        "sample_count": int(args.samples),
        "seed": int(args.seed),
        "indices": indices.tolist(),
        "meaning": "Fast-vs-exact comparison at each stored label altitude; not an audit of the outer altitude search.",
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
