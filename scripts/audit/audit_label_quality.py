"""Audit stored synthetic labels against an exact CVXPY altitude search.

The original labels use the documented fast FISTA/Dykstra inner solver and a
10+5 point deterministic altitude search.  This script measures, on a fixed
random subset, how far those labels are from a much denser *exact* P2|h search.
It therefore addresses label quality without changing or cherry-picking the
training dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fast import solve_fast
from kang import (
    angle_grid,
    channel_stats,
    desired_pattern,
    max_desired_power,
    steer_matrix,
)
from reference_solver import FixedAltitudeSDP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=202409)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "gap1" / "label_quality_audit.json")
    args = parser.parse_args()
    data = np.load(ROOT / "data" / "ds3.npz")
    features, label_h, label_obj = data["X"], data["yH"], data["yO"]
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(features), size=args.samples, replace=False)
    theta = angle_grid(1.0)
    Agrid = steer_matrix(theta)
    ref = FixedAltitudeSDP(Agrid, 2)
    rows = []

    for index in indices:
        z = features[index]
        chi = desired_pattern(theta, np.radians(z[:3]))
        r = z[3:5]
        phi = np.radians(z[5:7])
        rho = float(z[7])
        pdes = np.array([0.25 * max_desired_power(x) for x in r])

        # A global 5 m fast scan identifies the basin without making the audit
        # prohibitively expensive.  Only exact CVXPY values are eligible for
        # the reported audit optimum.
        coarse_h = np.arange(50.0, 250.0 + 1e-9, 5.0)
        coarse_obj = []
        X0 = None
        for h in coarse_h:
            G = [channel_stats(h, rr, pp)[0] for rr, pp in zip(r, phi)]
            value, X0, _ = solve_fast(Agrid, chi, G, pdes, rho, X0=X0)
            coarse_obj.append(value)
        coarse_obj = np.asarray(coarse_obj)
        center = coarse_h[int(np.argmin(coarse_obj))]
        fine_h = np.arange(max(50.0, center - 8.0), min(250.0, center + 8.0) + 1e-9, 1.0)
        fine_obj = np.array(
            [ref.solve(h, chi, r, phi, pdes, rho).objective for h in fine_h]
        )
        j = int(np.argmin(fine_obj))
        exact_h = float(fine_h[j])
        exact_obj = float(fine_obj[j])
        exact_at_label = ref.solve(float(label_h[index]), chi, r, phi, pdes, rho).objective
        rows.append(
            {
                "index": int(index),
                "stored_altitude": float(label_h[index]),
                "audit_altitude": exact_h,
                "absolute_altitude_error_m": float(abs(label_h[index] - exact_h)),
                "stored_fast_objective": float(label_obj[index]),
                "exact_objective_at_stored_altitude": float(exact_at_label),
                "audit_objective": exact_obj,
                "stored_altitude_gap_percent": float(100.0 * (exact_at_label - exact_obj) / exact_obj),
                "stored_fast_vs_exact_at_same_h_percent": float(100.0 * (label_obj[index] - exact_at_label) / exact_at_label),
            }
        )
        print(index, rows[-1]["absolute_altitude_error_m"], rows[-1]["stored_altitude_gap_percent"], flush=True)

    alt = np.array([x["absolute_altitude_error_m"] for x in rows])
    gap = np.array([x["stored_altitude_gap_percent"] for x in rows])
    inner = np.array([x["stored_fast_vs_exact_at_same_h_percent"] for x in rows])
    summary = {
        "altitude_mae_m": float(np.mean(alt)),
        "altitude_median_absolute_error_m": float(np.median(alt)),
        "altitude_max_absolute_error_m": float(np.max(alt)),
        "objective_gap_mean_percent": float(np.mean(gap)),
        "objective_gap_median_percent": float(np.median(gap)),
        "objective_gap_p95_percent": float(np.percentile(gap, 95)),
        "objective_gap_max_percent": float(np.max(gap)),
        "stored_fast_vs_exact_same_h_mean_percent": float(np.mean(inner)),
        "stored_fast_vs_exact_same_h_max_abs_percent": float(np.max(np.abs(inner))),
    }
    output = {
        "sample_count": int(args.samples),
        "seed": int(args.seed),
        "indices": indices.tolist(),
        "audit_grid": "5 m global fast basin scan + 1 m exact CVXPY refinement over +/-8 m",
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
