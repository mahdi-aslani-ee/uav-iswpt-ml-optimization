"""Close gap 1: benchmark Kang Algorithm 2 against all required baselines.

The script is intentionally explicit about fidelity:

* Hmin/Hmax: exact CVXPY P2|h reference solves.
* Algorithm 1 audit optimum: fast 2 m global scan followed by exact CVXPY
  refinement at 1 m spacing.  This deterministic search is a project audit,
  not Kang's stochastic PSO.
* Algorithm 2: Kang Eqs. (37), (55)--(56), P7, and M=8/J=8 PSO.
* ML surrogate: SVR predicts one altitude, followed by one exact P2|h solve.

Pass ``--exact-pso`` to additionally run Kang Algorithm 1 literally with
M=8/J=8 and an exact CVXPY inner solve at every particle evaluation.  It is
slow by design and is kept separate from the default reproducibility run.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from algorithm2 import deterministic_algorithm2, pso_minimize, solve_algorithm2
from evaluate_ml_corrected import canonicalise_ehds
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


def fit_locked_svr(dataset_path: Path) -> tuple[Pipeline, dict]:
    """Refit the final corrected SVR on the original 80% train split."""
    data = np.load(dataset_path)
    X, y = canonicalise_ehds(data["X"]), data["yH"]
    train, test = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42
    )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=300.0, gamma=0.03, epsilon=4.0)),
        ]
    )
    model.fit(X[train], y[train])
    return model, {
        "train_count": int(len(train)),
        "held_out_count": int(len(test)),
        "split_seed": 42,
        "C": 300.0,
        "gamma": 0.03,
        "epsilon": 4.0,
        "preprocessing": "EHDs canonicalised by ascending r with phi kept paired",
    }


def exact_refined_optimum(
    ref: FixedAltitudeSDP,
    Agrid: np.ndarray,
    chi: np.ndarray,
    r: np.ndarray,
    phi: np.ndarray,
    pdes: np.ndarray,
    rho: float,
) -> tuple[dict, list[dict]]:
    """Global fast scan plus local exact reference refinement."""
    scan = np.arange(HMIN, HMAX + 1e-9, 2.0)
    fast_values = []
    X0 = None
    for h in scan:
        G = [channel_stats(h, rr, pp)[0] for rr, pp in zip(r, phi)]
        value, X0, _ = solve_fast(Agrid, chi, G, pdes, rho, X0=X0)
        fast_values.append(value)
    center = float(scan[int(np.argmin(fast_values))])
    exact_h = np.arange(max(HMIN, center - 8), min(HMAX, center + 8) + 1e-9, 1.0)
    records = []
    for h in exact_h:
        sol = ref.solve(h, chi, r, phi, pdes, rho)
        records.append(
            {
                "altitude": float(h),
                "objective": sol.objective,
                "solve_seconds": sol.solve_seconds,
            }
        )
    best = min(records, key=lambda x: x["objective"])
    best["fast_scan_center"] = center
    best["fast_scan_spacing_m"] = 2.0
    best["exact_refine_spacing_m"] = 1.0
    return best, records


def exact_algorithm1_pso(
    ref: FixedAltitudeSDP,
    chi: np.ndarray,
    r: np.ndarray,
    phi: np.ndarray,
    pdes: np.ndarray,
    rho: float,
    seed: int,
) -> dict:
    cache = {}

    def fitness(h: float) -> float:
        if h not in cache:
            cache[h] = ref.solve(h, chi, r, phi, pdes, rho)
        return cache[h].objective

    start = time.perf_counter()
    h, _, evaluations = pso_minimize(fitness, seed=seed)
    final = ref.solve(h, chi, r, phi, pdes, rho)
    return {
        "objective": final.objective,
        "altitude": float(h),
        "evaluations": int(evaluations + 1),
        "seed": int(seed),
        "wall_seconds": float(time.perf_counter() - start),
        "meaning": "Literal Kang Algorithm 1: Table 1 PSO and exact P2|h per evaluation",
    }


def row(method: str, rho: float, objective: float, altitude: float, **extra) -> dict:
    return {
        "method": method,
        "rho": float(rho),
        "objective": float(objective),
        "altitude_m": float(altitude),
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--exact-pso", action="store_true")
    parser.add_argument("--exact-pso-rho", type=float, default=0.6)
    parser.add_argument("--exact-pso-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("gap1_outputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    theta = angle_grid(args.resolution)
    Agrid = steer_matrix(theta)
    target_deg = np.array([-40.0, 0.0, 40.0])
    chi = desired_pattern(theta, np.radians(target_deg))
    r = np.array([50.0, 150.0])
    phi_deg = np.array([-20.0, 20.0])
    phi = np.radians(phi_deg)
    pdes = np.array([0.25 * max_desired_power(x) for x in r])
    rhos = [0.1, 0.3, 0.5, 0.7, 0.9]

    ref = FixedAltitudeSDP(Agrid, len(r))
    svr, svr_meta = fit_locked_svr(Path("ds3.npz"))
    all_rows = []
    exact_curves = {}
    pso_repeatability = {}
    start_all = time.perf_counter()

    for rho in rhos:
        print(f"rho={rho:.1f}", flush=True)
        lo = ref.solve(HMIN, chi, r, phi, pdes, rho)
        hi = ref.solve(HMAX, chi, r, phi, pdes, rho)
        all_rows += [
            row("Fixed Hmin (P2 exact)", rho, lo.objective, HMIN, sdp_solves=1),
            row("Fixed Hmax (P2 exact)", rho, hi.objective, HMAX, sdp_solves=1),
        ]

        audit, curve = exact_refined_optimum(ref, Agrid, chi, r, phi, pdes, rho)
        exact_curves[f"{rho:.1f}"] = curve
        all_rows.append(
            row(
                "Algorithm 1 audit optimum",
                rho,
                audit["objective"],
                audit["altitude"],
                sdp_solves=len(curve),
                outer_method="2 m global fast scan + 1 m exact local refinement",
            )
        )

        dense_a2 = deterministic_algorithm2(
            Agrid,
            chi,
            r,
            phi,
            pdes,
            rho,
            np.arange(HMIN, HMAX + 1e-9, 1.0),
            radar_solver=ref,
        )
        all_rows.append(
            row(
                "Algorithm 2 deterministic audit",
                rho,
                dense_a2.objective,
                dense_a2.altitude,
                p7_evaluations=dense_a2.p7_evaluations,
                closed_form_evaluations=dense_a2.closed_form_evaluations,
                outer_method="1 m exhaustive altitude audit (not paper PSO)",
            )
        )

        a2_runs = []
        for seed in range(10):
            sol = solve_algorithm2(
                Agrid, chi, r, phi, pdes, rho, seed=seed, radar_solver=ref
            )
            a2_runs.append(
                {
                    "seed": seed,
                    "objective": sol.objective,
                    "altitude": sol.altitude,
                    "p7_evaluations": sol.p7_evaluations,
                    "closed_form_evaluations": sol.closed_form_evaluations,
                }
            )
        pso_repeatability[f"{rho:.1f}"] = a2_runs
        a2 = a2_runs[0]
        all_rows.append(
            row(
                "Kang Algorithm 2 PSO (seed 0)",
                rho,
                a2["objective"],
                a2["altitude"],
                p7_evaluations=a2["p7_evaluations"],
                pso_seed=0,
            )
        )

        features = np.array(
            [[*target_deg, *r, *phi_deg, rho]], dtype=float
        )
        h_ml = float(np.clip(svr.predict(features)[0], HMIN, HMAX))
        ml = ref.solve(h_ml, chi, r, phi, pdes, rho)
        all_rows.append(
            row(
                "SVR altitude + one exact P2 solve",
                rho,
                ml.objective,
                h_ml,
                sdp_solves=1,
                model="Final corrected RBF SVR, locked hyperparameters",
            )
        )
        print(
            "  opt=%.6f@%.1f | A2=%.6f@%.1f | ML=%.6f@%.1f"
            % (
                audit["objective"],
                audit["altitude"],
                dense_a2.objective,
                dense_a2.altitude,
                ml.objective,
                h_ml,
            ),
            flush=True,
        )

    exact_pso = None
    if args.exact_pso:
        rho = float(args.exact_pso_rho)
        exact_pso = exact_algorithm1_pso(
            ref, chi, r, phi, pdes, rho, seed=args.exact_pso_seed
        )
        all_rows.append(
            row(
                "Kang Algorithm 1 exact PSO",
                rho,
                exact_pso["objective"],
                exact_pso["altitude"],
                sdp_solves=exact_pso["evaluations"],
                pso_seed=args.exact_pso_seed,
            )
        )

    # Add gap-to-audit-oracle values only after every method is available.
    optimum = {
        x["rho"]: x["objective"]
        for x in all_rows
        if x["method"] == "Algorithm 1 audit optimum"
    }
    for item in all_rows:
        fstar = optimum.get(item["rho"])
        item["gap_to_audit_percent"] = (
            100.0 * (item["objective"] - fstar) / fstar if fstar else None
        )

    csv_path = args.output / "gap1_method_comparison.csv"
    fields = sorted({key for item in all_rows for key in item})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    metadata = {
        "scenario": {
            "target_angles_deg": target_deg.tolist(),
            "r_m": r.tolist(),
            "phi_deg": phi_deg.tolist(),
            "pdes": pdes.tolist(),
            "angle_resolution_deg": args.resolution,
        },
        "fidelity": {
            "reference_sdp_is_dcp": ref.problem.is_dcp(),
            "reference_sdp_is_dpp": ref.problem.is_dpp(),
            "algorithm2_equations": [37, 41, 42, 43, 44, 45, 46, 47, 51, 52, 55, 56, 59],
            "algorithm2_pso": {"sigma": 0.7, "z1": 2.0, "z2": 2.0, "M": 8, "J": 8},
            "algorithm1_audit_deviation": "deterministic scan/refinement instead of paper PSO",
        },
        "svr": svr_meta,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cvxpy": cp.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "wall_seconds": float(time.perf_counter() - start_all),
        "rows": all_rows,
        "algorithm2_pso_repeatability": pso_repeatability,
        "algorithm1_exact_pso": exact_pso,
        "exact_refinement_curves": exact_curves,
    }
    (args.output / "gap1_results.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    plot_methods = [
        "Fixed Hmin (P2 exact)",
        "Fixed Hmax (P2 exact)",
        "Algorithm 1 audit optimum",
        "Algorithm 2 deterministic audit",
        "SVR altitude + one exact P2 solve",
    ]
    styles = ["o--", "s--", "o-", "^-", "D-"]
    for method, style in zip(plot_methods, styles):
        values = sorted(
            [x for x in all_rows if x["method"] == method], key=lambda x: x["rho"]
        )
        ax.plot(
            [x["rho"] for x in values],
            [x["objective"] for x in values],
            style,
            linewidth=1.8,
            markersize=5,
            label=method,
        )
    ax.set_xlabel(r"Trade-off weight $\rho$")
    ax.set_ylabel("P1 objective (lower is better)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "gap1_objective_comparison.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for method, style in zip(plot_methods[2:], ["o-", "^-", "D-"]):
        values = sorted(
            [x for x in all_rows if x["method"] == method], key=lambda x: x["rho"]
        )
        ax.plot(
            [x["rho"] for x in values],
            [x["altitude_m"] for x in values],
            style,
            linewidth=1.8,
            markersize=5,
            label=method,
        )
    ax.set_xlabel(r"Trade-off weight $\rho$")
    ax.set_ylabel("Selected UAV altitude (m)")
    ax.set_ylim(HMIN - 5, HMAX + 5)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "gap1_altitude_comparison.png", dpi=220)
    plt.close(fig)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
