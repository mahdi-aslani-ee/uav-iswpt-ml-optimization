"""Professor-facing automated checks for gap 1 and the corrected solver."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from algorithm2 import deterministic_algorithm2
from kang import (
    A2,
    A3,
    B2,
    B3,
    HMIN,
    N,
    PT,
    angle_grid,
    desired_pattern,
    max_desired_power,
    objective,
    steer_matrix,
)
from reference_solver import FixedAltitudeSDP


def require(name: str, condition: bool, detail: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def main() -> None:
    require("A2 tripwire", np.isclose(A2, -1.5335, atol=5e-5), f"{A2:.10f}")
    require("B2 tripwire", np.isclose(B2, 3.5335, atol=5e-5), f"{B2:.10f}")
    require("A3 tripwire", np.isclose(A3, 5.0), f"{A3:.10f}")
    require("B3 tripwire", np.isclose(B3, 0.6994, atol=5e-5), f"{B3:.10f}")
    require(
        "Pdes tripwire",
        np.isclose(max_desired_power(50.0), 1.1598e-3, rtol=5e-5),
        f"{max_desired_power(50.0):.12e}",
    )

    theta = angle_grid(0.1)
    Agrid = steer_matrix(theta)
    chi = desired_pattern(theta, np.radians([-40.0, 0.0, 40.0]))
    r = np.array([50.0, 150.0])
    phi = np.radians([-20.0, 20.0])
    pdes = np.array([0.25 * max_desired_power(x) for x in r])
    rho = 0.6
    ref = FixedAltitudeSDP(Agrid, 2)
    require("Reference SDP is DCP", ref.problem.is_dcp(), str(ref.problem.is_dcp()))
    require("Reference SDP is DPP", ref.problem.is_dpp(), str(ref.problem.is_dpp()))

    full = ref.solve(127.0, chi, r, phi, pdes, rho)
    diagonal_error = float(np.max(np.abs(np.diag(full.X).real - PT / N)))
    min_eigenvalue = float(np.linalg.eigvalsh(full.X).min())
    direct = objective(full.X, full.gamma, 127.0, Agrid, chi, r, phi, pdes, rho)
    require("P2 per-antenna constraint", diagonal_error < 1e-7, f"max error {diagonal_error:.3e}")
    require("P2 PSD constraint", min_eigenvalue > -1e-7, f"lambda_min {min_eigenvalue:.3e}")
    require("P2 objective audit", np.isclose(full.objective, direct, rtol=1e-8), f"delta {direct-full.objective:.3e}")

    alg2 = deterministic_algorithm2(
        Agrid, chi, r, phi, pdes, rho, np.arange(50.0, 251.0, 1.0), radar_solver=ref
    )
    a2_diagonal_error = float(np.max(np.abs(np.diag(alg2.X).real - PT / N)))
    a2_min_eigenvalue = float(np.linalg.eigvalsh(alg2.X).min())
    require("Eq. (37) per-antenna constraint", a2_diagonal_error < 1e-7, f"max error {a2_diagonal_error:.3e}")
    require("Eq. (37) PSD constraint", a2_min_eigenvalue > -1e-7, f"lambda_min {a2_min_eigenvalue:.3e}")
    require("Algorithm 2 is a restricted design", alg2.objective >= full.objective - 1e-6, f"A2 {alg2.objective:.6f}, P2 {full.objective:.6f}")

    sweep = json.loads((ROOT / "gap1_outputs_paper_grid" / "exact_alg1_pso_sweep.json").read_text())
    max_gap = max(abs(x["gap_to_audit_percent"]) for x in sweep["runs"])
    require("Kang Algorithm 1 PSO repeatability audit", max_gap < 0.05, f"max |gap| {max_gap:.4f}%")

    paper = json.loads((ROOT / "gap1_outputs_paper_grid" / "gap1_results.json").read_text())
    rows = paper["rows"]
    required_methods = {
        "Fixed Hmin (P2 exact)",
        "Fixed Hmax (P2 exact)",
        "Algorithm 1 audit optimum",
        "Algorithm 2 deterministic audit",
        "SVR altitude + one exact P2 solve",
    }
    present = {x["method"] for x in rows}
    require("Five-method benchmark complete", required_methods <= present, str(sorted(required_methods)))
    print("All gap-1 verification checks passed.")


if __name__ == "__main__":
    main()
