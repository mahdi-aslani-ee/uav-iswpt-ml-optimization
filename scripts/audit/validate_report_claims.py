"""Machine-check the numerical claims used in the report and presentation.

This script does not recompute expensive experiments.  It independently reads
the saved JSON evidence, checks the protected Kang implementation hash, and
asserts the exact values that are allowed to appear in the final deliverables.
Use ``verify_gap1.py`` when an equation/constraint recomputation is also needed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from kang import A2, A3, B2, B3, max_desired_power


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def check_close(name: str, observed: float, expected: float, atol: float = 1e-10) -> None:
    if not np.isclose(observed, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"{name}: observed {observed!r}, expected {expected!r}")
    print(f"[PASS] {name}: {observed:.12g}")


def check_equal(name: str, observed, expected) -> None:
    if observed != expected:
        raise AssertionError(f"{name}: observed {observed!r}, expected {expected!r}")
    print(f"[PASS] {name}: {observed!r}")


def main() -> None:
    kang_hash = hashlib.sha256((SRC / "kang.py").read_bytes()).hexdigest()
    check_equal(
        "protected kang.py SHA-256",
        kang_hash,
        "f7d82bf1a67890e86ec230f3f95d004471e70764888af63690eb71e1d3bd59c2",
    )

    check_close("Kang A2", float(A2), -1.5335425964, 5e-11)
    check_close("Kang B2", float(B2), 3.5335425964, 5e-11)
    check_close("Kang A3", float(A3), 5.0, 1e-12)
    check_close("Kang B3", float(B3), 0.6993983051, 5e-11)
    check_close("max_desired_power(50 m)", max_desired_power(50.0), 1.159800804614e-3, 5e-16)

    paper = load("gap1_outputs_paper_grid/gap1_results.json")
    check_equal("reference SDP DCP", paper["fidelity"]["reference_sdp_is_dcp"], True)
    check_equal("reference SDP DPP", paper["fidelity"]["reference_sdp_is_dpp"], True)

    expected_objectives = {
        0.1: {
            "Fixed Hmin (P2 exact)": 0.4913083600058628,
            "Fixed Hmax (P2 exact)": 0.41691633587415233,
            "Algorithm 1 audit optimum": 0.1403283430016156,
            "Algorithm 2 deterministic audit": 0.14465643499123879,
            "SVR altitude + one exact P2 solve": 0.14075835631473355,
        },
        0.3: {
            "Fixed Hmin (P2 exact)": 0.4799164403837554,
            "Fixed Hmax (P2 exact)": 0.4825284980030272,
            "Algorithm 1 audit optimum": 0.30888524981153115,
            "Algorithm 2 deterministic audit": 0.31412772320371124,
            "SVR altitude + one exact P2 solve": 0.3095603121138565,
        },
        0.5: {
            "Fixed Hmin (P2 exact)": 0.4628114542253988,
            "Fixed Hmax (P2 exact)": 0.4834893893751781,
            "Algorithm 1 audit optimum": 0.38341478699922105,
            "Algorithm 2 deterministic audit": 0.3866637838790181,
            "SVR altitude + one exact P2 solve": 0.383429535393975,
        },
        0.7: {
            "Fixed Hmin (P2 exact)": 0.4376965409197404,
            "Fixed Hmax (P2 exact)": 0.45659261664749756,
            "Algorithm 1 audit optimum": 0.4035143413160932,
            "Algorithm 2 deterministic audit": 0.4052032369774844,
            "SVR altitude + one exact P2 solve": 0.4038465965568003,
        },
        0.9: {
            "Fixed Hmin (P2 exact)": 0.39945052840136436,
            "Fixed Hmax (P2 exact)": 0.4061799216390263,
            "Algorithm 1 audit optimum": 0.3899697572352807,
            "Algorithm 2 deterministic audit": 0.3905679439956197,
            "SVR altitude + one exact P2 solve": 0.39032703625191206,
        },
    }
    rows = {(float(row["rho"]), row["method"]): row for row in paper["rows"]}
    for rho, methods in expected_objectives.items():
        for method, expected in methods.items():
            check_close(f"paper-grid rho={rho} {method}", rows[(rho, method)]["objective"], expected)

    pso = load("gap1_outputs_paper_grid/exact_alg1_pso_sweep.json")
    max_abs_pso_gap = max(abs(row["gap_to_audit_percent"]) for row in pso["runs"])
    check_close("Algorithm 1 max absolute audit gap (%)", max_abs_pso_gap, 0.016783029681161066)
    check_equal("Algorithm 1 evaluations per rho", sorted({row["evaluations"] for row in pso["runs"]}), [73])

    fast = load("results/gap1/fast_solver_audit.json")["summary"]["fast120"]
    check_close("fast solver median absolute relative error", fast["median_absolute_relative_error"], 0.0002539748563927328)
    check_close("fast solver p95 absolute relative error", fast["p95_absolute_relative_error"], 0.0017577469395139353)
    check_close("fast solver max absolute relative error", fast["max_absolute_relative_error"], 0.002138026364139572)

    labels = load("results/gap1/label_quality_audit.json")["summary"]
    check_close("stored-label altitude MAE (m)", labels["altitude_mae_m"], 2.2685185185185177)
    check_close("stored-label mean P1 gap (%)", labels["objective_gap_mean_percent"], 0.06333874103359918)
    check_close("stored-label p95 P1 gap (%)", labels["objective_gap_p95_percent"], 0.22213208292673836)
    check_close("stored-label max P1 gap (%)", labels["objective_gap_max_percent"], 0.2427514380129119)

    ml = load("results/ml/ml_corrected_results.json")
    repeated = load("results/ml/repeated_split_metrics.json")
    expected_repeated_mae = {
        "Linear": (8.471580421862095, 0.6506032510212549),
        "Decision Tree": (10.496681881202313, 0.9573855358495924),
        "SVR (RBF)": (7.879970198418246, 0.6650744966608397),
        "MLP": (9.405659069850744, 0.9003116862127515),
    }
    for model, (mean, std) in expected_repeated_mae.items():
        check_close(f"{model} repeated MAE mean (m)", repeated["summary"][model]["MAE_m"]["mean"], mean)
        check_close(f"{model} repeated MAE std (m)", repeated["summary"][model]["MAE_m"]["std"], std)

    expected_fast_gap = {
        "Linear": (0.8672340930818568, 0.12813356218884098, 5.013335424169753),
        "Decision Tree": (2.144896053333001, 0.2643952033561782, 6.870389644281278),
        "SVR (RBF)": (0.7928597257663398, 0.08135678148771572, 3.3084763765399097),
        "MLP": (1.0194293789184248, 0.20271883269853236, 4.144743495947771),
    }
    for model, (mean, median, p95) in expected_fast_gap.items():
        summary = ml["fast_candidate_oracle"]["summary"][model]
        check_close(f"{model} fresh-oracle mean gap (%)", summary["mean_percent"], mean)
        check_close(f"{model} fresh-oracle median gap (%)", summary["median_percent"], median)
        check_close(f"{model} fresh-oracle p95 gap (%)", summary["p95_percent"], p95)
        check_equal(f"{model} negative gap count", summary["negative_count"], 0)

    ranked = sorted(
        ml["fast_candidate_oracle"]["summary"],
        key=lambda model: ml["fast_candidate_oracle"]["summary"][model]["mean_percent"],
    )
    check_equal("fresh-oracle model ranking by mean gap", ranked, ["SVR (RBF)", "Linear", "MLP", "Decision Tree"])
    print("\nAll report and presentation claims match the saved evidence.")


if __name__ == "__main__":
    main()
