"""Kang 2024 Algorithm 2: low-complexity waveform and altitude design.

Equation numbers refer to:
J. Kang, Electronics 13(21):4237, 2024.

The implementation follows Eq. (37), Eqs. (41)--(46), P7 in Eqs. (51)--(52),
the closed-form Eqs. (55)--(56), and the M=8, J=8 PSO in Algorithm 2.
``kang.py`` is intentionally imported, not modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cvxpy as cp
import numpy as np

from kang import HMAX, HMIN, N, PT, beampattern, channel_stats, objective, steer
from reference_solver import FixedAltitudeSDP


@dataclass
class P7Solution:
    objective: float
    weights: np.ndarray
    gamma: float
    used_closed_form: bool


@dataclass
class Algorithm2Solution:
    objective: float
    altitude: float
    X: np.ndarray
    gamma: float
    weights: np.ndarray
    X_rad: np.ndarray
    radar_gamma: float
    p7_evaluations: int
    closed_form_evaluations: int
    seed: int


def ehd_mrt_covariance(phi: float) -> np.ndarray:
    """PT * a_tilde(phi) a_tilde(phi)^H from Eq. (37)."""
    a = steer(float(phi))
    a_tilde = a / np.linalg.norm(a)
    return PT * np.outer(a_tilde, a_tilde.conj())


def proposed_waveform(X_rad: np.ndarray, phi_list: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Construct X_Prop exactly as Eq. (37)."""
    weights = np.asarray(weights, dtype=float)
    X = (1.0 - float(np.sum(weights))) * np.asarray(X_rad, dtype=complex)
    for w, phi in zip(weights, phi_list):
        X = X + float(w) * ehd_mrt_covariance(float(phi))
    return 0.5 * (X + X.conj().T)


def radar_components(
    X_rad: np.ndarray, Agrid: np.ndarray, phi_list: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return b_Rad and C_Rad from Eqs. (42)--(43)."""
    b_rad = beampattern(X_rad, Agrid)
    mrt_patterns = np.column_stack(
        [beampattern(ehd_mrt_covariance(phi), Agrid) for phi in phi_list]
    )
    return b_rad, mrt_patterns - b_rad[:, None]


def wpt_components(
    h: float, X_rad: np.ndarray, r_list: np.ndarray, phi_list: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return b_WPT(h) and C_WPT(h) from Eqs. (45)--(46)."""
    mrt = [ehd_mrt_covariance(phi) for phi in phi_list]
    b = []
    rows = []
    for r, phi in zip(r_list, phi_list):
        G, _, _ = channel_stats(float(h), float(r), float(phi))
        base = float(np.real(np.trace(G @ X_rad)))
        b.append(base)
        rows.append([float(np.real(np.trace(G @ Xk))) - base for Xk in mrt])
    return np.asarray(b), np.asarray(rows)


def p7_value(
    weights: np.ndarray,
    gamma: float,
    b_rad: np.ndarray,
    C_rad: np.ndarray,
    b_wpt: np.ndarray,
    C_wpt: np.ndarray,
    chi: np.ndarray,
    pdes: np.ndarray,
    rho: float,
) -> float:
    """Evaluate t(w, gamma, h) in Eq. (47) using its residual form."""
    radar_residual = gamma * chi - b_rad - C_rad @ weights
    wpt_residual = 1.0 - (b_wpt + C_wpt @ weights) / pdes
    return float(
        rho * np.mean(radar_residual**2)
        + (1.0 - rho) * np.mean(wpt_residual**2)
    )


def closed_form_p7(
    b_rad: np.ndarray,
    C_rad: np.ndarray,
    b_wpt: np.ndarray,
    C_wpt: np.ndarray,
    chi: np.ndarray,
    pdes: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, float]:
    """Unconstrained KKT solution in Kang Eqs. (55)--(56)."""
    L, U = len(chi), len(pdes)
    chi2 = float(chi @ chi)
    if chi2 <= 0:
        raise ValueError("The desired radar pattern must contain a nonzero sample.")
    projected_C = C_rad - np.outer(chi, (chi @ C_rad) / chi2)
    projected_b = b_rad - chi * float(chi @ b_rad) / chi2
    D = C_wpt / pdes[:, None]
    d = 1.0 - b_wpt / pdes

    H = (rho / L) * (projected_C.T @ projected_C)
    H += ((1.0 - rho) / U) * (D.T @ D)
    rhs = -(rho / L) * (projected_C.T @ projected_b)
    rhs += ((1.0 - rho) / U) * (D.T @ d)
    try:
        weights = np.linalg.solve(H, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(H, rcond=1e-12) @ rhs
    gamma = float((chi @ (b_rad + C_rad @ weights)) / chi2)
    return np.asarray(weights, dtype=float), gamma


def _weights_feasible(weights: np.ndarray, tol: float = 1e-9) -> bool:
    return bool(
        np.all(weights >= -tol)
        and np.all(weights <= 1.0 + tol)
        and np.sum(weights) <= 1.0 + tol
    )


def solve_p7(
    h: float,
    X_rad: np.ndarray,
    b_rad: np.ndarray,
    C_rad: np.ndarray,
    chi: np.ndarray,
    r_list: np.ndarray,
    phi_list: np.ndarray,
    pdes: np.ndarray,
    rho: float,
    solver: str = "CLARABEL",
) -> P7Solution:
    """Apply Eqs. (55)--(56), falling back to constrained convex P7."""
    b_wpt, C_wpt = wpt_components(h, X_rad, r_list, phi_list)
    weights, gamma = closed_form_p7(
        b_rad, C_rad, b_wpt, C_wpt, chi, pdes, rho
    )
    if _weights_feasible(weights):
        weights = np.clip(weights, 0.0, 1.0)
        return P7Solution(
            objective=p7_value(
                weights, gamma, b_rad, C_rad, b_wpt, C_wpt, chi, pdes, rho
            ),
            weights=weights,
            gamma=gamma,
            used_closed_form=True,
        )

    U = len(pdes)
    w = cp.Variable(U, name="w")
    gam = cp.Variable(name="gamma")
    radar_residual = gam * chi - b_rad - C_rad @ w
    wpt_residual = 1.0 - cp.multiply(1.0 / pdes, b_wpt + C_wpt @ w)
    prob = cp.Problem(
        cp.Minimize(
            (rho / len(chi)) * cp.sum_squares(radar_residual)
            + ((1.0 - rho) / U) * cp.sum_squares(wpt_residual)
        ),
        [w >= 0.0, w <= 1.0, cp.sum(w) <= 1.0],
    )
    prob.solve(solver=solver, warm_start=True)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"P7 failed with status {prob.status!r}.")
    return P7Solution(
        objective=float(prob.value),
        weights=np.asarray(w.value, dtype=float),
        gamma=float(gam.value),
        used_closed_form=False,
    )


def pso_minimize(
    fitness: Callable[[float], float],
    seed: int,
    particles: int = 8,
    iterations: int = 8,
    inertia: float = 0.7,
    cognitive: float = 2.0,
    social: float = 2.0,
) -> tuple[float, float, int]:
    """One-dimensional PSO using Kang Table 1 and Eqs. (26)--(27)."""
    rng = np.random.default_rng(seed)
    position = HMIN + rng.random(particles) * (HMAX - HMIN)
    velocity = np.zeros(particles)
    pbest = position.copy()
    pbest_value = np.asarray([fitness(float(h)) for h in position])
    evaluations = particles
    g_index = int(np.argmin(pbest_value))
    gbest = float(pbest[g_index])
    gbest_value = float(pbest_value[g_index])

    for _ in range(iterations):
        velocity = (
            inertia * velocity
            + cognitive * rng.random(particles) * (pbest - position)
            + social * rng.random(particles) * (gbest - position)
        )
        position = np.clip(position + velocity, HMIN, HMAX)
        values = np.asarray([fitness(float(h)) for h in position])
        evaluations += particles
        improved = values < pbest_value
        pbest[improved] = position[improved]
        pbest_value[improved] = values[improved]
        g_index = int(np.argmin(pbest_value))
        gbest = float(pbest[g_index])
        gbest_value = float(pbest_value[g_index])
    return gbest, gbest_value, evaluations


def solve_algorithm2(
    Agrid: np.ndarray,
    chi: np.ndarray,
    r_list: np.ndarray,
    phi_list: np.ndarray,
    pdes: np.ndarray,
    rho: float,
    seed: int = 0,
    radar_solver: FixedAltitudeSDP | None = None,
) -> Algorithm2Solution:
    """Run Kang Algorithm 2 end-to-end with the paper's PSO settings."""
    r_list = np.asarray(r_list, dtype=float)
    phi_list = np.asarray(phi_list, dtype=float)
    pdes = np.asarray(pdes, dtype=float)
    if radar_solver is None:
        radar_solver = FixedAltitudeSDP(Agrid, len(r_list))
    radar = radar_solver.solve(HMIN, chi, r_list, phi_list, pdes, rho=1.0)
    X_rad = radar.X
    b_rad, C_rad = radar_components(X_rad, Agrid, phi_list)
    cache: dict[float, P7Solution] = {}

    def evaluate(h: float) -> float:
        key = float(h)
        if key not in cache:
            cache[key] = solve_p7(
                key,
                X_rad,
                b_rad,
                C_rad,
                chi,
                r_list,
                phi_list,
                pdes,
                rho,
            )
        return cache[key].objective

    altitude, _, evaluations = pso_minimize(evaluate, seed=seed)
    p7 = solve_p7(
        altitude,
        X_rad,
        b_rad,
        C_rad,
        chi,
        r_list,
        phi_list,
        pdes,
        rho,
    )
    X = proposed_waveform(X_rad, phi_list, p7.weights)
    # Re-evaluate through the original P1 implementation as an equation audit.
    obj = objective(X, p7.gamma, altitude, Agrid, chi, r_list, phi_list, pdes, rho)
    if not np.isclose(obj, p7.objective, rtol=2e-8, atol=2e-10):
        raise RuntimeError("Eq. (47) does not match direct P1 evaluation.")
    return Algorithm2Solution(
        objective=float(obj),
        altitude=float(altitude),
        X=X,
        gamma=float(p7.gamma),
        weights=np.asarray(p7.weights),
        X_rad=X_rad,
        radar_gamma=float(radar.gamma),
        p7_evaluations=int(evaluations + 1),
        closed_form_evaluations=int(
            sum(sol.used_closed_form for sol in cache.values()) + p7.used_closed_form
        ),
        seed=int(seed),
    )


def deterministic_algorithm2(
    Agrid: np.ndarray,
    chi: np.ndarray,
    r_list: np.ndarray,
    phi_list: np.ndarray,
    pdes: np.ndarray,
    rho: float,
    altitudes: np.ndarray,
    radar_solver: FixedAltitudeSDP | None = None,
) -> Algorithm2Solution:
    """Dense altitude audit of P6; not part of Kang's stochastic Algorithm 2."""
    r_list = np.asarray(r_list, dtype=float)
    phi_list = np.asarray(phi_list, dtype=float)
    pdes = np.asarray(pdes, dtype=float)
    if radar_solver is None:
        radar_solver = FixedAltitudeSDP(Agrid, len(r_list))
    radar = radar_solver.solve(HMIN, chi, r_list, phi_list, pdes, rho=1.0)
    X_rad = radar.X
    b_rad, C_rad = radar_components(X_rad, Agrid, phi_list)
    solutions = [
        solve_p7(
            float(h), X_rad, b_rad, C_rad, chi, r_list, phi_list, pdes, rho
        )
        for h in altitudes
    ]
    i = int(np.argmin([s.objective for s in solutions]))
    altitude = float(altitudes[i])
    p7 = solutions[i]
    X = proposed_waveform(X_rad, phi_list, p7.weights)
    obj = objective(X, p7.gamma, altitude, Agrid, chi, r_list, phi_list, pdes, rho)
    return Algorithm2Solution(
        objective=float(obj),
        altitude=altitude,
        X=X,
        gamma=float(p7.gamma),
        weights=np.asarray(p7.weights),
        X_rad=X_rad,
        radar_gamma=float(radar.gamma),
        p7_evaluations=len(solutions),
        closed_form_evaluations=int(sum(s.used_closed_form for s in solutions)),
        seed=-1,
    )
