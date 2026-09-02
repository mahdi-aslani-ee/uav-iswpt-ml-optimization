"""Auditable CVXPY reference solver for Kang's fixed-altitude problem P2|h.

Unlike ``solver.py``, the sampled steering matrix is a constant in the CVXPY
graph.  The remaining scenario-dependent quantities enter as parameters in a
DPP-compliant way, so repeated solves do not trigger CVXPY's non-DPP warning.

No approximation is introduced here: the objective is Kang (6), (16), and
(20), subject to the per-antenna and PSD constraints (17)--(18).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from kang import N, PT, channel_stats


@dataclass
class SDPSolution:
    objective: float
    X: np.ndarray
    gamma: float
    status: str
    solve_seconds: float


class FixedAltitudeSDP:
    """Reusable DPP model for one fixed angle grid and number of EHDs."""

    def __init__(self, Agrid: np.ndarray, n_ehds: int, solver: str = "CLARABEL"):
        self.Agrid = np.asarray(Agrid, dtype=complex)
        self.L = self.Agrid.shape[0]
        self.U = int(n_ehds)
        self.solver = solver

        self.X_var = cp.Variable((N, N), hermitian=True, name="X")
        self.gamma_var = cp.Variable(name="gamma")

        # Combined parameters avoid products between two parameters.  This is
        # the key difference from the original non-DPP ``solver.py`` model.
        self.chi_scaled = cp.Parameter(self.L, name="sqrt_rho_over_L_times_chi")
        self.rad_scale = cp.Parameter(nonneg=True, name="sqrt_rho_over_L")
        self.G_scaled = [
            cp.Parameter((N, N), hermitian=True, name=f"scaled_G_{u}")
            for u in range(self.U)
        ]
        self.wpt_constant = cp.Parameter(self.U, name="wpt_constant")

        # Each row is vec(a_l^* a_l^T), so p = B vec(X) evaluates
        # a_l^H X a_l without ever materialising an L-by-L matrix.  The
        # seemingly simpler diag(A^H X A) graph becomes memory-prohibitive at
        # Kang's 0.1-degree grid (L=1801); this equivalent linear map is only
        # L-by-N^2.
        pattern_map = np.einsum(
            "li,lj->lij", self.Agrid.conj(), self.Agrid
        ).reshape(self.L, N * N)
        p = cp.real(
            pattern_map @ cp.reshape(self.X_var, (N * N,), order="C")
        )
        radar_residual = self.gamma_var * self.chi_scaled - self.rad_scale * p
        received_scaled = cp.hstack(
            [cp.real(cp.trace(G @ self.X_var)) for G in self.G_scaled]
        )
        wpt_residual = self.wpt_constant - received_scaled

        constraints = [self.X_var >> 0]
        constraints += [
            cp.real(self.X_var[n, n]) == PT / N for n in range(N)
        ]
        constraints += [cp.imag(self.X_var[n, n]) == 0 for n in range(N)]
        self.problem = cp.Problem(
            cp.Minimize(cp.sum_squares(radar_residual) + cp.sum_squares(wpt_residual)),
            constraints,
        )
        if not self.problem.is_dcp() or not self.problem.is_dpp():
            raise RuntimeError("Reference SDP must be both DCP and DPP.")

    def solve(
        self,
        h: float,
        chi: np.ndarray,
        r_list: np.ndarray,
        phi_list: np.ndarray,
        pdes: np.ndarray,
        rho: float,
        warm_start: bool = True,
    ) -> SDPSolution:
        """Solve Kang's convex P2|h exactly at the supplied altitude."""
        chi = np.asarray(chi, dtype=float)
        r_list = np.asarray(r_list, dtype=float)
        phi_list = np.asarray(phi_list, dtype=float)
        pdes = np.asarray(pdes, dtype=float)
        if len(r_list) != self.U or len(phi_list) != self.U or len(pdes) != self.U:
            raise ValueError("The number of EHD inputs must equal n_ehds.")
        if np.any(pdes <= 0):
            raise ValueError("All desired powers must be strictly positive.")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [0, 1].")

        radar_scale = float(np.sqrt(rho / self.L))
        wpt_scale = float(np.sqrt((1.0 - rho) / self.U))
        self.rad_scale.value = radar_scale
        self.chi_scaled.value = radar_scale * chi
        self.wpt_constant.value = wpt_scale * np.ones(self.U)
        for u, (r, phi, pd) in enumerate(zip(r_list, phi_list, pdes)):
            G, _, _ = channel_stats(float(h), float(r), float(phi))
            self.G_scaled[u].value = (wpt_scale / pd) * G

        start = time.perf_counter()
        self.problem.solve(solver=self.solver, warm_start=warm_start)
        elapsed = time.perf_counter() - start
        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"P2|h failed with status {self.problem.status!r}.")
        if self.X_var.value is None or self.gamma_var.value is None:
            raise RuntimeError("P2|h returned no primal solution.")
        X = np.asarray(self.X_var.value)
        X = 0.5 * (X + X.conj().T)
        return SDPSolution(
            objective=float(self.problem.value),
            X=X,
            gamma=float(self.gamma_var.value),
            status=str(self.problem.status),
            solve_seconds=float(elapsed),
        )
