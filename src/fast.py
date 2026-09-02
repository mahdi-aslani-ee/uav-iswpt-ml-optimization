"""
Fast solver for problem P2^h.
The inner problem is a convex quadratic in (X, gamma) over
C = {X Hermitian, X >= 0, [X]_nn = PT/N}.  We use FISTA with a Dykstra
projection onto C, and the closed-form minimiser for gamma at each step.
"""
import numpy as np
from kang import N, PT


def _proj_C(M, iters=25):
    """Dykstra projection onto {X>=0} ∩ {diag(X)=PT/N}."""
    c = PT / N
    X = 0.5 * (M + M.conj().T)
    p = np.zeros_like(X); q = np.zeros_like(X)
    for _ in range(iters):
        Y = X + p
        w, V = np.linalg.eigh(0.5 * (Y + Y.conj().T))
        Y2 = (V * np.maximum(w, 0.0)) @ V.conj().T          # PSD projection
        p = Y - Y2
        Z = Y2 + q
        Z2 = Z.copy()
        np.fill_diagonal(Z2, c)                              # diagonal projection
        q = Z - Z2
        X = Z2
    return X


def solve_fast(Agrid, chi, Glist, pdes, rho, n_iter=120, tol=1e-11, X0=None):
    """Returns (objective, X, gamma)."""
    L = len(chi); U = len(Glist)
    cr = rho / L
    cw = (1.0 - rho) / U
    pd = np.asarray(pdes, float)
    Gs = np.array(Glist)
    chi2 = float(np.sum(chi ** 2))

    Ac = Agrid.conj()

    def pattern(X):
        return np.real(np.einsum('li,ij,lj->l', Ac, X, Agrid))

    def harvest(X):
        return np.real(np.einsum('uij,ji->u', Gs, X))

    def best_gamma(p):
        return float(np.dot(chi, p) / chi2) if chi2 > 0 else 0.0

    def fval(X, g):
        p = pattern(X); q = harvest(X)
        return cr * np.sum((g * chi - p) ** 2) + cw * np.sum((1 - q / pd) ** 2)

    def grad(X, g):
        p = pattern(X); q = harvest(X)
        w = g * chi - p
        Gr = -2 * cr * (Agrid.T @ (w[:, None] * Ac))
        Gw = -2 * cw * np.einsum('u,uij->ij', (1 - q / pd) / pd, Gs)
        return Gr + Gw

    # Lipschitz estimate via power iteration on the Hessian (linear operator)
    R = np.random.default_rng(0).normal(size=(N, N)) + 1j * np.random.default_rng(1).normal(size=(N, N))
    R = 0.5 * (R + R.conj().T); R /= np.linalg.norm(R)
    Lip = 1.0
    for _ in range(30):
        pv = np.real(np.einsum('li,ij,lj->l', Ac, R, Agrid))
        qv = np.real(np.einsum('uij,ji->u', Gs, R))
        HR = 2 * cr * (Agrid.T @ (pv[:, None] * Ac)) + \
             2 * cw * np.einsum('u,uij->ij', qv / pd ** 2, Gs)
        HR = 0.5 * (HR + HR.conj().T)
        nr = np.linalg.norm(HR)
        if nr < 1e-300:
            break
        Lip = nr; R = HR / nr
    step = 1.0 / (Lip * 1.05 + 1e-30)

    X = _proj_C(np.eye(N) * (PT / N)) if X0 is None else _proj_C(X0)
    Y = X.copy(); t = 1.0
    g = best_gamma(pattern(X))
    prev = fval(X, g)
    for k in range(n_iter):
        g = best_gamma(pattern(Y))
        Xn = _proj_C(Y - step * grad(Y, g), iters=5)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        Y = Xn + ((t - 1) / tn) * (Xn - X)
        X, t = Xn, tn
        if k % 25 == 24:
            g = best_gamma(pattern(X)); cur = fval(X, g)
            if abs(prev - cur) < tol * max(1.0, abs(prev)):
                break
            prev = cur
    X = _proj_C(X, iters=40)
    g = best_gamma(pattern(X))
    return fval(X, g), X, g
