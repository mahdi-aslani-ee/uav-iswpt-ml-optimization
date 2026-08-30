"""
Faithful implementation of:
  J. Kang, "Joint Design of Transmit Waveform and Altitude for UAV-Enabled ISWPT
  Systems," Electronics 13(21):4237, 2024.
Equation numbers below refer to that paper.
"""
import numpy as np

# ---------------- Table 1 simulation settings ----------------
PT     = 1.0        # transmit sum power (W)
BETA0  = 1.0        # path loss at reference distance
ALPHA0 = 3.5        # path loss exponent, ground links
ALPHA_H= 2.0        # path loss exponent, aerial links (alpha_{pi/2})
A1, B1 = 9.61, 0.16 # LoS-probability params (urban), eq (10)
K0_DB, KH_DB = 5.0, 15.0   # Rician factors in dB at 0 and pi/2, eq (13)
HMIN, HMAX = 50.0, 250.0
N      = 8          # UAV antennas
DA_LC  = 0.5        # d_a / lambda_c
DELTA  = 10.0       # desired beam width (deg)

# derived coefficients, eq (12)
_e = A1 * np.exp(A1 * B1)
A2 = (ALPHA_H - ALPHA0) * (1.0 + _e) / _e
B2 = ALPHA0 - A2 / (1.0 + _e)
# derived coefficients, eq (13)
A3 = K0_DB
B3 = (2.0 / np.pi) * np.log(KH_DB / K0_DB)


def steer(theta_rad):
    """Array steering vector, eq (4)."""
    n = np.arange(N)
    return np.exp(-1j * 2 * np.pi * DA_LC * n * np.sin(theta_rad))


def p_los(psi):
    """LoS probability, eq (10). psi in radians."""
    return 1.0 / (1.0 + A1 * np.exp(-B1 * (np.degrees(psi) - A1)))


def channel_stats(h, r, phi):
    """G_u(h) from eq (15), plus alpha_u and K_u."""
    psi = np.arctan2(h, r)                      # eq (11)
    alpha = A2 * p_los(psi) + B2                # eq (12)
    K = 10.0 ** (0.1 * A3 * np.exp(B3 * psi))   # eq (13), linear
    d = np.sqrt(r ** 2 + h ** 2)                # eq (9)
    a = steer(phi)
    G = BETA0 * d ** (-alpha) * (
        K / (K + 1.0) * np.outer(a, a.conj()) + 1.0 / (K + 1.0) * np.eye(N))
    return G, alpha, K


def g_max(h, r):
    """Max harvested power with MRT beamforming, eq (36)."""
    psi = np.arctan2(h, r)
    alpha = A2 * p_los(psi) + B2
    K = 10.0 ** (0.1 * A3 * np.exp(B3 * psi))
    return BETA0 * (r ** 2 + h ** 2) ** (-alpha / 2.0) * PT * (N * K + 1.0) / (K + 1.0)


def max_desired_power(r, n_grid=400):
    """Solve P4'' by 1-D line search -> max desired power for an EHD."""
    hs = np.linspace(HMIN, HMAX, n_grid)
    return float(np.max([g_max(h, r) for h in hs]))


def angle_grid(res_deg=1.0):
    """Sample angle grid [theta_l], uniform over [-90, 90]."""
    return np.radians(np.arange(-90.0, 90.0 + 1e-9, res_deg))


def desired_pattern(theta_grid, target_angles_rad):
    """chi(theta_l), eq (60): unit gain within DELTA/2 of each target."""
    chi = np.zeros(len(theta_grid))
    half = np.radians(DELTA / 2.0)
    for t in target_angles_rad:
        chi[np.abs(theta_grid - t) <= half + 1e-12] = 1.0
    return chi


def steer_matrix(theta_grid):
    """A[l] = a(theta_l); returns (L, N) complex matrix."""
    return np.array([steer(t) for t in theta_grid])


def beampattern(X, Agrid):
    """P_Rad(theta_l; X) = a^H X a, eq (5), vectorised over the grid."""
    return np.real(np.einsum('li,ij,lj->l', Agrid.conj(), X, Agrid))


def loss_rad(X, Agrid, chi, gamma):
    """L_Rad(X, gamma), eq (6)."""
    return float(np.mean((gamma * chi - beampattern(X, Agrid)) ** 2))


def loss_wpt(X, h, r_list, phi_list, pdes):
    """L_WPT(X, h), eq (16)."""
    v = []
    for r, phi, pd in zip(r_list, phi_list, pdes):
        G, _, _ = channel_stats(h, r, phi)
        v.append((1.0 - np.real(np.trace(G @ X)) / pd) ** 2)
    return float(np.mean(v))


def objective(X, gamma, h, Agrid, chi, r_list, phi_list, pdes, rho):
    """Objective of problem P1."""
    return rho * loss_rad(X, Agrid, chi, gamma) + \
           (1 - rho) * loss_wpt(X, h, r_list, phi_list, pdes)
