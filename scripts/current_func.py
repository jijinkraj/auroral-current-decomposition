"""
current_func.py

Analytical stream function and current density for SECS basis functions.

    ψ_i(θ, φ) = −(1/4π) ln(1 − cos χ_i)        [A]
    J_E = −(1/r)       ∂ψ/∂θ       (A/m)
    J_N = −(1/r sinθ)  ∂ψ/∂φ       (A/m)
"""

import numpy as np

SIN_THETA_MIN = 0.01


def _cos_chi(pole_colat, pole_lon, eval_colat, eval_lon):
    cp = np.cos(pole_colat)[:, None]
    sp = np.sin(pole_colat)[:, None]
    ce = np.cos(eval_colat)[None, :]
    se = np.sin(eval_colat)[None, :]
    dlon = eval_lon[None, :] - pole_lon[:, None]
    return cp * ce + sp * se * np.cos(dlon)


def get_psi(sec_df_loc, sec_amps, eval_loc, eval_grid, times):
    """
    ψ(t, eval) = Σ_i I_i(t) [−(1/4π) ln(1 − cos χ_i)]

    sec_amps : (ntimes, nsec) current amplitudes [A]
    Returns  : (ntimes, npred) current [A]
    """
    pole_colat = np.radians(90.0 - sec_df_loc[:, 0])
    pole_lon = np.radians(sec_df_loc[:, 1])
    eval_colat = np.radians(90.0 - eval_loc[:, 0])
    eval_lon = np.radians(eval_loc[:, 1])

    cos_chi = _cos_chi(pole_colat, pole_lon, eval_colat, eval_lon)
    cos_chi = np.clip(cos_chi, -1.0 + 1e-12, 1.0 - 1e-12)

    psi_i = -1.0 / (4.0 * np.pi) * np.log(1.0 - cos_chi)
    psi = sec_amps @ psi_i

    nan_mask = np.isnan(psi)
    if nan_mask.any():
        valid = ~nan_mask
        for j in range(psi.shape[1]):
            if nan_mask[:, j].any():
                psi[nan_mask[:, j], j] = np.interp(
                    np.where(nan_mask[:, j])[0],
                    np.where(valid[:, j])[0],
                    psi[valid[:, j], j]
                )

    return psi, eval_grid, times


def psi_to_J(psi, sec_df_loc, eval_loc):
    """
    J from ψ by solving for SECS amplitudes, then analytical J.

    J_E = −(1/r)       ∂ψ/∂θ    (A/m)
    J_N = −(1/r sinθ)  ∂ψ/∂φ    (A/m)

    psi     : (ntimes, npred) or (npred,) current [A]
    Returns : (ntimes, npred, 2) current density [A/m]
              [..., 0] = eastward, [..., 1] = northward
    """
    psi = np.asarray(psi)
    single = psi.ndim == 1
    if single:
        psi = psi[np.newaxis, :]

    ntimes, npred = psi.shape
    nsec = sec_df_loc.shape[0]

    pole_colat = np.radians(90.0 - sec_df_loc[:, 0])
    pole_lon = np.radians(sec_df_loc[:, 1])
    eval_colat = np.radians(90.0 - eval_loc[:, 0])
    eval_lon = np.radians(eval_loc[:, 1])
    eval_r = eval_loc[:, 2]

    cos_chi = _cos_chi(pole_colat, pole_lon, eval_colat, eval_lon)
    cos_chi = np.clip(cos_chi, -1.0 + 1e-12, 1.0 - 1e-12)
    one_m_cos = 1.0 - cos_chi

    # ψ basis: (nsec, npred)
    psi_basis = -1.0 / (4.0 * np.pi) * np.log(one_m_cos)

    # J basis: (nsec, npred)
    sin_eval = np.sin(eval_colat)
    sin_eval = np.clip(sin_eval, SIN_THETA_MIN, None)

    dcos_dtheta = (-np.cos(pole_colat)[:, None] * sin_eval[None, :]
                   + np.sin(pole_colat)[:, None] * np.cos(eval_colat)[None, :]
                   * np.cos(eval_lon[None, :] - pole_lon[:, None]))

    dcos_dphi = (-np.sin(pole_colat)[:, None] * sin_eval[None, :]
                 * np.sin(eval_lon[None, :] - pole_lon[:, None]))

    dpsi_dtheta_i = 1.0 / (4.0 * np.pi) * dcos_dtheta / one_m_cos
    dpsi_dphi_i = 1.0 / (4.0 * np.pi) * dcos_dphi / one_m_cos

    inv_r = 1.0 / eval_r
    inv_r_sin = 1.0 / (eval_r * sin_eval)

    J_E_basis = -dpsi_dtheta_i * inv_r[None, :]
    J_N_basis = -dpsi_dphi_i * inv_r_sin[None, :]

    # solve psi = psi_basis.T @ amps  →  amps
    amps, *_ = np.linalg.lstsq(psi_basis.T, psi.T, rcond=None)
    amps = amps.T  # (ntimes, nsec)

    Je = amps @ J_E_basis
    Jn = amps @ J_N_basis

    J = np.stack([Je, Jn], axis=-1)

    if single:
        J = J[0]
    return J
