import numpy as np


def _fill_nan_columns(values):
    """Fill missing time samples in each spatial column by linear interpolation."""
    values = np.asarray(values, dtype=float).copy()
    if values.ndim != 2:
        raise ValueError("psi must have shape (time, evaluation_point)")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("psi must contain at least one time and one spatial point")

    for j in range(values.shape[1]):
        good = np.isfinite(values[:, j])
        if not good.any():
            raise ValueError(f"spatial column {j} contains no finite time samples")
        if not good.all():
            # np.interp holds the first/last valid value at the record ends
            values[~good, j] = np.interp(
                np.flatnonzero(~good), np.flatnonzero(good), values[good, j]
            )
    return values


def _validate_n_modes(n_modes):
    if not isinstance(n_modes, (int, np.integer)) or n_modes < 1:
        raise ValueError("n_modes must be a positive integer")


def mnoc(psi, n_modes, cell_area_m2=None):
    """
    psi: [n_time, n_space] current function (A).
    n_modes: number of natural components to retain (clipped to the rank).
    cell_area_m2: optional positive cell areas [m^2]. When supplied, use an
        area-weighted spatial inner product. ``None`` gives equal weights.

    Returns (j = mode index, descending eigenvalue order, same order in every output):
        T              [n_time, k]           time coefficients (A)
        X              [k, n_space]          unit-norm spatial modes (dimensionless)
        eigvals        [k]                   mean(T_j**2), power of each mode
        psi_modes      [k, n_time, n_space]  psi_j = T_j(t) X_j(x), in A
        power_fraction [k]                   eigvals / sum of all eigenvalues
    psi_modes.sum(axis=0) is the truncated reconstruction T @ X. The input
    is intentionally not centred, so the background current pattern remains
    part of the decomposition.
    """
    _validate_n_modes(n_modes)
    psi = _fill_nan_columns(psi)
    if cell_area_m2 is None:
        weights = np.ones(psi.shape[1])
    else:
        weights = np.asarray(cell_area_m2, dtype=float).ravel()
        if weights.size != psi.shape[1]:
            raise ValueError("cell_area_m2 must contain one value per evaluation point")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("cell_area_m2 must be finite and strictly positive")
        weights = weights / weights.mean()

    sqrt_weights = np.sqrt(weights)
    m = psi.shape[0]
    _, s, Vt = np.linalg.svd(psi * sqrt_weights, full_matrices=False)
    eigvals_all = s ** 2 / m                              # descending
    k = min(n_modes, len(s))
    eigvals = eigvals_all[:k]
    X = Vt[:k].copy() / sqrt_weights
    X *= np.sign(X[np.arange(k), np.abs(X).argmax(axis=1)])[:, None]  # largest element positive
    T = psi @ (X * weights).T
    psi_modes = T.T[:, :, None] * X[:, None, :]
    power_fraction = eigvals / eigvals_all.sum()
    return T, X, eigvals, psi_modes, power_fraction


def psi_to_current(psi):
    """
    psi: [n_time, n_space] or [n_time, n_lat, n_lon] (A): total field, one mode,
         reconstruction, anything with time on the first axis.
    Returns: [n_time] current (kA) = max - min of psi at each timestamp.
    """
    flat = np.asarray(psi, dtype=float).reshape(len(psi), -1)
    return ((np.nanmax(flat, axis=1)- np.nanmin(flat, axis=1)) / 1000)

