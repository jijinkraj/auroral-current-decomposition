"""Set up divergence-free SECS on the cubed-sphere grid."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pysecs import secs
from current_func import get_psi, psi_to_J
from generate_cubed_sphere import EARTH_RADIUS_M, generate_cubed_sphere

R_E = EARTH_RADIUS_M
IONO_ALT = 110e3


class SECSFit:
    """Fit divergence-free SECS using east and north ground fields."""

    def __init__(
        self,
        fpath,
        max_missing=50,
        epsilon=0.05,
        step=3.0,
        lat_min=50.0,
        lat_max=90.0,
        lon_min=0.0,
        lon_max=360.0,
        iono_height=IONO_ALT,
        crop=5.0,
    ):
        self.fpath = fpath
        self.max_missing = max_missing
        self.epsilon = epsilon
        self.step = step
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.iono_height = iono_height
        self.crop = crop

        self.model = None
        self.times = None
        self.obs_loc = None
        self.obs_B = None
        self.obs_std = None
        self.sec_amps = None
        self.sec_df_loc = None
        self.pole_grid = None
        self.pole_loc = None
        self.eval_grid = None
        self.eval_loc = None
        self.psi = None
        self.J = None
        self.B = None
        self.J_from_psi = None

        self._load_data()
        self._quality_control()

    def _load_data(self):
        ds = xr.open_dataset(self.fpath, engine="netcdf4")
        self.times = pd.to_datetime(
            {
                "year": ds["time_yr"].values,
                "month": ds["time_mo"].values,
                "day": ds["time_dy"].values,
                "hour": ds["time_hr"].values,
                "minute": ds["time_mt"].values,
                "second": ds["time_sc"].astype(int).values,
            },
            utc=True,
        ).to_numpy()
        self.dbe = ds["dbe_geo"].to_numpy()
        self.dbn = ds["dbn_geo"].to_numpy()
        self.glat = ds["glat"].to_numpy()
        self.glon = ds["glon"].to_numpy()
        ds.close()

    def _quality_control(self):
        nan_count = (~np.isfinite(self.dbe)).sum(axis=0) + (
            ~np.isfinite(self.dbn)
        ).sum(axis=0)
        good_mask = nan_count <= self.max_missing
        good_mask &= np.isfinite(self.glat[0]) & np.isfinite(self.glon[0])
        self.glat = self.glat[:, good_mask]
        self.glon = self.glon[:, good_mask]
        self.dbe = self.dbe[:, good_mask]
        self.dbn = self.dbn[:, good_mask]

        lat_mask = self.glat[0] >= self.lat_min - 10.0
        self.glat = self.glat[:, lat_mask]
        self.glon = self.glon[:, lat_mask]
        self.dbe = self.dbe[:, lat_mask]
        self.dbn = self.dbn[:, lat_mask]

        if self.glat.shape[1] == 0:
            raise ValueError("quality control removed every station")

        valid_per_time = np.isfinite(self.dbe).sum(axis=1) + np.isfinite(
            self.dbn
        ).sum(axis=1)
        if np.any(valid_per_time == 0):
            bad_time = int(np.flatnonzero(valid_per_time == 0)[0])
            raise ValueError(
                f"no finite horizontal observations at time index {bad_time}"
            )

    def _make_observations(self):
        ntime, nstations = self.dbe.shape
        self.obs_loc = np.column_stack(
            [
                self.glat[0],
                self.glon[0],
                np.full(nstations, R_E),
            ]
        )

        self.obs_B = np.zeros((ntime, nstations, 3), dtype=float)
        self.obs_B[:, :, 0] = np.nan_to_num(
            self.dbn, nan=0.0, posinf=0.0, neginf=0.0
        ) * 1e-9
        self.obs_B[:, :, 1] = np.nan_to_num(
            self.dbe, nan=0.0, posinf=0.0, neginf=0.0
        ) * 1e-9

        self.obs_std = np.ones_like(self.obs_B)
        self.obs_std[:, :, 2] = np.inf
        self.obs_std[:, :, 0][~np.isfinite(self.dbn)] = np.inf
        self.obs_std[:, :, 1][~np.isfinite(self.dbe)] = np.inf

    def _make_grid(self):
        pole_grid, eval_grid = generate_cubed_sphere(
            max_lat=self.lat_max,
            min_lat=self.lat_min,
            max_lon=self.lon_max,
            min_lon=self.lon_min,
            step=self.step,
            iono_height=self.iono_height,
            shift="on",
            crop=self.crop,
        )

        pole_mask = pole_grid.inside_box
        if not np.any(pole_mask):
            raise ValueError("the cubed-sphere grid contains no pole points")

        pole_lat = pole_grid.lat[pole_mask]
        pole_lon = pole_grid.lon[pole_mask] % 360.0
        pole_loc = np.column_stack(
            (pole_lat, pole_lon, np.full(pole_lat.size, pole_grid.radius))
        )

        eval_mask = eval_grid.inside_box
        if not np.any(eval_mask):
            raise ValueError("the shifted grid contains no evaluation points")

        eval_lat = eval_grid.lat[eval_mask]
        eval_lon = eval_grid.lon[eval_mask] % 360.0
        eval_loc = np.column_stack(
            (eval_lat, eval_lon, np.full(eval_lat.size, eval_grid.radius))
        )

        pole_grid.eval_lat = pole_lat
        pole_grid.eval_lon = pole_lon
        pole_grid.eval_loc = pole_loc
        pole_grid.eval_cell_area = pole_grid.cell_area_m2()[pole_mask]

        eval_grid.eval_lat = eval_lat
        eval_grid.eval_lon = eval_lon
        eval_grid.eval_loc = eval_loc
        eval_grid.eval_cell_area = eval_grid.cell_area_m2()[eval_mask]

        self.pole_grid = pole_grid
        self.pole_loc = pole_loc
        self.eval_grid = eval_grid
        self.eval_loc = eval_loc
        return eval_grid

    def fit(self):
        """Fit SECS amplitudes and predict J and B on the evaluation grid."""
        self._make_observations()
        if self.eval_grid is None:
            self._make_grid()

        self.model = secs.SECS(sec_df_loc=self.pole_loc)
        self.model.fit(
            self.obs_loc,
            self.obs_B,
            obs_std=self.obs_std,
            epsilon=self.epsilon,
        )
        self.sec_amps = self.model.sec_amps
        self.sec_df_loc = self.pole_loc
        self.psi = self.predict_psi()
        self.J = self.predict_J()
        self.B = self.predict_B()
        self.J_from_psi = self.predict_J_from_psi()
        return self

    def predict_psi(self, eval_loc=None, eval_grid=None):
        """Predict the stream function on the evaluation grid."""
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        if eval_loc is None:
            if self.eval_loc is None:
                self._make_grid()
            eval_loc = self.eval_loc
        if eval_grid is None:
            eval_grid = self.eval_grid
        self.psi, _, _ = get_psi(
            self.sec_df_loc,
            self.sec_amps,
            eval_loc,
            eval_grid,
            self.times,
        )
        return self.psi

    def predict_J_from_psi(self, psi=None, eval_loc=None):
        """Predict J by converting the stream function with ``psi_to_J``."""
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        if eval_loc is None:
            if self.eval_loc is None:
                self._make_grid()
            eval_loc = self.eval_loc
        if psi is None:
            if self.psi is None:
                psi = self.predict_psi(eval_loc=eval_loc)
            else:
                psi = self.psi
        self.J_from_psi = psi_to_J(psi, self.sec_df_loc, eval_loc)
        return self.J_from_psi

    def compare_J(self):
        """Compare pySECS ``J`` with ``psi_to_J`` currents."""
        if self.J is None:
            self.predict_J()
        if self.J_from_psi is None:
            self.predict_J_from_psi()
        difference = self.J - self.J_from_psi
        valid = np.isfinite(difference).all(axis=-1)
        if not np.any(valid):
            return {
                "difference": difference,
                "valid": valid,
                "max_abs": np.nan,
                "rms": np.nan,
            }
        values = difference[valid]
        return {
            "difference": difference,
            "valid": valid,
            "max_abs": float(np.max(np.abs(values))),
            "rms": float(np.sqrt(np.mean(values**2))),
        }

    def predict_J(self, eval_loc=None):
        """Predict flat ``(time, evaluation_point, east/north)`` currents."""
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        if eval_loc is None:
            if self.eval_loc is None:
                self._make_grid()
            eval_loc = self.eval_loc
        model_J = self.model.predict_J(eval_loc)
        self.J = model_J[..., [1, 0]]
        return self.J

    def predict_B(self, eval_loc=None):
        """Predict flat ``(time, evaluation_point, north/east/down)`` fields."""
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        if eval_loc is None:
            if self.eval_loc is None:
                self._make_grid()
            eval_loc = self.eval_loc
        self.B = self.model.predict_B(eval_loc)
        return self.B

    def run(self):
        """Fit and return ``(sec_amps, J, B, eval_grid)``."""
        self.fit()
        return self.sec_amps, self.J, self.B, self.eval_grid


def set_up_secs(*args, **kwargs):
    """Fit SECS and return ``(sec_amps, J, B, eval_grid)``."""
    return SECSFit(*args, **kwargs).run()
