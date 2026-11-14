"""Generate a regional cubed-sphere grid on a spherical Earth.

The projection and coordinate transformations are adapted from the ``secsy``
package by Karl Laundal (MIT licence), which implements the cubed-sphere
equations of Ronchi, Iacono & Paolucci (1996).

The main entry point is :func:`generate_cubed_sphere`.
"""

import numpy as np

D2R = np.pi / 180.0
R2D = 180.0 / np.pi
EARTH_RADIUS_M = 6371.2e3
CELL_SIZE_M = 111.0e3


def _enu_to_ecef(vector, lon, lat):
    ph = lon * D2R
    th = (90.0 - lat) * D2R
    east = np.array([-np.sin(ph), np.cos(ph), 0.0])
    north = np.array([-np.cos(th) * np.cos(ph), -np.cos(th) * np.sin(ph), np.sin(th)])
    up = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return vector[0] * east + vector[1] * north + vector[2] * up


class CSprojection:
    """Cubed-sphere projection centred on ``position = (lon, lat)``.

    Parameters
    ----------
    position : sequence of float
        Longitude and latitude in degrees of the cube-face centre.
    orientation : float or sequence of float, optional
        If scalar, the angle in degrees of the xi axis measured anticlockwise
        from east. If a two-element sequence, the eastward and northward
        components of a vector aligned with the xi axis.
    """

    def __init__(self, position, orientation=0.0):
        self.position = np.asarray(position, dtype=float)
        self.lon0, self.lat0 = self.position

        orientation = np.asarray(orientation, dtype=float)
        if orientation.size == 1:
            self.orientation = np.array(
                [np.cos(orientation * D2R), np.sin(orientation * D2R)]
            ).ravel()
        elif orientation.size == 2:
            self.orientation = orientation / np.linalg.norm(orientation)
        else:
            raise ValueError("orientation must be a scalar or a two-element vector")

        self.z = np.array(
            [
                np.cos(self.lat0 * D2R) * np.cos(self.lon0 * D2R),
                np.cos(self.lat0 * D2R) * np.sin(self.lon0 * D2R),
                np.sin(self.lat0 * D2R),
            ]
        )
        self.y = _enu_to_ecef(
            np.array([self.orientation[0], self.orientation[1], 0.0]),
            self.lon0,
            self.lat0,
        )
        self.x = np.cross(self.y, self.z)
        self.R_geo2local = np.vstack((self.x, self.y, self.z))
        self.R_local2geo = self.R_geo2local.T

    def geo2local(self, lon, lat, reverse=False):
        lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
        if lon.shape != lat.shape:
            raise ValueError("lon and lat must have the same shape")
        shape = lon.shape
        lat_r = lat.ravel() * D2R
        lon_r = lon.ravel() * D2R
        r = np.vstack(
            (
                np.cos(lat_r) * np.cos(lon_r),
                np.cos(lat_r) * np.sin(lon_r),
                np.sin(lat_r),
            )
        )
        r_rot = (self.R_local2geo if reverse else self.R_geo2local).dot(r)
        new_lat = np.arcsin(np.clip(r_rot[2], -1.0, 1.0)) * R2D
        new_lon = np.arctan2(r_rot[1], r_rot[0]) * R2D
        return new_lon.reshape(shape), new_lat.reshape(shape)

    def local2geo(self, lon, lat, reverse=False):
        return self.geo2local(lon, lat, reverse=not reverse)

    def geo2cube(self, lon, lat):
        lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
        if lon.shape != lat.shape:
            raise ValueError("lon and lat must have the same shape")
        shape = lon.shape
        local_lon, local_lat = self.geo2local(lon.ravel(), lat.ravel())
        theta = (90.0 - local_lat) * D2R
        phi = local_lon * D2R
        X = np.tan(theta) * np.sin(phi)
        Y = -np.tan(theta) * np.cos(phi)
        xi = np.arctan(X)
        eta = np.arctan(Y)
        xi[local_lat < -1e-12] = np.nan
        eta[local_lat < -1e-12] = np.nan
        return xi.reshape(shape), eta.reshape(shape)

    def cube2geo(self, xi, eta):
        xi, eta = np.asarray(xi, dtype=float), np.asarray(eta, dtype=float)
        if xi.shape != eta.shape:
            raise ValueError("xi and eta must have the same shape")
        shape = xi.shape
        xi_f, eta_f = xi.ravel(), eta.ravel()
        X, Y = np.tan(xi_f), np.tan(eta_f)
        phi = np.arctan2(X, -Y)
        theta = np.arctan2(np.hypot(X, Y), 1.0)
        local_lat = 90.0 - theta * R2D
        lon, lat = self.local2geo(phi * R2D, local_lat)
        return lon.reshape(shape), lat.reshape(shape)


class CubedSphereGrid:
    """A regular grid in cubed-sphere coordinates ``(xi, eta)``.

    Attributes
    ----------
    xi, eta : ndarray
        Cell-centre coordinates in radians.
    lon, lat : ndarray
        Geocentric cell-centre coordinates in degrees.
    xi_mesh, eta_mesh : ndarray
        Cell-corner coordinates in radians.
    lon_mesh, lat_mesh : ndarray
        Geocentric cell-corner coordinates in degrees.
    shape : tuple
        ``(n_eta, n_xi)`` number of grid cells.
    inside_box : ndarray of bool
        True for cell centres inside the requested lat/lon box.
    """

    def __init__(
        self, projection, xi_edges, eta_edges, radius, step_m, iono_height, bounds
    ):
        self.projection = projection
        self.radius = radius
        self.step_m = step_m
        self.iono_height = iono_height
        self.bounds = bounds
        self.xi_edges = np.asarray(xi_edges, dtype=float)
        self.eta_edges = np.asarray(eta_edges, dtype=float)
        self.dxi = self.xi_edges[1] - self.xi_edges[0]
        self.deta = self.eta_edges[1] - self.eta_edges[0]
        self.NW = self.xi_edges.size - 1
        self.NL = self.eta_edges.size - 1
        self.shape = (self.NL, self.NW)
        self.size = self.NL * self.NW

        self.xi_mesh, self.eta_mesh = np.meshgrid(
            self.xi_edges, self.eta_edges, indexing="xy"
        )
        self.lon_mesh, self.lat_mesh = projection.cube2geo(
            self.xi_mesh, self.eta_mesh
        )

        xi_centres = 0.5 * (self.xi_edges[:-1] + self.xi_edges[1:])
        eta_centres = 0.5 * (self.eta_edges[:-1] + self.eta_edges[1:])
        self.xi, self.eta = np.meshgrid(xi_centres, eta_centres, indexing="xy")
        self.lon, self.lat = projection.cube2geo(self.xi, self.eta)
        self._inside_box = np.ones(self.shape, dtype=bool)

    @property
    def inside_box(self):
        return self._inside_box

    def cell_area_m2(self):
        X, Y = np.tan(self.xi), np.tan(self.eta)
        delta = 1.0 + X**2 + Y**2
        dxi, deta = self.dxi, self.deta
        return (
            self.radius**2
            * dxi
            * deta
            / (delta ** 1.5 * np.cos(self.xi) ** 2 * np.cos(self.eta) ** 2)
        )

    def __repr__(self):
        lon0, lat0 = self.projection.lon0, self.projection.lat0
        return (
            "{} x {} cubed-sphere grid\n"
            "Centre: lon, lat = {:.2f}, {:.2f}\n"
            "xi range: [{:.4f}, {:.4f}] rad\n"
            "eta range: [{:.4f}, {:.4f}] rad\n"
            "Requested cell size: {:.1f} m"
        ).format(
            self.shape[0],
            self.shape[1],
            lon0,
            lat0,
            self.xi_edges[0],
            self.xi_edges[-1],
            self.eta_edges[0],
            self.eta_edges[-1],
            self.step_m,
        )


def _longitude_span(min_lon, max_lon):
    raw = max_lon - min_lon
    if abs(raw) >= 360.0 - 1e-9:
        span = 360.0
    elif raw < 0:
        span = raw + 360.0
    else:
        span = raw
    centre = min_lon + 0.5 * span
    centre = (centre + 180.0) % 360.0 - 180.0
    return centre, span


def _inside_longitude(lon, min_lon, span, tol=1e-10):
    if span >= 360.0 - 1e-9:
        return np.ones_like(lon, dtype=bool)
    rel = np.mod(lon - min_lon, 360.0)
    return (rel <= span + tol) | (np.isclose(rel, 360.0, atol=tol))


def _cube_extent(projection, min_lon, min_lat, max_lat, lon_span):
    n_lon = max(181, int(np.ceil(lon_span / 0.5)) + 1)
    n_lat = max(181, int(np.ceil((max_lat - min_lat) / 0.5)) + 1)

    edge_lons = min_lon + np.linspace(0.0, lon_span, n_lon)
    edge_lats = np.linspace(min_lat, max_lat, n_lat)
    sample_lons = np.concatenate(
        (
            edge_lons,
            edge_lons,
            np.full(n_lat, min_lon),
            np.full(n_lat, min_lon + lon_span),
        )
    )
    sample_lats = np.concatenate(
        (
            np.full(n_lon, min_lat),
            np.full(n_lon, max_lat),
            edge_lats,
            edge_lats,
        )
    )

    sample_xi, sample_eta = projection.geo2cube(sample_lons, sample_lats)
    valid = np.isfinite(sample_xi) & np.isfinite(sample_eta)
    if not np.any(valid):
        raise ValueError("the requested region is not on the visible cube face")
    if not np.all(valid):
        raise ValueError(
            "the requested region extends beyond the half-sphere of the "
            "cubed-sphere face; choose a smaller region or an explicit center"
        )

    return (
        float(np.nanmin(sample_xi)),
        float(np.nanmax(sample_xi)),
        float(np.nanmin(sample_eta)),
        float(np.nanmax(sample_eta)),
    )


def _shifted_edges(edges, lower, upper, spacing):
    centres = edges[1:-1]
    if centres.size == 0:
        raise ValueError("the grid is too small to create a shifted eval_grid")

    margin = 0.5 * spacing
    start = int(np.searchsorted(centres, lower - margin, side="left"))
    stop = int(np.searchsorted(centres, upper + margin, side="right"))
    start = max(0, start)
    stop = min(centres.size, stop)

    if stop <= start:
        nearest = int(
            np.clip(
                np.searchsorted(centres, 0.5 * (lower + upper)),
                0,
                centres.size - 1,
            )
        )
        start, stop = nearest, nearest + 1

    selected = centres[start:stop]
    return np.concatenate(([selected[0] - 0.5 * spacing], selected + 0.5 * spacing))


def _resolve_step(step):
    step = float(step)
    if step <= 0.0:
        raise ValueError("step must be positive")
    if step < CELL_SIZE_M:
        multiplier = int(round(step))
        if not np.isclose(step, multiplier, rtol=0.0, atol=1e-12):
            raise ValueError(
                "step values below 111000 m must be whole multiples of "
                "111000 m (for example step=1, step=2, ...)"
            )
        return multiplier * CELL_SIZE_M, multiplier
    return step, step / CELL_SIZE_M


def generate_cubed_sphere(
    max_lat,
    min_lat,
    max_lon,
    min_lon,
    step=111.0e3,
    iono_height=110.0e3,
    orientation=0.0,
    center=None,
    shift="off",
    crop=5.0,
):
    """Generate a cubed-sphere grid covering a latitude/longitude box.

    Parameters
    ----------
    max_lat, min_lat : float
        Northern and southern limits of the region in degrees.
    max_lon, min_lon : float
        Eastern and western limits in degrees. A region crossing the dateline
        can be given with ``min_lon > max_lon`` (for example ``170`` to
        ``-170``). ``min_lon=-180, max_lon=180`` denotes all longitudes.
    step : float, optional
        Cell size in metres. Values below 111000 are interpreted as multiples
        of the 111000 m reference cell, so ``step=1`` gives 111000 m and
        ``step=2`` gives 222000 m. Values of 111000 and above are interpreted
        as metres. The default, 111000, gives a 111000 m cell size.
    iono_height : float, optional
        Ionospheric height above the Earth's surface in metres. Default is
        110000 m. The grid is placed at ``EARTH_RADIUS_M + iono_height``.
    orientation : float or sequence of float, optional
        Orientation of the cubed-sphere xi axis. The default, 0, makes xi
        point eastward and eta northward at the projection centre.
    center : sequence of float, optional
        Explicit projection centre as ``(lon, lat)``. By default it is the
        centre of the requested box, snapped to a pole for polar boxes.
    shift : {"off", "on"}, optional
        If ``"on"``, also generate ``eval_grid`` with points shifted by half
        a cell in both xi and eta, so they lie between the points of ``grid``.
        Default is ``"off"``.
    crop : float, optional
        Degrees removed from each latitude and longitude bound when creating
        ``eval_grid``. Default is 5. Only used when ``shift="on"``.

    Returns
    -------
    grid : CubedSphereGrid
        Object containing cell-centre and cell-corner coordinates, together
        with ``grid.lon``, ``grid.lat`` and ``grid.inside_box``.
    eval_grid : CubedSphereGrid
        Additional shifted grid. Only returned when ``shift="on"``, as
        ``grid, eval_grid = generate_cubed_sphere(..., shift="on")``.
    """
    if shift not in ("off", "on"):
        raise ValueError("shift must be 'off' or 'on'")
    max_lat, min_lat = float(max_lat), float(min_lat)
    max_lon, min_lon = float(max_lon), float(min_lon)
    step, step_multiplier = _resolve_step(step)
    iono_height = float(iono_height)
    radius = EARTH_RADIUS_M + iono_height

    if not (-90.0 <= min_lat < max_lat <= 90.0):
        raise ValueError("require -90 <= min_lat < max_lat <= 90")
    if radius <= 0.0:
        raise ValueError("iono_height is too large and negative")

    lon_centre, lon_span = _longitude_span(min_lon, max_lon)
    if lon_span <= 0.0:
        raise ValueError("longitude span must be positive")

    if center is None:
        if max_lat >= 90.0 - 1e-12:
            centre_lat = 90.0
        elif min_lat <= -90.0 + 1e-12:
            centre_lat = -90.0
        elif lon_span >= 360.0 - 1e-9 and min_lat >= 0.0:
            centre_lat = 90.0
        elif lon_span >= 360.0 - 1e-9 and max_lat <= 0.0:
            centre_lat = -90.0
        else:
            centre_lat = 0.5 * (min_lat + max_lat)
        centre = (lon_centre, centre_lat)
    else:
        centre = center

    projection = CSprojection(centre, orientation)

    xi_min, xi_max, eta_min, eta_max = _cube_extent(
        projection, min_lon, min_lat, max_lat, lon_span
    )

    dxi = deta = step / radius
    if dxi >= np.pi / 2:
        raise ValueError("step is too large for the given radius")
    xi_mid = 0.5 * (xi_min + xi_max)
    eta_mid = 0.5 * (eta_min + eta_max)
    n_xi = max(1, int(np.ceil((xi_max - xi_min) / dxi)))
    n_eta = max(1, int(np.ceil((eta_max - eta_min) / deta)))
    xi_edges = xi_mid + (np.arange(n_xi + 1) - 0.5 * n_xi) * dxi
    eta_edges = eta_mid + (np.arange(n_eta + 1) - 0.5 * n_eta) * deta

    bounds = {
        "max_lat": max_lat,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "min_lon": min_lon,
        "lon_centre": lon_centre,
        "lon_span": lon_span,
        "step_multiplier": step_multiplier,
        "iono_height": iono_height,
    }
    grid = CubedSphereGrid(
        projection, xi_edges, eta_edges, radius, step, iono_height, bounds
    )
    grid._inside_box = (
        (grid.lat >= min_lat)
        & (grid.lat <= max_lat)
        & _inside_longitude(grid.lon, min_lon, lon_span)
    )

    if shift == "on":
        crop = float(crop)
        if crop < 0.0:
            raise ValueError("crop must be non-negative")

        eval_min_lat = min_lat + crop
        eval_max_lat = max_lat - crop
        if eval_min_lat >= eval_max_lat:
            raise ValueError("crop is too large for the latitude range")

        if lon_span - 2.0 * crop <= 0.0:
            raise ValueError("crop is too large for the longitude range")

        eval_min_lon = (min_lon + crop + 180.0) % 360.0 - 180.0
        eval_max_lon = (min_lon + lon_span - crop + 180.0) % 360.0 - 180.0
        eval_lon_centre, eval_lon_span = _longitude_span(eval_min_lon, eval_max_lon)

        eval_xi_min, eval_xi_max, eval_eta_min, eval_eta_max = _cube_extent(
            projection, eval_min_lon, eval_min_lat, eval_max_lat, eval_lon_span
        )
        eval_xi_edges = _shifted_edges(xi_edges, eval_xi_min, eval_xi_max, dxi)
        eval_eta_edges = _shifted_edges(eta_edges, eval_eta_min, eval_eta_max, deta)

        eval_bounds = {
            "max_lat": eval_max_lat,
            "min_lat": eval_min_lat,
            "max_lon": eval_max_lon,
            "min_lon": eval_min_lon,
            "lon_centre": eval_lon_centre,
            "lon_span": eval_lon_span,
            "step_multiplier": step_multiplier,
            "iono_height": iono_height,
            "crop": crop,
        }
        eval_grid = CubedSphereGrid(
            projection,
            eval_xi_edges,
            eval_eta_edges,
            radius,
            step,
            iono_height,
            eval_bounds,
        )
        eval_grid._inside_box = (
            (eval_grid.lat >= eval_min_lat)
            & (eval_grid.lat <= eval_max_lat)
            & _inside_longitude(eval_grid.lon, eval_min_lon, eval_lon_span)
        )
        return grid, eval_grid

    return grid


if __name__ == "__main__":
    example = generate_cubed_sphere(
        max_lat=90.0,
        min_lat=60.0,
        max_lon=180.0,
        min_lon=-180.0,
        step=1,
    )
    print(example)
    print("Number of grid cells inside the box:", example.inside_box.sum())
