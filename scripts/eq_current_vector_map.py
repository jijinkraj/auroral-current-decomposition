"""Plot equivalent-current vectors on a rotating UT polar map."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.patches import PathPatch
from cartopy.feature import NaturalEarthFeature


def plot_eq_current_vector_map(
    evaluation_locations,
    current_density,
    timestamp,
    reference_magnitude=3,
    reference_length_degrees=3.0,
    minimum_latitude=50.0,
    land_resolution="110m",
    marker_locations=None,
    save_path=None,
):
    """Plot one frame of equivalent-current vectors on a UT-rotated map.

    Parameters
    ----------
    evaluation_locations : ndarray (N, 3 [lat, lon, r])
        Evaluation locations in degrees and metres.
    current_density : ndarray (N, 2 [east, north])
        Horizontal sheet current density in A/m for one timestamp.
    timestamp : datetime
        Timestamp of the frame; sets the UT rotation.
    reference_magnitude : float, optional
        Common reference magnitude in A/m.
    reference_length_degrees : float, optional
        Arrow length in colatitude degrees for ``reference_magnitude``.
    minimum_latitude : float, optional
        Lowest latitude drawn.
    land_resolution : str, optional
        Natural Earth land resolution.
    marker_locations : None or list of (name, lat, lon), optional
        Optional labelled geographic markers.
    save_path : path-like, optional
        If given, save the figure instead of displaying it.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure when displaying, otherwise ``None``.
    """
    evaluation_locations = np.asarray(evaluation_locations, dtype=float)
    current_density = np.asarray(current_density, dtype=float)

    if evaluation_locations.ndim != 2 or evaluation_locations.shape[1] != 3:
        raise ValueError(
            "evaluation_locations must have shape (N, 3 [lat, lon, r])"
        )
    if current_density.ndim == 3 and current_density.shape[0] == 1:
        current_density = current_density[0]
    if current_density.ndim != 2 or current_density.shape[1] != 2:
        raise ValueError(
            "current_density must have shape (N, 2 [east, north]) for one frame"
        )
    if current_density.shape[0] != evaluation_locations.shape[0]:
        raise ValueError(
            "current_density and evaluation_locations must contain the same "
            "number of points"
        )
    if reference_magnitude <= 0.0:
        raise ValueError("reference_magnitude must be positive")
    if reference_length_degrees <= 0.0:
        raise ValueError("reference_length_degrees must be positive")

    latitude_degrees = evaluation_locations[:, 0]
    longitude_degrees = evaluation_locations[:, 1]
    eastward_current = current_density[:, 0]
    northward_current = current_density[:, 1]
    colatitude_degrees = 90.0 - latitude_degrees
    maximum_colatitude_degrees = 90.0 - minimum_latitude

    valid_point_mask = (
        np.isfinite(latitude_degrees)
        & np.isfinite(longitude_degrees)
        & np.isfinite(eastward_current)
        & np.isfinite(northward_current)
        & (colatitude_degrees >= 0.0)
        & (colatitude_degrees <= maximum_colatitude_degrees)
    )
    if not np.any(valid_point_mask):
        raise ValueError("no valid current vectors within minimum_latitude")

    latitude_degrees = latitude_degrees[valid_point_mask]
    longitude_degrees = longitude_degrees[valid_point_mask]
    eastward_current = eastward_current[valid_point_mask]
    northward_current = northward_current[valid_point_mask]
    colatitude_degrees = colatitude_degrees[valid_point_mask]

    current_magnitude = np.hypot(eastward_current, northward_current)
    maximum_frame_magnitude = float(np.max(current_magnitude))
    arrow_colors = np.where(eastward_current > 0.0, "red", "blue")

    rotation_shift_degrees = 15.0 * (
        timestamp.hour
        + timestamp.minute / 60.0
        + timestamp.second / 3600.0
    )
    theta_radians = np.radians(longitude_degrees + rotation_shift_degrees)
    arrow_positions = np.column_stack([theta_radians, colatitude_degrees])

    figure = plt.figure(figsize=(7, 7))
    axis = figure.add_subplot(projection="polar")
    figure.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)

    _setup_ut_dial(axis, minimum_latitude)
    axis.set_title(
        f"{timestamp.strftime('%Y-%m-%d %H:%M UT')}",
        pad=45,
        fontsize=13.5,
        fontweight="bold",
    )
    _draw_rotating_world_map(
        axis,
        rotation_shift_degrees,
        land_resolution,
    )

    figure.canvas.draw()
    center_pixel = axis.transData.transform([[0.0, 0.0]])
    rim_pixel = axis.transData.transform([[0.0, maximum_colatitude_degrees]])
    pixel_radius = float(np.hypot(*(rim_pixel - center_pixel)[0]))

    position_pixels = axis.transData.transform(arrow_positions)
    east_pixel_direction = (
        axis.transData.transform(arrow_positions + [1e-3, 0.0])
        - position_pixels
    ) / 1e-3
    north_pixel_direction = -(
        axis.transData.transform(arrow_positions + [0.0, 1.0])
        - position_pixels
    )
    vector_pixels = (
        eastward_current[:, None] * east_pixel_direction
        + northward_current[:, None] * north_pixel_direction
    )
    pixel_lengths = np.hypot(vector_pixels[:, 0], vector_pixels[:, 1])
    unit_pixel_vectors = vector_pixels / np.maximum(pixel_lengths, 1e-12)[:, None]

    target_pixel_lengths = (
        (reference_length_degrees / maximum_colatitude_degrees)
        * pixel_radius
        * (current_magnitude / reference_magnitude)
    )[:, None]
    arrow_tips = _display_offset_to_data_coordinates(
        axis,
        arrow_positions,
        unit_pixel_vectors * target_pixel_lengths,
    )
    arrow_deltas = arrow_tips - arrow_positions

    axis.quiver(
        theta_radians,
        colatitude_degrees,
        arrow_deltas[:, 0],
        arrow_deltas[:, 1],
        color=arrow_colors,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        pivot="tail",
        width=0.003,
        headwidth=3.5,
        headlength=5.0,
        headaxislength=4.5,
        zorder=4,
        clip_on=True,
    )

    if marker_locations is not None:
        locations = (
            [marker_locations]
            if isinstance(marker_locations[0], str)
            else marker_locations
        )
        for marker_name, marker_latitude, marker_longitude in locations:
            marker_colatitude = 90.0 - marker_latitude
            if marker_colatitude > maximum_colatitude_degrees:
                continue
            marker_theta = np.radians(
                marker_longitude + rotation_shift_degrees
            )
            axis.plot(
                marker_theta,
                marker_colatitude,
                marker="*",
                ms=14,
                mfc="yellow",
                mec="k",
                zorder=6,
            )
            axis.annotate(
                marker_name,
                (marker_theta, marker_colatitude),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white",
                    alpha=0.8,
                    lw=0.5,
                ),
            )

    reference_arrow_length = 2.0 * (
        reference_length_degrees / maximum_colatitude_degrees * pixel_radius
        / axis.get_window_extent().width
    )
    reference_arrow_start = (0.90, 0.0)
    reference_arrow_end = (
        reference_arrow_start[0] + reference_arrow_length,
        reference_arrow_start[1],
    )
    axis.annotate(
        "",
        xy=reference_arrow_end,
        xytext=reference_arrow_start,
        xycoords=axis.transAxes,
        textcoords=axis.transAxes,
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            lw=4,
            mutation_scale=20,
        ),
        annotation_clip=False,
        zorder=7,
    )
    axis.text(
        (reference_arrow_start[0] + reference_arrow_end[0]) / 2.0,
        reference_arrow_start[1] + 0.022,
        f"{maximum_frame_magnitude:.2f} A/m",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        zorder=7,
    )

    axis.set_thetalim(0, 2 * np.pi)
    axis.set_rlim(0, maximum_colatitude_degrees)

    if save_path:
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return None

    plt.show()
    return figure


def _setup_ut_dial(axis, minimum_latitude):
    maximum_colatitude_degrees = 90.0 - minimum_latitude

    axis.set_theta_zero_location("S")
    axis.set_theta_direction(1)
    axis.set_rlabel_position(157.5)

    latitude_ticks = np.arange(minimum_latitude, 90, 10)
    axis.set_yticks(90 - latitude_ticks)
    axis.set_yticklabels(
        [f"{int(latitude)}°" for latitude in latitude_ticks],
        fontsize=12,
        color="black",
        fontweight="bold",
    )
    axis.set_rlim(0, maximum_colatitude_degrees)
    axis.tick_params(axis="y", pad=10)

    hours = np.arange(0, 24, 3)
    hour_angles_degrees = 15 * hours
    axis.set_xticks(np.radians(hour_angles_degrees))
    axis.set_xticklabels([])

    for hour_angle, hour in zip(hour_angles_degrees, hours):
        label_rotation = (
            hour_angle + 180 if 120 <= hour_angle <= 240 else hour_angle
        )
        axis.text(
            np.radians(hour_angle),
            maximum_colatitude_degrees + 4,
            f"{hour:02d} LT",
            ha="center",
            va="center",
            fontsize=12,
            color="black",
            fontweight="bold",
            rotation=label_rotation,
            rotation_mode="anchor",
        )

    axis.grid(True, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)


def _draw_rotating_world_map(
    axis,
    rotation_shift_degrees,
    land_resolution="110m",
):
    def to_polar_path(coordinates):
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[0] < 2:
            return None

        longitude_degrees = coordinates[:, 0]
        latitude_degrees = coordinates[:, 1]
        finite = np.isfinite(longitude_degrees) & np.isfinite(
            latitude_degrees
        )
        if finite.sum() < 2:
            return None

        longitude_degrees = longitude_degrees[finite]
        latitude_degrees = latitude_degrees[finite]
        theta_radians = np.radians(
            longitude_degrees + rotation_shift_degrees
        )
        colatitude_degrees = 90.0 - latitude_degrees

        theta_radians = np.unwrap(theta_radians)
        theta_radians -= (
            2.0
            * np.pi
            * np.round(np.median(theta_radians) / (2.0 * np.pi))
        )

        vertices = np.column_stack(
            [theta_radians, colatitude_degrees]
        )
        codes = np.full(vertices.shape[0], mpath.Path.LINETO)
        codes[0] = mpath.Path.MOVETO
        return mpath.Path(vertices, codes)

    coastline_geometries = NaturalEarthFeature(
        "physical",
        "coastline",
        land_resolution,
    ).geometries()
    for coastline_geometry in coastline_geometries:
        for coastline_line in getattr(
            coastline_geometry,
            "geoms",
            [coastline_geometry],
        ):
            polar_path = to_polar_path(coastline_line.coords)
            if polar_path is not None:
                axis.add_patch(
                    PathPatch(
                        polar_path,
                        transform=axis.transData,
                        facecolor="none",
                        edgecolor="0.35",
                        linewidth=1.5,
                        zorder=1.2,
                        clip_on=True,
                    )
                )


def _display_offset_to_data_coordinates(axis, start_positions, pixel_offsets):
    """Return data coordinates for start positions plus display-pixel offsets."""

    def forward_transform(points):
        return axis.transData.transform(points)

    target_pixels = forward_transform(start_positions) + pixel_offsets
    current_positions = start_positions.astype(float)
    epsilon = 1e-6

    for _ in range(8):
        transformed_positions = forward_transform(current_positions)
        residuals = target_pixels - transformed_positions
        if np.max(np.abs(residuals)) < 1e-3:
            break

        theta_perturbed = current_positions.copy()
        theta_perturbed[:, 0] += epsilon
        radius_perturbed = current_positions.copy()
        radius_perturbed[:, 1] += epsilon
        jacobian_00 = (
            forward_transform(theta_perturbed)[:, 0]
            - transformed_positions[:, 0]
        ) / epsilon
        jacobian_01 = (
            forward_transform(radius_perturbed)[:, 0]
            - transformed_positions[:, 0]
        ) / epsilon
        jacobian_10 = (
            forward_transform(theta_perturbed)[:, 1]
            - transformed_positions[:, 1]
        ) / epsilon
        jacobian_11 = (
            forward_transform(radius_perturbed)[:, 1]
            - transformed_positions[:, 1]
        ) / epsilon
        determinant = jacobian_00 * jacobian_11 - jacobian_01 * jacobian_10
        invertible = np.abs(determinant) > 1e-12
        theta_steps = np.where(
            invertible,
            (
                residuals[:, 0] * jacobian_11
                - residuals[:, 1] * jacobian_01
            )
            / determinant,
            0.0,
        )
        radius_steps = np.where(
            invertible,
            (
                jacobian_00 * residuals[:, 1]
                - jacobian_10 * residuals[:, 0]
            )
            / determinant,
            0.0,
        )
        current_positions = current_positions + np.stack(
            [theta_steps, radius_steps],
            axis=1,
        )

    return np.stack(
        [current_positions[:, 0], current_positions[:, 1]],
        axis=1,
    )
