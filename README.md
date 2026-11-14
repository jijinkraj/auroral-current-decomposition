# Decomposition of Auroral Currents Using SECS and MNOC

This repository provides a framework for decomposing auroral equivalent currents using ground-based magnetometer observations.

The workflow consists of three main steps:

1. **Magnetic field inversion:** SuperMAG ground magnetometer observations are used to reconstruct the equivalent current system using the Spherical Elementary Current Systems (SECS) technique.
2. **Equivalent current function:** The reconstructed SECS amplitudes are used to derive the equivalent current function (stream function).
3. **Orthogonal decomposition:** The equivalent current function is decomposed using the area-weighted Method of Natural Orthogonal Components (MNOC) to identify dominant spatial and temporal current structures.

The resulting modes are examined in terms of their temporal evolution and potential association with different magnetospheric current-driving processes. In particular, the leading mode is investigated as a representation of directly driven (DD) currents, while the combined contributions of subsequent modes are examined in relation to substorm unloading processes.

This framework aims to investigate the evolution of auroral equivalent currents and distinguish their directly driven and unloading-related contributions using ground-based magnetometer observations.

## Key Components

- `current_func.py` — Calculates the equivalent current function (stream function) and returns the corresponding current series.
- `set_up_secs.py` — Fits the SECS model to SuperMAG observations and predicts the equivalent current function.
- `generate_cubed_sphere.py` — Generates the cubed-sphere grid for SECS poles and evaluation points.
- `mnoc.py` — Decomposes the equivalent current function into spatial modes and time coefficients using area-weighted MNOC.

## Citations

If you use this code in academic work, please cite the following packages and the methods they implement:

- **pysecs (software)** — Lucas, G., Rigler, E. J., & Murphy, B. (2024). *greglucas/pysecs: REL: v0.3.0* (v0.3.0). Zenodo. https://doi.org/10.5281/zenodo.14511596
- **SECS method** — Amm, O. (1997). Ionospheric Elementary Current Systems in Spherical Coordinates and Their Application. *Journal of Geomagnetism and Geoelectricity*, 49(7), 947–955. https://doi.org/10.5636/jgg.49.947
- **Cubed-sphere grid** — Ronchi, C., Iacono, R., & Paolucci, P. S. (1996). The "cubed sphere": A new method for the solution of partial differential equations in spherical geometry. *Journal of Computational Physics*, 124(1), 93–114. https://doi.org/10.1006/jcph.1996.0047
