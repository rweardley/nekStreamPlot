#!/usr/bin/env python3
"""
Read a uniform 2D slice exported by interpolate_uniform_grid.py, reconstruct
the velocity arrays, solve a stream-function Poisson problem, and plot both
velocity streamlines and stream-function contours.

For an x = constant plane, use y,z as the in-plane coordinates and v,w as the
corresponding velocity components:

  python plot_csv_streamfunction.py uniform.csv --shape 1 1001 1001

The CSV must contain a complete uniform Cartesian plane with columns x,y,z and
the requested velocity components.
"""

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import factorized


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot streamlines and stream-function contours from a 2D CSV slice."
    )
    parser.add_argument("input_csv", help="CSV from interpolate_uniform_grid.py")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        required=True,
        metavar=("NX", "NY", "NZ"),
        help="Numbers of CSV points in x, y, z. Exactly one must be 1.",
    )
    parser.add_argument(
        "--output",
        help="Output PNG path (default: <input>_streamfunction.png).",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1.5,
        help="Matplotlib streamline density (default: 1.5).",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=25,
        help="Number of stream-function contour levels (default: 25).",
    )
    parser.add_argument("--show", action="store_true", help="Also display the figure.")
    return parser.parse_args()


def check_uniform(values, name):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError(f"{name} needs at least two distinct coordinates.")
    spacing = np.diff(values)
    if not np.allclose(spacing, spacing[0], rtol=1.0e-8, atol=1.0e-12):
        raise ValueError(f"{name} coordinates are not uniformly spaced.")
    return spacing[0]


def plane_definition(shape):
    """Return the in-plane coordinates and velocity components from NX,NY,NZ."""
    names = ("x", "y", "z")
    velocity = ("u", "v", "w")
    collapsed = [index for index, size in enumerate(shape) if size == 1]
    if len(collapsed) != 1:
        raise ValueError(
            "This script plots one 2D plane, so exactly one entry in --shape "
            "must be 1; for example, --shape 1 ny nz for an x-constant plane."
        )
    in_plane = [index for index in range(3) if index != collapsed[0]]
    return (
        names[in_plane[0]],
        names[in_plane[1]],
        velocity[in_plane[0]],
        velocity[in_plane[1]],
    )


def make_plane_arrays(data, horizontal, vertical, u_name, v_name, shape):
    required = (horizontal, vertical, u_name, v_name)
    missing = [name for name in required if name not in data.dtype.names]
    if missing:
        available = ", ".join(data.dtype.names)
        raise ValueError(f"Missing CSV columns: {', '.join(missing)}. Available: {available}")

    x = np.unique(data[horizontal])
    y = np.unique(data[vertical])
    dx = check_uniform(x, horizontal)
    dy = check_uniform(y, vertical)
    nx, ny = len(x), len(y)

    expected_size = int(np.prod(shape))
    if data.size != expected_size:
        raise ValueError(
            f"--shape specifies {expected_size} points, but the CSV has {data.size} rows."
        )
    if (nx, ny) != (shape["xyz".index(horizontal)], shape["xyz".index(vertical)]):
        raise ValueError(
            f"Coordinate counts are {nx} in {horizontal} and {ny} in {vertical}, "
            f"which disagree with --shape {tuple(shape)}."
        )
    if data.size != nx * ny:
        raise ValueError(
            f"Expected one value at every {horizontal},{vertical} pair "
            f"({nx} x {ny} = {nx * ny}), but found {data.size} rows. "
            "Select a single 2D slice."
        )

    ix = np.searchsorted(x, data[horizontal])
    iy = np.searchsorted(y, data[vertical])
    if np.unique(iy * nx + ix).size != data.size:
        raise ValueError("The CSV has duplicate coordinate pairs.")

    # Matplotlib uses arrays indexed as [vertical, horizontal].
    u = np.full((ny, nx), np.nan)
    v = np.full((ny, nx), np.nan)
    u[iy, ix] = data[u_name]
    v[iy, ix] = data[v_name]
    if np.isnan(u).any() or np.isnan(v).any():
        raise ValueError("The CSV does not contain a complete rectangular plane.")

    return x, y, u, v, dx, dy


def solve_streamfunction(u, v, dx, dy):
    """Solve ∇²ψ = -(∂v/∂x - ∂u/∂y) with ψ = 0 on the outer boundary."""
    ny, nx = u.shape
    dv_dx = np.gradient(v, dx, axis=1, edge_order=2)
    du_dy = np.gradient(u, dy, axis=0, edge_order=2)
    omega = dv_dx - du_dy

    # Cell-centred second derivatives with zero ψ at the physical boundary.
    dxx = diags(
        [np.ones(nx - 1), np.r_[-3.0, -2.0 * np.ones(nx - 2), -3.0], np.ones(nx - 1)],
        [-1, 0, 1],
        format="csc",
    ) / dx**2
    dyy = diags(
        [np.ones(ny - 1), np.r_[-3.0, -2.0 * np.ones(ny - 2), -3.0], np.ones(ny - 1)],
        [-1, 0, 1],
        format="csc",
    ) / dy**2
    laplacian = kron(eye(ny, format="csc"), dxx, format="csc") + kron(
        dyy, eye(nx, format="csc"), format="csc"
    )

    psi = factorized(laplacian)(-omega.ravel()).reshape(ny, nx)
    return psi, omega


def main():
    args = parse_args()
    total_start = perf_counter()

    stage_start = perf_counter()
    data = np.genfromtxt(args.input_csv, delimiter=",", names=True, dtype=float)
    if data.ndim == 0:
        data = np.atleast_1d(data)
    print(f"CSV read time: {perf_counter() - stage_start:.3f} s")

    if min(args.shape) < 1:
        raise ValueError("All --shape entries must be positive.")
    horizontal, vertical, u_name, v_name = plane_definition(args.shape)
    print(f"Read {data.size:,} CSV rows from {args.input_csv}")
    print(f"Requested grid shape: nx, ny, nz = {tuple(args.shape)}")
    print(
        f"Plane definition: {horizontal}-{vertical} plane; "
        f"velocity components {u_name}, {v_name}"
    )

    stage_start = perf_counter()
    x, y, u, v, dx, dy = make_plane_arrays(
        data, horizontal, vertical, u_name, v_name, args.shape
    )
    print(f"Grid reconstruction time: {perf_counter() - stage_start:.3f} s")
    print(
        f"Reconstructed velocity arrays: {u.shape} "
        f"(rows={vertical}, columns={horizontal})"
    )
    print(f"Grid spacing: d{horizontal} = {dx:.16e}, d{vertical} = {dy:.16e}")

    stage_start = perf_counter()
    psi, omega = solve_streamfunction(u, v, dx, dy)
    print(f"Poisson solve time: {perf_counter() - stage_start:.3f} s")
    print(r"Solved ∇²ψ = -ω, with ω = ∂v/∂x - ∂u/∂y and ψ = 0 on the boundary.")
    print(f"Vorticity range: [{omega.min():.16e}, {omega.max():.16e}]")

    output = args.output
    if output is None:
        output = str(Path(args.input_csv).with_suffix("")) + "_streamfunction.png"

    stage_start = perf_counter()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    speed = np.hypot(u, v)

    stream = axes[0].streamplot(
        x,
        y,
        u,
        v,
        color=speed,
        density=args.density,
        cmap="viridis",
        linewidth=1.0,
    )
    figure.colorbar(stream.lines, ax=axes[0], label="Speed")
    axes[0].set(title="Velocity streamlines", xlabel=horizontal, ylabel=vertical)

    contours = axes[1].contour(x, y, psi, levels=args.levels, colors="black", linewidths=0.8)
    axes[1].clabel(contours, inline=True, fontsize=7)
    axes[1].set(title=r"Stream function $\psi$", xlabel=horizontal, ylabel=vertical)

    for axis in axes:
        axis.set_aspect("equal")
    figure.savefig(output, dpi=200)
    print(f"Plot and save time: {perf_counter() - stage_start:.3f} s")
    print(f"Stream function range: [{psi.min():.16e}, {psi.max():.16e}]")
    print(f"Wrote {output}")
    print(f"Total time: {perf_counter() - total_start:.3f} s")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
