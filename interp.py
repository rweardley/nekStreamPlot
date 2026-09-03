#!/usr/bin/env python3
"""
Interpolate fields from one Nek/pySEMTools field file onto a uniform Cartesian grid.

Example:
  mpiexec -n 8 python interpolate_uniform_grid.py field0.f00001 uniform.csv \
      --bounds 0 1 0 1 0 2 --shape 101 101 201 --fields u,v,w,p

All MPI ranks read and interpolate the field; only rank 0 writes the CSV.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from mpi4py import MPI
from pysemtools.datatypes.field import FieldRegistry
from pysemtools.datatypes.msh import Mesh
from pysemtools.interpolation.probes import Probes
from pysemtools.io.ppymech.neksuite import pynekread


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interpolate Nek fields onto a uniform Cartesian grid and write CSV."
    )
    parser.add_argument("input_file", help="Nek field file, e.g. case0.f00001")
    parser.add_argument("output_csv", help="CSV file written by MPI rank 0")
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        required=True,
        help="Cartesian interpolation box.",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        required=True,
        help="Numbers of uniformly spaced points in x, y, and z.",
    )
    parser.add_argument(
        "--fields",
        default="u,v,w,p",
        help="Comma-separated pySEMTools field names (default: u,v,w,p).",
    )
    return parser.parse_args()


def uniform_points(bounds, shape):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    nx, ny, nz = shape
    if min(shape) < 1:
        raise ValueError("--shape entries must be positive.")
    if not (xmin <= xmax and ymin <= ymax and zmin <= zmax):
        raise ValueError("Each lower bound must not exceed its upper bound.")

    x1 = np.linspace(xmin, xmax, nx)
    y1 = np.linspace(ymin, ymax, ny)
    z1 = np.linspace(zmin, zmax, nz)
    x, y, z = np.meshgrid(x1, y1, z1, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def write_csv(filename, points, values, field_names):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "z", *field_names])
        writer.writerows(np.column_stack((points, values)))


def global_range(array, comm):
    """Return the global minimum and maximum of a distributed NumPy array."""
    return (
        comm.allreduce(np.min(array), op=MPI.MIN),
        comm.allreduce(np.max(array), op=MPI.MAX),
    )


def print_mesh_and_velocity_ranges(mesh, fields, comm):
    """Print global coordinate and available velocity-component ranges on rank 0."""
    ranges = {
        "x": global_range(mesh.x, comm),
        "y": global_range(mesh.y, comm),
        "z": global_range(mesh.z, comm),
    }
    for name in ("u", "v", "w"):
        if name in fields.registry:
            ranges[name] = global_range(fields.registry[name], comm)

    if comm.Get_rank() == 0:
        print("Global mesh and velocity ranges:")
        for name, (minimum, maximum) in ranges.items():
            print(f"  {name}: [{minimum:.16e}, {maximum:.16e}]")


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    field_names = [name.strip() for name in args.fields.split(",") if name.strip()]
    if not field_names:
        raise ValueError("--fields must contain at least one field name.")

    # This deliberately uses the simple pySEMTools mode: rank 0 owns every
    # query point and receives the complete interpolated result at the end.
    mesh = Mesh(comm, create_connectivity=True)
    fields = FieldRegistry(comm)
    pynekread(args.input_file, comm, data_dtype=np.double, msh=mesh, fld=fields)
    print_mesh_and_velocity_ranges(mesh, fields, comm)

    if rank == 0:
        points = uniform_points(args.bounds, args.shape)
        print(f"Interpolating {len(points):,} points for: {', '.join(field_names)}")
    else:
        # Current pySEMTools expects None on non-I/O ranks when every query
        # point is supplied by rank 0.
        points = None

    try:
        source_fields = [fields.registry[name] for name in field_names]
    except KeyError as exc:
        available = ", ".join(fields.registry.keys())
        raise KeyError(
            f"Field {exc.args[0]!r} is absent. Available fields: {available}"
        ) from exc

    probes = Probes(
        comm,
        probes=points,
        msh=mesh,
        point_interpolator_type="multiple_point_legendre_numpy",
        max_pts=256,
        find_points_comm_pattern="point_to_point",
        write_coords=False,
    )
    probes.interpolate_from_field_list(
        fields.t, source_fields, comm, write_data=False, field_names=field_names
    )

    if rank == 0:
        # Column 0 is time; remaining columns correspond to source_fields.
        write_csv(args.output_csv, points, probes.interpolated_fields[:, 1:], field_names)
        print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()

