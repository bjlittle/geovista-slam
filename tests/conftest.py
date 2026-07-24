"""Shared fixtures for the ``slam`` test suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris.coord_systems import RotatedGeogCS
from iris.coords import AuxCoord
from iris.cube import Cube
from iris.mesh import Connectivity, MeshXY
import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from iris.coord_systems import CoordSystem


def _make_lam_cube(nx: int = 5, ny: int = 4, name: str = "air_temperature") -> Cube:
    """Build a tiny synthetic CF-UGRID quad-cell LAM cube.

    A regular ``(nx - 1) x (ny - 1)`` grid of quad faces over a small
    rectilinear region, enough to exercise the boundary solve without any
    external data files or network access.
    """
    xs = np.linspace(-2.0, 2.0, nx)
    ys = np.linspace(50.0, 54.0, ny)
    grid_x, grid_y = np.meshgrid(xs, ys)
    node_x, node_y = grid_x.ravel(), grid_y.ravel()

    faces, face_x, face_y = [], [], []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            nodes = [n0, n0 + 1, n0 + 1 + nx, n0 + nx]
            faces.append(nodes)
            face_x.append(node_x[nodes].mean())
            face_y.append(node_y[nodes].mean())
    faces = np.array(faces)

    node_lon = AuxCoord(node_x, standard_name="longitude", units="degrees")
    node_lat = AuxCoord(node_y, standard_name="latitude", units="degrees")
    face_lon = AuxCoord(np.array(face_x), standard_name="longitude", units="degrees")
    face_lat = AuxCoord(np.array(face_y), standard_name="latitude", units="degrees")
    connectivity = Connectivity(
        indices=faces, cf_role="face_node_connectivity", start_index=0
    )
    mesh = MeshXY(
        topology_dimension=2,
        node_coords_and_axes=[(node_lon, "x"), (node_lat, "y")],
        connectivities=[connectivity],
        face_coords_and_axes=[(face_lon, "x"), (face_lat, "y")],
    )

    mesh_x, mesh_y = mesh.to_MeshCoords("face")
    cube = Cube(np.arange(faces.shape[0], dtype=float), standard_name=name, units="K")
    cube.add_aux_coord(mesh_x, 0)
    cube.add_aux_coord(mesh_y, 0)
    return cube


@pytest.fixture
def lam_cube() -> Cube:
    """Return a fresh synthetic CF-UGRID LAM cube."""
    return _make_lam_cube()


@pytest.fixture
def lam_cube_factory() -> Callable[..., Cube]:
    """Return a factory that builds synthetic CF-UGRID LAM cubes."""
    return _make_lam_cube


@pytest.fixture
def crs() -> CoordSystem:
    """Return a rotated pole CRS to exercise coord_system round-tripping."""
    return RotatedGeogCS(38.12, 293.1, north_pole_grid_longitude=180)
