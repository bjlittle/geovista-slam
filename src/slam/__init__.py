"""TBD."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, TypeAlias
import warnings
import zipfile

import geovista as gv
from geovista.common import from_cartesian, wrap
from geovista.crs import WGS84
import iris
from iris.coord_systems import CoordSystem
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
import numpy as np
import pyvista as pv

try:
    from ._version import version as __version__
except ModuleNotFoundError:
    __version__ = "unknown"

if TYPE_CHECKING:
    from iris.mesh import MeshXY
    from numpy.typing import ArrayLike
    from pyvista import PolyData, UnstructuredGrid

__all__ = ("Transform",)

# aliases for types
CoordLike: TypeAlias = AuxCoord | DimCoord
PathLike: TypeAlias = str | Path
ShapeLike: TypeAlias = list[int] | tuple[int, ...]

BOUNDARY_N_CORNERS: int = 4
BOUNDARY_STRIP: int = 3
CELL_IDS: str = "slamIdsGlobal"
CELL_IDS_LOCAL: str = "slamIdsLocal"
DEFAULT_FAST_SOLVE: bool = False
DEFAULT_ROUNDING: bool = True
DEFAULT_SHARE_SPATIAL: bool = False
DEFAULT_CF_COORDINATE_SYSTEM: list[dict[str, Any]] = [
    {
        "standard_name": "longitude",
        "long_name": "longitude coordinate",
        "units": "degrees",
    },
    {
        "standard_name": "latitude",
        "long_name": "latitude coordinate",
        "units": "degrees",
    },
]
MDI: int = -1
SLAM_ARCHIVE_FORMAT: str = "slam-transform"
SLAM_ARCHIVE_VERSION: int = 1
# the slam archive members, shared by Transform.save and Transform.load so the
# two never drift apart
SLAM_ARCHIVE_MEMBERS: tuple[str, ...] = (
    "manifest.json",
    "grid.npy",
    "edge.npz",
    "mesh.vtp",
    "source.nc",
)
# the roles carried by the cubes within the source.nc archive member
SLAM_SOURCE_ROLES: tuple[str, ...] = ("source_mesh", "structure", "crs")
# a stable prefix of the benign netCDF4/pyvista/NumPy 2.5 deprecation emitted
# when (de)serialising arrays; matched as a prefix so it survives changes to the
# version-specific tail of the message
NUMPY_SHAPE_DEPRECATION: str = "Setting the shape on a NumPy array"


# TODO @bjlittle: trap for non-quad mesh
# TODO @bjlittle: add developer diagnostics (warnings)
# TODO @bjlittle: doc-strings
# TODO @bjlittle: add tox
# TODO @bjlittle: test coverage
# TODO @bjlittle: restructure factories and anything spanning mesh dimension
# TODO @bjlittle: discover crs from mesh
# TODO @bjlittle: enable ci services
# TODO @bjlittle: repr/str


def edge_factory() -> ArrayLike:
    """TBD.

    Returns
    -------
    ndarray

    Notes
    -----
    .. versionadded:: 0.1.0

    """
    return np.array([])


@dataclass
class Coords:
    x: CoordLike
    y: CoordLike


@dataclass
class Edge:
    top: ArrayLike = field(default_factory=edge_factory)
    bottom: ArrayLike = field(default_factory=edge_factory)
    left: ArrayLike = field(default_factory=edge_factory)
    right: ArrayLike = field(default_factory=edge_factory)
    generic: ArrayLike = field(default_factory=edge_factory)


@dataclass
class Points:
    x: ArrayLike
    y: ArrayLike


@dataclass
class Structure:
    coords: Coords
    edge: Edge
    grid: ArrayLike
    mesh: PolyData


def _array_equal(lhs: ArrayLike, rhs: ArrayLike) -> bool:
    """Compare two array-likes element-wise, tolerating dtype and shape.

    Parameters
    ----------
    lhs : ArrayLike
    rhs : ArrayLike

    Returns
    -------
    bool

    Notes
    -----
    .. versionadded:: 0.1.0

    """
    return np.array_equal(np.asarray(lhs), np.asarray(rhs))


class Transform:
    """TBD.

    Notes
    -----
    .. versionadded:: 0.1.0

    """

    def __init__(
        self,
        ucube: Cube,
        /,
        *,
        crs: CoordSystem | None = None,
        decimals: int | None = None,
        fast_solve: bool | None = None,
        rounding: bool | None = None,
    ) -> None:
        """TBD.

        Parameters
        ----------
        ucube : Cube
        crs : CoordSystem, optional
        decimals : int, optional
        fast_solve : bool, optional
        rounding : bool, optional

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        self._verify(ucube, crs=crs)
        structure = self._solver(
            ucube, crs=crs, decimals=decimals, fast_solve=fast_solve, rounding=rounding
        )
        self._assign(structure, ucube.mesh)

    def __call__(self, ucube: Cube, /, *, share: bool | None = None) -> Cube:
        """TBD.

        Parameters
        ----------
        ucube : Cube
        share : bool, optional

        Returns
        -------
        Cube

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        self._verify(ucube)

        if ucube.mesh != self._mesh:
            emsg = (
                f"The provided unstructured cube '{ucube.name()}' has a "
                "different CF-UGRID mesh."
            )
            raise ValueError(emsg)

        return self._restructure(ucube, self._structure, share=share)

    def __eq__(self, other: object) -> bool:
        """Determine whether two solved transforms are equivalent.

        Two transforms are equal when they carry the same solved structure
        (grid, boundary edges, restructured spatial coordinates and mesh) over
        the same source CF-UGRID mesh. This makes the equivalence of a
        :meth:`load` reconstruction to a freshly solved instance assertable.

        Parameters
        ----------
        other : object

        Returns
        -------
        bool

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        if not isinstance(other, Transform):
            return NotImplemented

        lhs, rhs = self._structure, other._structure
        edge_fields = ("top", "bottom", "left", "right", "generic")

        return bool(
            _array_equal(lhs.grid, rhs.grid)
            and all(
                _array_equal(getattr(lhs.edge, field), getattr(rhs.edge, field))
                for field in edge_fields
            )
            and lhs.coords.x == rhs.coords.x
            and lhs.coords.y == rhs.coords.y
            and _array_equal(lhs.mesh.points, rhs.mesh.points)
            and _array_equal(lhs.mesh.faces, rhs.mesh.faces)
            and _array_equal(lhs.mesh.cell_data[CELL_IDS], rhs.mesh.cell_data[CELL_IDS])
            and self._mesh == other._mesh
        )

    __hash__ = None

    def _assign(self, structure: Structure, mesh: MeshXY) -> None:
        """Assign the solved state onto the instance.

        The single choke-point through which both :meth:`__init__` and
        :meth:`load` populate instance state, so a reconstructed transform can
        never drift out of step with a freshly solved one.

        Parameters
        ----------
        structure : Structure
        mesh : MeshXY
            The source CF-UGRID mesh guarded by :meth:`__call__`.

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        self._structure = structure
        self._mesh = mesh

    @staticmethod
    def _boundary_shape(edge: Edge) -> ShapeLike:
        """TBD.

        Parameters
        ----------
        edge : Edge

        Returns
        -------
        Tuple

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        if edge.top.size != edge.bottom.size:
            emsg = (
                "Boundary edge top and bottom have difference sizes, got "
                f"{edge.top.size!r} and {edge.bottom.size!r}."
            )
            raise ValueError(emsg)

        if edge.left.size != edge.right.size:
            emsg = (
                "Boundary edge left and right have different sizes, got "
                f"{edge.left.size!r} and {edge.right.size!r}."
            )
            raise ValueError(emsg)

        if edge.top.size and edge.left.size:
            shape = (edge.left.size + 2, edge.top.size)
        elif edge.top.size and not edge.left.size:
            shape = (2, edge.top.size)
        elif edge.generic.size:
            shape = edge.generic.shape
        else:
            emsg = "Failed to determine the shape of the boundary."
            raise ValueError(emsg)

        return shape

    @staticmethod
    def _build_coords(points: Points, crs: CoordSystem | None = None) -> Coords:
        """TBD.

        Parameters
        ----------
        points : Points
        crs : CoordSystem, optional

        Returns
        -------
        Coords

        Notes
        -----
        .. versionadded:: 0.1.0

        """

        def build(values: ArrayLike, *, circular: bool | None = False) -> CoordLike:
            factory = DimCoord if values.ndim == 1 else AuxCoord
            cf = (
                DEFAULT_CF_COORDINATE_SYSTEM
                if crs is None
                else crs.as_cartopy_crs().cs_to_cf()
            )

            for metadata in cf:
                if metadata.get("units") in ("degrees_east", "degrees_north"):
                    metadata["units"] = "degrees"
                if metadata.get("axis"):
                    del metadata["axis"]
                metadata["coord_system"] = crs

            def build_coord(
                values: ArrayLike, base: float, kwargs: dict[str, Any]
            ) -> CoordLike:
                try:
                    coord = factory(wrap(values, base=base), **kwargs)
                except ValueError:
                    coord = None
                return coord

            if circular:
                kwargs = cf[0]
                for base in (-180, 0):
                    coord = build_coord(values, base, kwargs)
                    if coord is not None:
                        break
                if coord is None:
                    circular = False
                    factory = AuxCoord
            else:
                kwargs = cf[1]

            if not circular:
                coord = factory(values, **kwargs)

            return coord

        return Coords(x=build(points.x, circular=True), y=build(points.y))

    @staticmethod
    def _create_mesh(ucube: Cube) -> PolyData:
        """TBD.

        Parameters
        ----------
        ucube : Cube

        Returns
        -------
        PolyData

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        face_node = ucube.mesh.face_node_connectivity
        indices = face_node.indices_by_location()
        lons, lats = ucube.mesh.node_coords

        if crs := ucube.coord_system():
            crs = crs.as_cartopy_crs()

        mesh = gv.Transform.from_unstructured(
            lons.points,
            lats.points,
            connectivity=indices,
            start_index=face_node.start_index,
            crs=crs,
        )

        # attach the global cell indices
        mesh.cell_data[CELL_IDS] = np.arange(mesh.n_cells)

        return mesh

    @staticmethod
    def _extract_edges(mesh: PolyData, /, *, iteration: int | None = 0) -> Edge:
        """TBD.

        Parameters
        ----------
        mesh : PolyData
        iteration : int, optional

        Returns
        -------
        Edge

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        # refresh the local cell indices on the mesh
        mesh.cell_data[CELL_IDS_LOCAL] = np.arange(mesh.n_cells)

        if mesh.n_cells == 1:
            return Edge(generic=mesh.cell_data[CELL_IDS_LOCAL])

        edges = mesh.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            feature_edges=False,
            manifold_edges=False,
        )
        cell_ids = edges.cell_data[CELL_IDS_LOCAL]
        values, counts = np.unique(cell_ids, return_counts=True)
        (corners,) = np.where(counts > 1)

        if (
            corners.size == counts.size
            and counts[0] == BOUNDARY_STRIP
            and counts[-1] == BOUNDARY_STRIP
        ):
            edge = Edge(generic=values)
        else:
            if (ncorners := corners.size) != BOUNDARY_N_CORNERS:
                emsg = (
                    "Failed to extract corners of bounded region, expected 4 "
                    f"corners but found {ncorners} [iteration={iteration}]."
                )
                raise ValueError(emsg)

            # corners -> indices are relative to values
            # corner_values -> indices are relative to mesh
            corner_values = values[corners]

            def extract(
                id1: int,
                id2: int,
                offset1: int | None = 0,
                offset2: int | None = 0,
            ) -> ArrayLike:
                start = np.where(cell_ids == id1)[0][-1] + offset1
                end = np.where(cell_ids == id2)[0][-1] + offset2
                return cell_ids[start:end]

            # extract top and bottom edges (left to right)
            top = extract(*corner_values[:2])
            bottom = extract(*corner_values[2:])

            if top.shape != bottom.shape:
                emsg = (
                    "Cannot extract boundary, top and bottom edges have difference "
                    f"shapes, got {top.shape!r} and {bottom.shape!r}."
                )
                raise ValueError(emsg)

            # extract left and right edges (top to bottom)
            ids = extract(*corner_values[1:3], offset1=1, offset2=-1)
            left, right = ids[::2], ids[1::2]

            if left.shape != right.shape:
                emsg = (
                    "Cannot extract boundary, left and right edges have different "
                    f"shapes, got {left.shape!r} and {right.shape!r}."
                )
                raise ValueError(emsg)

            edge = Edge(top=top, bottom=bottom, left=left, right=right)

        return edge

    @staticmethod
    def _extract_points(
        ucube: Cube,
        mesh: PolyData,
        grid: ArrayLike,
        /,
        *,
        crs: CoordSystem | None = None,
        decimals: int | None = None,
        rounding: bool | None = None,
    ) -> Points:
        """TBD.

        Parameters
        ----------
        ucube : Cube
        mesh : PolyData
        grid : ArrayLike
        crs : CoordSystem, optional
        decimals : int, optional
        rounding : bool, optional

        Returns
        -------
        Points

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        if rounding is None:
            rounding = DEFAULT_ROUNDING

        if (src_crs := ucube.coord_system()) is None:
            src_crs = WGS84
        else:
            src_crs = src_crs.as_cartopy_crs()

        coords = ucube.mesh.face_coords

        if coords is None:
            face_centers = mesh.cell_centers()
            xy0 = from_cartesian(face_centers, stacked=False)
            face_x, face_y = xy0[0], xy0[1]
        else:
            face_x, face_y = coords.face_x.points, coords.face_y.points

        if crs is not None:
            transformed = crs.as_cartopy_crs().transform_points(src_crs, face_x, face_y)
            if decimals is not None:
                transformed = np.round(transformed, decimals=decimals)
            face_x, face_y = transformed[:, 0], transformed[:, 1]

        grid_x, grid_y = face_x[grid], face_y[grid]
        uniform_x = np.unique(np.abs(np.diff(grid_x, axis=0)))
        uniform_y = np.unique(np.abs(np.diff(grid_y, axis=1)))

        def is_uniform() -> bool:
            result = uniform_x.size == 1 and uniform_y.size == 1
            if not result:
                result = (
                    np.allclose(uniform_x[1:], np.mean(uniform_x[1:]))
                    if uniform_x.size > 1
                    else True
                )
                if result and uniform_y.size > 1:
                    result = np.allclose(uniform_y[1:], np.mean(uniform_y[1:]))
            return result

        if rounding and decimals is None and not is_uniform():
            hwm = np.max([uniform_x.max(), uniform_y.max()])
            decimals = np.trunc(np.log10(hwm))
            if decimals < 0:
                decimals = int(np.abs(decimals))
                round_grid_x = np.round(grid_x, decimals=decimals)
                round_grid_y = np.round(grid_y, decimals=decimals)
                uniform_x = np.unique(np.abs(np.diff(round_grid_x, axis=0)))
                uniform_y = np.unique(np.abs(np.diff(round_grid_y, axis=1)))
                if is_uniform():
                    wmsg = (
                        "Auto-rounding 1-D coordinate points to "
                        f'"{decimals}" decimal places.'
                    )
                    warnings.warn(wmsg, stacklevel=2)
                    grid_x, grid_y = round_grid_x, round_grid_y

        if is_uniform():
            grid_x, grid_y = grid_x[0], grid_y[:, 0]

        return Points(x=grid_x, y=grid_y)

    @staticmethod
    def _is_uniform(edge: Edge, shape: ShapeLike) -> bool:
        """TBD.

        Parameters
        ----------
        edge : Edge
        shape : ShapeLike

        Returns
        -------
        bool

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        result = edge.generic.size > 0

        if not result:
            rows, cols = shape
            size = np.prod(shape)

            def delta(arg: ArrayLike, step: int | None = 1) -> bool:
                udiff = np.unique(np.diff(arg))
                return (udiff.size == 1) and (udiff[0] == step)

            if edge.top.size and edge.bottom.size:
                top = (
                    (edge.top[0] == 0)
                    and (edge.top[-1] == (cols - 1))
                    and delta(edge.top)
                )
                bottom = (
                    (edge.bottom[0] == ((rows - 1) * cols))
                    and (edge.bottom[-1] == (size - 1))
                    and delta(edge.bottom)
                )
                result = top and bottom

            if result and edge.left.size and edge.right.size:
                left = (
                    (edge.left[0] == cols)
                    and (edge.left[-1] == ((rows - 2) * cols))
                    and delta(edge.left, step=cols)
                )
                right = (
                    (edge.right[0] == (2 * cols - 1))
                    and (edge.right[-1] == (((rows - 1) * cols) - 1))
                    and delta(edge.right, step=cols)
                )
                result = left and right

        return result

    @classmethod
    def _matryoshka(
        cls: type[Transform],
        mesh: PolyData,
        edge: Edge,
        grid: ArrayLike,
        /,
        *,
        iteration: int | None = 0,
    ) -> None:
        """TBD.

        Parameters
        ----------
        mesh : PolyData
        edge : Edge
        grid : ArrayLike
        iteration : int, optional

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        if edge.generic.size:
            indices = np.where(grid == MDI)
            grid[indices] = mesh.cell_data[CELL_IDS][edge.generic]
        else:
            grows, gcols = grid.shape

            # store top edge in grid
            if size := edge.top.size:
                indices = np.arange(0, size) + iteration
                grid[iteration, indices] = mesh.cell_data[CELL_IDS][edge.top]

            # store bottom edge in grid
            if size := edge.bottom.size:
                indices = np.arange(0, size) + iteration
                grid[grows - iteration - 1, indices] = mesh.cell_data[CELL_IDS][
                    edge.bottom
                ]

            # store left edge in grid
            if size := edge.left.size:
                indices = np.arange(0, size) + iteration + 1
                grid[indices, iteration] = mesh.cell_data[CELL_IDS][edge.left]

            # store right edge in grid
            if size := edge.right.size:
                indices = np.arange(0, size) + iteration + 1
                grid[indices, gcols - iteration - 1] = mesh.cell_data[CELL_IDS][
                    edge.right
                ]

            halo = np.concatenate([edge.top, edge.left, edge.right, edge.bottom])
            mesh = mesh.remove_cells(halo)

            if mesh.n_cells:
                iteration += 1
                edge = cls._extract_edges(mesh, iteration=iteration)
                cls._matryoshka(mesh, edge, grid, iteration=iteration)

    @staticmethod
    def _restructure(
        ucube: Cube,
        structure: Structure,
        /,
        *,
        share: bool | None = None,
    ) -> Cube:
        """TBD.

        Parameters
        ----------
        ucube : Cube
        structure : Structure
        share : bool, optional

        Returns
        -------
        Cube

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        if share is None:
            share = DEFAULT_SHARE_SPATIAL

        mesh_dim = ucube.mesh_dim()
        slicer = [slice(None)] * (ndim := ucube.ndim)
        slicer[mesh_dim] = structure.grid

        # TODO @bjlittle: make this a lazy operation
        data = ucube.data[tuple(slicer)]

        scube = Cube(data, **ucube.metadata._asdict())

        mapping = {}
        for dim in range(ndim):
            if dim < mesh_dim:
                mapping[dim] = dim
            elif dim == mesh_dim:
                ydim, xdim = dim, dim + 1
            else:
                mapping[dim] = dim + 1

        def remap(dims: tuple[int]) -> tuple[int]:
            return tuple(mapping[dim] for dim in dims)

        # add dim coordinates to structured cube
        for coord in ucube.dim_coords:
            dim = ucube.coord_dims(coord)
            scube.add_dim_coord(coord.copy(), remap(dim))

        # add all other non-mesh aux coords to structured cube
        for coord in ucube.coords(dim_coords=False, mesh_coords=False):
            dims = ucube.coord_dims(coord)
            scube.add_aux_coord(coord.copy(), remap(dims))

        # add cell methods to structured cube
        for cm in ucube.cell_methods:
            scube.add_cell_method(deepcopy(cm))

        # add cell measures to structured cube
        for cm in ucube.cell_measures():
            dims = ucube.cell_measure_dims(cm)
            scube.add_cell_measure(deepcopy(cm), remap(dims))

        # add restructured spatial coordinates to structured cube
        coords = structure.coords.x, structure.coords.y
        coords_dim = xdim, ydim

        for i, coord in enumerate(coords):
            dim = coords_dim[i] if coord.ndim == 1 else (ydim, xdim)
            factory = (
                scube.add_dim_coord
                if isinstance(coord, DimCoord)
                else scube.add_aux_coord
            )
            factory(coord if share else coord.copy(), dim)

        return scube

    @classmethod
    def _solver(
        cls: type[Transform],
        ucube: Cube,
        /,
        *,
        crs: CoordSystem | None = None,
        decimals: int | None = None,
        fast_solve: bool | None = None,
        rounding: bool | None = None,
    ) -> Structure:
        """TBD.

        Parameters
        ----------
        ucube : Cube
        crs : CRS, optional
        decimals : int, optional
        fast_solve: bool, optional
        rounding : bool, optional

        Returns
        -------
        Structure

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        if fast_solve is None:
            fast_solve = DEFAULT_FAST_SOLVE

        mesh = cls._create_mesh(ucube)
        edge = cls._extract_edges(mesh)
        shape = cls._boundary_shape(edge)

        if fast_solve and cls._is_uniform(edge, shape):
            grid = np.arange(np.prod(shape), dtype=int).reshape(shape)
        else:
            grid = np.ones(shape, dtype=int) * MDI
            cls._matryoshka(mesh, edge, grid)

        points = cls._extract_points(
            ucube, mesh, grid, crs=crs, decimals=decimals, rounding=rounding
        )
        coords = cls._build_coords(points, crs=crs)

        return Structure(coords=coords, edge=edge, grid=grid, mesh=mesh)

    @staticmethod
    def _verify(ucube: Cube, /, *, crs: CoordSystem | None = None) -> None:
        """TBD.

        Parameters
        ----------
        ucube : Cube
        crs : CoordSystem, optional

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        if ucube.mesh is None:
            emsg = (
                "Expected an unstructured cube, but no mesh found on "
                f"'{ucube.name()}' cube."
            )
            raise ValueError(emsg)

        if crs and not isinstance(crs, CoordSystem):
            emsg = (
                "Expected an 'iris.coord_system.CoordSystem' instance, "
                f"got {type(crs)} instead."
            )
            raise ValueError(emsg)

    @property
    def coords(self) -> Coords:
        """TBD.

        Returns
        -------
        Coords

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        return deepcopy(self._structure.coords)

    @classmethod
    def from_ugrid(
        cls: type[Transform],
        ucube: Cube,
        /,
        *,
        crs: CoordSystem | None = None,
        decimals: int | None = None,
        fast_solve: bool | None = None,
        rounding: bool | None = None,
    ) -> Cube:
        """TBD.

        Parameters
        ----------
        ucube : Cube
        crs : CoordSystem, optional
        decimals : int, optional
        fast_solve : bool, optional
        rounding : bool, optional

        Returns
        -------
        Cube

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        cls._verify(ucube, crs=crs)
        structure = cls._solver(
            ucube, crs=crs, decimals=decimals, fast_solve=fast_solve, rounding=rounding
        )
        return cls._restructure(ucube, structure)

    @property
    def grid(self) -> ArrayLike:
        """TBD.

        Returns
        -------
        ArrayLike

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        return self._structure.grid.copy()

    @property
    def halo(self) -> UnstructuredGrid:
        """TBD.

        Returns
        -------
        UnstructuredGrid

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        edge = self._structure.edge
        cells = (
            edge.generic
            if edge.generic.size
            else np.concatenate([edge.top, edge.left, edge.right, edge.bottom])
        )
        return self._structure.mesh.extract_cells(cells)

    @staticmethod
    def _load_source(path: Path, source: Path) -> tuple[Coords, MeshXY]:
        """Read the restructured coords and source mesh from a source.nc member.

        Parameters
        ----------
        path : Path
            The slam archive path, used only for error messages.
        source : Path
            The extracted ``source.nc`` archive member.

        Returns
        -------
        tuple of Coords, MeshXY

        Raises
        ------
        ValueError
            If the source cubes are missing, corrupt, or carry a duplicate role.

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        cubes = iris.load(source)

        # bin the source cubes by role, guarding against a naming collision in
        # the user's data producing two cubes with the same slam_role
        by_role: dict[str, Cube] = {}
        for cube in cubes:
            role = cube.attributes.get("slam_role")
            if role not in SLAM_SOURCE_ROLES:
                continue
            if role in by_role:
                emsg = (
                    f"Cannot load slam archive {str(path)!r}, found duplicate "
                    f"{role!r} source cubes."
                )
                raise ValueError(emsg)
            by_role[role] = cube

        if set(by_role) != set(SLAM_SOURCE_ROLES):
            emsg = (
                f"Cannot load slam archive {str(path)!r}, the source cubes are "
                "missing or corrupt."
            )
            raise ValueError(emsg)

        structure_cube, anchor_cube = by_role["structure"], by_role["crs"]

        x_coord = structure_cube.coord(axis="x")
        y_coord = structure_cube.coord(axis="y")
        # netcdf drops the coord_system from 2-D aux coords, so restore each
        # axis from the anchor (whose 1-D dim coords round-trip it reliably),
        # keeping the x and y coord systems independent
        x_coord.coord_system = anchor_cube.coord(axis="x").coord_system
        y_coord.coord_system = anchor_cube.coord(axis="y").coord_system
        # the solver builds spatial coords with no var_name, so clear the
        # netcdf-assigned var_name to match a freshly solved transform
        x_coord.var_name = y_coord.var_name = None
        coords = Coords(x=x_coord, y=y_coord)

        return coords, by_role["source_mesh"].mesh

    @classmethod
    def load(cls: type[Transform], fname: PathLike) -> Transform:
        """Load a solved transform from a slam archive on disk.

        Reconstruct a :class:`Transform` from an archive previously created by
        :meth:`save`, skipping the expensive solve. The reconstructed instance
        is equivalent to one freshly solved from the same source unstructured
        cube.

        Parameters
        ----------
        fname : PathLike
            The filename of the slam archive to load.

        Returns
        -------
        Transform
            The reconstructed transform.

        Raises
        ------
        ValueError
            If `fname` does not exist, is not a valid slam archive, or has an
            unsupported archive format or version.

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        path = Path(fname)

        if not path.is_file():
            emsg = f"Cannot load slam archive, no such file {str(path)!r}."
            raise ValueError(emsg)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            try:
                with zipfile.ZipFile(path, mode="r") as archive:
                    available = set(archive.namelist())
                    missing = [
                        member
                        for member in SLAM_ARCHIVE_MEMBERS
                        if member not in available
                    ]
                    if missing:
                        names = ", ".join(repr(member) for member in missing)
                        emsg = (
                            f"Cannot load slam archive {str(path)!r}, missing "
                            f"archive member(s) {names}."
                        )
                        raise ValueError(emsg)
                    # only the known members are extracted (also avoids ruff S202)
                    for member in SLAM_ARCHIVE_MEMBERS:
                        archive.extract(member, tmp)
            except zipfile.BadZipFile as err:
                emsg = f"Cannot load slam archive {str(path)!r}, not a valid zip file."
                raise ValueError(emsg) from err

            manifest = json.loads((tmp / "manifest.json").read_text())
            fmt, version = manifest.get("format"), manifest.get("format_version")
            if fmt != SLAM_ARCHIVE_FORMAT or version != SLAM_ARCHIVE_VERSION:
                # surface the writing slam version to aid diagnosis
                wrote = manifest.get("slam_version", "unknown")
                emsg = (
                    f"Cannot load slam archive {str(path)!r}, unsupported archive "
                    f"format {fmt!r} (version {version!r}) written by slam "
                    f"{wrote!r}."
                )
                raise ValueError(emsg)

            archived_class = manifest.get("class")
            if archived_class != cls.__name__:
                emsg = (
                    f"Cannot load slam archive {str(path)!r}, expected a "
                    f"{cls.__name__!r} archive but got {archived_class!r}."
                )
                raise ValueError(emsg)

            grid = np.load(tmp / "grid.npy")

            with np.load(tmp / "edge.npz") as npz:
                edge = Edge(
                    top=npz["top"],
                    bottom=npz["bottom"],
                    left=npz["left"],
                    right=npz["right"],
                    generic=npz["generic"],
                )

            mesh = pv.read(tmp / "mesh.vtp")

            coords, source_mesh = cls._load_source(path, tmp / "source.nc")

        structure = Structure(coords=coords, edge=edge, grid=grid, mesh=mesh)

        # reconstruct without __init__, which would re-run the expensive solve,
        # routing state through _assign so it cannot drift from a solved instance
        inst = cls.__new__(cls)
        inst._assign(structure, source_mesh)  # noqa: SLF001

        return inst

    @property
    def mesh(self) -> PolyData:
        """TBD.

        Returns
        -------
        PolyData

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        return self._structure.mesh.copy()

    def save(self, fname: PathLike) -> None:
        """Save the solved transform to a slam archive on disk.

        Persist the entire state of the transform to a single ``zip`` archive of
        ecosystem-native artifacts, so it may later be reloaded with :meth:`load`
        in a separate runtime without re-running the expensive solve. An existing
        archive at `fname` is overwritten.

        Parameters
        ----------
        fname : PathLike
            The filename of the slam archive to create.

        Raises
        ------
        ValueError
            If the transform has not been solved, or `fname` is an existing
            directory.

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        if getattr(self, "_structure", None) is None:
            emsg = "Cannot save an unsolved transform."
            raise ValueError(emsg)

        path = Path(fname)

        if path.is_dir():
            emsg = f"Cannot save slam archive, {str(path)!r} is a directory."
            raise ValueError(emsg)

        structure = self._structure
        edge = structure.edge

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # the expensive matryoshka solve output
            np.save(tmp / "grid.npy", structure.grid)

            # the boundary edges
            np.savez(
                tmp / "edge.npz",
                top=edge.top,
                bottom=edge.bottom,
                left=edge.left,
                right=edge.right,
                generic=edge.generic,
            )

            # the geovista/pyvista mesh, preserving cell_data[CELL_IDS]
            structure.mesh.save(tmp / "mesh.vtp")

            # a dummy 1-D cube over the mesh dimension, carrying the source
            # UGRID mesh needed by the __call__ guard
            mesh_coords = self._mesh.to_MeshCoords("face")
            n_faces = mesh_coords[0].shape[0]
            mesh_cube = Cube(
                np.arange(n_faces), attributes={"slam_role": "source_mesh"}
            )
            for mesh_coord in mesh_coords:
                mesh_cube.add_aux_coord(mesh_coord, 0)

            # a cube carrying the restructured spatial coordinates, using the
            # same dim/aux logic as _restructure (grid is shaped (ydim, xdim))
            x_coord, y_coord = structure.coords.x, structure.coords.y
            structure_cube = Cube(structure.grid, attributes={"slam_role": "structure"})
            ydim, xdim = 0, 1
            for coord, cdim in ((x_coord, xdim), (y_coord, ydim)):
                dims = cdim if coord.ndim == 1 else (ydim, xdim)
                add_coord = (
                    structure_cube.add_dim_coord
                    if isinstance(coord, DimCoord)
                    else structure_cube.add_aux_coord
                )
                add_coord(coord.copy(), dims)

            # a tiny anchor cube whose 1-D dim coords reliably round-trip the
            # coord_system (netcdf drops it from 2-D aux coords); reattached on
            # load. Reuse the spatial standard_names so the grid_mapping matches
            crs = x_coord.coord_system
            anchor_cube = Cube(
                np.zeros((1, 1), dtype=int), attributes={"slam_role": "crs"}
            )
            anchor_cube.add_dim_coord(
                DimCoord(
                    [0.0],
                    standard_name=y_coord.standard_name,
                    units=y_coord.units,
                    coord_system=crs,
                ),
                0,
            )
            anchor_cube.add_dim_coord(
                DimCoord(
                    [0.0],
                    standard_name=x_coord.standard_name,
                    units=x_coord.units,
                    coord_system=crs,
                ),
                1,
            )

            with warnings.catch_warnings():
                # benign netCDF4/NumPy 2.5 shape-setting deprecation on write;
                # matched on a stable prefix so it survives changes to the
                # version-specific tail of the message
                warnings.filterwarnings(
                    "ignore",
                    message=NUMPY_SHAPE_DEPRECATION,
                    category=DeprecationWarning,
                )
                with iris.FUTURE.context(save_split_attrs=True):
                    iris.save(
                        [mesh_cube, structure_cube, anchor_cube], tmp / "source.nc"
                    )

            manifest = {
                "format": SLAM_ARCHIVE_FORMAT,
                "format_version": SLAM_ARCHIVE_VERSION,
                "slam_version": __version__,
                "class": type(self).__name__,
            }
            (tmp / "manifest.json").write_text(json.dumps(manifest))

            # mode="w" truncates any existing archive at path (overwrite)
            with zipfile.ZipFile(
                path, mode="w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for member in SLAM_ARCHIVE_MEMBERS:
                    archive.write(tmp / member, arcname=member)

    @property
    def shape(self) -> ShapeLike:
        """TBD.

        Returns
        -------
        ShapeLike

        Notes
        -----
        .. versionadded:: 0.1.0

        """
        return self._structure.grid.shape
