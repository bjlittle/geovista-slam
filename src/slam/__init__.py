"""TBD."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias
import warnings

import geovista as gv
from geovista.common import from_cartesian, wrap
from geovista.crs import WGS84
from iris.coord_systems import CoordSystem
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
import numpy as np

try:
    from ._version import version as __version__
except ModuleNotFoundError:
    __version__ = "unknown"

if TYPE_CHECKING:
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


# TODO @bjlittle: trap for non-quad mesh
# TODO @bjlittle: add developer diagnostics (warnings)
# TODO @bjlittle: doc-strings
# TODO @bjlittle: add tox
# TODO @bjlittle: test coverage
# TODO @bjlittle: restructure factories and anything spanning mesh dimension
# TODO @bjlittle: discover crs from mesh
# TODO @bjlittle: load/save
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
        self._structure = self._solver(
            ucube, crs=crs, decimals=decimals, fast_solve=fast_solve, rounding=rounding
        )
        self._mesh = ucube.mesh

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
        cls: Transform,
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
        cls: Transform,
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
        cls: Transform,
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
        return cls._restructure(ucube, share=structure)

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

    @classmethod
    def load(cls: Transform, fname: PathLike) -> Transform:
        """TBD.

        Parameters
        ----------
        fname : PathLike

        Returns
        -------
        Transform

        Notes
        -----
        .. versionadded:: 0.1.0

        """

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
        """TBD.

        Notes
        -----
        .. versionadded:: 0.1.0

        """

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
