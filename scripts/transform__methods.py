import iris
from iris.coord_systems import RotatedGeogCS
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD
import pyvista as pv

import slam

fname = "/project/avd/ng-vat/data/Anke-Finnenkoetter/examples/falklands_startdump.nc"
with PARSE_UGRID_ON_LOAD.context():
    ucube = iris.load_cube(fname, "air_potential_temperature")

print(ucube.summary(shorten=True))

crs = RotatedGeogCS(38.12, 293.1, north_pole_grid_longitude=180)
transform = slam.Transform(ucube, crs=crs, fast_solve=True)

print(f"\n{transform.coords=}")
print(f"\n{transform.grid=}")
print(f"\n{transform.shape=}")
print(f"\n{transform.mesh=}")
print(f"\n{transform.halo=}")

cmap = "balance"
transform.mesh.plot(cmap=cmap)
transform.halo.plot(cmap=cmap, show_edges=True)

plotter = pv.Plotter()
plotter.add_mesh(transform.halo, cmap=cmap, show_edges=True)
plotter.add_point_labels(
    transform.halo.cell_centers(),
    transform.halo.cell_data["slamIdsLocal"],
    always_visible=True,
)
plotter.add_axes()
plotter.show()
