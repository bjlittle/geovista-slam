from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime

import iris
from iris.coord_systems import RotatedGeogCS
import iris.quickplot as qplt
import matplotlib.pyplot as plt

import slam


@contextmanager
def timeit(name: str | None = None) -> None:
    start = datetime.now()
    try:
        yield
    finally:
        msg = "" if name is None else f" [{name}]"
        print(f"\ntimeit = {datetime.now() - start}{msg}\n")


fname = "falklands_startdump.nc"
ucube = iris.load_cube(fname, "air_potential_temperature")
print(ucube)

crs = RotatedGeogCS(38.12, 293.1, north_pole_grid_longitude=180)
with timeit("fast-solve"):
    scube = slam.Transform.from_ugrid(ucube, crs=crs, fast_solve=True)
print(scube)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=scube.coord_system().as_cartopy_projection())
qplt.pcolormesh(scube[0], axes=ax)
ax = plt.gca()
ax.coastlines()
ax.gridlines()
plt.show()
