from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import iris
from iris.coord_systems import RotatedGeogCS
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD
import iris.quickplot as qplt
import matplotlib.pyplot as plt

import slam


@contextmanager
def timeit(name: Optional[str] = None) -> None:
    start = datetime.now()
    try:
        yield
    finally:
        msg = "" if name is None else f" [{name}]"
        print(f"\ntimeit = {datetime.now() - start}{msg}\n")


fname = "/project/avd/ng-vat/data/Anke-Finnenkoetter/examples/falklands_startdump.nc"
with PARSE_UGRID_ON_LOAD.context():
    ucube = iris.load_cube(fname, "air_potential_temperature")
print(ucube)

crs = RotatedGeogCS(38.12, 293.1, north_pole_grid_longitude=180)
with timeit("full-solve"):
    scube = slam.Transform.from_ugrid(ucube, crs=crs)
print(scube)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=scube.coord_system().as_cartopy_projection())
qplt.pcolormesh(scube[0], axes=ax)
ax = plt.gca()
ax.coastlines()
ax.gridlines()
plt.show()
