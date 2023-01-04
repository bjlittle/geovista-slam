from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Optional

import iris
from iris.coord_systems import RotatedGeogCS
from iris.cube import Cube
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD

import slam


@contextmanager
def timeit(name: Optional[str] = None) -> None:
    start = datetime.now()
    try:
        yield
    finally:
        msg = "" if name is None else f" [{name}]"
        print(f"\ntimeit = {datetime.now() - start}{msg}")


def show(cubes: Dict[str, Cube]) -> None:
    for cube in cubes.values():
        print(cube.summary(shorten=True))


fname = "/project/avd/ng-vat/data/Anke-Finnenkoetter/examples/falklands_startdump.nc"
with PARSE_UGRID_ON_LOAD.context():
    ucubes = iris.load(fname)

with timeit("extract"):
    names = (
        "air_density",
        "rain_mixing_ratio",
        "Snow soot content",
        "air_potential_temperature",
    )
    samples = {name: ucubes.extract_cube(name) for name in names}

show(samples)

ucube = samples["air_potential_temperature"]

crs = RotatedGeogCS(38.12, 293.1, north_pole_grid_longitude=180)
with timeit("Transform"):
    transform = slam.Transform(ucube, crs=crs)

with timeit("transform"):
    results = {name: transform(samples[name]) for name in names}

show(results)
