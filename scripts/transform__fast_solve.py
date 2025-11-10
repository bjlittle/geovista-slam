from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime

import iris
from iris.coord_systems import RotatedGeogCS
from iris.cube import Cube

import slam


@contextmanager
def timeit(name: str | None = None) -> None:
    start = datetime.now()
    try:
        yield
    finally:
        msg = "" if name is None else f" [{name}]"
        print(f"\ntimeit = {datetime.now() - start}{msg}")


def show(cubes: dict[str, Cube]) -> None:
    for cube in cubes.values():
        print(cube.summary(shorten=True))


fname = "falklands_startdump.nc"
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
    transform = slam.Transform(ucube, crs=crs, fast_solve=True)

with timeit("transform"):
    results = {name: transform(samples[name]) for name in names}
show(results)
