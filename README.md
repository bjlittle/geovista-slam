# geovista-slam

[GeoVista](https://github.com/bjlittle/geovista) utility to convert [CF UGRID](https://ugrid-conventions.github.io/ugrid-conventions/) Local Area Model quad-cell meshes
into structured rectilinear or curvilinear grids.

|              |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ⚙️ CI        | [![ci-citation](https://github.com/bjlittle/geovista-slam/actions/workflows/ci-citation.yml/badge.svg)](https://github.com/bjlittle/geovista-slam/actions/workflows/ci-citation.yml) [![ci-locks](https://github.com/bjlittle/geovista-slam/actions/workflows/ci-locks.yml/badge.svg)](https://github.com/bjlittle/geovista-slam/actions/workflows/ci-locks.yml) [![ci-manifest](https://github.com/bjlittle/geovista-slam/actions/workflows/ci-manifest.yml/badge.svg)](https://github.com/bjlittle/geovista-slam/actions/workflows/ci-manifest.yml) [![ci-wheels](https://github.com/bjlittle/geovista-slam/actions/workflows/ci-wheels.yml/badge.svg)](https://github.com/bjlittle/geovista-slam/actions/workflows/ci-wheels.yml) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/bjlittle/geovista-slam/main.svg)](https://results.pre-commit.ci/latest/github/bjlittle/geovista-slam/main) |
| 💬 Community | [![Contributor Covenant](https://img.shields.io/badge/contributor%20covenant-2.1-4baaaa.svg)](https://github.com/bjlittle/geovista-slam/blob/main/CODE_OF_CONDUCT.md) [![GH Discussions](https://img.shields.io/badge/github-discussions%20%F0%9F%92%AC-yellow?logo=github&logoColor=lightgrey)](https://github.com/bjlittle/geovista-slam/discussions)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ✨ Meta       | [![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![license - bds-3-clause](https://img.shields.io/github/license/bjlittle/geovista-slam)](https://github.com/bjlittle/geovista-slam/blob/main/LICENSE) [![conda platform](https://img.shields.io/conda/pn/conda-forge/geovista-slam.svg)](https://anaconda.org/conda-forge/geovista-slam)                                                                                                                                                                                                                                                                                                            |
| 📦 Package   | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7837322.svg)](https://doi.org/10.5281/zenodo.7837322) [![conda-forge](https://img.shields.io/conda/vn/conda-forge/geovista-slam?color=orange&label=conda-forge&logo=conda-forge&logoColor=white)](https://anaconda.org/conda-forge/geovista-slam) [![pypi](https://img.shields.io/pypi/v/geovista-slam?color=orange&label=pypi&logo=python&logoColor=white)](https://pypi.org/project/geovista-slam/) [![pypi - python version](https://img.shields.io/pypi/pyversions/geovista-slam.svg?color=orange&logo=python&label=python&logoColor=white)](https://pypi.org/project/geovista-slam/)                                                                                                                                                                                                                                                                       |
| 🧰 Repo      | [![commits-since](https://img.shields.io/github/commits-since/bjlittle/geovista-slam/latest.svg)](https://github.com/bjlittle/geovista-slam/commits/main) [![contributors](https://img.shields.io/github/contributors/bjlittle/geovista-slam)](https://github.com/bjlittle/geovista-slam/graphs/contributors) [![release](https://img.shields.io/github/v/release/bjlittle/geovista-slam)](https://github.com/bjlittle/geovista-slam/releases)                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|              |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

## Installation

`geovista-slam` is available on [conda-forge](https://anaconda.org/conda-forge/geovista-slam) and [PyPI](https://pypi.org/project/geovista-slam/).

We recommend using [mamba](https://github.com/mamba-org/mamba) to install `geovista-slam` 👍

### conda

`geovista-slam` is available on [conda-forge](https://anaconda.org/conda-forge/geovista-slam), and can be easily installed with [conda](https://docs.conda.io/projects/conda/en/latest/index.html):
```shell
conda install -c conda-forge geovista-slam
```
or alternatively with [mamba](https://github.com/mamba-org/mamba):
```shell
mamba install -c conda-forge geovista-slam
```
For more information see our [conda-forge feedstock](https://github.com/conda-forge/geovista-slam-feedstock).

### pip

`geovista-slam` is also available on [PyPI](https://pypi.org/project/geovista/):

```shell
pip install geovista-slam
```

However, complications may arise due to the [cartopy](https://pypi.org/project/cartopy/) package dependencies.


## License

`geovista-slam` is distributed under the terms of the [BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) license.


## [#ShowYourStripes](https://showyourstripes.info/s/globe)

<h4 align="center">
  <a href="https://showyourstripes.info/s/globe">
    <img src="https://raw.githubusercontent.com/ed-hawkins/show-your-stripes/master/2021/GLOBE---1850-2021-MO.png"
         height="50" width="800"
         alt="#showyourstripes Global 1850-2021"></a>
</h4>

**Graphics and Lead Scientist**: [Ed Hawkins](http://www.met.reading.ac.uk/~ed/home/index.php), National Centre for Atmospheric Science, University of Reading.

**Data**: Berkeley Earth, NOAA, UK Met Office, MeteoSwiss, DWD, SMHI, UoR, Meteo France & ZAMG.

<p>
<a href="https://showyourstripes.info/s/globe">#ShowYourStripes</a> is distributed under a
<a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>
<a href="https://creativecommons.org/licenses/by/4.0/">
  <img src="https://i.creativecommons.org/l/by/4.0/80x15.png" alt="creative-commons-by" style="border-width:0"></a>
</p>
