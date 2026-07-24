"""Tests for :meth:`slam.Transform.save` / :meth:`slam.Transform.load`."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING
import warnings
import zipfile

import iris
import numpy as np
import pytest

import slam
from slam import CELL_IDS, SLAM_ARCHIVE_MEMBERS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from iris.coord_systems import CoordSystem
    from iris.cube import Cube


def _roundtrip_nc(cube: Cube, path: Path) -> Cube:
    """Save `cube` to netcdf and reload it, mimicking a fresh runtime.

    In practice a caller reloads their (regularly regenerated) LFric file in a
    new runtime, so its mesh carries netcdf-assigned metadata just like the mesh
    persisted inside a slam archive.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with iris.FUTURE.context(save_split_attrs=True):
            iris.save(cube, path)
        return iris.load_cube(path)


def _repack(
    src: Path,
    dst: Path,
    *,
    drop: Iterable[str] = (),
    manifest_update: Mapping[str, object] | None = None,
) -> None:
    """Rewrite a slam archive, optionally dropping members or editing manifest."""
    drop = set(drop)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        with zipfile.ZipFile(src) as archive:
            names = archive.namelist()
            archive.extractall(tmp)

        if manifest_update is not None:
            manifest = json.loads((tmp / "manifest.json").read_text())
            manifest.update(manifest_update)
            (tmp / "manifest.json").write_text(json.dumps(manifest))

        with zipfile.ZipFile(
            dst, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for name in names:
                if name not in drop:
                    archive.write(tmp / name, arcname=name)


def _duplicate_structure_role(src: Path, dst: Path) -> None:
    """Rewrite a slam archive so source.nc carries two ``structure`` cubes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        with zipfile.ZipFile(src) as archive:
            names = archive.namelist()
            archive.extractall(tmp)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cubes = list(iris.load(tmp / "source.nc"))
            (structure,) = [
                c for c in cubes if c.attributes.get("slam_role") == "structure"
            ]
            duplicate = structure.copy()
            # a distinct name keeps iris from merging the pair back into one cube
            duplicate.long_name = "structure duplicate"
            with iris.FUTURE.context(save_split_attrs=True):
                iris.save([*cubes, duplicate], tmp / "source.nc")

        with zipfile.ZipFile(
            dst, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for name in names:
                archive.write(tmp / name, arcname=name)


@pytest.fixture
def solved(lam_cube: Cube, crs: CoordSystem) -> slam.Transform:
    """Return a freshly solved transform over the synthetic LAM cube."""
    return slam.Transform(lam_cube, crs=crs)


@pytest.fixture
def archive(tmp_path: Path, solved: slam.Transform) -> Path:
    """Return the path to a saved slam archive of the solved transform."""
    path = tmp_path / "transform.slam.zip"
    solved.save(path)
    return path


def test_roundtrip_equivalence(
    tmp_path: Path,
    lam_cube: Cube,
    crs: CoordSystem,
    solved: slam.Transform,
    archive: Path,
) -> None:
    """A loaded transform equals, and behaves identically to, a solved one."""
    # a new runtime reloads the (netcdf) source cube for the same mesh
    reloaded = _roundtrip_nc(lam_cube, tmp_path / "source.nc")
    fresh = slam.Transform(reloaded, crs=crs)

    loaded = slam.Transform.load(archive)

    # equivalence, exercised through __eq__
    assert loaded == fresh

    # behavioural equivalence: identical restructured output
    expected = solved(lam_cube)
    actual = loaded(reloaded)
    assert np.array_equal(actual.data, expected.data)
    assert actual.coord(axis="x") == expected.coord(axis="x")
    assert actual.coord(axis="y") == expected.coord(axis="y")


def test_load_reconstructs_all_state(solved: slam.Transform, archive: Path) -> None:
    """Load populates exactly the instance state a solved transform holds."""
    loaded = slam.Transform.load(archive)
    assert vars(loaded).keys() == vars(solved).keys()


def test_equality_semantics(
    crs: CoordSystem,
    archive: Path,
    lam_cube_factory: Callable[..., Cube],
) -> None:
    """Equality is reflexive, rejects non-transforms and distinguishes grids."""
    loaded = slam.Transform.load(archive)
    assert loaded == loaded  # noqa: PLR0124
    assert (loaded == "not a transform") is False

    other = slam.Transform(lam_cube_factory(nx=6, ny=5), crs=crs)
    assert loaded != other


def test_coord_system_restored(crs: CoordSystem, archive: Path) -> None:
    """Both restructured spatial coords regain their coord_system on load."""
    loaded = slam.Transform.load(archive)
    assert loaded.coords.x.coord_system == crs
    assert loaded.coords.y.coord_system == crs


def test_cell_ids_preserved(solved: slam.Transform, archive: Path) -> None:
    """The mesh.vtp round-trip preserves cell_data[CELL_IDS]."""
    loaded = slam.Transform.load(archive)
    ids = loaded.mesh.cell_data[CELL_IDS]
    assert np.array_equal(ids, np.arange(loaded.mesh.n_cells))
    assert np.array_equal(ids, solved.mesh.cell_data[CELL_IDS])


def test_save_overwrites_existing(tmp_path: Path, solved: slam.Transform) -> None:
    """Save truncates any pre-existing file and leaves no stale members."""
    path = tmp_path / "transform.slam.zip"
    path.write_bytes(b"stale content that is not a valid zip archive")

    solved.save(path)
    with zipfile.ZipFile(path) as archive:
        assert tuple(archive.namelist()) == SLAM_ARCHIVE_MEMBERS

    # a valid archive can itself be overwritten and remains loadable
    solved.save(path)
    assert slam.Transform.load(path) == slam.Transform.load(path)
    with zipfile.ZipFile(path) as archive:
        assert sorted(archive.namelist()) == sorted(SLAM_ARCHIVE_MEMBERS)


def test_save_rejects_directory(tmp_path: Path, solved: slam.Transform) -> None:
    """Saving over an existing directory raises a clear error."""
    with pytest.raises(ValueError, match="is a directory"):
        solved.save(tmp_path)


def test_save_rejects_unsolved(tmp_path: Path) -> None:
    """Saving a transform that was never solved raises a clear error."""
    inst = slam.Transform.__new__(slam.Transform)
    with pytest.raises(ValueError, match="unsolved"):
        inst.save(tmp_path / "transform.slam.zip")


def test_load_missing_file(tmp_path: Path) -> None:
    """Loading a non-existent archive raises a clear error."""
    with pytest.raises(ValueError, match="no such file"):
        slam.Transform.load(tmp_path / "does-not-exist.zip")


def test_load_not_a_zip(tmp_path: Path) -> None:
    """Loading a file that is not a zip archive raises a clear error."""
    path = tmp_path / "not-a-zip.zip"
    path.write_bytes(b"this is plainly not a zip archive")
    with pytest.raises(ValueError, match="not a valid zip"):
        slam.Transform.load(path)


def test_load_missing_member(tmp_path: Path, archive: Path) -> None:
    """A truncated archive names the missing member, not a bare KeyError."""
    truncated = tmp_path / "truncated.zip"
    _repack(archive, truncated, drop=("source.nc",))
    with pytest.raises(ValueError, match=r"missing archive member.*'source.nc'"):
        slam.Transform.load(truncated)


def test_load_unsupported_version(tmp_path: Path, archive: Path) -> None:
    """An archive with an unknown format version is rejected."""
    tampered = tmp_path / "future.zip"
    _repack(archive, tampered, manifest_update={"format_version": 999})
    with pytest.raises(ValueError, match="unsupported archive"):
        slam.Transform.load(tampered)


def test_load_wrong_class(tmp_path: Path, archive: Path) -> None:
    """An archive whose manifest class does not match is rejected."""
    tampered = tmp_path / "wrong-class.zip"
    _repack(archive, tampered, manifest_update={"class": "NotATransform"})
    with pytest.raises(ValueError, match="expected a 'Transform'"):
        slam.Transform.load(tampered)


def test_load_duplicate_role(tmp_path: Path, archive: Path) -> None:
    """An archive with two cubes sharing a slam_role is rejected."""
    duped = tmp_path / "duplicate-role.zip"
    _duplicate_structure_role(archive, duped)
    with pytest.raises(ValueError, match="duplicate 'structure'"):
        slam.Transform.load(duped)
