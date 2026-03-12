"""Unit tests for recovar.simulation.pdb_utils."""

import os
import tempfile

import numpy as np
import pytest

from recovar.simulation.pdb_utils import AtomGroup, write_pdb, _parse_pdb_file

pytestmark = pytest.mark.unit


class TestAtomGroup:
    def test_empty_default(self):
        ag = AtomGroup()
        assert ag.numAtoms() == 0
        assert ag.getCoords().shape == (0, 3)

    def test_set_and_get(self):
        ag = AtomGroup()
        coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        ag.setCoords(coords)
        np.testing.assert_array_equal(ag.getCoords(), coords)
        assert ag.numAtoms() == 2

        elems = np.array(["C", "N"])
        ag.setElements(elems)
        np.testing.assert_array_equal(ag.getElements(), elems)

        chains = np.array(["A", "B"])
        ag.setChids(chains)
        np.testing.assert_array_equal(ag.getChids(), chains)

    def test_getData(self):
        ag = AtomGroup()
        ag.setElements(np.array(["C"]))
        assert ag.getData("element")[0] == "C"
        with pytest.raises(KeyError):
            ag.getData("unknown_key")


class TestPdbWriteReadRoundtrip:
    def test_roundtrip(self):
        ag = AtomGroup()
        coords = np.array([
            [1.234, 5.678, 9.012],
            [10.0, -20.5, 30.75],
            [0.0, 0.0, 0.0],
        ], dtype=np.float64)
        ag.setCoords(coords)
        ag.setNames(np.array(["CA", "CB", "N"]))
        ag.setElements(np.array(["C", "C", "N"]))

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            pdb_path = f.name

        try:
            write_pdb(pdb_path, ag)
            assert os.path.isfile(pdb_path)

            loaded = _parse_pdb_file(pdb_path)
            assert loaded.numAtoms() == 3
            np.testing.assert_allclose(loaded.getCoords(), coords, atol=0.002)
            np.testing.assert_array_equal(loaded.getElements(), ["C", "C", "N"])
        finally:
            os.unlink(pdb_path)

    def test_many_atoms(self):
        n = 500
        ag = AtomGroup()
        rng = np.random.default_rng(99)
        coords = rng.standard_normal((n, 3)) * 50
        ag.setCoords(coords)
        ag.setNames(np.array(["CA"] * n))
        ag.setElements(np.array(["C"] * n))

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            pdb_path = f.name
        try:
            write_pdb(pdb_path, ag)
            loaded = _parse_pdb_file(pdb_path)
            assert loaded.numAtoms() == n
            np.testing.assert_allclose(loaded.getCoords(), coords, atol=0.002)
        finally:
            os.unlink(pdb_path)
