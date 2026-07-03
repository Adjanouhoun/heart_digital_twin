"""Tests pour opencarp_config.py et units.py — contrats d'interface."""
import numpy as np
import pytest
from app.solver.ep.opencarp_config import VALIDATED, generate_par_file
from app.core.units import (
    mm_to_um, fix_element_orientation, tet_volume,
    filter_small_elements, mesh_quality_report
)


class TestOpenCARPConfig:

    def test_ionic_model_name(self):
        assert VALIDATED.ionic_model == "tenTusscherPanfilov"

    def test_conductivities_reference(self):
        assert VALIDATED.g_il == 0.3544
        assert VALIDATED.g_it == 0.024
        assert VALIDATED.g_el == 1.2700
        assert VALIDATED.g_et == 0.0862

    def test_dt_standard(self):
        assert VALIDATED.dt_ms == 0.02

    def test_mesh_units_um(self):
        assert VALIDATED.mesh_units == "um"

    def test_par_keywords(self):
        assert VALIDATED.par_keyword_mesh == "meshname"
        assert VALIDATED.par_keyword_output == "simID"
        assert VALIDATED.par_keyword_stim_prefix == "stimulus[0]"

    def test_generate_par_contains_meshname(self):
        par = generate_par_file("/tmp/m", "/tmp/o", 50.0, (0, 0, 0))
        assert "meshname = /tmp/m" in par
        assert "simID = /tmp/o" in par

    def test_generate_par_bcl_clamped(self):
        par = generate_par_file("/tmp/m", "/tmp/o", 10.0, (0, 0, 0), bcl_ms=500)
        assert "bcl = 10.0" in par

    def test_generate_par_stim_duration_error(self):
        with pytest.raises(ValueError):
            generate_par_file("/tmp/m", "/tmp/o", 1.0, (0, 0, 0))

    def test_generate_par_conductivities(self):
        par = generate_par_file("/tmp/m", "/tmp/o", 50.0, (0, 0, 0))
        assert "g_il = 0.3544" in par
        assert "g_el = 1.27" in par


class TestUnits:

    def test_mm_to_um(self):
        nodes = np.array([[1.0, 2.0, 3.0]])
        result = mm_to_um(nodes)
        np.testing.assert_array_equal(result, [[1000, 2000, 3000]])

    def test_tet_volume_positive(self):
        nodes = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
        vol = tet_volume(nodes, [0, 1, 2, 3])
        assert vol > 0

    def test_fix_orientation_negative(self):
        nodes = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
        bad = [[0, 2, 1, 3]]
        fixed, n = fix_element_orientation(nodes, bad)
        assert n == 1
        assert tet_volume(nodes, fixed[0]) > 0

    def test_fix_orientation_already_correct(self):
        nodes = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
        good = [[0, 1, 2, 3]]
        fixed, n = fix_element_orientation(nodes, good)
        assert n == 0

    def test_filter_small_elements(self):
        nodes = np.array([
            [0,0,0],[1,0,0],[0,1,0],[0,0,1],
            [0,0,0],[0.01,0,0],[0,0.01,0],[0,0,0.01]
        ], dtype=float)
        elements = [[0,1,2,3], [4,5,6,7]]
        kept, n_removed = filter_small_elements(nodes, elements, min_edge_mm=0.1)
        assert n_removed == 1
        assert len(kept) == 1

    def test_mesh_quality_report(self):
        nodes = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
        report = mesh_quality_report(nodes, [[0,1,2,3]])
        assert report["n_nodes"] == 4
        assert report["n_elements"] == 1
        assert report["h_min_mm"] > 0
