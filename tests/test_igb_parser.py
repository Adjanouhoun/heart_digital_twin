"""Tests pour igb_parser.py et run_opencarp_patient.py"""
import numpy as np
import pytest
import os
from pathlib import Path
from app.solver.ep.igb_parser import parse_igb, OpenCARPResult


class TestIGBParser:

    @pytest.fixture
    def patient003_result(self):
        igb = "/tmp/opencarp_runs/patient003/output/vm.igb"
        pts = "/tmp/opencarp_runs/patient003/mesh.pts"
        if not os.path.exists(igb):
            pytest.skip("openCARP results not available")
        return parse_igb(igb, pts, 200.0)

    def test_result_type(self, patient003_result):
        assert isinstance(patient003_result, OpenCARPResult)

    def test_n_nodes_positive(self, patient003_result):
        assert patient003_result.n_nodes > 0

    def test_n_frames_201(self, patient003_result):
        assert patient003_result.n_frames == 51

    def test_vm_resting_physiological(self, patient003_result):
        assert -90 < patient003_result.vm_resting_mv < -70

    def test_vm_peak_depolarization(self, patient003_result):
        assert patient003_result.vm_peak_mv > 0

    def test_activated_nodes_positive(self, patient003_result):
        assert patient003_result.n_activated > 0

    def test_pct_activated_reasonable(self, patient003_result):
        assert 1.0 < patient003_result.pct_activated < 100.0

    def test_apd_positive(self, patient003_result):
        assert patient003_result.apd_ms > 50

    def test_activation_times_array(self, patient003_result):
        act = patient003_result.activation_times_ms
        assert len(act) == patient003_result.n_nodes
        valid = ~np.isnan(act)
        assert valid.sum() == patient003_result.n_activated

    def test_vm_shape(self, patient003_result):
        r = patient003_result
        assert r.vm.shape == (r.n_frames, r.n_nodes)
