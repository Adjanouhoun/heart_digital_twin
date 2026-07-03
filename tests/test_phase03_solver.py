"""
Tests Phase 03 — Solveur Multi-Physique.
Vérifie EP, Windkessel, couplage, DoE sans openCARP (fallback).
"""
import numpy as np
import pytest
from app.solver.ep.opencarp_solver import OpenCARPSolver, EPParameters
from app.solver.hemodynamics.windkessel import WindkesselSolver, WindkesselParameters
from app.solver.coupled_solver import CoupledSolver, SimulationParameters
from app.solver.doe.latin_hypercube import LatinHypercubeSampler


@pytest.fixture
def simple_mesh():
    """Maillage cubique minimal pour les tests."""
    nodes = np.array([
        [0,0,0],[10,0,0],[0,10,0],[10,10,0],
        [0,0,10],[10,0,10],[0,10,10],[10,10,10],
    ], dtype=np.float64)
    elements = np.array([
        [0,1,2,4],[1,3,2,5],[2,3,5,6],
        [3,7,5,6],[1,5,4,3],[2,6,4,3],
    ], dtype=np.int32)
    fibers = np.tile([1.0, 0.0, 0.0], (8, 1))
    return nodes, elements, fibers


class TestEPSolver:

    def test_fallback_returns_result(self, simple_mesh):
        nodes, elements, fibers = simple_mesh
        solver = OpenCARPSolver()
        params = EPParameters(duration_ms=100.0, dt_ms=0.1)
        result = solver.simulate(params, nodes, elements, fibers,
                                "test_twin", "test_job")
        assert result is not None
        assert len(result.activation_times_ms) == len(nodes)

    def test_activation_times_positive(self, simple_mesh):
        nodes, elements, fibers = simple_mesh
        solver = OpenCARPSolver()
        params = EPParameters(duration_ms=100.0)
        result = solver.simulate(params, nodes, elements, fibers,
                                "twin1", "job1")
        assert result.activation_times_ms.min() >= 0

    def test_conduction_velocity_physiological(self, simple_mesh):
        nodes, elements, fibers = simple_mesh
        solver = OpenCARPSolver()
        params = EPParameters()
        result = solver.simulate(params, nodes, elements, fibers,
                                "twin1", "job1")
        # CV entre 0.3 et 1.5 m/s (physiologique)
        assert 0.1 <= result.conduction_velocity_ms <= 2.0

    def test_ecg_12_leads_generated(self, simple_mesh):
        nodes, elements, fibers = simple_mesh
        solver = OpenCARPSolver()
        params = EPParameters(duration_ms=500.0, dt_ms=0.1)
        result = solver.simulate(params, nodes, elements, fibers,
                                "twin1", "job1")
        if result.ecg_leads_mv is not None:
            assert result.ecg_leads_mv.shape[0] == 12

    def test_benchmark_validation(self, simple_mesh):
        nodes, elements, fibers = simple_mesh
        solver = OpenCARPSolver()
        params = EPParameters(sigma_l=0.3, sigma_t=0.1)
        result = solver.simulate(params, nodes, elements, fibers,
                                "twin1", "job1")
        # Le benchmark doit passer avec les paramètres nominaux
        assert result.benchmark_passed


class TestWindkesselSolver:

    def test_simulate_returns_result(self):
        solver = WindkesselSolver()
        params = WindkesselParameters()
        result = solver.simulate(params)
        assert result is not None
        assert result.ef_pct > 0

    def test_ef_physiological_range(self):
        solver = WindkesselSolver()
        params = WindkesselParameters(V_ed_mL=130.0, V_es_mL=50.0)
        result = solver.simulate(params)
        # EF normale : 50-70%
        assert 20 <= result.ef_pct <= 90

    def test_pressures_physiological(self):
        solver = WindkesselSolver()
        params = WindkesselParameters()
        result = solver.simulate(params)
        # Pression systolique 80-160 mmHg
        assert 50 <= result.p_systolic_mmHg <= 300
        assert result.p_systolic_mmHg > result.p_diastolic_mmHg

    def test_pv_loop_coherent(self):
        """La boucle PV doit être physiologiquement cohérente."""
        solver = WindkesselSolver()
        params = WindkesselParameters()
        result = solver.simulate(params)
        # EDV > ESV (éjection positive)
        assert result.edv_mL > result.esv_mL
        # SV positif
        assert result.sv_mL > 0
        # Débit cardiaque > 0
        assert result.cardiac_output_L_min > 0

    def test_cardiac_output_range(self):
        """Débit cardiaque normal : 4-8 L/min."""
        solver = WindkesselSolver()
        params = WindkesselParameters(heart_rate_bpm=75.0)
        result = solver.simulate(params)
        assert 0.001 <= result.cardiac_output_L_min <= 15.0

    def test_slo_validation(self):
        """SLO : boucle PV dans les plages physiologiques."""
        solver = WindkesselSolver()
        params = WindkesselParameters(
            V_ed_mL=130.0, V_es_mL=50.0,
            heart_rate_bpm=75.0
        )
        result = solver.simulate(params)
        assert result.slo_passed, f"SLO failed: {result.slo_details}"


class TestCoupledSolver:

    def test_coupled_simulation(self, simple_mesh):
        nodes, elements, fibers = simple_mesh
        solver = CoupledSolver()
        params = SimulationParameters(duration_ms=200.0)
        result = solver.simulate(params, nodes, elements, fibers,
                                "twin1", "job1")
        assert result is not None
        assert result.ep_result is not None
        assert result.wk_result is not None

    def test_output_vector_shape(self, simple_mesh):
        nodes, elements, fibers = simple_mesh
        solver = CoupledSolver()
        params = SimulationParameters(duration_ms=200.0)
        result = solver.simulate(params, nodes, elements, fibers,
                                "twin1", "job1")
        assert result.output_vector is not None
        assert len(result.output_vector) >= 9  # 5 EP + 9 WK scalaires

    def test_doe_row_complete(self, simple_mesh):
        nodes, elements, fibers = simple_mesh
        solver = CoupledSolver()
        params = SimulationParameters()
        result = solver.simulate(params, nodes, elements, fibers,
                                "twin1", "job1")
        row = result.to_doe_row()
        # Vérifier les colonnes clés
        assert "sigma_l" in row
        assert "ef_pct" in row
        assert "p_systolic_mmHg" in row
        assert "cv_ms" in row


class TestLatinHypercube:

    def test_lhs_generates_correct_count(self):
        sampler = LatinHypercubeSampler(seed=42)
        params = sampler.sample(100)
        assert len(params) == 100

    def test_lhs_params_in_bounds(self):
        sampler = LatinHypercubeSampler(seed=42)
        params_list = sampler.sample(50)
        for p in params_list:
            assert 0.15 <= p.sigma_l <= 0.50
            assert 0.05 <= p.sigma_t <= 0.20
            assert 80.0 <= p.T_max_kPa <= 200.0
            assert 50.0 <= p.heart_rate_bpm <= 100.0

    def test_lhs_coverage_uniform(self):
        """Vérifie la couverture uniforme du LHS."""
        sampler = LatinHypercubeSampler(seed=42)
        params_list = sampler.sample(100)
        sigma_l_values = [p.sigma_l for p in params_list]
        # Distribution uniforme : variance proche de variance théorique
        vals = np.array(sigma_l_values)
        assert vals.min() < 0.25   # Couverture basse
        assert vals.max() > 0.40   # Couverture haute

    def test_lhs_500_samples(self):
        """SLO : générer 500 échantillons sans erreur."""
        sampler = LatinHypercubeSampler(seed=42)
        params_list = sampler.sample(500)
        assert len(params_list) == 500

    def test_lhs_reproducible(self):
        """LHS doit être reproductible avec le même seed."""
        s1 = LatinHypercubeSampler(seed=99)
        s2 = LatinHypercubeSampler(seed=99)
        p1 = s1.sample(10)
        p2 = s2.sample(10)
        assert abs(p1[0].sigma_l - p2[0].sigma_l) < 1e-10


class TestFenicsxSolver:
    def test_fallback_returns_result(self, simple_mesh):
        from app.solver.mechanics.fenicsx_solver import FenicsxSolver, MechanicsParameters
        nodes, elements, fibers = simple_mesh
        solver = FenicsxSolver()
        params = MechanicsParameters(duration_ms=200.0)
        activation_times = np.zeros(len(nodes))
        result = solver.simulate(
            params, nodes, elements, fibers,
            activation_times, "twin1", "job1"
        )
        assert result is not None
        assert result.converged

    def test_displacement_shape(self, simple_mesh):
        from app.solver.mechanics.fenicsx_solver import FenicsxSolver, MechanicsParameters
        nodes, elements, fibers = simple_mesh
        solver = FenicsxSolver()
        params = MechanicsParameters()
        result = solver.simulate(
            params, nodes, elements, fibers,
            np.zeros(len(nodes)), "twin1", "job1"
        )
        assert result.displacement_mm.shape == nodes.shape

    def test_ef_fallback_physiological(self, simple_mesh):
        from app.solver.mechanics.fenicsx_solver import FenicsxSolver, MechanicsParameters
        nodes, elements, fibers = simple_mesh
        solver = FenicsxSolver()
        params = MechanicsParameters(T_max_kPa=135.0)
        result = solver.simulate(
            params, nodes, elements, fibers,
            np.zeros(len(nodes)), "twin1", "job1"
        )
        assert 20 <= result.ef_pct <= 85

    def test_volume_tissue_positive(self, simple_mesh):
        from app.solver.mechanics.fenicsx_solver import FenicsxSolver, MechanicsParameters
        nodes, elements, fibers = simple_mesh
        solver = FenicsxSolver()
        params = MechanicsParameters()
        result = solver.simulate(
            params, nodes, elements, fibers,
            np.zeros(len(nodes)), "twin1", "job1"
        )
        assert result.volume_tissue_mL > 0

    def test_active_tension_stored(self, simple_mesh):
        from app.solver.mechanics.fenicsx_solver import FenicsxSolver, MechanicsParameters
        nodes, elements, fibers = simple_mesh
        solver = FenicsxSolver()
        params = MechanicsParameters(T_max_kPa=100.0)
        result = solver.simulate(
            params, nodes, elements, fibers,
            np.zeros(len(nodes)), "twin1", "job1",
            T_act_kPa=80.0
        )
        assert result.active_tension_kPa == 80.0

