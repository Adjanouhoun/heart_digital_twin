import numpy as np
import pytest

from app.solver.mechanics.fenicsx_solver import (
    FenicsxSolver,
    MechanicsParameters,
)


@pytest.fixture
def simple_mesh():
    nodes = np.array([
        [0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0],
        [0, 0, 10], [10, 0, 10], [0, 10, 10], [10, 10, 10],
    ], dtype=np.float64)
    elements = np.array([
        [0, 1, 2, 4], [1, 3, 2, 5], [2, 3, 5, 6],
        [3, 7, 5, 6], [1, 5, 4, 3], [2, 6, 4, 3],
    ], dtype=np.int32)
    fibers = np.tile([1.0, 0.0, 0.0], (8, 1))
    return nodes, elements, fibers


def test_endocardial_pressure_is_blocked_by_mesh_quality_gate(simple_mesh):
    nodes, elements, fibers = simple_mesh
    with pytest.raises(RuntimeError, match="porte qualité du maillage"):
        FenicsxSolver().simulate(
            MechanicsParameters(p_endo_kPa=1.0),
            nodes,
            elements,
            fibers,
            np.zeros(len(nodes)),
            "twin1",
            "job1",
        )


def test_default_continuation_threshold_is_preserved():
    assert MechanicsParameters().easy_iteration_threshold == 4


def test_orthotropic_microstructure_contract():
    field = np.array([
        [2.0, 0.0, 0.0, 0.0, 3.0, 0.0],
        [0.0, 4.0, 0.0, 0.0, 0.0, 5.0],
    ])
    fibers, sheets = FenicsxSolver._split_microstructure(field, 2)
    assert np.allclose(np.linalg.norm(fibers, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(sheets, axis=1), 1.0)


def test_orthotropic_microstructure_rejects_fiber_only():
    with pytest.raises(ValueError, match="fibre\\+sheet"):
        FenicsxSolver._split_microstructure(np.ones((4, 3)), 4)


def test_element_projection_is_invariant_to_axis_sign():
    fibers = np.array([
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    ])
    sheets = np.array([
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
    ])
    f0, s0 = FenicsxSolver._project_microstructure_to_elements(
        fibers, sheets, np.array([[0, 1, 2, 3]])
    )
    assert np.allclose(np.abs(f0[0]), [1.0, 0.0, 0.0])
    assert np.allclose(np.abs(s0[0]), [0.0, 1.0, 0.0])
    assert np.isclose(np.dot(f0[0], s0[0]), 0.0)


def test_orthotropic_microstructure_rejects_nonorthogonal_basis():
    field = np.array([[1.0, 0.0, 0.0, 1.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="orthogonale"):
        FenicsxSolver._split_microstructure(field, 1)
