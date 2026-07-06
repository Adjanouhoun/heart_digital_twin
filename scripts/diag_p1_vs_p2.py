"""
Test isole : P1 vs P2 sur un cube synthetique (rapide) pour confirmer que le
verrouillage volumetrique (P1 + penalisation quasi-incompressible) est la
cause du pivot nul, avant de relancer sur le vrai maillage (2min/essai).
"""
import numpy as np
import dolfinx
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import dolfinx.mesh
import ufl
from mpi4py import MPI
from petsc4py import PETSc


def run_test(degree, T_act_kPa=135.0, kappa_vol=10000.0):
    print(f"\n=== Test degree={degree} (P{degree}), T_act={T_act_kPa}kPa ===")
    domain = dolfinx.mesh.create_box(
        MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]],
        [4, 4, 4], dolfinx.mesh.CellType.tetrahedron
    )
    V = dolfinx.fem.functionspace(domain, ("Lagrange", degree, (3,)))
    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)

    I = ufl.Identity(3)
    F_def = I + ufl.grad(u)
    C = F_def.T * F_def
    J = ufl.det(F_def)

    a_Pa, b = 496.0, 7.209  # a_kPa=0.496 -> Pa
    W_passive = (a_Pa) / (2 * b) * (ufl.exp(b * (ufl.tr(C) - 3)) - 1)
    W_vol = kappa_vol * (J - 1) ** 2

    f0 = ufl.as_vector([1.0, 0.0, 0.0])  # fibre uniforme (test simplifie)
    T_act = dolfinx.fem.Constant(domain, 0.0)

    F_passive = ufl.derivative((W_passive + W_vol) * ufl.dx, u, v)
    F_active = T_act * ufl.inner(F_def * ufl.outer(f0, f0), ufl.grad(v)) * ufl.dx
    F_form = F_passive + F_active

    base_dofs = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[2], 0.0))
    bc = dolfinx.fem.dirichletbc(np.zeros(3), base_dofs, V)

    dF = ufl.derivative(F_form, u)
    problem = dolfinx.fem.petsc.NonlinearProblem(F_form, u, bcs=[bc], J=dF)
    solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
    solver.atol, solver.rtol, solver.max_it = 1e-6, 1e-4, 50
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    prefix = ksp.getOptionsPrefix()
    opts[f"{prefix}ksp_type"] = "preonly"
    opts[f"{prefix}pc_type"] = "lu"
    ksp.setFromOptions()

    n_steps = 10
    for step in range(1, n_steps + 1):
        T_act.value = T_act_kPa * 1000.0 * (step / n_steps)
        try:
            n_its, converged = solver.solve(u)
        except RuntimeError as e:
            print(f"  ECHEC au palier {step} (T={T_act.value/1000:.1f}kPa): {e}")
            return False
    print(f"  SUCCES: tous les {n_steps} paliers convergent")
    return True


print("Test P1 (formulation actuelle, deplacement pur, attend un ECHEC)...")
ok_p1 = run_test(degree=1)

print("\nTest P2 (fix propose, attend un SUCCES)...")
ok_p2 = run_test(degree=2)

print(f"\n=== RESUME === P1: {'OK' if ok_p1 else 'ECHEC'} | P2: {'OK' if ok_p2 else 'ECHEC'}")
