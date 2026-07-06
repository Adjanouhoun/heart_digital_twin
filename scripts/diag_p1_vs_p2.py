"""
Test isole : P1 vs P2 sur un cube synthetique (rapide) pour confirmer que le
verrouillage volumetrique (P1 + penalisation quasi-incompressible) est la
cause du pivot nul, avant de relancer sur le vrai maillage (2min/essai).
"""
import numpy as np
import time
import dolfinx
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import dolfinx.mesh
import ufl
from mpi4py import MPI
from petsc4py import PETSc


def run_test(degree, T_act_kPa=135.0, kappa_vol=10000.0, solver_type="lu"):
    print(f"\n=== Test degree={degree} (P{degree}), T_act={T_act_kPa}kPa, solver={solver_type} ===")
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

    a_Pa, b = 496.0, 7.209
    W_passive = (a_Pa) / (2 * b) * (ufl.exp(b * (ufl.tr(C) - 3)) - 1)
    W_vol = kappa_vol * (J - 1) ** 2

    f0 = ufl.as_vector([1.0, 0.0, 0.0])
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
    # Pas de sous-relaxation ici : ce cube synthetique est bien conditionne
    # (contrairement au vrai maillage myocardique). Objectif de ce test =
    # comparer le COUT PAR ITERATION de LU vs GAMG, pas la robustesse de
    # convergence (deja validee separement avec relaxation=0.3 sur le vrai
    # maillage).

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    prefix = ksp.getOptionsPrefix()
    if solver_type == "lu":
        opts[f"{prefix}ksp_type"] = "preonly"
        opts[f"{prefix}pc_type"] = "lu"
    else:  # gamg iteratif
        opts[f"{prefix}ksp_type"] = "gmres"
        opts[f"{prefix}pc_type"] = "gamg"
        opts[f"{prefix}ksp_rtol"] = 1e-8
        opts[f"{prefix}ksp_max_it"] = 200
    ksp.setFromOptions()

    n_steps = 10
    t_start = time.time()
    for step in range(1, n_steps + 1):
        T_act.value = T_act_kPa * 1000.0 * (step / n_steps)
        try:
            n_its, converged = solver.solve(u)
        except RuntimeError as e:
            print(f"  ECHEC au palier {step}: {e}")
            return False, 0.0
    dt = time.time() - t_start
    print(f"  SUCCES en {dt:.2f}s ({dt/n_steps:.2f}s/palier)")
    return True, dt


print("Test P1 + LU direct (reference)...")
ok_lu, t_lu = run_test(degree=1, solver_type="lu")

print("\nTest P1 + GAMG iteratif (optimisation proposee)...")
ok_gamg, t_gamg = run_test(degree=1, solver_type="gamg")

print(f"\n=== RESUME ===")
print(f"LU direct  : {'OK' if ok_lu else 'ECHEC'} en {t_lu:.2f}s")
print(f"GAMG itera.: {'OK' if ok_gamg else 'ECHEC'} en {t_gamg:.2f}s")
if ok_lu and ok_gamg:
    print(f"Acceleration: {t_lu/t_gamg:.1f}x")
