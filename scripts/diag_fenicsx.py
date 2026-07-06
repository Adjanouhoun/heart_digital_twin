"""
Diagnostic FEniCSx : reproduit _run_fenicsx avec KSP verbeux (setFromOptions)
pour voir la VRAIE raison de l'echec PETSc KSPSolve.
"""
import sys, time
sys.path.insert(0, "/cdt")
import numpy as np
from app.core.units import filter_small_elements
from app.solver.mechanics.fenicsx_solver import MechanicsParameters

import dolfinx
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import dolfinx.mesh
import ufl
from mpi4py import MPI
from petsc4py import PETSc

MESH_DIR = "/tmp/gmsh_test"
PATIENT = "patient001"

print("=== Chargement maillage ===")
with open(f"{MESH_DIR}/{PATIENT}.pts") as f:
    f.readline()
    nodes_mm = np.array([list(map(float, l.split())) for l in f])
with open(f"{MESH_DIR}/{PATIENT}.elem") as f:
    f.readline()
    elements = [[int(x) for x in l.split()[1:5]] for l in f]
with open(f"{MESH_DIR}/{PATIENT}_fibers.lon") as f:
    f.readline()
    fibers_raw = [l.strip().split() for l in f]

elements_f, n_removed = filter_small_elements(nodes_mm, elements, 0.3)
keep = sorted(set(v for e in elements_f for v in e))
node_map = {old: new for new, old in enumerate(keep)}
nodes = nodes_mm[keep]
elements_np = np.array([[node_map[v] for v in e] for e in elements_f], dtype=np.int64)
fibers = np.array([
    list(map(float, fibers_raw[i][:3])) if i < len(fibers_raw) else [1.0, 0.0, 0.0]
    for i in keep
])
print(f"Final: {len(nodes)} nodes, {len(elements_np)} tets")

params = MechanicsParameters()
T_act_kPa = params.T_max_kPa

print("\n=== Construction du probleme FEM ===")
domain = ufl.Mesh(ufl.VectorElement("Lagrange", ufl.tetrahedron, 1))
msh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, elements_np, nodes, domain)

V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (3,)))
u = dolfinx.fem.Function(V)
v = ufl.TestFunction(V)

I = ufl.Identity(3)
F_def = I + ufl.grad(u)
C = F_def.T * F_def
J = ufl.det(F_def)

a_Pa = params.a_kPa * 1000.0
b = params.b
W_passive = (a_Pa) / (2 * b) * (ufl.exp(b * (ufl.tr(C) - 3)) - 1)
W_vol = params.kappa_vol * (J - 1)**2

DG = dolfinx.fem.functionspace(msh, ("DG", 0, (3,)))
f0_func = dolfinx.fem.Function(DG)
for i in range(min(len(elements_np), len(fibers))):
    f0_func.x.array[i*3:(i+1)*3] = fibers[i]
f0 = ufl.as_vector([f0_func[0], f0_func[1], f0_func[2]])

T_act = dolfinx.fem.Constant(msh, float(T_act_kPa * 1000.0))

F_passive = ufl.derivative((W_passive + W_vol) * ufl.dx, u, v)
F_active = T_act * ufl.inner(F_def * ufl.outer(f0, f0), ufl.grad(v)) * ufl.dx
F_form = F_passive + F_active

z_max = nodes[:, 2].max()
z_threshold = z_max - params.base_bc_depth_mm
base_dofs = dolfinx.fem.locate_dofs_geometrical(V, lambda x: x[2] > z_threshold)
print(f"BC : {len(base_dofs)} DOFs fixes a la base (z > {z_threshold:.1f})")
bc = dolfinx.fem.dirichletbc(np.zeros(3), base_dofs, V)

dF = ufl.derivative(F_form, u)
problem = dolfinx.fem.petsc.NonlinearProblem(F_form, u, bcs=[bc], J=dF)
solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
solver.atol = 1e-6
solver.rtol = 1e-4
solver.max_it = 50

# --- ICI : activer explicitement la verbosite KSP ---
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}ksp_converged_reason"] = None
ksp.setFromOptions()

print("\n=== Newton solve (verbeux) ===")
try:
    n_its, converged = solver.solve(u)
    print(f"converged={converged}, n_its={n_its}")
except Exception as e:
    print(f"\nERREUR: {e}")
    print(f"KSP converged reason: {ksp.getConvergedReason()}")
    import traceback; traceback.print_exc()
