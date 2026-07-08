"""
Diagnostic P15 v2 : UN SEUL solve mixte P2-P1, avec les 3 correctifs :
  1. SNESSetFunctionDomainError : verif J AVANT assemblage complet, sortie
     immediate si J<=0/NaN (evite les 52s d'assemblage sur config invalide)
  2. Pression adimensionnee p_hat = p/kappa (equilibre l'echelle du residu)
  3. Continuation ultra-fine : dlam initial 1e-4 (pas 2%)
snes_linesearch_monitor active pour VERIFIER empiriquement si domain error
fait decroitre lambda en interne (bt) sans interrompre tout SNESSolve.
A lancer DANS le conteneur dolfinx:v0.7.3.
"""
import sys, time
sys.path.insert(0, "/cdt")
import numpy as np

from app.core.units import filter_small_elements, filter_degenerate_dihedral

MESH_DIR = "/cdt/reports/meshes_acdc/meshes"
PATIENT = "patient001"

print("=== Chargement maillage ===", flush=True)
with open(f"{MESH_DIR}/{PATIENT}.pts") as f:
    f.readline()
    nodes_mm = np.array([list(map(float, l.split())) for l in f])
with open(f"{MESH_DIR}/{PATIENT}.elem") as f:
    f.readline()
    elements = [[int(x) for x in l.split()[1:5]] for l in f]
with open(f"{MESH_DIR}/{PATIENT}_fibers.lon") as f:
    f.readline()
    fibers_raw = [l.strip().split() for l in f]

elements_f, _ = filter_small_elements(nodes_mm, elements, 0.3)
keep = sorted(set(v for e in elements_f for v in e))
node_map = {old: new for new, old in enumerate(keep)}
nodes = nodes_mm[keep]
elements_final = np.array([[node_map[v] for v in e] for e in elements_f], dtype=np.int64)
fibers = np.array([
    list(map(float, fibers_raw[i][:3])) if i < len(fibers_raw) else [1.0, 0.0, 0.0]
    for i in keep
])
elements_final, _, _ = filter_degenerate_dihedral(nodes, elements_final, min_angle_deg=15.0)
elements_final = np.array(elements_final, dtype=np.int64)
keep2 = sorted(set(v for e in elements_final for v in e))
node_map2 = {old: new for new, old in enumerate(keep2)}
nodes = nodes[keep2]
elements_final = np.array([[node_map2[v] for v in e] for e in elements_final], dtype=np.int64)
fibers = fibers[keep2]
print(f"Maillage: {len(nodes)} nodes, {len(elements_final)} tets", flush=True)

import dolfinx
import dolfinx.fem, dolfinx.fem.petsc, dolfinx.mesh
import basix.ufl
import ufl
from mpi4py import MPI
from petsc4py import PETSc

msh = dolfinx.mesh.create_mesh(
    MPI.COMM_WORLD, elements_final, nodes,
    ufl.Mesh(ufl.VectorElement("Lagrange", ufl.tetrahedron, 1)))

P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(3,))
P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P2, P1]))
w = dolfinx.fem.Function(W)
u, p_hat = ufl.split(w)          # CORRECTIF 2 : p_hat = p/kappa, adimensionne
w_test = ufl.TestFunction(W)
v, q_hat = ufl.split(w_test)

I = ufl.Identity(3)
F_def = I + ufl.grad(u)
C = F_def.T * F_def
J = ufl.det(F_def)

a_Pa, b = 0.496 * 1000.0, 7.209
kappa = 1.0e6
J_reg = ufl.max_value(J, 1.0e-3)
I1_bar = J_reg ** (-2.0 / 3.0) * ufl.tr(C)
W_iso = (a_Pa / (2.0 * b)) * (ufl.exp(b * (I1_bar - 3.0)) - 1.0)

p = kappa * p_hat  # pression physique reconstruite pour l'energie
Pi_vol = kappa * p_hat * (J - 1.0) - 0.5 * kappa * p_hat * p_hat
Pi = (W_iso + Pi_vol) * ufl.dx

DG = dolfinx.fem.functionspace(msh, ("DG", 0, (3,)))
f0_func = dolfinx.fem.Function(DG)
fpe = np.zeros((len(elements_final), 3))
for e_idx, tet in enumerate(elements_final):
    avg = fibers[tet].mean(axis=0)
    nrm = np.linalg.norm(avg)
    fpe[e_idx] = avg / nrm if nrm > 1e-8 else np.array([1.0, 0.0, 0.0])
f0_func.x.array[:] = fpe.flatten()
f0 = ufl.as_vector([f0_func[0], f0_func[1], f0_func[2]])

T_act = dolfinx.fem.Constant(msh, PETSc.ScalarType(0.0))
F_passive = ufl.derivative(Pi, w, w_test)
F_active = T_act * ufl.inner(F_def * ufl.outer(f0, f0), ufl.grad(v)) * ufl.dx
F_form = F_passive + F_active
dF = ufl.derivative(F_form, w)

z_max = nodes[:, 2].max()
W0, _ = W.sub(0).collapse()
base_dofs = dolfinx.fem.locate_dofs_geometrical(
    (W.sub(0), W0), lambda x: x[2] > z_max - 5.0)
u_bc = dolfinx.fem.Function(W0)
u_bc.x.array[:] = 0.0
bc = dolfinx.fem.dirichletbc(u_bc, base_dofs, W.sub(0))

DG0s = dolfinx.fem.functionspace(msh, ("DG", 0))
J_expr = dolfinx.fem.Expression(J, DG0s.element.interpolation_points())


class P_:
    def __init__(s, F, w, bc, Jf, msh, J_expr, DG0s):
        s.L = dolfinx.fem.form(F); s.a = dolfinx.fem.form(Jf); s.bc = bc; s.w = w
        s.msh = msh
        s.J_expr = J_expr
        s.J_func = dolfinx.fem.Function(DG0s)
        s.n_domain_errors = 0

    def F(s, snes, x, b):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(s.w.vector)
        s.w.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # --- CORRECTIF 1 : verif J AVANT assemblage complet ---
        try:
            s.J_func.interpolate(s.J_expr)
            arr = s.J_func.x.array
            finite = bool(np.isfinite(arr).all())
            local_min = float(arr.min()) if finite and arr.size else -np.inf
        except Exception:
            finite, local_min = False, -np.inf
        min_j = s.msh.comm.allreduce(local_min, op=MPI.MIN)

        if (not finite) or (min_j <= 1.0e-2):
            s.n_domain_errors += 1
            print(f"    [DOMAIN ERROR #{s.n_domain_errors}] min_J={min_j:.4g} "
                  f"-> snes.setFunctionDomainError()", flush=True)
            snes.setFunctionDomainError()
            return

        with b.localForm() as bl: bl.set(0.0)
        dolfinx.fem.petsc.assemble_vector(b, s.L)
        dolfinx.fem.petsc.apply_lifting(b, [s.a], bcs=[[s.bc]], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, [s.bc], x, -1.0)

    def J(s, snes, x, A, P):
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, s.a, bcs=[s.bc]); A.assemble()


pde = P_(F_form, w, bc, dF, msh, J_expr, DG0s)
b_vec = dolfinx.fem.petsc.create_vector(pde.L)
A_mat = dolfinx.fem.petsc.create_matrix(pde.a)

n_dofs = W.dofmap.index_map.size_global * W.dofmap.index_map_bs
print(f"DOFs totaux (u P2 + p_hat P1): {n_dofs}", flush=True)

snes = PETSc.SNES().create(msh.comm)
snes.setFunction(pde.F, b_vec)
snes.setJacobian(pde.J, A_mat)
snes.setType("newtonls")
snes.setTolerances(atol=1e-6, rtol=1e-6, stol=0.0, max_it=30)

opts = PETSc.Options()
opts["snes_monitor"] = None
opts["snes_linesearch_monitor"] = None   # CLE : voir lambda decroitre en interne
opts["snes_linesearch_type"] = "basic"       # pas de backtracking iteratif
opts["snes_linesearch_damping"] = 0.10        # pas FIXE a 10% du pas Newton (realiste)
opts["snes_max_it"] = 1                       # UNE seule iteration SNES suffit
opts["ksp_monitor"] = None                    # voir OU part le temps (KSP vs assemblage)
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"
snes.setFromOptions()

# --- CORRECTIF 3 : charge quasi-nulle pour ce test isole (0.01%) ---
T_act.value = 0.0001 * 135.0 * 1000.0
print(f"\n=== SOLVE unique a T_act = {T_act.value/1000:.4f} kPa (0.01% de 135) ===", flush=True)
print("Surveiller: 'Line search: lambda = ...' doit apparaitre et DECROITRE\n", flush=True)

t0 = time.time()
try:
    snes.solve(None, w.vector)
except Exception as e:
    print(f"EXCEPTION: {e}", flush=True)
dt = time.time() - t0

print(f"\n=== RESULTAT ({dt:.1f}s) ===", flush=True)
print(f"reason           : {snes.getConvergedReason()}", flush=True)
print(f"iterations       : {snes.getIterationNumber()}", flush=True)
print(f"domain_errors_vus: {pde.n_domain_errors}", flush=True)
try:
    print(f"residu final     : {snes.getFunctionNorm():.3e}", flush=True)
except Exception:
    pass
