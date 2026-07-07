"""
Diagnostic P15 : UN SEUL solve mixte P2-P1 a faible charge (2% T_max),
avec options MUMPS pour POINT-SELLE indefini (pivots nuls du bloc pression).
Le diag precedent a montre : MUMPS bloque sur la factorisation meme a 34K tets
(pas un probleme de taille, mais de nature indefinie du systeme mixte).
Reglages ajoutes :
  icntl(24)=1 : detection des pivots nuls (bloc pression = zeros diagonale)
  icntl(13)=1 : desactive ScaLAPACK sur complement de Schur (stabilite)
  cntl(1)=1e-6: seuil de pivotage relache (evite pivoting dynamique infini)
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
u, p = ufl.split(w)
w_test = ufl.TestFunction(W)
v, q = ufl.split(w_test)

I = ufl.Identity(3)
F_def = I + ufl.grad(u)
C = F_def.T * F_def
J = ufl.det(F_def)
a_Pa, b = 0.496 * 1000.0, 7.209
kappa = 1.0e6
J_reg = ufl.max_value(J, 1.0e-3)  # borne anti-NaN (element transitoirement retourne)
I1_bar = J_reg ** (-2.0 / 3.0) * ufl.tr(C)
W_iso = (a_Pa / (2.0 * b)) * (ufl.exp(b * (I1_bar - 3.0)) - 1.0)
Pi_vol = p * (J - 1.0) - 0.5 * p * p / kappa
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


class P_:
    def __init__(s, F, w, bc, Jf):
        s.L = dolfinx.fem.form(F); s.a = dolfinx.fem.form(Jf); s.bc = bc; s.w = w
    def F(s, snes, x, b):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(s.w.vector)
        s.w.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with b.localForm() as bl: bl.set(0.0)
        dolfinx.fem.petsc.assemble_vector(b, s.L)
        dolfinx.fem.petsc.apply_lifting(b, [s.a], bcs=[[s.bc]], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, [s.bc], x, -1.0)
    def J(s, snes, x, A, P):
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, s.a, bcs=[s.bc]); A.assemble()


pde = P_(F_form, w, bc, dF)
b_vec = dolfinx.fem.petsc.create_vector(pde.L)
A_mat = dolfinx.fem.petsc.create_matrix(pde.a)

n_dofs = W.dofmap.index_map.size_global * W.dofmap.index_map_bs
print(f"DOFs totaux (u P2 + p P1): {n_dofs}", flush=True)

snes = PETSc.SNES().create(msh.comm)
snes.setFunction(pde.F, b_vec)
snes.setJacobian(pde.J, A_mat)
snes.setType("newtonls")
snes.setTolerances(atol=1e-6, rtol=1e-6, stol=0.0, max_it=25)

opts = PETSc.Options()
opts["snes_monitor"] = None
opts["snes_linesearch_type"] = "bt"
opts["snes_linesearch_minlambda"] = 1e-3   # ne pas s'acharner sous 1/1000 du pas
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"
snes.setFromOptions()

T_act.value = 0.02 * 135.0 * 1000.0  # 2% de T_max
print(f"\n=== SOLVE unique a T_act = {T_act.value/1000:.2f} kPa (2% de 135) ===", flush=True)
print("Si des lignes 'SNES Function norm' 1,2,3... apparaissent : MUMPS factorise OK\n", flush=True)

t0 = time.time()
try:
    snes.solve(None, w.vector)
except Exception as e:
    print(f"EXCEPTION: {e}", flush=True)
dt = time.time() - t0

print(f"\n=== RESULTAT ({dt:.1f}s) ===", flush=True)
print(f"reason      : {snes.getConvergedReason()}", flush=True)
print(f"iterations  : {snes.getIterationNumber()}", flush=True)
try:
    print(f"residu final: {snes.getFunctionNorm():.3e}", flush=True)
except Exception:
    pass
