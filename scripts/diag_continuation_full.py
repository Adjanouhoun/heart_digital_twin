"""
Test de continuation COMPLETE, mixte P2-P1, formulation p DIRECTE (Pa).
CORRECTIF vs version precedente : la substitution p_hat=p/kappa amplifiait
le desequilibre d'echelle d'un facteur kappa (erreur de derivation en chaine :
dPi/dp_hat = kappa * dPi/dp), au lieu de l'attenuer. On revient a p brut,
que MUMPS gere deja bien seul (confirme le 07/07 : KSP 1 iteration, 2e6->1.5e-8).
Correctifs conserves :
  1. SNESSetFunctionDomainError (garde-fou J avant assemblage complet)
  3. Continuation ultra-fine (dlam initial 1e-4) + restauration immediate
A lancer en arriere-plan (nohup + caffeinate).
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
import dolfinx.log
dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)
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
u, p = ufl.split(w)                # p DIRECT (Pa), plus de p_hat
w_test = ufl.TestFunction(W)
v, q = ufl.split(w_test)

I = ufl.Identity(3)
F_def = I + ufl.grad(u)
C = F_def.T * F_def
J = ufl.det(F_def)

a_Pa, b = 0.496 * 1000.0, 7.209
kappa = 1.0e6
J_reg = ufl.max_value(J, 1.0e-3)
I1_bar = J_reg ** (-2.0 / 3.0) * ufl.tr(C)
W_iso = (a_Pa / (2.0 * b)) * (ufl.exp(b * (I1_bar - 3.0)) - 1.0)

# Terme volumetrique en p DIRECT (Pa) -- formulation validee le 07/07
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
print(f"DOFs totaux (u P2 + p P1): {n_dofs}", flush=True)

snes = PETSc.SNES().create(msh.comm)
snes.setFunction(pde.F, b_vec)
snes.setJacobian(pde.J, A_mat)
snes.setType("newtonls")
snes.setTolerances(atol=1e-4, rtol=1e-4, stol=0.0, max_it=15)

opts = PETSc.Options()
opts["snes_monitor"] = None
opts["snes_linesearch_monitor"] = None
opts["snes_linesearch_type"] = "bt"
opts["snes_linesearch_order"] = 1
opts["snes_linesearch_max_it"] = 6
opts["snes_linesearch_minlambda"] = 1e-3
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"
snes.setFromOptions()

T_target_Pa = 135.0 * 1000.0
lam = 0.0
dlam = 1.0e-4
DLAM_MIN = 1.0e-6
DLAM_MAX = 0.05
J_MIN = 0.1
MAX_STEPS = 300
t_start = time.time()

w_accepted = w.x.array.copy()
n_its_total = 0
n_steps = 0
min_J_global = 1.0
consecutive_easy = 0

print(f"\n=== CONTINUATION COMPLETE p-direct (T_max=135kPa, dlam0={dlam}) ===", flush=True)

while lam < 1.0 - 1e-9 and n_steps < MAX_STEPS:
    n_steps += 1
    target = min(lam + dlam, 1.0)
    T_act.value = T_target_Pa * target
    t_step = time.time()

    try:
        snes.solve(None, w.vector)
    except Exception as e:
        print(f"[{time.time()-t_start:7.1f}s] EXCEPTION step={n_steps} target={target:.6f}: {e}", flush=True)
        w.x.array[:] = w_accepted
        w.x.scatter_forward()
        dlam *= 0.3
        if dlam < DLAM_MIN:
            print("CONTINUATION STALLED (exception)", flush=True)
            break
        continue

    w.x.scatter_forward()
    its = snes.getIterationNumber()
    reason = snes.getConvergedReason()
    finite = bool(np.isfinite(w.x.array).all())

    if finite:
        pde.J_func.interpolate(pde.J_expr)
        arr = pde.J_func.x.array
        local_min = float(arr.min()) if arr.size else np.inf
        min_j = msh.comm.allreduce(local_min, op=MPI.MIN)
    else:
        min_j = float("-inf")

    accept = (reason > 0) and finite and (min_j > J_MIN)
    dt_step = time.time() - t_step

    if accept:
        lam = target
        w_accepted = w.x.array.copy()
        n_its_total += its
        min_J_global = min_j
        print(f"[{time.time()-t_start:7.1f}s] step={n_steps:3d} OK   "
              f"lam={lam:.6f} T={T_act.value/1000:.4f}kPa its={its} "
              f"min_J={min_j:.4f} dlam={dlam:.6f} dt={dt_step:.1f}s "
              f"domain_err_total={pde.n_domain_errors}", flush=True)
        consecutive_easy = consecutive_easy + 1 if its <= 4 else 0
        if consecutive_easy >= 2:
            dlam = min(dlam * 1.5, DLAM_MAX)
    else:
        w.x.array[:] = w_accepted
        w.x.scatter_forward()
        dlam *= 0.3
        consecutive_easy = 0
        print(f"[{time.time()-t_start:7.1f}s] step={n_steps:3d} REJECT "
              f"target={target:.6f} reason={int(reason)} "
              f"min_J={min_j if np.isfinite(min_j) else 'nan'} "
              f"finite={finite} new_dlam={dlam:.8f} dt={dt_step:.1f}s", flush=True)
        if dlam < DLAM_MIN:
            print(f"CONTINUATION STALLED at lam={lam:.6f}", flush=True)
            break

converged = lam >= 1.0 - 1e-9
print(f"\n=== FIN CONTINUATION ({time.time()-t_start:.1f}s total) ===", flush=True)
print(f"converged: {converged}  lam_final: {lam:.6f}  n_steps: {n_steps}  "
      f"n_its_total: {n_its_total}  domain_errors: {pde.n_domain_errors}", flush=True)

if converged:
    u_sub = w.sub(0).collapse()
    V1 = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (3,)))
    u1 = dolfinx.fem.Function(V1)
    u1.interpolate(u_sub)
    u_arr_dof_order = u1.x.array.reshape(-1, 3)
    dof_coords = V1.tabulate_dof_coordinates()
    coord_to_dof_idx = {tuple(np.round(c, 6)): i for i, c in enumerate(dof_coords)}
    remap = np.array([coord_to_dof_idx[tuple(np.round(n, 6))] for n in nodes])
    u_arr = u_arr_dof_order[remap]

    vol = 0.0
    nodes_def = nodes + u_arr
    for tet in elements_final:
        vv = nodes_def[tet]
        mat = np.array([vv[1]-vv[0], vv[2]-vv[0], vv[3]-vv[0]])
        vol += abs(np.linalg.det(mat)) / 6.0
    vol_mL = vol / 1000.0

    z_mid = (nodes[:, 2].min() + nodes[:, 2].max()) / 2
    mid_mask = np.abs(nodes[:, 2] - z_mid) < 5.0
    mid_nodes = nodes[mid_mask]
    center_xy = mid_nodes[:, :2].mean(0)
    radii = np.linalg.norm(mid_nodes[:, :2] - center_xy, axis=1)
    r_med = np.median(radii)
    endo_r, epi_r = [], []
    for i in np.where(mid_mask)[0]:
        rd_dir = nodes[i, :2] - center_xy
        r = np.linalg.norm(rd_dir)
        if r > 0.1:
            rd_dir /= r
            rd = np.dot(u_arr[i, :2], rd_dir)
            (endo_r if r < r_med else epi_r).append(rd)
    endo_disp = np.mean(endo_r) if endo_r else 0.0
    epi_disp = np.mean(epi_r) if epi_r else 0.0

    print(f"\n=== RESULTAT PHYSIOLOGIQUE ===", flush=True)
    print(f"volume_tissue_mL : {vol_mL:.2f}", flush=True)
    print(f"endo_radial_disp_mm : {endo_disp:.4f}", flush=True)
    print(f"epi_radial_disp_mm  : {epi_disp:.4f}", flush=True)
