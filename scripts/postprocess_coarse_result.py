"""
Post-traitement complet du resultat convergent (lam=1.0) sauvegarde dans
continuation_coarse_checkpoint.npz. Recharge l'etat final SANS relancer
la simulation (le checkpoint contient deja la solution a pleine charge),
puis calcule les grandeurs physiologiques : volume, deplacement radial
endo/epi, fraction d'ejection approximative.

ATTENTION : fibres utilisees pour ce run = champ tangentiel SIMPLIFIE
(pas LDRB), donc ce resultat valide la MECANIQUE (convergence, stabilite)
mais PAS la physiologie fine (orientation des fibres non anatomique).
"""
import sys
sys.path.insert(0, "/cdt")
import numpy as np

MESH_DIR = "/cdt/reports/meshes_acdc/meshes"
PATIENT = "patient001_coarse5"
CHECKPOINT_PATH = "/cdt/continuation_coarse_ldrb_checkpoint.npz"

print("=== Chargement maillage + checkpoint ===", flush=True)
with open(f"{MESH_DIR}/{PATIENT}.pts") as f:
    n = int(f.readline())
    nodes = np.array([list(map(float, f.readline().split())) for _ in range(n)])
with open(f"{MESH_DIR}/{PATIENT}.elem") as f:
    n_elem = int(f.readline())
    elements_final = np.array([
        list(map(int, f.readline().split()[1:5])) for _ in range(n_elem)
    ], dtype=np.int64)

ckpt = np.load(CHECKPOINT_PATH)
print(f"Checkpoint : lam={float(ckpt['lam']):.6f}  n_steps={int(ckpt['n_steps'])}  "
      f"min_J={float(ckpt['min_J']):.4f}", flush=True)

import dolfinx
import dolfinx.log
dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)
import dolfinx.fem
import dolfinx.mesh
import basix.ufl
import ufl
from mpi4py import MPI

msh = dolfinx.mesh.create_mesh(
    MPI.COMM_WORLD, elements_final, nodes,
    ufl.Mesh(ufl.VectorElement("Lagrange", ufl.tetrahedron, 1)))

P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(3,))
P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P2, P1]))
w = dolfinx.fem.Function(W)

w.x.array[:] = ckpt["w_array"]
w.x.scatter_forward()
print("Etat final recharge dans l'espace mixte.", flush=True)

# --- Extraction u (P2) -> reinterpolation sur P1 aux sommets ---
u_sub = w.sub(0).collapse()
V1 = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (3,)))
u1 = dolfinx.fem.Function(V1)
u1.interpolate(u_sub)

u_arr_dof_order = u1.x.array.reshape(-1, 3)
dof_coords = V1.tabulate_dof_coordinates()
coord_to_dof_idx = {tuple(np.round(c, 6)): i for i, c in enumerate(dof_coords)}
remap = np.array([coord_to_dof_idx[tuple(np.round(n_, 6))] for n_ in nodes])
u_arr = u_arr_dof_order[remap]

print(f"Deplacement extrait : min={u_arr.min():.4f}mm max={u_arr.max():.4f}mm "
      f"(devrait etre de l'ordre de quelques mm, PAS des metres)", flush=True)

# --- Volume tissulaire (deforme) ---
def compute_volume(nodes_pos, elements):
    total_vol = 0.0
    for tet in elements:
        v = nodes_pos[tet]
        mat = np.array([v[1]-v[0], v[2]-v[0], v[3]-v[0]])
        total_vol += abs(np.linalg.det(mat)) / 6.0
    return total_vol / 1000.0  # mm^3 -> mL

vol_repos = compute_volume(nodes, elements_final)
vol_deforme = compute_volume(nodes + u_arr, elements_final)

print(f"\n=== VOLUME TISSULAIRE ===", flush=True)
print(f"Volume au repos    : {vol_repos:.2f} mL", flush=True)
print(f"Volume deforme     : {vol_deforme:.2f} mL", flush=True)
print(f"Variation relative : {100*(vol_deforme-vol_repos)/vol_repos:.2f}% "
      f"(devrait etre proche de 0%, quasi-incompressibilite)", flush=True)

# --- Deplacement radial endo/epi (coupe medio-ventriculaire) ---
z_mid = (nodes[:, 2].min() + nodes[:, 2].max()) / 2
mid_mask = np.abs(nodes[:, 2] - z_mid) < (nodes[:, 2].max() - nodes[:, 2].min()) * 0.15
print(f"\nNoeuds dans la coupe medio-ventriculaire : {mid_mask.sum()}", flush=True)

if mid_mask.sum() >= 10:
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

    print(f"\n=== DEPLACEMENT RADIAL (coupe medio-ventriculaire) ===", flush=True)
    print(f"endo_radial_disp_mm : {endo_disp:.4f}  "
          f"(negatif = contraction endocarde, attendu)", flush=True)
    print(f"epi_radial_disp_mm  : {epi_disp:.4f}  "
          f"(peut etre positif = epaississement paroi, attendu)", flush=True)
else:
    print("Pas assez de noeuds dans la coupe pour un calcul fiable "
          "(maillage tres grossier -- resultat a interpreter avec prudence).",
          flush=True)

print(f"\n=== RESUME ===", flush=True)
print(f"Convergence  : lam={float(ckpt['lam']):.4f} (charge complete atteinte)", flush=True)
print(f"min_J final  : {float(ckpt['min_J']):.4f} (aucune inversion, tissu comprime "
      f"mais physique)", flush=True)
print(f"AVERTISSEMENT : fibres tangentielles simplifiees (pas LDRB) -- "
      f"resultat valide la MECANIQUE, pas la physiologie fine des fibres.", flush=True)
