"""
Verification rapide : quelle forme retourne V.tabulate_dof_coordinates()
pour un espace vectoriel (Lagrange 1, bloc 3) ? Necessaire avant de faire
confiance au fix de _run_fenicsx (remap coord->dof).
"""
import numpy as np
import dolfinx
import dolfinx.fem
import dolfinx.mesh
from mpi4py import MPI

domain = dolfinx.mesh.create_box(
    MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]],
    [3, 3, 3], dolfinx.mesh.CellType.tetrahedron
)
V = dolfinx.fem.functionspace(domain, ("Lagrange", 1, (3,)))
u = dolfinx.fem.Function(V)

n_nodes_geom = domain.geometry.x.shape[0]
n_dofs_total = u.x.array.shape[0]
dof_coords = V.tabulate_dof_coordinates()

print(f"n_nodes (geometrie du maillage) = {n_nodes_geom}")
print(f"n_dofs_total (u.x.array)        = {n_dofs_total}  (attendu: 3 x n_nodes = {3*n_nodes_geom})")
print(f"dof_coords.shape                = {dof_coords.shape}")
print(f"  -> Si dof_coords.shape[0] == n_nodes_geom       : 1 ligne par NOEUD (pas de [0::3] a faire)")
print(f"  -> Si dof_coords.shape[0] == n_dofs_total        : 1 ligne par DOF scalaire (besoin de [0::3])")

# Verification directe : les coordonnees dans dof_coords[0::3] correspondent-elles
# bien aux memes points que domain.geometry.x (a une permutation pres) ?
geom_coords = domain.geometry.x
if dof_coords.shape[0] == n_dofs_total:
    candidate = dof_coords[0::3]
elif dof_coords.shape[0] == n_nodes_geom:
    candidate = dof_coords
else:
    candidate = None
    print("FORME INATTENDUE — ni n_nodes ni n_dofs_total, investiguer davantage")

if candidate is not None:
    set_geom = set(tuple(np.round(c, 6)) for c in geom_coords)
    set_cand = set(tuple(np.round(c, 6)) for c in candidate)
    print(f"Coordonnees geometrie : {len(set_geom)} points uniques")
    print(f"Coordonnees candidates: {len(set_cand)} points uniques")
    print(f"Match exact (memes ensembles de points) : {set_geom == set_cand}")
