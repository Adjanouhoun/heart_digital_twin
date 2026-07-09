"""
Test cible : create_mesh en MPI avec GhostMode.shared_facet explicite,
au lieu du GhostMode.none automatique choisi par dolfinx quand comm.size>1
et qu'aucun partitionneur n'est precise. Fonde sur inspection du code
source dolfinx.mesh.create_mesh (v0.7.x) + issue GitHub FEniCS/dolfinx#994
("Bus errors when running in parallel inside Docker containers", SCOTCH).
"""
import sys
sys.path.insert(0, "/cdt")
import numpy as np

from app.core.units import filter_small_elements, filter_degenerate_dihedral

MESH_DIR = "/cdt/reports/meshes_acdc/meshes"
PATIENT = "patient001"

with open(f"{MESH_DIR}/{PATIENT}.pts") as f:
    f.readline()
    nodes_mm = np.array([list(map(float, l.split())) for l in f])
with open(f"{MESH_DIR}/{PATIENT}.elem") as f:
    f.readline()
    elements = [[int(x) for x in l.split()[1:5]] for l in f]

elements_f, _ = filter_small_elements(nodes_mm, elements, 0.3)
keep = sorted(set(v for e in elements_f for v in e))
node_map = {old: new for new, old in enumerate(keep)}
nodes = nodes_mm[keep]
elements_final = np.array([[node_map[v] for v in e] for e in elements_f], dtype=np.int64)
elements_final, _, _ = filter_degenerate_dihedral(nodes, elements_final, min_angle_deg=15.0)
elements_final = np.array(elements_final, dtype=np.int64)
keep2 = sorted(set(v for e in elements_final for v in e))
node_map2 = {old: new for new, old in enumerate(keep2)}
nodes = nodes[keep2]
elements_final = np.array([[node_map2[v] for v in e] for e in elements_final], dtype=np.int64)

import dolfinx
import dolfinx.mesh
import ufl
from mpi4py import MPI

print(f"n_ranks: {MPI.COMM_WORLD.size}", flush=True)
print(f"Maillage: {len(nodes)} nodes, {len(elements_final)} tets", flush=True)

domain = ufl.Mesh(ufl.VectorElement("Lagrange", ufl.tetrahedron, 1))

# --- Partitionneur EXPLICITE avec GhostMode.shared_facet ---
# (au lieu du GhostMode.none automatique de dolfinx quand comm.size>1
# et qu'aucun partitionneur n'est fourni -- source du bug suspecte)
partitioner = dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)

try:
    msh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, elements_final, nodes, domain,
                                     partitioner=partitioner)
    print(f"OK cree avec succes (GhostMode.shared_facet), "
          f"n_cells locales: {msh.topology.index_map(3).size_local}", flush=True)
except Exception as e:
    print(f"ECHEC avec GhostMode.shared_facet: {e}", flush=True)
    import traceback
    traceback.print_exc()
