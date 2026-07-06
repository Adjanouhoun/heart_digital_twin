"""
Qualite geometrique reelle du maillage Gmsh (angle diedre minimal par tet).
Le filtre h_min>=0.3mm ne detecte que les aretes courtes, PAS les tetraedres
"plats" (mauvais angle) qui degenerent les formulations hyperelastiques.

Angle diedre : angle entre 2 faces partageant une arete. Un tet regulier a
un angle diedre de ~70.5 degres. Un tet degenere (sliver) a un angle proche
de 0 ou 180 degres.
"""
import numpy as np
import time

MESH_DIR = "/tmp/gmsh_test"
PATIENT = "patient001"

print("=== Chargement maillage ===")
with open(f"{MESH_DIR}/{PATIENT}.pts") as f:
    f.readline()
    nodes = np.array([list(map(float, l.split())) for l in f])
with open(f"{MESH_DIR}/{PATIENT}.elem") as f:
    f.readline()
    elements = np.array([[int(x) for x in l.split()[1:5]] for l in f], dtype=np.int64)

print(f"{len(nodes)} nodes, {len(elements)} tets")

# Les 4 faces d'un tet (i,j,k,l) : face opposee a chaque sommet
# F0=(1,2,3) F1=(0,2,3) F2=(0,1,3) F3=(0,1,2)
face_defs = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
face_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # partagent 1 arete chacune

t0 = time.time()
verts = nodes[elements]  # (n_tets, 4, 3)
centroids = verts.mean(axis=1)  # (n_tets, 3)

# Normales des 4 faces (non normalisees d'abord)
face_normals = np.zeros((len(elements), 4, 3))
face_centroids = np.zeros((len(elements), 4, 3))
for f_idx, (a, b, c) in enumerate(face_defs):
    va, vb, vc = verts[:, a], verts[:, b], verts[:, c]
    n = np.cross(vb - va, vc - va)
    face_normals[:, f_idx] = n
    face_centroids[:, f_idx] = (va + vb + vc) / 3.0

# Orienter les normales vers l'exterieur (loin du centroide du tet)
for f_idx in range(4):
    to_face = face_centroids[:, f_idx] - centroids
    dot = np.sum(face_normals[:, f_idx] * to_face, axis=1)
    flip = dot < 0
    face_normals[flip, f_idx] *= -1

norms = np.linalg.norm(face_normals, axis=2, keepdims=True)
norms[norms < 1e-12] = 1e-12
face_normals_unit = face_normals / norms

# Angle diedre pour chaque paire de faces partageant une arete
min_dihedral_deg = np.full(len(elements), 180.0)
for (i, j) in face_pairs:
    cos_angle = np.sum(face_normals_unit[:, i] * face_normals_unit[:, j], axis=1)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    # angle diedre = pi - angle(normales exterieures)
    dihedral_rad = np.pi - np.arccos(cos_angle)
    dihedral_deg = np.degrees(dihedral_rad)
    min_dihedral_deg = np.minimum(min_dihedral_deg, dihedral_deg)

print(f"Calcul termine en {time.time()-t0:.1f}s")

print(f"\n=== Distribution angle diedre minimal ===")
print(f"min={min_dihedral_deg.min():.2f} deg, max={min_dihedral_deg.max():.2f} deg, "
      f"median={np.median(min_dihedral_deg):.2f} deg")

for threshold in [1, 2, 5, 10, 15, 20]:
    n_bad = (min_dihedral_deg < threshold).sum()
    pct = 100 * n_bad / len(elements)
    print(f"  tets avec angle diedre min < {threshold:2d} deg : {n_bad:6d} ({pct:.2f}%)")

print(f"\n=== Pire tet (plus degenere) ===")
worst_idx = np.argmin(min_dihedral_deg)
print(f"element #{worst_idx}, angle={min_dihedral_deg[worst_idx]:.4f} deg, "
      f"noeuds={elements[worst_idx]}")
