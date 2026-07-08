"""
Diagnostic P15 (MPI) : localiser la ou les facettes partagees par PLUS DE
DEUX tetraedres dans le maillage filtre (elements_final), cause du
RuntimeError dolfinx "A facet is connected to more than two cells" lors
du partitionnement parallele (SCOTCH). Ce defaut topologique est TOLERE
par dolfinx en sequentiel (n=1) mais casse la construction du graphe dual
en parallele (n>1).

Purement geometrique -- pas besoin de dolfinx, execution locale rapide.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict
from app.core.units import filter_small_elements, filter_degenerate_dihedral

MESH_DIR = "reports/meshes_acdc/meshes"
PATIENT = "patient001"

print("=== Chargement + filtrage (identique au pipeline solveur) ===")
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

print(f"Maillage final : {len(nodes)} noeuds, {len(elements_final)} tetraedres\n")

# --- Construction du dictionnaire facette -> liste de cellules ---
# Chaque tet (a,b,c,d) a 4 faces triangulaires. On identifie une face par
# le tuple TRIE de ses 3 indices de noeuds (independant de l'orientation).
face_to_cells = defaultdict(list)

for cell_idx, tet in enumerate(elements_final):
    a, b, c, d = tet
    faces = [
        tuple(sorted((a, b, c))),
        tuple(sorted((a, b, d))),
        tuple(sorted((a, c, d))),
        tuple(sorted((b, c, d))),
    ]
    for f in faces:
        face_to_cells[f].append(cell_idx)

print(f"Nombre total de facettes distinctes : {len(face_to_cells)}")

counts = np.array([len(v) for v in face_to_cells.values()])
print(f"Facettes avec 1 cellule (bord)     : {(counts == 1).sum()}")
print(f"Facettes avec 2 cellules (interne) : {(counts == 2).sum()}")
print(f"Facettes avec 3+ cellules (BUG)    : {(counts >= 3).sum()}")

bad_faces = {f: cells for f, cells in face_to_cells.items() if len(cells) >= 3}

if not bad_faces:
    print("\nAucune facette pathologique trouvee -- le defaut est ailleurs "
          "(peut-etre lie a la renumerotation MPI elle-meme, pas aux donnees).")
else:
    print(f"\n=== {len(bad_faces)} FACETTE(S) PATHOLOGIQUE(S) TROUVEE(S) ===\n")
    coupable_cells = set()
    for face, cells in bad_faces.items():
        print(f"Facette (noeuds {face}) partagee par {len(cells)} cellules : {cells}")
        coupable_cells.update(cells)

    print(f"\n{len(coupable_cells)} tetraedre(s) implique(s) au total : "
          f"{sorted(coupable_cells)}")

    print("\n--- Detail des tetraedres coupables ---")
    for c in sorted(coupable_cells):
        tet = elements_final[c]
        v = nodes[tet]
        vol = abs(np.linalg.det(np.array([v[1]-v[0], v[2]-v[0], v[3]-v[0]]))) / 6.0
        edges = [
            np.linalg.norm(v[1]-v[0]), np.linalg.norm(v[2]-v[0]),
            np.linalg.norm(v[3]-v[0]), np.linalg.norm(v[2]-v[1]),
            np.linalg.norm(v[3]-v[1]), np.linalg.norm(v[3]-v[2]),
        ]
        print(f"  tet[{c}] noeuds={tet.tolist()} volume={vol:.6e} "
              f"aretes_min={min(edges):.4f} aretes_max={max(edges):.4f}")

    # Sauvegarde pour reutilisation (filtre correctif ulterieur)
    np.save("bad_cells_indices.npy", np.array(sorted(coupable_cells)))
    print("\nIndices sauvegardes dans bad_cells_indices.npy")
