"""
Genere des fibres via LDRBFiberGenerator (app/fibers/ldrb.py, deja utilise
en production dans le pipeline Celery) pour patient001_coarse5_fixed.
Remplace le champ tangentiel simplifie d'hier soir par une vraie variation
transmurale (helice +60 endo -> -60 epi, regle de Bayer 2012), meme si les
champs scalaires sous-jacents (apex-base, endo-epi) sont des proxys
geometriques simplifies (Z-height, distance au centroide) plutot que de
vraies resolutions de Laplace -- suffisant pour une geometrie simple
type VG, mais a noter pour toute publication scientifique future.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from app.fibers.ldrb import LDRBFiberGenerator

MESH_DIR = "reports/meshes_acdc/meshes"
PATIENT = "patient001_coarse5_fixed"

with open(f"{MESH_DIR}/{PATIENT}.pts") as f:
    n = int(f.readline())
    nodes = np.array([list(map(float, f.readline().split())) for _ in range(n)])
with open(f"{MESH_DIR}/{PATIENT}.elem") as f:
    n_elem = int(f.readline())
    elements = np.array([
        list(map(int, f.readline().split()[1:5])) for _ in range(n_elem)
    ], dtype=np.int32)

print(f"Maillage : {len(nodes)} noeuds, {len(elements)} tets")

element_tags = np.ones(len(elements), dtype=np.int32)

ldrb = LDRBFiberGenerator()
result = ldrb.generate(nodes, elements, element_tags)

print(f"LDRB termine en {result.duration_seconds:.3f}s")
print(f"fiber_vectors shape: {result.fiber_vectors.shape}")
print(f"Norme moyenne des fibres (doit etre ~1.0) : "
      f"{np.linalg.norm(result.fiber_vectors, axis=1).mean():.4f}")

# --- Ecriture au format attendu par le solveur (3 colonnes fx fy fz par
#     ligne, PAS le format .lon complet a 6 colonnes fibre+sheet, pour
#     rester compatible avec le parsing existant de test_fenicsx_real.py
#     et diag_continuation_coarse.py qui lisent 3 colonnes) ---
OUT_PATH = f"{MESH_DIR}/{PATIENT}_fibers_ldrb.lon"
with open(OUT_PATH, "w") as f:
    f.write(f"{len(nodes)}\n")
    for fv in result.fiber_vectors:
        f.write(f"{fv[0]:.6f} {fv[1]:.6f} {fv[2]:.6f}\n")

print(f"\nEcrit : {OUT_PATH}")
