"""
Genere un champ de fibres SIMPLIFIE (tangentiel autour de l'axe long du
VG) pour un maillage de test. PAS physiologiquement precis (pas de LDRB,
pas de variation transmurale epicarde/endocarde) -- utilise uniquement
pour valider la VITESSE du solveur sur un maillage reduit, pas la
physiologie des resultats.
"""
import sys
sys.path.insert(0, ".")
import numpy as np

PTS_PATH = "reports/meshes_acdc/meshes/patient001_coarse5.pts"
OUT_PATH = "reports/meshes_acdc/meshes/patient001_coarse5_fibers.lon"

with open(PTS_PATH) as f:
    n = int(f.readline())
    nodes = np.array([list(map(float, f.readline().split())) for _ in range(n)])

print(f"Noeuds charges : {len(nodes)}")

center_xy = nodes[:, :2].mean(axis=0)
print(f"Centre XY (axe long approx.) : {center_xy}")

# Fibre tangentielle : perpendiculaire au vecteur radial dans le plan XY
radial = nodes[:, :2] - center_xy
radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
radial_norm[radial_norm < 1e-6] = 1.0
radial_unit = radial / radial_norm

# Tangente = rotation 90 deg du radial dans le plan XY, composante Z nulle
tangent = np.zeros((len(nodes), 3))
tangent[:, 0] = -radial_unit[:, 1]
tangent[:, 1] = radial_unit[:, 0]
tangent[:, 2] = 0.0

with open(OUT_PATH, "w") as f:
    f.write(f"{len(nodes)}\n")
    for t in tangent:
        f.write(f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}\n")

print(f"Fibres ecrites : {OUT_PATH}")
