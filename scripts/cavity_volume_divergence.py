"""
Calcul EXACT du volume de cavite ventriculaire par le theoreme de la
divergence, sur la surface endocardique fermee du maillage.

  V = (1/3) * somme_facettes ( centroide . normale_sortante ) * aire

Aucune hypothese de forme (ni circularite, ni percentile) -- exact pour
toute geometrie fermee.

VALIDATION : applique au maillage au repos, doit retrouver le volume de
cavite mesure directement dans la segmentation ACDC (label 3 = cavite VG),
soit 295.51 mL pour patient001 (mesure par comptage de voxels sur la
verite terrain ACDC, verifiee 2026-07-13).

Contexte : la methode precedente (empilement de disques, rayon = 10e
percentile par tranche) donnait 209 mL au lieu de 295 mL -- biais de -29%,
cause directe de l'EF erronee (24% au lieu de la valeur physiologique).
"""
import sys
sys.path.insert(0, "/cdt" if len(sys.argv) > 1 and sys.argv[1] == "docker" else ".")
import numpy as np
from collections import defaultdict


def extract_boundary_faces(elements):
    """Facettes de bord = celles appartenant a UN SEUL tetraedre."""
    face_count = defaultdict(list)
    for cell_idx, tet in enumerate(elements):
        a, b, c, d = tet
        for f in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            face_count[tuple(sorted(f))].append((cell_idx, f))
    boundary = [v[0] for v in face_count.values() if len(v) == 1]
    return boundary  # liste de (cell_idx, (n0,n1,n2)) avec ordre d'origine


def outward_normal(nodes, tet, face):
    """Normale sortante du tetraedre (pointe hors de l'element)."""
    p0, p1, p2 = nodes[list(face)]
    n = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return None, 0.0
    area = 0.5 * norm
    n = n / norm
    # Le 4e sommet du tet (celui hors de la facette) sert de reference interne
    opposite = [v for v in tet if v not in face][0]
    to_inside = nodes[opposite] - p0
    if np.dot(n, to_inside) > 0:
        n = -n  # on veut la normale SORTANTE du tet
    return n, area


def classify_and_compute(nodes, elements, base_tol_mm=1.0, verbose=True):
    boundary = extract_boundary_faces(elements)
    z_max = nodes[:, 2].max()
    center_xy = nodes[:, :2].mean(axis=0)

    endo_faces, epi_faces, base_faces = [], [], []

    for cell_idx, face in boundary:
        tet = elements[cell_idx]
        n, area = outward_normal(nodes, tet, face)
        if n is None:
            continue
        centroid = nodes[list(face)].mean(axis=0)

        # Base : facette dont tous les noeuds sont au sommet (plan basal)
        if np.all(nodes[list(face)][:, 2] > z_max - base_tol_mm):
            base_faces.append((face, n, area, centroid))
            continue

        # Endocarde vs epicarde : la normale sortante de l'endocarde pointe
        # VERS l'axe central (produit scalaire negatif avec le vecteur radial)
        radial = centroid[:2] - center_xy
        r = np.linalg.norm(radial)
        if r < 1e-9:
            continue
        radial_unit = np.array([radial[0] / r, radial[1] / r, 0.0])
        if np.dot(n, radial_unit) < 0:
            endo_faces.append((face, n, area, centroid))
        else:
            epi_faces.append((face, n, area, centroid))

    if verbose:
        print(f"  Facettes de bord : {len(boundary)}")
        print(f"    endocarde : {len(endo_faces)}")
        print(f"    epicarde  : {len(epi_faces)}")
        print(f"    base      : {len(base_faces)}")

    # Volume enferme par la surface endocardique + couvercle basal.
    # Pour la cavite, la normale doit pointer VERS L'EXTERIEUR DE LA CAVITE,
    # c'est-a-dire l'oppose de la normale sortante du tissu.
    V = 0.0
    for face, n, area, centroid in endo_faces:
        n_cav = -n  # normale sortante de la CAVITE
        V += np.dot(centroid, n_cav) * area

    # Couvercle basal : disque plan fermant la cavite au niveau z_max.
    # Sa normale sortante de la cavite pointe vers +z.
    endo_base_nodes = []
    for face, n, area, centroid in endo_faces:
        for v in face:
            if nodes[v, 2] > z_max - base_tol_mm:
                endo_base_nodes.append(v)
    endo_base_nodes = sorted(set(endo_base_nodes))

    if len(endo_base_nodes) >= 3:
        ring = nodes[endo_base_nodes]
        c_ring = ring.mean(axis=0)
        ang = np.arctan2(ring[:, 1] - c_ring[1], ring[:, 0] - c_ring[0])
        order = np.argsort(ang)
        ring = ring[order]
        for i in range(len(ring)):
            p1 = ring[i]
            p2 = ring[(i + 1) % len(ring)]
            tri = np.array([c_ring, p1, p2])
            nn = np.cross(p1 - c_ring, p2 - c_ring)
            a = 0.5 * np.linalg.norm(nn)
            if a < 1e-12:
                continue
            nn = nn / (2 * a)
            if nn[2] < 0:
                nn = -nn  # sortante de la cavite = +z
            V += np.dot(tri.mean(axis=0), nn) * a
        if verbose:
            print(f"    couvercle basal : {len(ring)} noeuds")

    return V / 3.0 / 1000.0  # mm^3 -> mL


if __name__ == "__main__":
    MESH_DIR = "reports/meshes_acdc/meshes"
    PATIENT = "patient001_coarse5"

    with open(f"{MESH_DIR}/{PATIENT}.pts") as f:
        n = int(f.readline())
        nodes = np.array([list(map(float, f.readline().split())) for _ in range(n)])
    with open(f"{MESH_DIR}/{PATIENT}.elem") as f:
        ne = int(f.readline())
        elements = np.array([list(map(int, f.readline().split()[1:5]))
                              for _ in range(ne)], dtype=np.int64)

    print(f"Maillage : {len(nodes)} noeuds, {len(elements)} tets\n")
    V = classify_and_compute(nodes, elements)

    print(f"\n=== VOLUME DE CAVITE (theoreme de la divergence) ===")
    print(f"  Calcule    : {V:.2f} mL")
    print(f"  Reference  : 295.51 mL  (segmentation ACDC, label 3, comptage voxels)")
    print(f"  Ecart      : {100*(V-295.51)/295.51:+.1f} %")
    print(f"\n  (Ancienne methode par empilement de disques : 209.21 mL, ecart -29.2 %)")
