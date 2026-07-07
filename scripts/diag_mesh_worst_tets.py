"""
Diagnostic qualite maillage AU REPOS (avant deformation).

Objectif P15 : identifier les tetraedres quasi-degeneres qui s'inversent
au moindre chargement (min_J<0 des 0.8% de charge). L'angle diedre seul
(filtre actuel, seuil 15deg) rate les slivers plats a grand volume mais
faible epaisseur. On ajoute deux metriques plus severes :
  - radius_ratio  = 3 * r_in / r_circ   (1 = regulier, ->0 = degenere)
  - min_dihedral  = plus petit angle diedre (deg)
On liste les pires elements et on compte ceux sous des seuils critiques.
"""
import numpy as np
import json
import sys
from pathlib import Path

MESH_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/patient001/mesh.json"


def load_mesh(path):
    p = Path(path)
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        nodes = np.asarray(d["nodes"], dtype=float)
        elements = np.asarray(d["elements"], dtype=int)
    elif p.suffix == ".npz":
        d = np.load(p)
        nodes = d["nodes"].astype(float)
        elements = d["elements"].astype(int)
    else:
        raise SystemExit(f"Format non gere: {p.suffix} (attendu .json ou .npz)")
    return nodes, elements


def tet_metrics(v):
    a, b, c, d = v[0], v[1], v[2], v[3]
    vol = np.dot(np.cross(b - a, c - a), d - a) / 6.0

    edges = [b - a, c - a, d - a, c - b, d - b, d - c]
    edge_len = [np.linalg.norm(e) for e in edges]

    faces = [(a, b, c), (a, b, d), (a, c, d), (b, c, d)]
    area_sum = 0.0
    for f in faces:
        area_sum += 0.5 * np.linalg.norm(
            np.cross(f[1] - f[0], f[2] - f[0]))
    r_in = 3.0 * abs(vol) / area_sum if area_sum > 0 else 0.0

    la, lb, lc = edge_len[0], edge_len[1], edge_len[2]
    ld, le, lf = edge_len[5], edge_len[4], edge_len[3]
    prod = (la * ld + lb * le + lc * lf)
    r_circ = np.sqrt(
        abs((la * ld + lb * le + lc * lf)
            * (-la * ld + lb * le + lc * lf)
            * (la * ld - lb * le + lc * lf)
            * (la * ld + lb * le - lc * lf))
    ) / (24.0 * abs(vol)) if abs(vol) > 1e-14 else np.inf

    radius_ratio = (3.0 * r_in / r_circ) if np.isfinite(r_circ) and r_circ > 0 else 0.0

    normals = []
    for f in faces:
        n = np.cross(f[1] - f[0], f[2] - f[0])
        nn = np.linalg.norm(n)
        normals.append(n / nn if nn > 1e-14 else n)
    min_dih = 180.0
    for i in range(4):
        for j in range(i + 1, 4):
            cosang = np.clip(-np.dot(normals[i], normals[j]), -1.0, 1.0)
            ang = np.degrees(np.arccos(cosang))
            min_dih = min(min_dih, ang)

    return vol, radius_ratio, min_dih


def main():
    nodes, elements = load_mesh(MESH_PATH)
    print(f"Maillage : {len(nodes)} noeuds, {len(elements)} tetraedres")
    print(f"Source   : {MESH_PATH}\n")

    vols = np.zeros(len(elements))
    rr = np.zeros(len(elements))
    dih = np.zeros(len(elements))
    for i, tet in enumerate(elements):
        vols[i], rr[i], dih[i] = tet_metrics(nodes[tet])

    n_neg = int((vols <= 0).sum())
    print(f"Tetraedres a volume <= 0 (deja retournes au repos !) : {n_neg}")
    print(f"Volume    : min={vols.min():.4e}  median={np.median(vols):.4e}")
    print(f"RadiusRatio (1=parfait, 0=degenere) :")
    print(f"           min={rr.min():.4f}  median={np.median(rr):.4f}")
    for thr in (0.05, 0.1, 0.2):
        print(f"           < {thr} : {(rr < thr).sum()} tets")
    print(f"MinDihedral (deg) :")
    print(f"           min={dih.min():.2f}  median={np.median(dih):.2f}")
    for thr in (1, 5, 10, 15):
        print(f"           < {thr:2d}deg : {(dih < thr).sum()} tets")

    print("\n--- 15 PIRES tetraedres (par radius_ratio) ---")
    worst = np.argsort(rr)[:15]
    print(f"{'elem':>7} {'radiusRatio':>12} {'minDihedral':>12} {'volume':>12}")
    for e in worst:
        print(f"{e:>7} {rr[e]:>12.5f} {dih[e]:>12.3f} {vols[e]:>12.4e}")


if __name__ == "__main__":
    main()
