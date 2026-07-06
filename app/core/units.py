"""
CDT Units — Conversions et controle qualite maillage.
Regle 4 : contrats d'interface explicites.
"""
import numpy as np


def mm_to_um(nodes_mm: np.ndarray) -> np.ndarray:
    return nodes_mm * 1000.0


def fix_element_orientation(nodes, elements):
    fixed = []
    n_flipped = 0
    for e in elements:
        vol = tet_volume(nodes, e)
        if vol < 0:
            fixed.append([e[0], e[2], e[1], e[3]])
            n_flipped += 1
        else:
            fixed.append(e)
    return fixed, n_flipped


def tet_volume(nodes, tet):
    a = nodes[tet[1]] - nodes[tet[0]]
    b = nodes[tet[2]] - nodes[tet[0]]
    c = nodes[tet[3]] - nodes[tet[0]]
    return np.dot(a, np.cross(b, c)) / 6.0


def filter_small_elements(nodes, elements, min_edge_mm=0.3):
    kept = []
    n_removed = 0
    for e in elements:
        h_min = min(
            np.linalg.norm(nodes[e[i]] - nodes[e[j]])
            for i in range(4) for j in range(i+1, 4)
        )
        if h_min >= min_edge_mm:
            kept.append(e)
        else:
            n_removed += 1
    return kept, n_removed


def filter_degenerate_dihedral(nodes, elements, min_angle_deg=15.0):
    """Filtre les tetraedres a angle diedre minimal trop faible (slivers
    geometriques). DECOUVERT (6 juillet) : le filtre h_min (longueur d'arete)
    ne detecte PAS ces elements — un tet peut avoir des aretes assez longues
    tout en etant quasi plat (angle diedre proche de 0). Meme 1 seul element
    a angle tres faible (<1 deg) suffit a singulariser la matrice tangente
    globale en hyperelasticite (pivot nul PETSc). Seuil 15 deg standard.

    Retourne (elements_filtres, n_removed, min_dihedral_deg_array).
    """
    elements = np.asarray(elements)
    verts = nodes[elements]  # (n_tets, 4, 3)
    centroids = verts.mean(axis=1)

    face_defs = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
    face_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    face_normals = np.zeros((len(elements), 4, 3))
    face_centroids = np.zeros((len(elements), 4, 3))
    for f_idx, (a, b, c) in enumerate(face_defs):
        va, vb, vc = verts[:, a], verts[:, b], verts[:, c]
        n = np.cross(vb - va, vc - va)
        face_normals[:, f_idx] = n
        face_centroids[:, f_idx] = (va + vb + vc) / 3.0

    for f_idx in range(4):
        to_face = face_centroids[:, f_idx] - centroids
        dot = np.sum(face_normals[:, f_idx] * to_face, axis=1)
        face_normals[dot < 0, f_idx] *= -1

    norms = np.linalg.norm(face_normals, axis=2, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    face_normals_unit = face_normals / norms

    min_dihedral_deg = np.full(len(elements), 180.0)
    for (i, j) in face_pairs:
        cos_angle = np.clip(
            np.sum(face_normals_unit[:, i] * face_normals_unit[:, j], axis=1),
            -1.0, 1.0)
        dihedral_deg = np.degrees(np.pi - np.arccos(cos_angle))
        min_dihedral_deg = np.minimum(min_dihedral_deg, dihedral_deg)

    keep_mask = min_dihedral_deg >= min_angle_deg
    n_removed = int((~keep_mask).sum())
    return elements[keep_mask].tolist(), n_removed, min_dihedral_deg


def mesh_quality_report(nodes, elements):
    edges = []
    for e in elements:
        for i in range(4):
            for j in range(i+1, 4):
                edges.append(np.linalg.norm(nodes[e[i]] - nodes[e[j]]))
    edges = np.array(edges)
    return {
        "n_nodes": len(nodes),
        "n_elements": len(elements),
        "h_min_mm": float(edges.min()),
        "h_max_mm": float(edges.max()),
        "h_median_mm": float(np.median(edges)),
        "edges_below_100um": int((edges < 0.1).sum()),
        "pct_below_100um": float((edges < 0.1).mean() * 100),
    }
