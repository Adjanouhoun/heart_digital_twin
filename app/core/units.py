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
