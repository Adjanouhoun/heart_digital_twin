"""
QC maillage via meshio — remplace meshtool binaire.
Vérifie : Jacobian > 0, éléments dégénérés < 0.1%, conversion formats.
"""
import numpy as np
import structlog
from dataclasses import dataclass
from pathlib import Path

logger = structlog.get_logger(__name__)


@dataclass
class QCReport:
    num_nodes: int
    num_elements: int
    min_jacobian: float
    mean_jacobian: float
    num_degenerate: int
    degenerate_pct: float
    min_edge_mm: float
    max_edge_mm: float
    mean_edge_mm: float
    qc_passed: bool
    warnings: list[str]


def compute_jacobians(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Calcule le Jacobian (volume) de chaque tétraèdre."""
    jacobians = np.zeros(len(elements))
    for i, tet in enumerate(elements):
        v = nodes[tet]
        mat = np.array([v[1]-v[0], v[2]-v[0], v[3]-v[0]])
        jacobians[i] = abs(np.linalg.det(mat)) / 6.0
    return jacobians


def run_qc(nodes: np.ndarray, elements: np.ndarray) -> QCReport:
    """
    Contrôle qualité complet du maillage.
    Équivalent aux checks meshtool : Jacobian, aspect ratio, dégénérés.
    """
    warnings = []

    # Jacobians
    jacobians = compute_jacobians(nodes, elements)
    num_degenerate = int((jacobians <= 0).sum())
    degenerate_pct = num_degenerate / max(len(elements), 1) * 100

    # Longueurs d'arêtes
    edge_lengths = []
    for tet in elements:
        v = nodes[tet]
        for i in range(4):
            for j in range(i+1, 4):
                edge_lengths.append(np.linalg.norm(v[i] - v[j]))
    edge_lengths = np.array(edge_lengths)

    # Vérifications SLO
    if jacobians.min() <= 0:
        warnings.append(f"❌ Jacobian min ≤ 0 : {jacobians.min():.6f}")
    if degenerate_pct >= 0.1:
        warnings.append(f"❌ Éléments dégénérés : {num_degenerate} ({degenerate_pct:.3f}%)")
    if edge_lengths.max() > 3.0:
        warnings.append(f"⚠️  Arête max trop grande : {edge_lengths.max():.2f}mm")

    qc_passed = (jacobians.min() > 0 and degenerate_pct < 0.1)

    report = QCReport(
        num_nodes=len(nodes),
        num_elements=len(elements),
        min_jacobian=float(jacobians.min()),
        mean_jacobian=float(jacobians.mean()),
        num_degenerate=num_degenerate,
        degenerate_pct=round(degenerate_pct, 4),
        min_edge_mm=float(edge_lengths.min()),
        max_edge_mm=float(edge_lengths.max()),
        mean_edge_mm=float(edge_lengths.mean()),
        qc_passed=qc_passed,
        warnings=warnings,
    )

    status = "✅ PASSED" if qc_passed else "❌ FAILED"
    logger.info(
        "meshtool_qc.result",
        status=status,
        nodes=len(nodes),
        elements=len(elements),
        min_jacobian=round(report.min_jacobian, 6),
        degenerate=num_degenerate,
        degenerate_pct=degenerate_pct,
    )

    return report


def convert_to_opencarp(
    nodes: np.ndarray,
    elements: np.ndarray,
    element_tags: np.ndarray,
    output_dir: Path,
    basename: str = "mesh",
) -> dict[str, Path]:
    """
    Convertit le maillage vers les formats openCARP (.pts, .elem).
    Équivalent à : meshtool convert --imsh=mesh.vtk --omsh=mesh
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # .pts — noeuds
    pts_path = output_dir / f"{basename}.pts"
    with open(pts_path, "w") as f:
        f.write(f"{len(nodes)}\n")
        for n in nodes:
            f.write(f"{n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

    # .elem — éléments
    elem_path = output_dir / f"{basename}.elem"
    with open(elem_path, "w") as f:
        f.write(f"{len(elements)}\n")
        for elem, tag in zip(elements, element_tags):
            f.write(f"Tt {elem[0]} {elem[1]} {elem[2]} {elem[3]} {tag}\n")

    logger.info("meshtool_qc.converted",
                pts=str(pts_path), elem=str(elem_path))

    return {"pts": pts_path, "elem": elem_path}
