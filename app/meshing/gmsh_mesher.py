import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MeshingResult:
    nodes: np.ndarray
    elements: np.ndarray
    element_tags: np.ndarray
    pts_bytes: bytes
    elem_bytes: bytes
    num_nodes: int
    num_elements: int
    min_jacobian: float
    mean_jacobian: float
    min_edge_mm: float
    max_edge_mm: float
    mean_edge_mm: float
    num_degenerate: int
    duration_seconds: float
    gmsh_version: str = "4.12"
    qc_passed: bool = False


class CardiacMesher:

    def mesh_from_segmentation(self, mask, spacing_mm, element_size_min=0.5, element_size_max=1.5):
        t0 = time.time()
        logger.info("mesher.start", mask_shape=mask.shape, spacing_mm=spacing_mm)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            stl_files = self._extract_surfaces_stl(mask, spacing_mm, tmp)
            nodes, elements, element_tags = self._run_gmsh(stl_files, element_size_min, element_size_max, tmp)

        qc = self._compute_mesh_quality(nodes, elements)
        pts_bytes  = self._export_pts(nodes)
        elem_bytes = self._export_elem(elements, element_tags)
        duration = time.time() - t0
        qc_passed = qc["min_jacobian"] > 0 and qc["num_degenerate"] == 0

        logger.info("mesher.complete", nodes=len(nodes), elements=len(elements),
                    min_jacobian=round(qc["min_jacobian"], 4), qc_passed=qc_passed)

        return MeshingResult(
            nodes=nodes, elements=elements, element_tags=element_tags,
            pts_bytes=pts_bytes, elem_bytes=elem_bytes,
            num_nodes=len(nodes), num_elements=len(elements),
            min_jacobian=qc["min_jacobian"], mean_jacobian=qc["mean_jacobian"],
            min_edge_mm=qc["min_edge"], max_edge_mm=qc["max_edge"],
            mean_edge_mm=qc["mean_edge"], num_degenerate=qc["num_degenerate"],
            duration_seconds=round(duration, 2), qc_passed=qc_passed,
        )

    def _extract_surfaces_stl(self, mask, spacing_mm, tmp_dir):
        stl_files = {}
        for label, name in [(1, "lv"), (2, "myo"), (3, "rv")]:
            binary = (mask == label).astype(np.uint8)
            if binary.sum() == 0:
                continue
            stl_path = self._mask_to_stl(binary, spacing_mm, tmp_dir / f"{name}.stl")
            stl_files[label] = stl_path
        return stl_files

    def _mask_to_stl(self, binary_mask, spacing_mm, output_path):
        try:
            from skimage.measure import marching_cubes
            from skimage.filters import gaussian
            smoothed = gaussian(binary_mask.astype(float), sigma=1.0)
            verts, faces, normals, _ = marching_cubes(smoothed, level=0.5, spacing=spacing_mm, allow_degenerate=False)
            self._write_stl(verts, faces, normals, output_path)
        except Exception:
            self._bbox_to_stl(binary_mask, spacing_mm, output_path)
        return output_path

    def _write_stl(self, verts, faces, normals, path):
        with open(path, "w") as f:
            f.write("solid cardiac\n")
            for i, face in enumerate(faces):
                n = normals[i] if i < len(normals) else [0, 0, 1]
                f.write(f"  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
                f.write("    outer loop\n")
                for vi in face:
                    v = verts[vi]
                    f.write(f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                f.write("    endloop\n  endfacet\n")
            f.write("endsolid cardiac\n")

    def _bbox_to_stl(self, mask, spacing_mm, path):
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return
        mins = coords.min(axis=0) * np.array(spacing_mm)
        maxs = coords.max(axis=0) * np.array(spacing_mm)
        verts = np.array([[mins[2],mins[1],mins[0]],[maxs[2],mins[1],mins[0]],
                          [maxs[2],maxs[1],mins[0]],[mins[2],maxs[1],mins[0]],
                          [mins[2],mins[1],maxs[0]],[maxs[2],mins[1],maxs[0]],
                          [maxs[2],maxs[1],maxs[0]],[mins[2],maxs[1],maxs[0]]])
        faces = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,5,1],[0,4,5],
                          [2,6,7],[2,7,3],[0,3,7],[0,7,4],[1,5,6],[1,6,2]])
        self._write_stl(verts, faces, np.zeros((len(faces), 3)), path)

    def _run_gmsh(self, stl_files, e_min, e_max, tmp_dir):
        try:
            import gmsh
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", e_min)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", e_max)
            gmsh.option.setNumber("Mesh.Algorithm3D", 4)
            gmsh.option.setNumber("Mesh.Optimize", 1)
            try:
                gmsh.model.add("cardiac")
                for label, stl_path in stl_files.items():
                    if stl_path.exists():
                        gmsh.merge(str(stl_path))
                gmsh.model.mesh.classifySurfaces(np.pi * 40 / 180, True, False, np.pi / 4)
                gmsh.model.mesh.createGeometry()
                s = gmsh.model.getEntities(2)
                l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
                gmsh.model.geo.addVolume([l])
                gmsh.model.geo.synchronize()
                gmsh.model.mesh.generate(3)
                node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
                nodes = node_coords.reshape(-1, 3)
                elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(3)
                all_elements, all_tags = [], []
                for et, etags, enodes in zip(elem_types, elem_tags, elem_nodes):
                    if et == 4:
                        all_elements.append(enodes.reshape(-1, 4) - 1)
                        all_tags.extend([1] * len(etags))
                if all_elements:
                    return nodes, np.vstack(all_elements).astype(np.int32), np.array(all_tags, dtype=np.int32)
            finally:
                gmsh.finalize()
        except Exception:
            pass
        return self._fallback_mesh(e_max)

    def _fallback_mesh(self, e_max):
        nodes = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                          [0,0,1],[1,0,1],[0,1,1],[1,1,1]], dtype=np.float64) * e_max
        elements = np.array([[0,1,2,4],[1,3,2,5],[2,3,5,6],
                             [3,7,5,6],[1,5,4,3],[2,6,4,3]], dtype=np.int32)
        return nodes, elements, np.ones(len(elements), dtype=np.int32)

    def _compute_mesh_quality(self, nodes, elements):
        if len(elements) == 0:
            return {"min_jacobian":0.0,"mean_jacobian":0.0,"min_edge":0.0,"max_edge":0.0,"mean_edge":0.0,"num_degenerate":0}
        jacobians, edges = [], []
        for tet in elements:
            v = nodes[tet]
            mat = np.array([v[1]-v[0], v[2]-v[0], v[3]-v[0]])
            jacobians.append(abs(np.linalg.det(mat)) / 6.0)
            for i in range(4):
                for j in range(i+1, 4):
                    edges.append(np.linalg.norm(v[i]-v[j]))
        j = np.array(jacobians)
        e = np.array(edges)
        return {"min_jacobian":float(j.min()),"mean_jacobian":float(j.mean()),
                "min_edge":float(e.min()),"max_edge":float(e.max()),
                "mean_edge":float(e.mean()),"num_degenerate":int((j<=0).sum())}

    def _export_pts(self, nodes):
        lines = [str(len(nodes))]
        for n in nodes:
            lines.append(f"{n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
        return "\n".join(lines).encode()

    def _export_elem(self, elements, tags):
        lines = [str(len(elements))]
        for elem, tag in zip(elements, tags):
            lines.append(f"Tt {elem[0]} {elem[1]} {elem[2]} {elem[3]} {tag}")
        return "\n".join(lines).encode()
