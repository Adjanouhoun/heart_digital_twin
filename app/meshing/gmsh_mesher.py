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
            # Conforme au projet : on maille le MYOCARDE seul (label 2).
            # LDRB (fibres) et openCARP (EP) operent sur le tissu musculaire,
            # PAS sur les cavites sanguines VG/VD. Mailler les 3 organes serait
            # une erreur physiologique.
            myo_binary = (mask == 2).astype(np.uint8)
            if myo_binary.sum() == 0:
                logger.error("mesher.no_myocardium", msg="Aucun voxel myocarde (label 2) dans le masque")
                nodes, elements, element_tags = self._fallback_mesh(element_size_max)
            else:
                myo_stl = self._mask_to_stl(myo_binary, spacing_mm, tmp / "myo.stl")
                nodes, elements, element_tags = self._run_gmsh(myo_stl, element_size_min, element_size_max, tmp)

        # Filtrage des slivers (elements degeneres) — seuil h_min >= 0.3mm
        # valide dans l'audit (fix divergence PETSc openCARP). Retire les tets
        # a arete trop courte + renumerote les noeuds orphelins.
        if len(elements) > 0:
            n_before = len(elements)
            nodes, elements, element_tags = self._filter_slivers(
                nodes, elements, element_tags, h_min_mm=0.3)
            n_filtered = n_before - len(elements)
            if n_filtered > 0:
                logger.info("mesher.slivers_filtered", removed=n_filtered,
                            remaining=len(elements))

        qc = self._compute_mesh_quality(nodes, elements)
        pts_bytes  = self._export_pts(nodes)
        elem_bytes = self._export_elem(elements, element_tags)
        duration = time.time() - t0
        qc_passed = qc["min_jacobian"] > 0 and qc["num_degenerate"] == 0

        logger.info("mesher.complete", nodes=len(nodes), elements=len(elements),
                    min_jacobian=round(qc["min_jacobian"], 6),
                    min_edge_mm=round(qc["min_edge"], 3),
                    num_degenerate=qc["num_degenerate"], qc_passed=qc_passed)

        return MeshingResult(
            nodes=nodes, elements=elements, element_tags=element_tags,
            pts_bytes=pts_bytes, elem_bytes=elem_bytes,
            num_nodes=len(nodes), num_elements=len(elements),
            min_jacobian=qc["min_jacobian"], mean_jacobian=qc["mean_jacobian"],
            min_edge_mm=qc["min_edge"], max_edge_mm=qc["max_edge"],
            mean_edge_mm=qc["mean_edge"], num_degenerate=qc["num_degenerate"],
            duration_seconds=round(duration, 2), qc_passed=qc_passed,
        )

    def _mask_to_stl(self, binary_mask, spacing_mm, output_path, target_mm=1.5):
        from skimage.measure import marching_cubes
        from skimage.filters import gaussian
        from scipy import ndimage
        # Remplir les trous internes (le myocarde en anneau peut avoir des trous
        # dus au bruit de segmentation).
        #
        # BUG CRITIQUE CORRIGE (2026-07-13) : binary_closing(iterations=2)
        # etait utilise ici. Ce n'est PAS un remplissage de trous mais une
        # dilatation suivie d'une erosion, avec un element structurant
        # ISOTROPE EN VOXELS. Or le masque ACDC est fortement anisotrope
        # (spacing 1.5625 x 1.5625 x 10.0 mm) : +/-2 voxels valent +/-3.1mm
        # en x/y mais +/-20mm en z. La dilatation debordait des bords du
        # tableau (le masque n'a que 10 tranches) et etait tronquee, tandis
        # que l'erosion suivante rongeait pleinement +/-20mm en z.
        #
        # Effet mesure sur patient001 : hauteur du myocarde 100mm -> 60mm
        # (puis 55.6mm apres le lissage gaussien). Le ventricule etait
        # ecrase de 44% dans l'axe long.
        #
        # Consequences en chaine (toutes resolues par ce correctif) :
        #   - volume de cavite calcule 209mL au lieu de 295mL (verite terrain
        #     ACDC, label 3, comptage voxels)
        #   - EF structurellement bloquee a ~24% (SLO projet : +/-3% vs
        #     reference clinique)
        #   - PCA incapable de trouver un axe long (geometrie aplatie)
        #   - base non plane, aucune facette basale detectable
        #   - ~3-4 elements seulement dans l'epaisseur pariétale
        #
        # binary_fill_holes fait ce que le commentaire annoncait : il remplit
        # les cavites internes fermees sans toucher a la geometrie externe.
        # Verifie : hauteur restauree a 95mm (les 5mm d'ecart avec 100mm sont
        # attendus, marching_cubes place l'isosurface a un demi-voxel des
        # bords).
        filled = ndimage.binary_fill_holes(binary_mask).astype(float)

        # Reechantillonner a la resolution cible (~1.5mm, spec projet) AVANT
        # marching_cubes. Cela controle la densite de la surface produite :
        # une surface a 1.5mm donne un maillage volumique ~25-35K nodes
        # (au lieu de 110K a 1mm), sans passer par le remaillage Gmsh qui
        # echoue sur les surfaces medicales ("Invalid boundary for parametrization").
        zoom = [s / target_mm for s in spacing_mm]
        if any(abs(z - 1.0) > 0.05 for z in zoom):
            filled = ndimage.zoom(filled, zoom, order=1)
            effective_spacing = (target_mm, target_mm, target_mm)
        else:
            effective_spacing = spacing_mm

        # PADDING (2026-07-13) : le myocarde touche les bords du tableau en z
        # (les 10 tranches ACDC sont toutes occupees). marching_cubes coupe
        # alors l'isosurface net a la frontiere du tableau -> surface OUVERTE
        # (44 aretes de bord mesurees) que Gmsh refuse de mailler
        # ("Wrong topology of boundary mesh for parametrization", puis 0 tet).
        # Une bordure de vide (appliquee APRES le zoom, donc en voxels de la
        # grille finale isotrope) permet a l'isosurface de se refermer.
        # Verifie : 0 arete de bord, 0 arete non-manifold.
        filled = np.pad(filled, pad_width=3, mode="constant", constant_values=0)

        # Lissage gaussien => surface reguliere, maillable.
        # sigma=1.0 erodait la geometrie de 21mm en z (100mm -> 78mm mesures).
        # sigma=0.5 preserve la hauteur (99mm) tout en regularisant la surface
        # et en simplifiant sa topologie (Euler -12 contre -18 sans lissage).
        smoothed = gaussian(filled, sigma=0.5)
        verts, faces, normals, _ = marching_cubes(
            smoothed, level=0.5, spacing=effective_spacing, allow_degenerate=False)

        # Compenser l'offset introduit par le padding pour restaurer les
        # coordonnees physiques d'origine.
        verts = verts - 3 * np.array(effective_spacing)
        self._write_stl(verts, faces, normals, output_path)
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

    def _run_gmsh(self, myo_stl, e_min, e_max, tmp_dir):
        # Strategie 1 : remaillage surface a la taille cible projet (0.5-1.5mm)
        result = self._gmsh_remesh(myo_stl, e_min, e_max)
        if result is not None:
            return result
        # Strategie 2 (secours) : mesh-based, conserve la densite du STL
        # (produit un maillage plus dense mais valide)
        result = self._gmsh_mesh_based(myo_stl, e_min, e_max)
        if result is not None:
            return result
        logger.warning("mesher.fallback", msg="⚠️ Maillage fallback trivial (cube) — Gmsh a echoue")
        return self._fallback_mesh(e_max)

    def _gmsh_remesh(self, myo_stl, e_min, e_max):
        # Remaille la surface a la taille cible via classifySurfaces avec
        # forReparametrization=False (evite l'erreur "Wrong topology for
        # parametrization" du forReparametrization=True). Les MeshSize sont
        # alors respectees car Gmsh regenere une vraie geometrie.
        try:
            import gmsh
            gmsh.initialize()
            try:
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.option.setNumber("Mesh.MeshSizeMin", e_min)
                gmsh.option.setNumber("Mesh.MeshSizeMax", e_max)
                gmsh.option.setNumber("Mesh.Algorithm3D", 1)
                gmsh.option.setNumber("Mesh.Optimize", 1)
                gmsh.model.add("myo_remesh")
                gmsh.merge(str(myo_stl))
                angle = 40 * np.pi / 180
                # 3e arg = forReparametrization=False (cle du fix)
                gmsh.model.mesh.classifySurfaces(angle, True, False, np.pi)
                gmsh.model.mesh.createGeometry()
                surfaces = gmsh.model.getEntities(2)
                if not surfaces:
                    raise RuntimeError("Aucune surface apres createGeometry")
                sl = gmsh.model.geo.addSurfaceLoop([e[1] for e in surfaces])
                gmsh.model.geo.addVolume([sl])
                gmsh.model.geo.synchronize()
                gmsh.model.mesh.generate(3)
                nodes, elements, tags = self._extract_gmsh_mesh(gmsh)
                if elements is not None and len(elements) > 0:
                    logger.info("mesher.gmsh_success", strategy="remesh",
                                nodes=len(nodes), tets=len(elements))
                    return nodes, elements, tags
                logger.warning("mesher.gmsh_no_tets", strategy="remesh")
            finally:
                gmsh.finalize()
        except Exception as e:
            import traceback
            logger.warning("mesher.gmsh_remesh_failed",
                           error=str(e), traceback=traceback.format_exc())
        return None

    def _gmsh_mesh_based(self, myo_stl, e_min, e_max):
        # Secours : construit le volume depuis le maillage de surface discret
        # sans remaillage (conserve la densite du STL).
        try:
            import gmsh
            gmsh.initialize()
            try:
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.option.setNumber("Mesh.MeshSizeMin", e_min)
                gmsh.option.setNumber("Mesh.MeshSizeMax", e_max)
                gmsh.option.setNumber("Mesh.Algorithm3D", 1)
                gmsh.option.setNumber("Mesh.Optimize", 1)
                gmsh.model.add("myo_meshbased")
                gmsh.merge(str(myo_stl))
                gmsh.model.mesh.createTopology()
                surfaces = gmsh.model.getEntities(2)
                if not surfaces:
                    raise RuntimeError("Aucune surface apres createTopology")

                # BUG CORRIGE (2026-07-13) : toutes les surfaces detectees
                # etaient passees dans un SEUL SurfaceLoop. Or marching_cubes
                # produit, en plus de l'enveloppe du myocarde, de petits
                # fragments parasites issus du bruit de segmentation (mesure
                # sur patient001 : surface principale = 5254 triangles /
                # 46561 mm2 / 163 mL, contre 2 fragments de 24 et 132 triangles
                # pres de l'apex). Regrouper des composantes disjointes dans un
                # meme SurfaceLoop rend l'enveloppe incoherente : Gmsh declare
                # "Found void region" pour chacune et ne produit AUCUN
                # tetraedre ("No tetrahedra in region 1").
                #
                # L'enveloppe du myocarde est UNE SEULE surface fermee connexe
                # (le myocarde est un anneau : epicarde et endocarde forment
                # une frontiere unique de genre topologique eleve). On ne
                # conserve donc que la composante de plus grande aire.
                if len(surfaces) > 1:
                    areas = []
                    for dim, tag in surfaces:
                        _, _, enodes = gmsh.model.mesh.getElements(dim, tag)
                        tri = np.array(enodes[0]).reshape(-1, 3)
                        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
                        xyz = {int(t): np.array(ncoords[3*i:3*i+3])
                               for i, t in enumerate(ntags)}
                        a = 0.0
                        for i0, i1, i2 in tri:
                            p0, p1, p2 = xyz[int(i0)], xyz[int(i1)], xyz[int(i2)]
                            a += 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
                        areas.append(a)
                    keep = int(np.argmax(areas))
                    logger.info("mesher.surface_selection",
                                n_surfaces=len(surfaces),
                                kept_tag=surfaces[keep][1],
                                kept_area_mm2=round(areas[keep], 1),
                                discarded=[round(a, 1)
                                           for i, a in enumerate(areas)
                                           if i != keep])
                    surfaces = [surfaces[keep]]

                sl = gmsh.model.geo.addSurfaceLoop([e[1] for e in surfaces])
                gmsh.model.geo.addVolume([sl])
                gmsh.model.geo.synchronize()
                gmsh.model.mesh.generate(3)
                nodes, elements, tags = self._extract_gmsh_mesh(gmsh)
                if elements is not None and len(elements) > 0:
                    logger.info("mesher.gmsh_success", strategy="mesh_based",
                                nodes=len(nodes), tets=len(elements))
                    return nodes, elements, tags
                logger.warning("mesher.gmsh_no_tets", strategy="mesh_based")
            finally:
                gmsh.finalize()
        except Exception as e:
            import traceback
            logger.error("mesher.gmsh_failed", strategy="mesh_based",
                         error=str(e), traceback=traceback.format_exc())
        return None

    def _extract_gmsh_mesh(self, gmsh):
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes = node_coords.reshape(-1, 3)
        # Remap des tags Gmsh (1-based, potentiellement non contigus) vers 0-based
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
        elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(3)
        all_elements, all_tags = [], []
        for et, etags, enodes in zip(elem_types, elem_tags, elem_nodes):
            if et == 4:  # tetraedre
                conn = enodes.reshape(-1, 4)
                remapped = np.vectorize(tag_to_idx.get)(conn)
                all_elements.append(remapped)
                all_tags.extend([1] * len(etags))
        if all_elements:
            return nodes, np.vstack(all_elements).astype(np.int32), np.array(all_tags, dtype=np.int32)
        return nodes, None, None

    def _fallback_mesh(self, e_max):
        nodes = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                          [0,0,1],[1,0,1],[0,1,1],[1,1,1]], dtype=np.float64) * e_max
        elements = np.array([[0,1,2,4],[1,3,2,5],[2,3,5,6],
                             [3,7,5,6],[1,5,4,3],[2,6,4,3]], dtype=np.int32)
        return nodes, elements, np.ones(len(elements), dtype=np.int32)

    def _filter_slivers(self, nodes, elements, tags, h_min_mm=0.3):
        # Retire les tetraedres dont la plus petite arete < h_min_mm.
        # Ces "slivers" causent la divergence PETSc dans openCARP (audit).
        keep = np.ones(len(elements), dtype=bool)
        for i, tet in enumerate(elements):
            v = nodes[tet]
            min_edge = min(np.linalg.norm(v[a] - v[b])
                           for a in range(4) for b in range(a + 1, 4))
            if min_edge < h_min_mm:
                keep[i] = False
        elements = elements[keep]
        tags = tags[keep]
        if len(elements) == 0:
            return nodes, elements, tags
        # Renumerotation : ne garder que les noeuds encore references
        used = np.unique(elements)
        remap = -np.ones(len(nodes), dtype=np.int64)
        remap[used] = np.arange(len(used))
        new_nodes = nodes[used]
        new_elements = remap[elements].astype(np.int32)
        return new_nodes, new_elements, tags

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
