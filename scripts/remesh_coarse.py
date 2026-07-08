"""
Regenere un maillage PLUS GROSSIER (target_mm=2.5 au lieu de 1.5) pour
patient001, a partir du masque de segmentation deja produit. Objectif :
reduire le nombre de tetraedres pour des cycles de developpement FEniCSx
rapides (le cout d'assemblage FFCx scale ~lineairement avec le nombre
de cellules). Ne touche PAS au maillage de production (1.5mm) existant.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import nibabel as nib

from app.meshing.gmsh_mesher import CardiacMesher

SEG_PATH = "reports/meshes_acdc/segmentations/patient001_seg.nii.gz"
TARGET_MM = 5.0
OUT_PREFIX = "reports/meshes_acdc/meshes/patient001_coarse5"

print(f"=== Chargement segmentation : {SEG_PATH} ===")
img = nib.load(SEG_PATH)
mask = np.asarray(img.dataobj)
spacing_mm = img.header.get_zooms()[:3]
print(f"Mask shape: {mask.shape}, spacing_mm: {spacing_mm}")
print(f"Labels presents: {np.unique(mask)}")

mesher = CardiacMesher()
myo_binary = (mask == 2).astype(np.uint8)
print(f"Voxels myocarde (label 2): {myo_binary.sum()}")

import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    stl_path = mesher._mask_to_stl(myo_binary, spacing_mm, tmp / "myo_coarse.stl",
                                     target_mm=TARGET_MM)
    print(f"STL genere : {stl_path}")
    nodes, elements, tags = mesher._run_gmsh(stl_path, 0.5, 3.0, tmp)

print(f"\nMaillage brut : {len(nodes)} noeuds, {len(elements)} tets")

nodes_f, elements_f, tags_f = mesher._filter_slivers(nodes, elements, tags, h_min_mm=0.3)
print(f"Apres filtre slivers (h_min=0.3mm) : {len(nodes_f)} noeuds, {len(elements_f)} tets")

qc = mesher._compute_mesh_quality(nodes_f, elements_f)
print(f"\n=== Qualite ===")
print(f"min_jacobian: {qc['min_jacobian']:.6f}")
print(f"num_degenerate: {qc['num_degenerate']}")
print(f"min_edge: {qc['min_edge']:.4f} mm, max_edge: {qc['max_edge']:.4f} mm")

pts_bytes = mesher._export_pts(nodes_f)
elem_bytes = mesher._export_elem(elements_f, tags_f)

with open(f"{OUT_PREFIX}.pts", "wb") as f:
    f.write(pts_bytes)
with open(f"{OUT_PREFIX}.elem", "wb") as f:
    f.write(elem_bytes)

print(f"\n=== Ecrit : {OUT_PREFIX}.pts / .elem ===")
print(f"NOTE : pas de fichier .lon (fibres) genere -- a produire separement "
      f"si besoin (LDRB) ou reutiliser un champ radial simplifie pour les tests.")
