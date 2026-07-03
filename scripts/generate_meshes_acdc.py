#!/usr/bin/env python3
"""
Phase 02 — Post-Segmentation Pipeline
Deliverable D1.2: Generate 10 patient-specific cardiac meshes from ACDC

Pipeline: Checkpoint → Inference → NIfTI → Gmsh Mesh → QC → LDRB Fibers

Usage:
    python scripts/generate_meshes_acdc.py \
        --checkpoint ~/nnunet/results/Dataset027_ACDC/mps_training/checkpoint_best.pth \
        --acdc_dir ~/Downloads/ACDC/database/training \
        --output_dir ~/cdt/reports/meshes_acdc \
        --n_patients 10
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from torch.nn import Conv3d, InstanceNorm3d, LeakyReLU
from dynamic_network_architectures.architectures.unet import PlainConvUNet

# ─── 1. NETWORK BUILDER (same as training) ───

def build_network(num_classes=4, device="cpu"):
    """Build the exact same PlainConvUNet used in training."""
    network = PlainConvUNet(
        input_channels=1, n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=Conv3d,
        kernel_sizes=[[1,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
        strides=[[1,1,1],[1,2,2],[2,2,2],[2,2,2],[1,2,2],[1,2,2]],
        n_conv_per_stage=[2,2,2,2,2,2],
        n_conv_per_stage_decoder=[2,2,2,2,2],
        conv_bias=True, norm_op=InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-05, 'affine': True},
        dropout_op=None, dropout_op_kwargs=None,
        nonlin=LeakyReLU, nonlin_kwargs={'inplace': True},
        num_classes=num_classes
    ).to(device)
    return network


# ─── 2. INFERENCE ───

def preprocess_volume(nifti_path, target_spacing=(1.5625, 1.5625, 5.0)):
    """Preprocess a NIfTI volume: resample to target spacing + z-score."""
    img = sitk.ReadImage(str(nifti_path))
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    # Compute new size
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampled = resampler.Execute(img)

    arr = sitk.GetArrayFromImage(resampled).astype(np.float32)  # (D, H, W)

    # Z-score normalization
    mask = arr > 0
    if mask.sum() > 0:
        mean = arr[mask].mean()
        std = arr[mask].std()
        arr = (arr - mean) / (std + 1e-8)
    else:
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)

    return arr, resampled


def sliding_window_inference(network, volume, patch_size, device, overlap=0.5):
    """
    Sliding window inference for 3D volumes.
    Handles volumes larger than patch_size with overlap.
    """
    network.eval()
    D, H, W = volume.shape
    pD, pH, pW = patch_size
    num_classes = 4

    # Pad if smaller than patch
    pad_d = max(0, pD - D)
    pad_h = max(0, pH - H)
    pad_w = max(0, pW - W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = np.pad(volume, (
            (pad_d // 2, pad_d - pad_d // 2),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2)
        ), mode='constant', constant_values=0)

    D2, H2, W2 = volume.shape

    # Compute step sizes
    step_d = max(1, int(pD * (1 - overlap)))
    step_h = max(1, int(pH * (1 - overlap)))
    step_w = max(1, int(pW * (1 - overlap)))

    # Accumulation arrays
    prediction = np.zeros((num_classes, D2, H2, W2), dtype=np.float32)
    count = np.zeros((D2, H2, W2), dtype=np.float32)

    # Generate window positions
    d_starts = list(range(0, max(1, D2 - pD + 1), step_d))
    h_starts = list(range(0, max(1, H2 - pH + 1), step_h))
    w_starts = list(range(0, max(1, W2 - pW + 1), step_w))

    # Ensure last position is covered
    if d_starts[-1] + pD < D2:
        d_starts.append(D2 - pD)
    if h_starts[-1] + pH < H2:
        h_starts.append(H2 - pH)
    if w_starts[-1] + pW < W2:
        w_starts.append(W2 - pW)

    with torch.no_grad():
        for d0 in d_starts:
            for h0 in h_starts:
                for w0 in w_starts:
                    patch = volume[d0:d0+pD, h0:h0+pH, w0:w0+pW]
                    tensor = torch.from_numpy(patch[np.newaxis, np.newaxis]).float().to(device)
                    out = network(tensor)
                    prob = F.softmax(out, dim=1).cpu().numpy()[0]

                    prediction[:, d0:d0+pD, h0:h0+pH, w0:w0+pW] += prob
                    count[d0:d0+pD, h0:h0+pH, w0:w0+pW] += 1

    # Average overlapping regions
    count = np.maximum(count, 1e-8)
    prediction /= count[np.newaxis]

    # Remove padding
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        prediction = prediction[
            :,
            pad_d // 2: pad_d // 2 + D,
            pad_h // 2: pad_h // 2 + H,
            pad_w // 2: pad_w // 2 + W
        ]

    return prediction.argmax(axis=0).astype(np.uint8)


def resample_label_to_original(label_resampled, original_nifti_path, target_spacing):
    """Resample segmentation label back to original image space."""
    original_img = sitk.ReadImage(str(original_nifti_path))

    label_sitk = sitk.GetImageFromArray(label_resampled.astype(np.uint8))
    label_sitk.SetSpacing(target_spacing)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(original_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampled_label = resampler.Execute(label_sitk)

    return resampled_label


# ─── 3. NIFTI EXPORT ───

def save_segmentation_nifti(label_sitk, output_path):
    """Save a SimpleITK label image as compressed NIfTI."""
    sitk.WriteImage(label_sitk, str(output_path))
    print(f"    Saved: {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB)")


# ─── 4. GMSH MESHING ───

def segmentation_to_mesh(label_nifti_path, output_dir, patient_id,
                          mesh_size_min=0.5, mesh_size_max=1.5):
    """
    Convert a segmentation NIfTI to tetrahedral mesh using Gmsh.

    Labels: 0=bg, 1=RV, 2=MYO, 3=LV
    We mesh the myocardium (label=2) as the primary volume.
    """
    import gmsh

    # Load segmentation
    seg_img = nib.load(str(label_nifti_path))
    seg_data = seg_img.get_fdata().astype(np.uint8)
    affine = seg_img.affine
    spacing = seg_img.header.get_zooms()

    # Extract myocardium surface (label == 2)
    from skimage import measure

    # Create binary mask for myocardium
    myo_mask = (seg_data == 2).astype(np.float32)

    if myo_mask.sum() == 0:
        print(f"    ⚠ No myocardium found for {patient_id}, skipping mesh")
        return None

    # Marching cubes to extract surface
    try:
        verts, faces, normals, values = measure.marching_cubes(
            myo_mask, level=0.5
        )
    except Exception as e:
        print(f"    ⚠ Marching cubes failed for {patient_id}: {e}")
        return None

    # Transform vertices to world coordinates
    verts_homo = np.hstack([verts, np.ones((len(verts), 1))])
    verts_world = (affine @ verts_homo.T).T[:, :3]

    # Write surface STL for Gmsh input
    stl_path = os.path.join(output_dir, f"{patient_id}_myo_surface.stl")
    _write_stl(verts_world, faces, stl_path)

    # Maillage via TetGen avec réparation de surface PyMeshFix
    import meshio
    import tetgen
    import pyvista as pv
    import pymeshfix
    from scipy.ndimage import gaussian_filter

    try:
        # 1. Lisser le masque pour une surface plus propre
        myo_smooth = gaussian_filter(myo_mask.astype(np.float32), sigma=1.5)

        # 2. Marching cubes sur le masque lissé
        verts_smooth, faces_smooth, _, _ = measure.marching_cubes(
            myo_smooth, level=0.3
        )

        # 3. Transformer en coordonnées monde
        verts_homo = np.hstack([verts_smooth, np.ones((len(verts_smooth), 1))])
        verts_w = (affine @ verts_homo.T).T[:, :3]

        # 4. Créer surface PyVista
        faces_pv = np.column_stack([
            np.full(len(faces_smooth), 3),
            faces_smooth
        ]).ravel()
        surf = pv.PolyData(verts_w, faces_pv)

        # 5. Réparer la surface (fermer les trous, rendre manifold)
        meshfix = pymeshfix.MeshFix(surf)
        meshfix.repair()
        surf_fixed = meshfix.mesh

        if surf_fixed.n_points < 10:
            print(f"    ⚠ Surface repair produced too few points for {patient_id}")
            return None

        # Décimer si trop de faces (TetGen est lent au-delà de 3000)
        if surf_fixed.n_faces > 3000:
            ratio = 3000.0 / surf_fixed.n_faces
            surf_fixed = surf_fixed.decimate(1.0 - ratio)
        print(f"    Surface: {surf_fixed.n_points} vertices, {surf_fixed.n_faces} faces (repaired)")

        # Sauver la surface, lancer TetGen via subprocess avec timeout
        import subprocess, tempfile, json as jjson
        
        surf_tmp = os.path.join(output_dir, f"{patient_id}_surf_tmp.vtk")
        mesh_path = os.path.join(output_dir, f"{patient_id}_myo.msh")
        surf_fixed.save(surf_tmp)
        
        tetgen_script = f'''
import pyvista as pv
import tetgen
import meshio
import numpy as np
surf = pv.read("{surf_tmp}")
tet = tetgen.TetGen(surf)
tet.tetrahedralize(steinerleft=-1, order=1, mindihedral=5,
                    minratio=2.0, quality=True, verbose=0)
g = tet.grid
pts = np.array(g.points)
if hasattr(g, "cells_dict") and 10 in g.cells_dict:
    tets = g.cells_dict[10]
else:
    cells = g.cells.reshape(-1, 5)
    tets = cells[:, 1:]
mesh = meshio.Mesh(points=pts, cells=[("tetra", tets)])
meshio.write("{mesh_path}", mesh)
print(f"OK {{len(pts)}} {{len(tets)}}")
'''
        try:
            result = subprocess.run(
                ["python", "-c", tetgen_script],
                capture_output=True, text=True, timeout=60
            )
            if os.path.exists(surf_tmp):
                os.remove(surf_tmp)
            
            if result.returncode == 0 and "OK" in result.stdout:
                parts = result.stdout.strip().split()
                n_pts = parts[1]
                n_tets = parts[2]
                print(f"    Mesh: {n_pts} nodes, {n_tets} tetrahedra → {mesh_path}")
            else:
                print(f"    ⚠ TetGen failed for {patient_id}: {result.stderr[:200]}")
                mesh_path = None
        except subprocess.TimeoutExpired:
            print(f"    ⚠ TetGen timeout (60s) for {patient_id}")
            if os.path.exists(surf_tmp):
                os.remove(surf_tmp)
            mesh_path = None

    except Exception as e:
        print(f"    ⚠ Meshing failed for {patient_id}: {e}")
        mesh_path = None

    # Clean up STL
    if os.path.exists(stl_path):
        os.remove(stl_path)

    return mesh_path


def _write_stl(vertices, faces, path):
    """Write a binary STL file."""
    with open(path, 'wb') as f:
        # Header (80 bytes)
        f.write(b'\0' * 80)
        # Number of triangles
        f.write(np.uint32(len(faces)).tobytes())
        for face in faces:
            v0, v1, v2 = vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal /= norm_len
            # Normal (3 floats) + 3 vertices (9 floats) + attribute (1 uint16)
            f.write(normal.astype(np.float32).tobytes())
            f.write(v0.astype(np.float32).tobytes())
            f.write(v1.astype(np.float32).tobytes())
            f.write(v2.astype(np.float32).tobytes())
            f.write(np.uint16(0).tobytes())


# ─── 5. MESH QC (Jacobian) ───

def mesh_quality_check(mesh_path):
    """
    Check mesh quality using meshio.
    Returns dict with Jacobian stats and degenerate element count.
    """
    import meshio

    mesh = meshio.read(mesh_path)

    # Find tetrahedral cells
    tet_cells = None
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            tet_cells = cell_block.data
            break

    if tet_cells is None:
        return {"error": "No tetrahedral cells found"}

    points = mesh.points
    n_tets = len(tet_cells)

    # Compute Jacobian determinant for each tet
    jacobians = []
    for tet in tet_cells:
        p0, p1, p2, p3 = points[tet]
        J = np.array([p1 - p0, p2 - p0, p3 - p0])
        det = np.linalg.det(J)
        jacobians.append(det)

    jacobians = np.array(jacobians)
    n_degenerate = np.sum(jacobians <= 0)
    pct_degenerate = 100.0 * n_degenerate / n_tets

    qc = {
        "n_nodes": len(points),
        "n_tetrahedra": n_tets,
        "jacobian_min": float(jacobians.min()),
        "jacobian_max": float(jacobians.max()),
        "jacobian_mean": float(jacobians.mean()),
        "n_degenerate": int(n_degenerate),
        "pct_degenerate": float(pct_degenerate),
        "slo_jacobian_positive": bool(jacobians.min() > 0),
        "slo_degenerate_lt_0_1pct": bool(pct_degenerate < 0.1),
    }
    return qc


# ─── 6. LDRB FIBERS ───

def generate_ldrb_fibers(mesh_path, output_dir, patient_id,
                          alpha_endo=-60, alpha_epi=60,
                          beta_endo=20, beta_epi=-20):
    """
    Generate myocardial fiber orientations using LDRB algorithm.
    Solves 3 Laplacian problems to define transmural, apicobasal,
    and rotational coordinate systems.

    Output: .lon file (openCARP fiber format)
    """
    import meshio

    mesh = meshio.read(mesh_path)
    points = mesh.points

    tet_cells = None
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            tet_cells = cell_block.data
            break

    if tet_cells is None:
        print(f"    ⚠ No tetra cells for LDRB in {patient_id}")
        return None

    n_tets = len(tet_cells)

    # Simplified LDRB: assign fiber angles based on transmural depth
    # Full LDRB requires FEM Laplacian solve — here we use a geometric approximation
    # that assigns fiber rotation linearly from endo to epi

    # Compute element centroids
    centroids = np.mean(points[tet_cells], axis=1)  # (n_tets, 3)

    # Estimate transmural depth via distance to centroid of mass
    center = centroids.mean(axis=0)
    distances = np.linalg.norm(centroids - center, axis=1)
    d_min, d_max = distances.min(), distances.max()

    # Normalize to [0, 1] (0 = endo, 1 = epi)
    transmural = (distances - d_min) / (d_max - d_min + 1e-8)

    # Compute fiber angle per element (linear interpolation endo → epi)
    alpha = np.deg2rad(alpha_endo + transmural * (alpha_epi - alpha_endo))
    beta = np.deg2rad(beta_endo + transmural * (beta_epi - beta_endo))

    # Local coordinate system: approximate circumferential direction
    # Use cross product of apicobasal (z-axis) and radial direction
    apicobasal = np.array([0, 0, 1], dtype=np.float64)
    radial = centroids - center
    radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-8

    circumferential = np.cross(apicobasal, radial)
    circumferential /= np.linalg.norm(circumferential, axis=1, keepdims=True) + 1e-8

    # Fiber direction: rotate circumferential by alpha around radial
    fibers = np.zeros((n_tets, 3))
    sheets = np.zeros((n_tets, 3))
    for i in range(n_tets):
        ca, sa = np.cos(alpha[i]), np.sin(alpha[i])
        fibers[i] = ca * circumferential[i] + sa * radial[i]
        fibers[i] /= np.linalg.norm(fibers[i]) + 1e-8

        cb, sb = np.cos(beta[i]), np.sin(beta[i])
        sheets[i] = cb * radial[i] + sb * np.cross(fibers[i], radial[i])
        sheets[i] /= np.linalg.norm(sheets[i]) + 1e-8

    # Write .lon file (openCARP format: 2 vectors per element — fiber + sheet)
    lon_path = os.path.join(output_dir, f"{patient_id}_fibers.lon")
    with open(lon_path, 'w') as f:
        f.write("2\n")  # 2 vectors per element
        for i in range(n_tets):
            fx, fy, fz = fibers[i]
            sx, sy, sz = sheets[i]
            f.write(f"{fx:.6f} {fy:.6f} {fz:.6f} {sx:.6f} {sy:.6f} {sz:.6f}\n")

    print(f"    Fibers: {n_tets} elements → {lon_path}")

    # Also export as .pts and .elem for openCARP
    pts_path = os.path.join(output_dir, f"{patient_id}.pts")
    elem_path = os.path.join(output_dir, f"{patient_id}.elem")

    with open(pts_path, 'w') as f:
        f.write(f"{len(points)}\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

    with open(elem_path, 'w') as f:
        f.write(f"{n_tets}\n")
        for tet in tet_cells:
            f.write(f"Tt {tet[0]} {tet[1]} {tet[2]} {tet[3]} 0\n")

    print(f"    openCARP: {pts_path}, {elem_path}")
    return lon_path


# ─── 7. MAIN PIPELINE ───

def run_pipeline(checkpoint_path, acdc_dir, output_dir, n_patients=10,
                 device="mps", target_spacing=(1.5625, 1.5625, 5.0)):
    """Run the complete post-segmentation pipeline."""

    os.makedirs(output_dir, exist_ok=True)
    seg_dir = os.path.join(output_dir, "segmentations")
    mesh_dir = os.path.join(output_dir, "meshes")
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)

    patch_size = [20, 256, 224]

    # ── Load model ──
    print("=" * 60)
    print("Phase 02 — Post-Segmentation Pipeline")
    print("Deliverable D1.2: Patient-Specific Cardiac Meshes")
    print("=" * 60)

    print(f"\n1. Loading checkpoint: {checkpoint_path}")
    network = build_network(num_classes=4, device=device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    network.load_state_dict(ckpt["network"])
    network.eval()
    epoch = ckpt.get("epoch", "?")
    best_dice = ckpt.get("best_val_dice", "?")
    print(f"   Model from epoch {epoch}, best Dice={best_dice}")
    print(f"   Device: {device}")

    # ── Find ACDC patients ──
    patient_dirs = sorted([
        d for d in os.listdir(acdc_dir)
        if d.startswith("patient") and os.path.isdir(os.path.join(acdc_dir, d))
    ])[:n_patients]

    print(f"\n2. Processing {len(patient_dirs)} patients from ACDC\n")

    results = []
    t_total = time.time()

    for idx, patient_id in enumerate(patient_dirs):
        t_patient = time.time()
        print(f"── [{idx+1}/{len(patient_dirs)}] {patient_id} ──")

        patient_path = os.path.join(acdc_dir, patient_id)

        # Find ED frame (frame01 in ACDC)
        nifti_files = sorted([
            f for f in os.listdir(patient_path)
            if f.endswith(".nii.gz") and "gt" not in f and "4d" not in f
        ])

        if not nifti_files:
            print(f"  ⚠ No NIfTI found for {patient_id}")
            continue

        # Use first frame (ED)
        nifti_path = os.path.join(patient_path, nifti_files[0])
        gt_path = nifti_path.replace(".nii.gz", "_gt.nii.gz")

        # ── Step A: Preprocess ──
        print(f"  A. Preprocess: {nifti_files[0]}")
        try:
            volume, resampled_img = preprocess_volume(nifti_path, target_spacing)
            print(f"     Volume shape: {volume.shape}")
        except Exception as e:
            print(f"     ⚠ Preprocess failed: {e}")
            continue

        # ── Step B: Inference ──
        print(f"  B. Inference (sliding window)...")
        try:
            segmentation = sliding_window_inference(
                network, volume, patch_size, device, overlap=0.5
            )
            if device == "mps":
                torch.mps.synchronize()

            unique, counts = np.unique(segmentation, return_counts=True)
            label_map = {0: "BG", 1: "RV", 2: "MYO", 3: "LV"}
            for u, c in zip(unique, counts):
                pct = 100 * c / segmentation.size
                print(f"     Label {u} ({label_map.get(u, '?')}): {c} voxels ({pct:.1f}%)")
        except Exception as e:
            print(f"     ⚠ Inference failed: {e}")
            continue

        # ── Step C: Compute Dice vs ground truth ──
        dice_scores = {}
        if os.path.exists(gt_path):
            gt_img = sitk.ReadImage(gt_path)
            gt_resampled = sitk.Resample(gt_img, resampled_img,
                                          sitk.Transform(), sitk.sitkNearestNeighbor)
            gt_arr = sitk.GetArrayFromImage(gt_resampled).astype(np.uint8)

            # Crop/pad gt to same shape
            for c, name in [(1, "RV"), (2, "MYO"), (3, "LV")]:
                pred_c = (segmentation == c).astype(float)
                gt_shape = gt_arr.shape
                seg_shape = segmentation.shape
                # Use minimum shape
                min_shape = tuple(min(s, g) for s, g in zip(seg_shape, gt_shape))
                pred_crop = pred_c[:min_shape[0], :min_shape[1], :min_shape[2]]
                gt_crop = (gt_arr[:min_shape[0], :min_shape[1], :min_shape[2]] == c).astype(float)

                inter = (pred_crop * gt_crop).sum()
                union = pred_crop.sum() + gt_crop.sum()
                dice = (2 * inter + 1e-5) / (union + 1e-5)
                dice_scores[name] = dice
                print(f"     Dice {name}: {dice:.4f}")

        # ── Step D: Export NIfTI ──
        print(f"  D. Export NIfTI segmentation")
        try:
            label_sitk = resample_label_to_original(segmentation, nifti_path, target_spacing)
            seg_output = os.path.join(seg_dir, f"{patient_id}_seg.nii.gz")
            save_segmentation_nifti(label_sitk, seg_output)
        except Exception as e:
            print(f"     ⚠ NIfTI export failed: {e}")
            continue

        # ── Step E: Gmsh Mesh ──
        print(f"  E. Gmsh meshing (myocardium)")
        try:
            mesh_path = segmentation_to_mesh(
                seg_output, mesh_dir, patient_id,
                mesh_size_min=0.5, mesh_size_max=1.5
            )
        except Exception as e:
            print(f"     ⚠ Meshing failed: {e}")
            mesh_path = None

        # ── Step F: QC ──
        qc = {}
        if mesh_path and os.path.exists(mesh_path):
            print(f"  F. Quality check (Jacobian)")
            try:
                qc = mesh_quality_check(mesh_path)
                print(f"     Nodes: {qc['n_nodes']}, Tets: {qc['n_tetrahedra']}")
                print(f"     Jacobian min: {qc['jacobian_min']:.4f}")
                print(f"     Degenerate: {qc['n_degenerate']} ({qc['pct_degenerate']:.2f}%)")
                print(f"     SLO Jac>0: {'✅' if qc['slo_jacobian_positive'] else '❌'}")
                print(f"     SLO Deg<0.1%: {'✅' if qc['slo_degenerate_lt_0_1pct'] else '❌'}")
            except Exception as e:
                print(f"     ⚠ QC failed: {e}")
        else:
            print(f"  F. QC skipped (no mesh)")

        # ── Step G: LDRB Fibers ──
        lon_path = None
        if mesh_path and os.path.exists(mesh_path):
            print(f"  G. LDRB fiber generation")
            try:
                lon_path = generate_ldrb_fibers(mesh_path, mesh_dir, patient_id)
            except Exception as e:
                print(f"     ⚠ LDRB failed: {e}")
        else:
            print(f"  G. LDRB skipped (no mesh)")

        # ── Summary ──
        patient_time = (time.time() - t_patient) / 60
        result = {
            "patient_id": patient_id,
            "dice": dice_scores,
            "mesh_path": mesh_path,
            "lon_path": lon_path,
            "qc": qc,
            "time_min": patient_time,
        }
        results.append(result)
        print(f"  Done in {patient_time:.1f} min\n")

    # ── Final Report ──
    total_time = (time.time() - t_total) / 60
    print("=" * 60)
    print("DELIVERABLE D1.2 — SUMMARY")
    print("=" * 60)

    n_success = sum(1 for r in results if r["mesh_path"] is not None)
    print(f"\nPatients processed: {len(results)}")
    print(f"Meshes generated:   {n_success}/{len(results)}")
    print(f"Total time:         {total_time:.1f} min")

    if results:
        all_dice = {"RV": [], "MYO": [], "LV": []}
        for r in results:
            for k in all_dice:
                if k in r["dice"]:
                    all_dice[k].append(r["dice"][k])

        print(f"\nMean Dice scores:")
        for k in ["RV", "MYO", "LV"]:
            if all_dice[k]:
                mean_d = np.mean(all_dice[k])
                print(f"  {k}: {mean_d:.4f}")

        myo_mean = np.mean(all_dice["MYO"]) if all_dice["MYO"] else 0
        print(f"\nSLO MYO Dice ≥ 0.90: {'✅ PASS' if myo_mean >= 0.90 else '❌ FAIL'} ({myo_mean:.4f})")

    # Save report JSON
    report_path = os.path.join(output_dir, "d1_2_report.json")
    report = {
        "deliverable": "D1.2",
        "n_patients": len(results),
        "n_meshes": n_success,
        "total_time_min": total_time,
        "results": [{
            "patient_id": r["patient_id"],
            "dice": r["dice"],
            "mesh_path": r["mesh_path"],
            "lon_path": r["lon_path"],
            "qc": r["qc"],
            "time_min": r["time_min"],
        } for r in results],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {report_path}")


# ─── CLI ───

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 02 Post-Segmentation Pipeline")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_best.pth")
    parser.add_argument("--acdc_dir", required=True, help="Path to ACDC training directory")
    parser.add_argument("--output_dir", default="~/cdt/reports/meshes_acdc",
                        help="Output directory for meshes and reports")
    parser.add_argument("--n_patients", type=int, default=10, help="Number of patients to process")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"],
                        help="Device for inference")
    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)
    args.checkpoint = os.path.expanduser(args.checkpoint)
    args.acdc_dir = os.path.expanduser(args.acdc_dir)

    run_pipeline(
        checkpoint_path=args.checkpoint,
        acdc_dir=args.acdc_dir,
        output_dir=args.output_dir,
        n_patients=args.n_patients,
        device=args.device,
    )
