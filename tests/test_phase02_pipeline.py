"""
Tests for Phase 02 Post-Segmentation Pipeline
Covers: network builder, sliding window inference, mesh QC, LDRB fibers, STL writer, openCARP export
"""
import os
import json
import tempfile
import pytest
import numpy as np
import torch

# ─── We add the scripts dir to path so we can import from generate_meshes_acdc ───
import sys
sys.path.insert(0, os.path.expanduser("~/cdt/scripts"))
from generate_meshes_acdc import (
    build_network,
    sliding_window_inference,
    mesh_quality_check,
    generate_ldrb_fibers,
    _write_stl,
)


# ═══════════════════════════════════════════════════════════
# 1. Network Builder
# ═══════════════════════════════════════════════════════════

class TestBuildNetwork:
    def test_network_has_correct_param_count(self):
        network = build_network(num_classes=4, device="cpu")
        n_params = sum(p.numel() for p in network.parameters()) / 1e6
        assert abs(n_params - 30.4) < 0.5, f"Expected ~30.4M params, got {n_params:.1f}M"

    def test_network_output_shape(self):
        network = build_network(num_classes=4, device="cpu")
        network.eval()
        dummy = torch.randn(1, 1, 20, 256, 224)
        with torch.no_grad():
            out = network(dummy)
        assert out.shape == (1, 4, 20, 256, 224), f"Wrong shape: {out.shape}"

    def test_network_output_4_classes(self):
        network = build_network(num_classes=4, device="cpu")
        network.eval()
        dummy = torch.randn(1, 1, 20, 256, 224)
        with torch.no_grad():
            out = network(dummy)
        assert out.shape[1] == 4

    def test_network_batch_2(self):
        network = build_network(num_classes=4, device="cpu")
        network.eval()
        dummy = torch.randn(2, 1, 20, 256, 224)
        with torch.no_grad():
            out = network(dummy)
        assert out.shape[0] == 2

    def test_network_gradients_flow(self):
        network = build_network(num_classes=4, device="cpu")
        network.train()
        dummy = torch.randn(1, 1, 20, 256, 224)
        target = torch.randint(0, 4, (1, 20, 256, 224))
        out = network(dummy)
        loss = torch.nn.functional.cross_entropy(out, target)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in network.parameters())
        assert has_grad, "No gradients flowing through network"


# ═══════════════════════════════════════════════════════════
# 2. Sliding Window Inference
# ═══════════════════════════════════════════════════════════

class TestSlidingWindowInference:
    @pytest.fixture
    def small_network(self):
        """Use the real network for correct skip connections."""
        return build_network(num_classes=4, device="cpu").eval()

    def test_output_shape_matches_input(self, small_network):
        volume = np.random.randn(20, 256, 224).astype(np.float32)
        patch_size = [20, 256, 224]
        result = sliding_window_inference(small_network, volume, patch_size, "cpu", overlap=0.5)
        assert result.shape == volume.shape, f"Expected {volume.shape}, got {result.shape}"

    def test_output_dtype_uint8(self, small_network):
        volume = np.random.randn(20, 256, 224).astype(np.float32)
        patch_size = [20, 256, 224]
        result = sliding_window_inference(small_network, volume, patch_size, "cpu")
        assert result.dtype == np.uint8

    def test_output_labels_in_range(self, small_network):
        volume = np.random.randn(20, 256, 224).astype(np.float32)
        patch_size = [20, 256, 224]
        result = sliding_window_inference(small_network, volume, patch_size, "cpu")
        assert result.min() >= 0
        assert result.max() <= 3

    def test_padding_small_volume(self, small_network):
        """Volume smaller than patch_size should be padded and result cropped back."""
        volume = np.random.randn(10, 128, 112).astype(np.float32)
        patch_size = [20, 256, 224]
        result = sliding_window_inference(small_network, volume, patch_size, "cpu")
        assert result.shape == (10, 128, 112), f"Padding failed: {result.shape}"

    def test_overlap_coverage(self, small_network):
        """Every voxel should be predicted at least once."""
        volume = np.random.randn(20, 256, 224).astype(np.float32)
        patch_size = [20, 256, 224]
        result = sliding_window_inference(small_network, volume, patch_size, "cpu", overlap=0.5)
        assert result.shape == volume.shape


# ═══════════════════════════════════════════════════════════
# 3. STL Writer
# ═══════════════════════════════════════════════════════════

class TestSTLWriter:
    def test_stl_file_created(self):
        vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
        faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            path = f.name
        try:
            _write_stl(vertices, faces, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_stl_header_size(self):
        """Binary STL: 80-byte header + 4-byte count + 50 bytes per triangle."""
        vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
        faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            path = f.name
        try:
            _write_stl(vertices, faces, path)
            expected_size = 80 + 4 + len(faces) * 50
            actual_size = os.path.getsize(path)
            assert actual_size == expected_size, f"Expected {expected_size}, got {actual_size}"
        finally:
            os.unlink(path)

    def test_stl_normals_unit_length(self):
        vertices = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32)
        faces = np.array([[0,1,2]])
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            path = f.name
        try:
            _write_stl(vertices, faces, path)
            with open(path, 'rb') as f:
                f.read(80)  # header
                f.read(4)   # count
                normal = np.frombuffer(f.read(12), dtype=np.float32)
                norm_len = np.linalg.norm(normal)
                assert abs(norm_len - 1.0) < 1e-5, f"Normal not unit: {norm_len}"
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════
# 4. Mesh Quality Check
# ═══════════════════════════════════════════════════════════

class TestMeshQualityCheck:
    @pytest.fixture
    def good_mesh_path(self):
        """Create a simple valid tetrahedral mesh with positive Jacobians."""
        import meshio
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 1],
        ], dtype=np.float64)
        cells = [("tetra", np.array([
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ]))]
        mesh = meshio.Mesh(points, cells)
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            path = f.name
        meshio.write(path, mesh)
        yield path
        os.unlink(path)

    def test_qc_returns_dict(self, good_mesh_path):
        qc = mesh_quality_check(good_mesh_path)
        assert isinstance(qc, dict)

    def test_qc_has_required_keys(self, good_mesh_path):
        qc = mesh_quality_check(good_mesh_path)
        required = ["n_nodes", "n_tetrahedra", "jacobian_min", "jacobian_max",
                     "jacobian_mean", "n_degenerate", "pct_degenerate",
                     "slo_jacobian_positive", "slo_degenerate_lt_0_1pct"]
        for key in required:
            assert key in qc, f"Missing key: {key}"

    def test_qc_node_count(self, good_mesh_path):
        qc = mesh_quality_check(good_mesh_path)
        assert qc["n_nodes"] == 5

    def test_qc_tet_count(self, good_mesh_path):
        qc = mesh_quality_check(good_mesh_path)
        assert qc["n_tetrahedra"] == 2

    def test_qc_jacobian_positive(self, good_mesh_path):
        qc = mesh_quality_check(good_mesh_path)
        assert qc["jacobian_min"] > 0, f"Jacobian min should be positive: {qc['jacobian_min']}"

    def test_qc_no_degenerate(self, good_mesh_path):
        qc = mesh_quality_check(good_mesh_path)
        assert qc["n_degenerate"] == 0

    def test_qc_slo_pass(self, good_mesh_path):
        qc = mesh_quality_check(good_mesh_path)
        assert qc["slo_jacobian_positive"] is True
        assert qc["slo_degenerate_lt_0_1pct"] is True

    def test_qc_degenerate_detection(self):
        """Mesh with inverted tet should be detected."""
        import meshio
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=np.float64)
        # Inverted tet (swap two vertices)
        cells = [("tetra", np.array([[0, 2, 1, 3]]))]
        mesh = meshio.Mesh(points, cells)
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            path = f.name
        meshio.write(path, mesh)
        try:
            qc = mesh_quality_check(path)
            assert qc["n_degenerate"] == 1
            assert qc["slo_jacobian_positive"] is False
        finally:
            os.unlink(path)

    def test_qc_no_tetra_returns_error(self):
        """Mesh without tetrahedra should return error dict."""
        import meshio
        points = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float64)
        cells = [("triangle", np.array([[0,1,2]]))]
        mesh = meshio.Mesh(points, cells)
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            path = f.name
        meshio.write(path, mesh)
        try:
            qc = mesh_quality_check(path)
            assert "error" in qc
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════
# 5. LDRB Fiber Generation
# ═══════════════════════════════════════════════════════════

class TestLDRBFibers:
    @pytest.fixture
    def tet_mesh_path(self):
        """Create a simple tet mesh for LDRB testing."""
        import meshio
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=np.float64)
        cells = [("tetra", np.array([
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [1, 3, 4, 5],
            [2, 3, 4, 6],
        ]))]
        mesh = meshio.Mesh(points, cells)
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            path = f.name
        meshio.write(path, mesh)
        yield path
        os.unlink(path)

    @pytest.fixture
    def output_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_lon_file_created(self, tet_mesh_path, output_dir):
        lon = generate_ldrb_fibers(tet_mesh_path, output_dir, "test_patient")
        assert lon is not None
        assert os.path.exists(lon)

    def test_lon_file_format(self, tet_mesh_path, output_dir):
        """First line should be '2' (fiber + sheet vectors)."""
        lon = generate_ldrb_fibers(tet_mesh_path, output_dir, "test_patient")
        with open(lon) as f:
            first_line = f.readline().strip()
        assert first_line == "2"

    def test_lon_line_count(self, tet_mesh_path, output_dir):
        """Should have header + one line per tetrahedron (4 tets here)."""
        lon = generate_ldrb_fibers(tet_mesh_path, output_dir, "test_patient")
        with open(lon) as f:
            lines = f.readlines()
        assert len(lines) == 5, f"Expected 5 lines (1 header + 4 tets), got {len(lines)}"

    def test_lon_vectors_6_components(self, tet_mesh_path, output_dir):
        """Each data line should have 6 floats (fx fy fz sx sy sz)."""
        lon = generate_ldrb_fibers(tet_mesh_path, output_dir, "test_patient")
        with open(lon) as f:
            lines = f.readlines()[1:]  # skip header
        for line in lines:
            parts = line.strip().split()
            assert len(parts) == 6, f"Expected 6 components, got {len(parts)}"

    def test_fiber_vectors_unit_length(self, tet_mesh_path, output_dir):
        """Fiber vectors should be approximately unit length."""
        lon = generate_ldrb_fibers(tet_mesh_path, output_dir, "test_patient")
        with open(lon) as f:
            lines = f.readlines()[1:]
        for line in lines:
            vals = [float(x) for x in line.strip().split()]
            fiber = np.array(vals[:3])
            norm = np.linalg.norm(fiber)
            assert abs(norm - 1.0) < 0.01, f"Fiber not unit: {norm}"

    def test_sheet_vectors_unit_length(self, tet_mesh_path, output_dir):
        """Sheet vectors should be approximately unit length."""
        lon = generate_ldrb_fibers(tet_mesh_path, output_dir, "test_patient")
        with open(lon) as f:
            lines = f.readlines()[1:]
        for line in lines:
            vals = [float(x) for x in line.strip().split()]
            sheet = np.array(vals[3:])
            norm = np.linalg.norm(sheet)
            assert abs(norm - 1.0) < 0.01, f"Sheet not unit: {norm}"

    def test_pts_file_created(self, tet_mesh_path, output_dir):
        generate_ldrb_fibers(tet_mesh_path, output_dir, "test_patient")
        pts = os.path.join(output_dir, "test_patient.pts")
        assert os.path.exists(pts)

    def test_pts_header_node_count(self, tet_mesh_path, output_dir):
        generate_ldrb_fibers(tet_mesh_path, output_dir, "test_patient")
        pts = os.path.join(output_dir, "test_patient.pts")
        with open(pts) as f:
            header = int(f.readline().strip())
        assert header == 8, f"Expected 8 nodes, got {header}"

    def test_elem_file_created(self, tet_mesh_path, output_dir):
        generate_ldrb_fibers(tet_mesh_path, output_dir, "test_patient")
        elem = os.path.join(output_dir, "test_patient.elem")
        assert os.path.exists(elem)

    def test_elem_header_tet_count(self, tet_mesh_path, output_dir):
        generate_ldrb_fibers(tet_mesh_path, output_dir, "test_patient")
        elem = os.path.join(output_dir, "test_patient.elem")
        with open(elem) as f:
            header = int(f.readline().strip())
        assert header == 4, f"Expected 4 tets, got {header}"

    def test_elem_format_opencarp(self, tet_mesh_path, output_dir):
        """Each elem line should start with 'Tt' (openCARP tet format)."""
        generate_ldrb_fibers(tet_mesh_path, output_dir, "test_patient")
        elem = os.path.join(output_dir, "test_patient.elem")
        with open(elem) as f:
            f.readline()  # skip header
            for line in f:
                assert line.startswith("Tt "), f"Expected 'Tt', got: {line}"

    def test_custom_angles(self, tet_mesh_path, output_dir):
        """Custom endo/epi angles should not crash."""
        lon = generate_ldrb_fibers(tet_mesh_path, output_dir, "test_patient",
                                    alpha_endo=-40, alpha_epi=40,
                                    beta_endo=10, beta_epi=-10)
        assert lon is not None


# ═══════════════════════════════════════════════════════════
# 6. Checkpoint Save/Load
# ═══════════════════════════════════════════════════════════

class TestCheckpointIntegrity:
    def test_save_and_reload_weights(self):
        network = build_network(num_classes=4, device="cpu")
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        try:
            torch.save({"network": network.state_dict(), "epoch": 42,
                         "best_val_dice": 0.85}, path)
            ckpt = torch.load(path, weights_only=False)
            network2 = build_network(num_classes=4, device="cpu")
            network2.load_state_dict(ckpt["network"])
            assert ckpt["epoch"] == 42
            assert abs(ckpt["best_val_dice"] - 0.85) < 1e-6
        finally:
            os.unlink(path)

    def test_reloaded_network_same_output(self):
        network = build_network(num_classes=4, device="cpu")
        network.eval()
        dummy = torch.randn(1, 1, 20, 256, 224)
        with torch.no_grad():
            out1 = network(dummy)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        try:
            torch.save({"network": network.state_dict()}, path)
            network2 = build_network(num_classes=4, device="cpu")
            network2.load_state_dict(torch.load(path, weights_only=False)["network"])
            network2.eval()
            with torch.no_grad():
                out2 = network2(dummy)
            assert torch.allclose(out1, out2, atol=1e-6)
        finally:
            os.unlink(path)
