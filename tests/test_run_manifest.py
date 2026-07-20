import hashlib

from app.solver.mechanics.run_manifest import build_manifest, sha256_file


def test_sha256_file(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"cardiac-digital-twin")
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_contains_scientific_input_hash(tmp_path):
    path = tmp_path / "mesh.pts"
    path.write_text("1\n0 0 0\n")
    manifest = build_manifest(
        tmp_path,
        {"T_max_kPa": 30.0},
        {"mesh_nodes": path},
    )
    entry = manifest["scientific_inputs"]["mesh_nodes"]
    assert entry["path"] == "mesh.pts"
    assert entry["size_bytes"] == path.stat().st_size
    assert len(entry["sha256"]) == 64
    assert manifest["parameters"]["T_max_kPa"] == 30.0
