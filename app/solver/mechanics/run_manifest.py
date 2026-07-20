"""Construction de manifestes reproductibles pour les runs mécaniques."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def build_manifest(root: str | Path, parameters: dict,
                   scientific_inputs: dict[str, str | Path]) -> dict:
    root = Path(root).resolve()
    files = {}
    for label, raw_path in scientific_inputs.items():
        path = Path(raw_path).resolve()
        files[label] = {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    try:
        commit = _git_value(root, "rev-parse", "HEAD")
        dirty = bool(_git_value(root, "status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, None

    packages = {}
    for name in ("numpy", "scipy", "nibabel", "scikit-image"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"commit": commit, "dirty": dirty},
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
        },
        "parameters": parameters,
        "scientific_inputs": files,
    }


def write_manifest(path: str | Path, manifest: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
