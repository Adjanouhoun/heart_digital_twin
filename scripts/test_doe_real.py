"""
Test DoE : 5 sims sur le VRAI maillage patient001 + vrai openCARP.
But : confirmer que le DoE produit une variance PHYSIQUE reelle (pas du fallback).

Usage : PYTHONPATH=~/cdt python scripts/test_doe_real.py
"""
import base64
import numpy as np
from app.solver.tasks.doe_task import run_doe_batch


def load_mesh(mesh_dir, patient):
    with open(f"{mesh_dir}/{patient}.pts") as f:
        f.readline()
        nodes = np.array([list(map(float, l.split())) for l in f])
    with open(f"{mesh_dir}/{patient}.elem") as f:
        f.readline()
        elements = np.array([[int(x) for x in l.split()[1:5]] for l in f], dtype=np.int32)
    with open(f"{mesh_dir}/{patient}_fibers.lon") as f:
        f.readline()
        fibers = np.array([list(map(float, l.split()[:3])) for l in f])
    return nodes, elements, fibers


def main():
    mesh_dir = "/tmp/gmsh_test"
    patient = "patient001"
    n_sims = 5

    nodes, elements, fibers = load_mesh(mesh_dir, patient)
    print(f"Maillage {patient}: {len(nodes)} nodes, {len(elements)} tets, {len(fibers)} fibres")
    print(f"Lancement DoE {n_sims} sims sur VRAI openCARP...\n")

    summary = run_doe_batch(
        twin_id="a" * 64,
        n_simulations=n_sims,
        nodes_b64=base64.b64encode(nodes.tobytes()).decode(),
        elements_b64=base64.b64encode(elements.astype(np.int32).tobytes()).decode(),
        nodes_shape=list(nodes.shape),
        elements_shape=list(elements.shape),
        fiber_b64=base64.b64encode(fibers.tobytes()).decode(),
    )

    print("\n=== RESUME DoE ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
