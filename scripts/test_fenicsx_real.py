"""
Test FEniCSx reel : simulation mecanique sur le maillage myocardique patient001.
A lancer DANS le conteneur dolfinx/dolfinx:v0.7.3 (dolfinx installe).

Usage :
docker run --rm -v ~/cdt:/cdt -w /cdt dolfinx/dolfinx:v0.7.3 sh -c \
  "pip install structlog --quiet && python3 scripts/test_fenicsx_real.py"
"""
import sys, time
sys.path.insert(0, "/cdt")
import numpy as np

from app.core.units import filter_small_elements, filter_degenerate_dihedral
from app.solver.mechanics.fenicsx_solver import FenicsxSolver, MechanicsParameters

MESH_DIR = "/tmp/gmsh_test"
PATIENT = "patient001"

print("=== Chargement maillage (mm) ===")
with open(f"{MESH_DIR}/{PATIENT}.pts") as f:
    f.readline()
    nodes_mm = np.array([list(map(float, l.split())) for l in f])
with open(f"{MESH_DIR}/{PATIENT}.elem") as f:
    f.readline()
    elements = [[int(x) for x in l.split()[1:5]] for l in f]
with open(f"{MESH_DIR}/{PATIENT}_fibers.lon") as f:
    f.readline()
    fibers_raw = [l.strip().split() for l in f]

print(f"Original: {len(nodes_mm)} nodes, {len(elements)} tets")

# Filtrer les slivers (meme seuil valide pour openCARP)
elements_f, n_removed = filter_small_elements(nodes_mm, elements, 0.3)
print(f"Filtered (h_min>=0.3mm): removed {n_removed} tets")

keep = sorted(set(v for e in elements_f for v in e))
node_map = {old: new for new, old in enumerate(keep)}
nodes = nodes_mm[keep]
elements_final = np.array([[node_map[v] for v in e] for e in elements_f], dtype=np.int64)
fibers = np.array([
    list(map(float, fibers_raw[i][:3])) if i < len(fibers_raw) else [1.0, 0.0, 0.0]
    for i in keep
])

print(f"Final: {len(nodes)} nodes, {len(elements_final)} tets")

print("\n=== Filtrage dièdre (slivers géométriques) ===")
elements_final, n_removed_dihedral, _ = filter_degenerate_dihedral(
    nodes, elements_final, min_angle_deg=15.0)
elements_final = np.array(elements_final, dtype=np.int64)
print(f"Removed {n_removed_dihedral} tets (angle diedre < 15 deg)")

# Reduire les noeuds au sous-ensemble encore utilise
keep2 = sorted(set(v for e in elements_final for v in e))
node_map2 = {old: new for new, old in enumerate(keep2)}
nodes = nodes[keep2]
elements_final = np.array([[node_map2[v] for v in e] for e in elements_final], dtype=np.int64)
fibers = fibers[keep2]
print(f"Apres filtrage diedre: {len(nodes)} nodes, {len(elements_final)} tets")

print("\n=== Lancement FEniCSx (Holzapfel-Ogden + tension active) ===")
solver = FenicsxSolver()
print("fenicsx_available:", solver._fenicsx_available)

params = MechanicsParameters()  # valeurs par defaut (Land 2015)
activation_times = np.zeros(len(nodes))  # non utilise par _run_fenicsx directement

t0 = time.time()
result = solver.simulate(
    params=params,
    nodes=nodes,
    elements=elements_final,
    fibers=fibers,
    activation_times_ms=activation_times,
    twin_id="a" * 64,
    job_id="test_fenicsx_real",
)
dt = time.time() - t0

print(f"\n=== Resultat ({dt:.1f}s) ===")
print(f"converged: {result.converged}")
print(f"n_iterations: {result.n_iterations}")
print(f"endo_radial_disp_mm: {result.endo_radial_disp_mm:.3f}")
print(f"epi_radial_disp_mm: {result.epi_radial_disp_mm:.3f}")
print(f"volume_tissue_mL: {result.volume_tissue_mL:.1f}")
print(f"solver_version: {result.solver_version}")
