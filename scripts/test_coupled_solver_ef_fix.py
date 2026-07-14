"""
Test de validation : verifie que la correction du calcul d'EF dans
coupled_solver.py (cavite via empilement de disques, pas volume de tissu)
fonctionne dans l'architecture CoupledSolver prescrite par le projet.

EP tourne en fallback analytique (openCARP absent de cette image de test) --
verifie sans impact sur ce test car activation_times n'est pas utilise
dans _compute_volume_waveform (confirme par lecture directe du code).
"""
import sys, time
sys.path.insert(0, "/cdt")
import numpy as np

MESH_DIR = "/cdt/reports/meshes_acdc/meshes"
PATIENT = "patient001_coarse5"

with open(f"{MESH_DIR}/{PATIENT}.pts") as f:
    n = int(f.readline())
    nodes = np.array([list(map(float, f.readline().split())) for _ in range(n)])
with open(f"{MESH_DIR}/{PATIENT}.elem") as f:
    n_elem = int(f.readline())
    elements = np.array([
        list(map(int, f.readline().split()[1:5])) for _ in range(n_elem)
    ], dtype=np.int64)
with open(f"{MESH_DIR}/{PATIENT}_fibers_ldrb.lon") as f:
    n_fib = int(f.readline())
    fibers = np.array([list(map(float, f.readline().split())) for _ in range(n_fib)])

print(f"Maillage : {len(nodes)} noeuds, {len(elements)} tets", flush=True)

from app.solver.coupled_solver import CoupledSolver, SimulationParameters

params = SimulationParameters(
    sigma_l=0.30, sigma_t=0.12, sigma_n=0.05,
    a_kPa=0.496, b=7.209, a_f_kPa=15.193, b_f=20.417,
    T_max_kPa=135.0, tau_r_ms=75.0, tau_d_ms=150.0,
    R_p=1.2e8, C_a=1.0e-8, Z_c=1.0e7,
    heart_rate_bpm=70.0,
)

solver = CoupledSolver()
t0 = time.time()
result = solver.simulate(
    params=params, nodes=nodes, elements=elements, fiber_vectors=fibers,
    twin_id="test_ef_fix", job_id="test_coupled_0000",
)
dt = time.time() - t0

print(f"\n=== RESULTAT ({dt:.1f}s) ===", flush=True)
print(f"error_message: {result.error_message}", flush=True)
print(f"benchmark_passed: {result.benchmark_passed}", flush=True)
if result.mech_result:
    print(f"mech_result.converged: {result.mech_result.converged}", flush=True)
    print(f"mech_result.min_jacobian: {result.mech_result.min_jacobian}", flush=True)
if result.wk_result:
    print(f"wk_result.ef_pct: {result.wk_result.ef_pct}", flush=True)
    print(f"wk_result.edv_mL: {result.wk_result.edv_mL}", flush=True)
    print(f"wk_result.esv_mL: {result.wk_result.esv_mL}", flush=True)
    print(f"wk_result.slo_passed: {result.wk_result.slo_passed}", flush=True)
    print(f"wk_result.slo_details: {result.wk_result.slo_details}", flush=True)
