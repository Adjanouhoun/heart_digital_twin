"""
Traite UN SEUL point du DoE : charge le maillage grossier valide,
lance FenicsxSolver.simulate(), calcule le VRAI volume de CAVITE
(pas le volume de tissu) par empilement de disques radiaux, sauvegarde
le champ de deplacement complet pour analyse future.

CORRECTIF (2026-07-12) : les runs precedents utilisaient par erreur
volume_tissue_mL (volume du MUSCLE, quasi-incompressible, ~3-5% de
variation) comme s'il s'agissait du volume de la CAVITE ventriculaire
(qui varie de ~50-60% entre diastole et systole) -- EF calculee etait
donc fausse (2-5% au lieu de 50-70% physiologique).

Usage : python3 run_single_doe_point.py <job_index> <params_json_path>
"""
import sys, json, time, os
sys.path.insert(0, "/cdt")
import numpy as np

job_index = int(sys.argv[1])
params_path = sys.argv[2]

with open(params_path) as f:
    all_params = json.load(f)
row = all_params[job_index]
job_id = f"doe_{job_index:04d}"

print(f"=== DoE point {job_index} (job_id={job_id}) ===", flush=True)
print(f"Parametres : {row}", flush=True)

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


def _integrate(y, x):
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)


def cavity_volume_fixed_z(nodes_pos, z_min_ref, z_max_ref, center_xy_ref, n_slices=40):
    """
    Approxime le volume de CAVITE (pas le tissu) par empilement de
    disques, integre sur une plage Z et un centre XY FIXES (ceux de
    l'etat AU REPOS). CRITIQUE : integrer sur le meme reperage anatomique
    pour l'etat deforme evite un artefact ou l'allongement axial (l'apex
    est libre, la base fixee) fausse completement le calcul si on
    recalcule z_min/z_max sur l'etat deforme (bug diagnostique et
    corrige le 2026-07-12 : donnait EF~2% au lieu de ~24% reel).

    APPROXIMATION : sections transversales quasi-circulaires (rayon
    endocardique = 10e percentile des rayons de noeuds par tranche).
    """
    z_edges = np.linspace(z_min_ref, z_max_ref, n_slices + 1)
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    dz = z_edges[1] - z_edges[0]
    areas = []
    for zc in z_centers:
        mask = np.abs(nodes_pos[:, 2] - zc) < dz / 2
        if mask.sum() < 4:
            areas.append(0.0)
            continue
        radii = np.linalg.norm(nodes_pos[mask][:, :2] - center_xy_ref, axis=1)
        areas.append(np.pi * np.percentile(radii, 10) ** 2)
    areas = np.array(areas)
    return _integrate(areas, z_centers) / 1000.0  # mm^3 -> mL


Z_MIN_REF = nodes[:, 2].min()
Z_MAX_REF = nodes[:, 2].max()
CENTER_XY_REF = nodes[:, :2].mean(axis=0)

V_ed_cavity = cavity_volume_fixed_z(nodes, Z_MIN_REF, Z_MAX_REF, CENTER_XY_REF)
print(f"V_ed (cavite, repos) = {V_ed_cavity:.2f} mL", flush=True)

from app.solver.mechanics.fenicsx_solver import FenicsxSolver, MechanicsParameters

mech_params = MechanicsParameters(
    a_kPa=row["a_kPa"], b=row["b"],
    T_max_kPa=row["T_max_kPa"],
    checkpoint_dir=f"/cdt/doe_checkpoints/{job_id}",
)

solver = FenicsxSolver()
t0 = time.time()
result = solver.simulate(
    params=mech_params, nodes=nodes, elements=elements, fibers=fibers,
    activation_times_ms=np.zeros(len(nodes)),
    twin_id="doe", job_id=job_id,
)
dt = time.time() - t0

if result.converged:
    nodes_deformed = nodes + result.displacement_mm
    V_es_cavity = cavity_volume_fixed_z(nodes_deformed, Z_MIN_REF, Z_MAX_REF, CENTER_XY_REF)
    ef_pct = 100 * (V_ed_cavity - V_es_cavity) / V_ed_cavity
else:
    V_es_cavity = None
    ef_pct = None

# --- Sauvegarde du champ COMPLET (permanent, pas ecrase par checkpoint) ---
os.makedirs("/cdt/doe_full_results", exist_ok=True)
np.savez(f"/cdt/doe_full_results/{job_id}.npz",
         nodes=nodes, elements=elements,
         displacement_mm=result.displacement_mm,
         converged=result.converged, min_jacobian=result.min_jacobian)

output = {
    "job_index": job_index, "job_id": job_id, "params": row,
    "converged": result.converged, "duration_s": round(dt, 1),
    "V_ed_cavity_mL": round(V_ed_cavity, 3),
    "V_es_cavity_mL": round(V_es_cavity, 3) if V_es_cavity else None,
    "ef_pct": round(ef_pct, 2) if ef_pct else None,
    "min_jacobian": round(result.min_jacobian, 4),
    "load_fraction": round(result.load_fraction, 4),
    "n_iterations": result.n_iterations,
}
os.makedirs("/cdt/doe_results", exist_ok=True)
with open(f"/cdt/doe_results/{job_id}.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"=== TERMINE : converged={result.converged} "
      f"EF_cavite={output['ef_pct']}% duree={dt:.1f}s ===", flush=True)
