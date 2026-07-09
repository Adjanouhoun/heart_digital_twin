"""
Validation formelle openCARP contre le benchmark N-version (Niederer et al.
2011). Geometrie officielle : coin 20x7x3mm, stimule dans un coin (sphere),
propagation diagonale mesuree en 8 points (P1 pres du stimulus -> P8 au
coin oppose).

Reutilise l'infrastructure VALIDEE existante (app/solver/ep/opencarp_config.py,
generate_par_file) au lieu de reconstruire un .par a la main -- evite le
bug dt (doit etre en MICROSECONDES, pas ms) et utilise la detection LAT
native (depol.dat) plutot qu'un seuillage manuel sur vm.igb.
"""
import sys, subprocess, time, os
sys.path.insert(0, "/cdt")
import numpy as np
from app.solver.ep.opencarp_config import generate_par_file, VALIDATED

WORK_DIR = "/tmp/opencarp_niederer_bench"
os.makedirs(WORK_DIR, exist_ok=True)

print("=== Generation du maillage (carputils.mesh.Block) ===", flush=True)
import carputils.mesh as cmesh

RESOLUTION_MM = 0.5
geom = cmesh.Block(centre=(10.0, 3.5, 1.5), size=(20.0, 7.0, 3.0),
                    resolution=RESOLUTION_MM, etype="tetra")

mesh_prefix = f"{WORK_DIR}/wedge"
opts = [str(o) for o in geom.mesher_opts(mesh_prefix)]

t0 = time.time()
result = subprocess.run(["mesher"] + opts, capture_output=True, text=True, timeout=120)
print(f"mesher termine en {time.time()-t0:.1f}s, code={result.returncode}", flush=True)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:], flush=True)
    sys.exit(1)

with open(f"{mesh_prefix}.pts") as f:
    n_nodes = int(f.readline())
    nodes = np.array([list(map(float, f.readline().split())) for _ in range(n_nodes)])
print(f"Maillage : {n_nodes} noeuds", flush=True)
print(f"Etendue coordonnees : x=[{nodes[:,0].min():.1f},{nodes[:,0].max():.1f}] "
      f"(verif unites : doit etre ~0-20000 si um, ~0-20 si mm)", flush=True)

# --- Fibres longitudinales selon X ---
with open(f"{mesh_prefix}.lon", "w") as f:
    f.write("1\n")
    n_elem_check = sum(1 for _ in open(f"{mesh_prefix}.elem")) - 1
    for _ in range(n_elem_check):
        f.write("1.000000 0.000000 0.000000\n")

# --- Determination de l'unite reelle du maillage genere ---
coord_max = nodes[:, 0].max()
is_um = coord_max > 100  # si > 100, c'est forcement des um (le coin fait 20mm max)
unit_scale = 1000.0 if is_um else 1.0  # convertit mm -> um si besoin (CORRIGE)
print(f"Unite detectee : {'um' if is_um else 'mm (conversion en um appliquee)'}", flush=True)

apex_um = (0.0, 0.0, 0.0)  # coin stimule, coordonnees en um
stim_radius_um = 2 * RESOLUTION_MM * 1000.0  # rayon ~2x la resolution

par_content = generate_par_file(
    mesh_path=mesh_prefix,
    output_path=f"{WORK_DIR}/output",
    tend_ms=130.0,
    apex_um=apex_um,
    stim_radius_um=stim_radius_um,
    bcl_ms=130.0,
    g_mult=1.0,  # conductivite de reference deja validee
)
with open(f"{WORK_DIR}/sim.par", "w") as f:
    f.write(par_content)
print(f"\n.par genere via generate_par_file() (config validee, dt en us)", flush=True)

print("\n=== Lancement openCARP (mpirun -n 4) ===", flush=True)
t0 = time.time()
proc = subprocess.Popen(
    ["mpirun", "-n", "4", "openCARP", "--param-fallback=legacy", "+F", f"{WORK_DIR}/sim.par"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
)
for line in proc.stdout:
    elapsed = time.time() - t0
    print(f"[{elapsed:7.1f}s] {line.rstrip()}", flush=True)
    if elapsed > 300:
        proc.kill()
        print("TIMEOUT 300s -- process tue", flush=True)
        break
proc.wait()
print(f"openCARP termine en {time.time()-t0:.1f}s, code={proc.returncode}", flush=True)

# --- Lecture des temps d'activation depuis depol.dat (LAT natif openCARP) ---
depol_path = f"{WORK_DIR}/output.depol.dat"
print(f"\n=== Recherche fichier LAT ===", flush=True)
for candidate in [f"{WORK_DIR}/output/init_acts_depol-thresh.dat",
                   depol_path, f"{WORK_DIR}/output/depol.dat",
                   f"{WORK_DIR}/output_depol.dat"]:
    if os.path.exists(candidate):
        print(f"Trouve : {candidate}", flush=True)
        depol_path = candidate
        break
else:
    print("depol.dat non trouve aux emplacements attendus, listing du dossier :", flush=True)
    for root, dirs, files in os.walk(WORK_DIR):
        for fn in files:
            print(f"  {os.path.join(root, fn)}", flush=True)
    sys.exit(1)

act_times = np.loadtxt(depol_path)
print(f"Temps d'activation charges : {len(act_times)} valeurs", flush=True)

diag_start = np.array([0.0, 0.0, 0.0]) * unit_scale
diag_end = np.array([20.0, 7.0, 3.0]) * unit_scale
print(f"\n{'Point':>6} {'Position (um)':>30} {'Act. time (ms)':>15}")
for k in range(1, 9):
    frac = (k - 1) / 7.0
    target = diag_start + frac * (diag_end - diag_start)
    dists = np.linalg.norm(nodes - target, axis=1)
    nearest = np.argmin(dists)
    t_act = act_times[nearest] if nearest < len(act_times) else float('nan')
    print(f"P{k:<5} {str(np.round(nodes[nearest],1)):>30} {t_act:>15.3f}")
