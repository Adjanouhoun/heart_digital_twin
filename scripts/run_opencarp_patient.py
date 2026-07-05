"""
Lance une simulation openCARP sur un patient.
Usage: python scripts/run_opencarp_patient.py --patient patient003 --tend 200
"""
import numpy as np
import subprocess
import time
import argparse
import os
import json
from pathlib import Path
from app.core.units import mm_to_um, fix_element_orientation, filter_small_elements, mesh_quality_report
from app.solver.ep.opencarp_config import generate_par_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="patient003")
    parser.add_argument("--tend", type=float, default=200.0)
    parser.add_argument("--mpi", type=int, default=4)
    parser.add_argument("--mesh-dir", default=os.path.expanduser("~/cdt/reports/meshes_acdc/meshes"))
    parser.add_argument("--output-dir", default="/tmp/opencarp_runs")
    parser.add_argument("--min-edge", type=float, default=0.3)
    args = parser.parse_args()

    work_dir = Path(args.output_dir) / args.patient
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== openCARP Simulation ===")
    print(f"  Patient: {args.patient}")
    print(f"  tend: {args.tend} ms")
    print(f"  MPI: {args.mpi} cores")

    with open(f"{args.mesh_dir}/{args.patient}.pts") as f:
        f.readline()
        nodes_mm = np.array([list(map(float, l.split())) for l in f])

    with open(f"{args.mesh_dir}/{args.patient}.elem") as f:
        f.readline()
        elements = [[int(x) for x in l.split()[1:5]] for l in f]

    with open(f"{args.mesh_dir}/{args.patient}_fibers.lon") as f:
        f.readline()
        fibers_raw = [l.strip() for l in f]

    print(f"  Original: {len(nodes_mm)} nodes, {len(elements)} tets")

    elements_filtered, n_removed = filter_small_elements(nodes_mm, elements, args.min_edge)
    print(f"  Filtered (h_min >= {args.min_edge}mm): removed {n_removed} tets")

    keep_nodes = sorted(set(v for e in elements_filtered for v in e))
    node_map = {old: new for new, old in enumerate(keep_nodes)}
    new_nodes_mm = nodes_mm[keep_nodes]
    new_elements = [[node_map[v] for v in e] for e in elements_filtered]

    nodes_um = mm_to_um(new_nodes_mm)
    new_elements, n_flip = fix_element_orientation(nodes_um, new_elements)

    report = mesh_quality_report(new_nodes_mm, new_elements)
    print(f"  Final: {report['n_nodes']} nodes, {report['n_elements']} tets")
    print(f"  h_min: {report['h_min_mm']:.3f} mm ({report['h_min_mm']*1000:.0f} um)")
    print(f"  Flipped: {n_flip}")

    pts_lines = [str(len(nodes_um))]
    for n in nodes_um:
        pts_lines.append(f"{n[0]:.1f} {n[1]:.1f} {n[2]:.1f}")
    (work_dir / "mesh.pts").write_text("\n".join(pts_lines))

    elem_lines = [str(len(new_elements))]
    for e in new_elements:
        elem_lines.append(f"Tt {e[0]} {e[1]} {e[2]} {e[3]} 1")
    (work_dir / "mesh.elem").write_text("\n".join(elem_lines))

    lon_lines = ["2"]
    for idx in keep_nodes:
        if idx < len(fibers_raw):
            parts = fibers_raw[idx].split()
            if len(parts) >= 6:
                lon_lines.append(fibers_raw[idx])
            elif len(parts) >= 3:
                lon_lines.append(f"{parts[0]} {parts[1]} {parts[2]} 0.0 1.0 0.0")
            else:
                lon_lines.append("1.0 0.0 0.0 0.0 1.0 0.0")
        else:
            lon_lines.append("1.0 0.0 0.0 0.0 1.0 0.0")
    (work_dir / "mesh.lon").write_text("\n".join(lon_lines))

    z_10pct = np.percentile(nodes_um[:, 2], 10)
    stim_nodes = (nodes_um[:, 2] < z_10pct).sum()

    apex_um = (
        float(nodes_um[:, 0].mean()),
        float(nodes_um[:, 1].mean()),
        float(nodes_um[:, 2].min() - 100)
    )

    # Stimulus local a l'apex : sphere de 3mm de rayon (physiologique).
    # L'onde d'activation part de l'apex et se propage — un stimulus local
    # est requis pour mesurer le temps d'activation (SLO benchmark +/-5ms).
    stim_radius = 3000.0  # um = 3mm

    par = generate_par_file(
        mesh_path=str(work_dir / "mesh"),
        output_path=str(work_dir / "output"),
        tend_ms=args.tend,
        apex_um=apex_um,
        stim_radius_um=stim_radius,
        bcl_ms=args.tend,
    )

    (work_dir / "sim.par").write_text(par)
    print(f"  Stimulus: {stim_nodes} nodes ({stim_nodes/len(nodes_um)*100:.0f}%)")
    print(f"\nLancement openCARP (tend={args.tend}ms)...")

    t0 = time.time()
    proc = subprocess.Popen(
        ["mpirun", "-n", str(args.mpi), "openCARP.par", "+F", str(work_dir / "sim.par")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    log_path = work_dir / "opencarp.log"
    with open(log_path, "w") as log:
        for line in proc.stdout:
            elapsed = time.time() - t0
            log.write(line)
            log.flush()
            l = line.rstrip()
            if l and ("%" in l or "error" in l.lower() or "Launch" in l
                      or "Timing" in l or "Destroy" in l or "diverge" in l):
                print(f"[{elapsed/60:.1f}min] {l}")

    proc.wait()
    elapsed = time.time() - t0

    result = {
        "patient": args.patient,
        "return_code": proc.returncode,
        "time_seconds": elapsed,
        "tend_ms": args.tend,
        "n_nodes": report["n_nodes"],
        "n_elements": report["n_elements"],
        "h_min_mm": report["h_min_mm"],
    }

    vm_path = work_dir / "output" / "vm.igb"
    if vm_path.exists():
        result["vm_igb_size"] = vm_path.stat().st_size
        print(f"\nvm.igb: {result['vm_igb_size'] / 1024 / 1024:.1f} MB")

    print(f"\nCode: {proc.returncode} | Temps: {elapsed/60:.1f} min")

    with open(work_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Resultat: {work_dir / 'result.json'}")

if __name__ == "__main__":
    main()
