"""
Parseur de fichiers IGB (openCARP output).
Extrait Vm, activation times, CV, APD depuis vm.igb.
"""
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class OpenCARPResult:
    n_nodes: int
    n_frames: int
    dt_ms: float
    tend_ms: float
    vm: np.ndarray
    activation_times_ms: np.ndarray
    n_activated: int
    pct_activated: float
    cv_cm_s: float
    apd_ms: float
    vm_resting_mv: float
    vm_peak_mv: float


def parse_igb(igb_path: str, mesh_pts_path: str, tend_ms: float) -> OpenCARPResult:
    """Parse un fichier vm.igb et extrait les metriques EP."""
    igb_path = Path(igb_path)
    mesh_pts_path = Path(mesh_pts_path)

    with open(mesh_pts_path) as f:
        n_mesh_nodes = int(f.readline().strip())
        nodes = np.array([list(map(float, l.split())) for l in f])

    with open(igb_path, "rb") as f:
        raw = f.read(1024)
        header = ""
        for b in raw:
            if b == 0:
                break
            if b < 128:
                header += chr(b)

    n_igb = None
    n_frames_igb = None
    for part in header.replace("\r\n", " ").replace("\n", " ").split():
        if part.startswith("x:"):
            n_igb = int(part.split(":")[1])
        if part.startswith("t:"):
            n_frames_igb = int(part.split(":")[1])

    if n_igb is None or n_frames_igb is None:
        raise ValueError(f"Cannot parse IGB header: {header[:100]}")

    frame_bytes = n_igb * 4
    file_size = igb_path.stat().st_size
    offset = file_size - (n_frames_igb * frame_bytes)

    with open(igb_path, "rb") as f:
        f.seek(offset)
        vm = np.frombuffer(f.read(), dtype=np.float32).reshape(n_frames_igb, n_igb)

    dt_ms = tend_ms / (n_frames_igb - 1) if n_frames_igb > 1 else 1.0

    activated = vm.max(axis=0) > -10
    n_act = int(activated.sum())

    act_times = np.full(n_igb, np.nan)
    for i in range(n_igb):
        above = np.where(vm[:, i] > -10)[0]
        if len(above) > 0:
            act_times[i] = above[0] * dt_ms

    cv = 0.0
    valid = ~np.isnan(act_times)
    if valid.sum() > 100 and n_igb <= len(nodes):
        act_nodes = nodes[:n_igb][valid[:n_igb]]
        act_t = act_times[valid]
        early = act_t < np.percentile(act_t, 5)
        late = act_t > np.percentile(act_t, 95)
        if early.sum() > 0 and late.sum() > 0:
            dist_cm = np.linalg.norm(
                act_nodes[late].mean(0) - act_nodes[early].mean(0)
            ) / 10000
            time_s = (act_t[late].mean() - act_t[early].mean()) / 1000
            if time_s > 0:
                cv = dist_cm / time_s

    apd_list = []
    for i in range(n_igb):
        above = np.where(vm[:, i] > -10)[0]
        if len(above) > 5:
            start = above[0]
            after = vm[start:, i]
            below = np.where(after < -60)[0]
            if len(below) > 0 and below[0] > 2:
                apd_list.append(below[0] * dt_ms)

    apd = float(np.mean(apd_list)) if apd_list else 0.0

    return OpenCARPResult(
        n_nodes=n_igb,
        n_frames=n_frames_igb,
        dt_ms=dt_ms,
        tend_ms=tend_ms,
        vm=vm,
        activation_times_ms=act_times,
        n_activated=n_act,
        pct_activated=n_act / n_igb * 100,
        cv_cm_s=cv,
        apd_ms=apd,
        vm_resting_mv=float(vm[0].mean()),
        vm_peak_mv=float(vm.max()),
    )
