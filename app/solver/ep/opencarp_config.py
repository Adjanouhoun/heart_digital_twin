"""
openCARP Configuration — Paramètres validés
Source: opencarp.org/documentation/examples/02_ep_tissue/03c_tuning_wavelength
Benchmark: slab 1575 nodes, 50ms, Code=0 (34.5s)
Patient: patient003, 35374 nodes, 50ms, Code=0 (9h28)
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenCARPValidatedConfig:
    """Paramètres validés par benchmark et documentation officielle."""

    ionic_model: str = "tenTusscherPanfilov"

    g_il: float = 0.3544
    g_it: float = 0.024
    g_el: float = 1.2700
    g_et: float = 0.0862

    dt_ms: float = 0.02
    spacedt: int = 1

    stim_strength: float = 250.0
    stim_duration_ms: float = 2.0
    stim_npls: int = 1

    mesh_units: str = "um"
    mesh_min_edge_mm: float = 0.3

    par_keyword_mesh: str = "meshname"
    par_keyword_output: str = "simID"
    par_keyword_stim_prefix: str = "stimulus[0]"
    par_keyword_stim_type: str = "stimtype"


VALIDATED = OpenCARPValidatedConfig()


def generate_par_file(
    mesh_path: str,
    output_path: str,
    tend_ms: float,
    apex_um: tuple[float, float, float],
    stim_radius_um: float = 5000.0,
    bcl_ms: float = None,
) -> str:
    """Génère un fichier .par openCARP avec les paramètres validés.
    
    Args:
        mesh_path: chemin du maillage (sans extension)
        output_path: chemin de sortie (simID)
        tend_ms: durée simulation en ms
        apex_um: coordonnées de l'apex en µm (x, y, z)
        stim_radius_um: rayon du stimulus en µm
        bcl_ms: basic cycle length (doit être <= tend_ms)
    """
    c = VALIDATED

    if bcl_ms is None:
        bcl_ms = tend_ms

    if bcl_ms > tend_ms:
        bcl_ms = tend_ms

    if c.stim_duration_ms > tend_ms:
        raise ValueError(
            f"stim_duration ({c.stim_duration_ms}) > tend ({tend_ms})"
        )

    ax, ay, az = apex_um

    return f"""{c.par_keyword_mesh} = {mesh_path}
{c.par_keyword_output} = {output_path}
num_imp_regions = 1
imp_region[0].im = {c.ionic_model}
imp_region[0].num_IDs = 1
imp_region[0].ID[0] = 1
num_gregions = 1
gregion[0].num_IDs = 1
gregion[0].ID[0] = 1
gregion[0].g_il = {c.g_il}
gregion[0].g_it = {c.g_it}
gregion[0].g_el = {c.g_el}
gregion[0].g_et = {c.g_et}
num_stim = 1
{c.par_keyword_stim_prefix}.{c.par_keyword_stim_type} = 0
{c.par_keyword_stim_prefix}.strength = {c.stim_strength}
{c.par_keyword_stim_prefix}.duration = {c.stim_duration_ms}
{c.par_keyword_stim_prefix}.start = 0.0
{c.par_keyword_stim_prefix}.npls = {c.stim_npls}
{c.par_keyword_stim_prefix}.bcl = {bcl_ms}
{c.par_keyword_stim_prefix}.x0 = {ax - stim_radius_um:.0f}
{c.par_keyword_stim_prefix}.xd = {2 * stim_radius_um:.0f}
{c.par_keyword_stim_prefix}.y0 = {ay - stim_radius_um:.0f}
{c.par_keyword_stim_prefix}.yd = {2 * stim_radius_um:.0f}
{c.par_keyword_stim_prefix}.z0 = {az - stim_radius_um:.0f}
{c.par_keyword_stim_prefix}.zd = {2 * stim_radius_um:.0f}
tend = {tend_ms}
dt = {c.dt_ms}
spacedt = {c.spacedt}
timedt = {tend_ms}
"""
