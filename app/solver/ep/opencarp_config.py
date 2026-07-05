"""
openCARP Configuration — Parametres valides pour openCARP git a86f7c4.

Source syntaxe : /usr/local/lib/opencarp/share/examples/02_EP_tissue/
Corrections vs ancien generateur (documentees dans CDT_JOURNAL_TECHNIQUE.md) :
  - dt en MICROSECONDES (openCARP lit dt en us, pas en ms) — ancien dt=0.02
    etait interprete 0.02us => 1000x trop de pas => 37min/50ms. Corrige : dt=20us.
  - bidomain = 0 + parab_solve = 1 : monodomaine explicite (Crank-Nicolson).
  - imp_region[0].ID = 1 / gregion[0].ID = 1 (PAS .ID[0], syntaxe obsolete).
  - Stimulus : nouvelle syntaxe stim[0].* (name / crct.type / pulse.strength /
    ptcl.start|duration|npls|bcl) au lieu de stimulus[0].stimtype|strength...
  - Localisation stimulus : stim[0].elec.geom_type=0 (sphere) + p0 (centre um) +
    radius (um), au lieu de l'ancienne boite x0/xd/y0/yd/z0/zd.

Conductivites (monodomaine, tenTusscherPanfilov) : inchangees, validees.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenCARPValidatedConfig:
    """Parametres valides — openCARP a86f7c4."""

    ionic_model: str = "tenTusscherPanfilov"

    # Conductivites — validees
    g_il: float = 0.3544
    g_it: float = 0.024
    g_el: float = 1.2700
    g_et: float = 0.0862

    # dt en MICROSECONDES (openCARP lit dt en us). 20us = 0.02ms.
    dt_us: int = 20
    spacedt: float = 1.0        # sortie tous les 1 ms
    timedt: float = 100.0       # rafraichissement console tous les 100 ms

    # Mode de resolution : monodomaine
    bidomain: int = 0
    parab_solve: int = 1        # Crank-Nicolson

    # Stimulus
    stim_strength: float = 250.0    # uA/cm2
    stim_duration_ms: float = 2.0
    stim_npls: int = 1
    stim_geom_type: int = 1         # 1 = sphere (valeurs openCARP: 1=sphere,2=block,3=cylinder)

    mesh_units: str = "um"
    mesh_min_edge_mm: float = 0.3


VALIDATED = OpenCARPValidatedConfig()


def generate_par_file(
    mesh_path: str,
    output_path: str,
    tend_ms: float,
    apex_um: tuple,
    stim_radius_um: float = 5000.0,
    bcl_ms: float = None,
) -> str:
    """Genere un fichier .par openCARP (syntaxe a86f7c4) avec parametres valides.

    Args:
        mesh_path: chemin du maillage (sans extension)
        output_path: chemin de sortie (simID)
        tend_ms: duree simulation en ms
        apex_um: coordonnees de l'apex en um (x, y, z) — centre du stimulus
        stim_radius_um: rayon de la sphere de stimulation en um
        bcl_ms: basic cycle length (doit etre <= tend_ms)
    """
    c = VALIDATED

    if bcl_ms is None:
        bcl_ms = tend_ms
    if bcl_ms > tend_ms:
        bcl_ms = tend_ms
    if c.stim_duration_ms > tend_ms:
        raise ValueError(f"stim_duration ({c.stim_duration_ms}) > tend ({tend_ms})")

    # timedt (rafraichissement console) doit etre <= tend
    timedt = min(c.timedt, tend_ms)

    ax, ay, az = apex_um

    return f"""meshname = {mesh_path}
simID = {output_path}

# --- Resolution numerique (dt en MICROSECONDES) ---
dt = {c.dt_us}
tend = {tend_ms}
spacedt = {c.spacedt}
timedt = {timedt}
bidomain = {c.bidomain}
parab_solve = {c.parab_solve}

# --- Modele ionique ---
num_imp_regions = 1
imp_region[0].im = {c.ionic_model}
imp_region[0].num_IDs = 1
imp_region[0].ID = 1

# --- Conductivites (monodomaine) ---
num_gregions = 1
gregion[0].num_IDs = 1
gregion[0].ID = 1
gregion[0].g_il = {c.g_il}
gregion[0].g_it = {c.g_it}
gregion[0].g_el = {c.g_el}
gregion[0].g_et = {c.g_et}

# --- Stimulus (sphere a l'apex, coordonnees en um) ---
num_stim = 1
stim[0].name = "apex"
stim[0].crct.type = 0
stim[0].pulse.strength = {c.stim_strength}
stim[0].ptcl.start = 0.0
stim[0].ptcl.duration = {c.stim_duration_ms}
stim[0].ptcl.npls = {c.stim_npls}
stim[0].ptcl.bcl = {bcl_ms}
stim[0].elec.geom_type = {c.stim_geom_type}
stim[0].elec.p0[0] = {ax:.0f}
stim[0].elec.p0[1] = {ay:.0f}
stim[0].elec.p0[2] = {az:.0f}
stim[0].elec.radius = {stim_radius_um:.0f}
"""
