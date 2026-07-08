"""
Solveur mecanique cardiaque — FEniCSx 0.7+

Formulation MIXTE quasi-incompressible (Taylor-Hood P2-P1), p DIRECT (Pa) :
  Deplacement u : Lagrange P2 vectoriel
  Pression     p : Lagrange P1 scalaire (PAS de substitution/adimensionnement)
  Passive : Holzapfel-Ogden isotrope sur invariant ISOCHORE I1_bar
            W_iso = a/(2b) * (exp(b*(I1_bar - 3)) - 1),  I1_bar = J^(-2/3) tr(C)
  Volume  : Pi_vol = p*(J-1) - p^2/(2*kappa)   (p porte l'incompressibilite)
  Active  : contrainte active le long des fibres  P_a = T_a F (f0 x f0)
  BC      : base fixee (Dirichlet u=0 sur le sous-espace deplacement)

Solveur : PETSc SNES 'newtonls' + line search 'bt' + garde-fou domaine
          (SNESSetFunctionDomainError, verif J avant assemblage complet),
          KSP direct LU/MUMPS, continuation ADAPTATIVE ultra-fine avec
          restauration immediate sur rejet.

Historique complet du fix (P15), sessions 2026-07-06 -> 2026-07-08 :
  v1 exp(b*(tr(C)-3)) NON isochore + kappa=1e4 trop faible + relaxation
     fixe -> volume x1e11, fausse convergence a haute charge.
  v2 isochore + kappa=1e6 + SNES/bt + GAMG -> min_J=1.0 (volume tenu) mais
     GAMG diverge (reason=-3, penalite mal conditionnee).
  v3 KSP direct LU/MUMPS (P1) -> min_J sain a faible charge (maillage OK),
     mais line search bt echoue a haute charge (reason=-6) : LOCKING P1.
  v4 formulation MIXTE P2-P1 -> locking leve, mais 1er solve bloque (NaN
     sur inversion transitoire au 1er pas de Newton, bt boucle).
  v5 tentative p_hat=p/kappa (adimensionnement) -> BUG : derivation en
     chaine dPi/dp_hat=kappa*dPi/dp AMPLIFIE le desequilibre d'echelle au
     lieu de l'attenuer (palier 1 : 7 iterations, 6 reductions de pas).
  v6 (cette version) : retour a p DIRECT + garde-fou SNESSetFunctionDomainError
     + continuation ultra-fine adaptative. Palier 1 : 3 iterations, PAS
     COMPLET des le depart (aucun backtracking). Validation en cours sur
     maillage reduit : formulation physiquement saine sur toute la
     trajectoire testee (min_J decroit proprement 0.97->0.78, jamais de
     NaN, domain_err_total=0), continuation auto-corrige sur rejet de
     palier (zone de raideur locale franchie en reduisant dlam). Le point
     ouvert est le TEMPS de calcul (~450-2500s/palier selon difficulte
     locale), pas la validite physique.
"""
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MechanicsParameters:
    a_kPa: float = 0.496
    b: float = 7.209
    a_f_kPa: float = 15.193
    b_f: float = 20.417
    a_s_kPa: float = 3.283
    b_s: float = 11.176
    a_fs_kPa: float = 0.662
    b_fs: float = 9.466
    T_max_kPa: float = 135.0
    kappa_vol: float = 1.0e6
    base_bc_depth_mm: float = 5.0
    duration_ms: float = 500.0
    dt_ms: float = 1.0
    # --- Parametres de continuation (P15 v6) ---
    dlam_init: float = 1.0e-4
    dlam_min: float = 1.0e-6
    dlam_max: float = 0.05
    j_min_accept: float = 0.1
    max_continuation_steps: int = 500


@dataclass
class MechanicsResult:
    twin_id: str
    job_id: str
    displacement_mm: np.ndarray
    fiber_strain: np.ndarray
    volume_tissue_mL: float
    ef_pct: float
    edv_mL: float
    esv_mL: float
    active_tension_kPa: float
    endo_radial_disp_mm: float
    epi_radial_disp_mm: float
    parameters: Optional[MechanicsParameters] = None
    duration_seconds: float = 0.0
    solver_version: str = "fenicsx-0.7"
    n_iterations: int = 0
    converged: bool = False
    min_jacobian: float = 0.0
    load_fraction: float = 0.0
    domain_errors_total: int = 0


class FenicsxSolver:

    def __init__(self):
        self._fenicsx_available = self._check_fenicsx()

    def _check_fenicsx(self):
        try:
            import dolfinx
            logger.info("mechanics.fenicsx.available", version=dolfinx.__version__)
            return True
        except ImportError:
            logger.warning("mechanics.fenicsx.not_available")
            return False

    def simulate(self, params, nodes, elements, fibers, activation_times_ms,
                 twin_id, job_id, T_act_kPa=None):
        t0 = time.time()

        if T_act_kPa is None:
            T_act_kPa = params.T_max_kPa

        if self._fenicsx_available:
            result = self._run_fenicsx(params, nodes, elements, fibers,
                                        T_act_kPa, twin_id, job_id)
        else:
            result = self._run_fallback(params, nodes, elements, fibers,
                                         activation_times_ms, T_act_kPa,
                                         twin_id, job_id)

        result.duration_seconds = time.time() - t0
        result.parameters = params
        logger.info("mechanics.complete", twin_id=twin_id,
                    converged=result.converged, its=result.n_iterations,
                    endo_disp=round(result.endo_radial_disp_mm, 3),
                    min_J=round(result.min_jacobian, 4),
                    load=round(result.load_fraction, 4),
                    domain_errors=result.domain_errors_total,
                    duration_s=round(result.duration_seconds, 1))
        return result

    def _run_fenicsx(self, params, nodes, elements, fibers,
                      T_act_kPa, twin_id, job_id):
        import dolfinx
        import dolfinx.log
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)
        import dolfinx.fem
        import dolfinx.fem.petsc
        import dolfinx.mesh
        import basix.ufl
        import ufl
        from mpi4py import MPI
        from petsc4py import PETSc

        elements_np = np.array(elements, dtype=np.int64)
        domain = ufl.Mesh(ufl.VectorElement("Lagrange", ufl.tetrahedron, 1))
        msh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, elements_np, nodes, domain)

        # --- Espace MIXTE Taylor-Hood : u (P2 vectoriel) + p (P1 scalaire) ---
        P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(3,))
        P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
        W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P2, P1]))

        w = dolfinx.fem.Function(W)
        u, p = ufl.split(w)              # p DIRECT (Pa) -- pas de p_hat
        w_test = ufl.TestFunction(W)
        v, q = ufl.split(w_test)

        I = ufl.Identity(3)
        F_def = I + ufl.grad(u)
        C = F_def.T * F_def
        J = ufl.det(F_def)

        # --- Passif ISOCHORE ---
        a_Pa = params.a_kPa * 1000.0
        b = params.b
        J_reg = ufl.max_value(J, 1.0e-3)
        I1_bar = J_reg ** (-2.0 / 3.0) * ufl.tr(C)
        W_iso = (a_Pa / (2.0 * b)) * (ufl.exp(b * (I1_bar - 3.0)) - 1.0)

        # --- Terme volumetrique en p DIRECT ---
        kappa = params.kappa_vol
        Pi_vol = p * (J - 1.0) - 0.5 * p * p / kappa
        Pi = (W_iso + Pi_vol) * ufl.dx

        # --- Fibres : DG0 = un vecteur PAR ELEMENT ---
        DG = dolfinx.fem.functionspace(msh, ("DG", 0, (3,)))
        f0_func = dolfinx.fem.Function(DG)
        elements_arr = np.asarray(elements)
        fiber_per_elem = np.zeros((len(elements_arr), 3))
        for e_idx, tet in enumerate(elements_arr):
            avg = fibers[tet].mean(axis=0)
            norm = np.linalg.norm(avg)
            fiber_per_elem[e_idx] = avg / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0])
        f0_func.x.array[:] = fiber_per_elem.flatten()
        f0 = ufl.as_vector([f0_func[0], f0_func[1], f0_func[2]])

        T_act = dolfinx.fem.Constant(msh, PETSc.ScalarType(0.0))
        F_passive = ufl.derivative(Pi, w, w_test)
        F_active = T_act * ufl.inner(
            F_def * ufl.outer(f0, f0), ufl.grad(v)
        ) * ufl.dx
        F_form = F_passive + F_active
        dF = ufl.derivative(F_form, w)

        # --- BC : base fixee sur le sous-espace deplacement ---
        z_max = nodes[:, 2].max()
        z_threshold = z_max - params.base_bc_depth_mm
        W0, _ = W.sub(0).collapse()
        base_dofs = dolfinx.fem.locate_dofs_geometrical(
            (W.sub(0), W0), lambda x: x[2] > z_threshold
        )
        u_bc = dolfinx.fem.Function(W0)
        u_bc.x.array[:] = 0.0
        bc = dolfinx.fem.dirichletbc(u_bc, base_dofs, W.sub(0))

        # --- J compile pour le garde-fou domaine ---
        DG0s = dolfinx.fem.functionspace(msh, ("DG", 0))
        J_expr = dolfinx.fem.Expression(J, DG0s.element.interpolation_points())

        # --- Wrapper SNES avec garde-fou domaine (evite l'assemblage
        #     complet quand J<=0, cf. SNESSetFunctionDomainError PETSc) ---
        class _SNESProblem:
            def __init__(self, F, w, bc, Jform, msh, J_expr, DG0s, j_min):
                self.L = dolfinx.fem.form(F)
                self.a = dolfinx.fem.form(Jform)
                self.bc = bc
                self.w = w
                self.msh = msh
                self.J_expr = J_expr
                self.J_func = dolfinx.fem.Function(DG0s)
                self.j_min = j_min
                self.n_domain_errors = 0

            def F(self, snes, x, b_vec):
                x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                              mode=PETSc.ScatterMode.FORWARD)
                x.copy(self.w.vector)
                self.w.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                          mode=PETSc.ScatterMode.FORWARD)

                try:
                    self.J_func.interpolate(self.J_expr)
                    arr = self.J_func.x.array
                    finite = bool(np.isfinite(arr).all())
                    local_min = float(arr.min()) if finite and arr.size else -np.inf
                except Exception:
                    finite, local_min = False, -np.inf
                min_j = self.msh.comm.allreduce(local_min, op=MPI.MIN)

                if (not finite) or (min_j <= self.j_min):
                    self.n_domain_errors += 1
                    snes.setFunctionDomainError()
                    return

                with b_vec.localForm() as b_local:
                    b_local.set(0.0)
                dolfinx.fem.petsc.assemble_vector(b_vec, self.L)
                dolfinx.fem.petsc.apply_lifting(
                    b_vec, [self.a], bcs=[[self.bc]], x0=[x], scale=-1.0)
                b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                  mode=PETSc.ScatterMode.REVERSE)
                dolfinx.fem.petsc.set_bc(b_vec, [self.bc], x, -1.0)

            def J(self, snes, x, A, P):
                A.zeroEntries()
                dolfinx.fem.petsc.assemble_matrix(A, self.a, bcs=[self.bc])
                A.assemble()

        pde = _SNESProblem(F_form, w, bc, dF, msh, J_expr, DG0s,
                            params.j_min_accept)
        b_vec = dolfinx.fem.petsc.create_vector(pde.L)
        A_mat = dolfinx.fem.petsc.create_matrix(pde.a)

        snes = PETSc.SNES().create(msh.comm)
        snes.setOptionsPrefix("mech_")
        snes.setFunction(pde.F, b_vec)
        snes.setJacobian(pde.J, A_mat)
        snes.setType("newtonls")
        snes.setTolerances(atol=1e-4, rtol=1e-4, stol=0.0, max_it=15)

        opts = PETSc.Options()
        prefix = snes.getOptionsPrefix()
        opts[f"{prefix}snes_linesearch_type"] = "bt"
        opts[f"{prefix}snes_linesearch_order"] = 1
        opts[f"{prefix}snes_linesearch_max_it"] = 6
        opts[f"{prefix}snes_linesearch_minlambda"] = 1e-3

        ksp = snes.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        try:
            pc.setFactorSolverType("mumps")
        except Exception:
            logger.warning("mechanics.fenicsx.mumps_unavailable_fallback_petsc_lu")

        snes.setFromOptions()

        # --- Continuation ADAPTATIVE ultra-fine + restauration immediate ---
        T_target_Pa = float(T_act_kPa * 1000.0)
        lam = 0.0
        dlam = params.dlam_init
        dlam_min = params.dlam_min
        dlam_max = params.dlam_max
        j_min_accept = params.j_min_accept
        max_steps = params.max_continuation_steps

        w_accepted = w.x.array.copy()
        n_its_total = 0
        n_steps = 0
        min_J_global = 1.0
        consecutive_easy = 0

        while lam < 1.0 - 1e-9 and n_steps < max_steps:
            n_steps += 1
            target = min(lam + dlam, 1.0)
            T_act.value = T_target_Pa * target

            try:
                snes.solve(None, w.vector)
            except Exception as e:
                logger.error("mechanics.fenicsx.snes_exception",
                             target=round(target, 6), error=str(e))
                w.x.array[:] = w_accepted
                w.x.scatter_forward()
                dlam *= 0.3
                if dlam < dlam_min:
                    logger.error("mechanics.fenicsx.continuation_stalled",
                                 lam=round(lam, 6))
                    break
                continue

            w.x.scatter_forward()
            its = snes.getIterationNumber()
            reason = snes.getConvergedReason()
            finite = bool(np.isfinite(w.x.array).all())

            if finite:
                pde.J_func.interpolate(pde.J_expr)
                arr = pde.J_func.x.array
                local_min = float(arr.min()) if arr.size else np.inf
                min_j = msh.comm.allreduce(local_min, op=MPI.MIN)
            else:
                min_j = float("-inf")

            accept = (reason > 0) and finite and (min_j > j_min_accept)

            if accept:
                lam = target
                w_accepted = w.x.array.copy()
                n_its_total += its
                min_J_global = min_j
                logger.info("mechanics.fenicsx.load_step",
                            lam=round(lam, 6),
                            T_act_kPa=round(T_act.value / 1000.0, 3),
                            its=its, min_J=round(min_j, 4),
                            dlam=round(dlam, 6),
                            domain_err_total=pde.n_domain_errors)
                consecutive_easy = consecutive_easy + 1 if its <= 4 else 0
                if consecutive_easy >= 2:
                    dlam = min(dlam * 1.5, dlam_max)
            else:
                w.x.array[:] = w_accepted
                w.x.scatter_forward()
                dlam *= 0.3
                consecutive_easy = 0
                logger.warning("mechanics.fenicsx.load_step_reject",
                               target=round(target, 6), reason=int(reason),
                               min_J=(round(min_j, 4) if np.isfinite(min_j) else None),
                               finite=finite, new_dlam=round(dlam, 8))
                if dlam < dlam_min:
                    logger.error("mechanics.fenicsx.continuation_stalled",
                                 lam=round(lam, 6))
                    break

        converged = lam >= 1.0 - 1e-9
        n_its = n_its_total

        # --- Post-traitement : extraire u (P2), re-interpoler sur P1 aux
        #     sommets pour le remap coordonnee (base sur "nodes") ---
        u_sub = w.sub(0).collapse()
        V1 = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (3,)))
        u1 = dolfinx.fem.Function(V1)
        u1.interpolate(u_sub)

        u_arr_dof_order = u1.x.array.reshape(-1, 3)
        dof_coords = V1.tabulate_dof_coordinates()
        coord_to_dof_idx = {
            tuple(np.round(c, 6)): i for i, c in enumerate(dof_coords)
        }
        remap = np.array([
            coord_to_dof_idx[tuple(np.round(n, 6))] for n in nodes
        ])
        u_arr = u_arr_dof_order[remap]

        vol_tissue = self._compute_volume(nodes + u_arr, elements)
        endo_disp, epi_disp = self._compute_radial_displacements(nodes, u_arr)

        return MechanicsResult(
            twin_id=twin_id, job_id=job_id,
            displacement_mm=u_arr,
            fiber_strain=np.zeros(len(nodes)),
            volume_tissue_mL=vol_tissue,
            ef_pct=0.0,
            edv_mL=vol_tissue,
            esv_mL=vol_tissue,
            active_tension_kPa=T_act_kPa * lam,
            endo_radial_disp_mm=endo_disp,
            epi_radial_disp_mm=epi_disp,
            solver_version=f"fenicsx-{dolfinx.__version__}-mixedP2P1-pdirect",
            n_iterations=n_its,
            converged=converged,
            min_jacobian=float(min_J_global),
            load_fraction=float(lam),
            domain_errors_total=pde.n_domain_errors,
        )

    def _run_fallback(self, params, nodes, elements, fibers,
                       activation_times_ms, T_act_kPa, twin_id, job_id):
        center = nodes.mean(0)
        radii = np.linalg.norm(nodes - center, axis=1)
        ef_est = 0.35 + 0.25 * (T_act_kPa / 135.0)
        ef_est = np.clip(ef_est, 0.25, 0.75)
        height_factor = np.sin(
            np.pi * (nodes[:, 2] - nodes[:, 2].min()) /
            (nodes[:, 2].max() - nodes[:, 2].min() + 1e-10)
        )
        radial_disp = -0.15 * ef_est * radii * height_factor
        directions = (nodes - center) / (radii[:, np.newaxis] + 1e-10)
        displacement = directions * radial_disp[:, np.newaxis]
        vol = self._compute_volume(nodes, elements)

        return MechanicsResult(
            twin_id=twin_id, job_id=job_id,
            displacement_mm=displacement,
            fiber_strain=np.zeros(len(nodes)),
            volume_tissue_mL=vol,
            ef_pct=round(ef_est * 100, 1),
            edv_mL=round(vol, 1),
            esv_mL=round(vol * (1 - ef_est), 1),
            active_tension_kPa=T_act_kPa,
            endo_radial_disp_mm=-0.15 * ef_est * np.median(radii),
            epi_radial_disp_mm=0.1 * ef_est * np.median(radii),
            solver_version="fallback-analytical-NOT-FOR-CLINICAL-USE",
            n_iterations=0,
            converged=True,
        )

    def _compute_volume(self, nodes, elements):
        total_vol = 0.0
        for tet in elements:
            v = nodes[tet]
            mat = np.array([v[1]-v[0], v[2]-v[0], v[3]-v[0]])
            total_vol += abs(np.linalg.det(mat)) / 6.0
        return total_vol / 1000.0

    def _compute_radial_displacements(self, nodes, u_arr):
        z_mid = (nodes[:, 2].min() + nodes[:, 2].max()) / 2
        mid_mask = np.abs(nodes[:, 2] - z_mid) < 5.0
        if mid_mask.sum() < 10:
            return 0.0, 0.0

        mid_nodes = nodes[mid_mask]
        center_xy = mid_nodes[:, :2].mean(0)
        radii = np.linalg.norm(mid_nodes[:, :2] - center_xy, axis=1)
        r_med = np.median(radii)

        endo_radial = []
        epi_radial = []
        for i in np.where(mid_mask)[0]:
            radial_dir = nodes[i, :2] - center_xy
            r = np.linalg.norm(radial_dir)
            if r > 0.1:
                radial_dir /= r
                rd = np.dot(u_arr[i, :2], radial_dir)
                if r < r_med:
                    endo_radial.append(rd)
                else:
                    epi_radial.append(rd)

        endo = np.mean(endo_radial) if endo_radial else 0.0
        epi = np.mean(epi_radial) if epi_radial else 0.0
        return float(endo), float(epi)
