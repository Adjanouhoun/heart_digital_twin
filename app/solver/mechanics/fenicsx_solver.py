"""
Solveur mecanique cardiaque — FEniCSx 0.7+

Formulation MIXTE quasi-incompressible (deux champs, Taylor-Hood P2-P1) :
  Deplacement u : Lagrange P2 vectoriel
  Pression     p : Lagrange P1 scalaire
  Passive : Holzapfel-Ogden isotrope sur invariant ISOCHORE I1_bar
            W_iso = a/(2b) * (exp(b*(I1_bar - 3)) - 1),  I1_bar = J^(-2/3) tr(C)
  Volume  : Pi_vol = p*(J-1) - p^2/(2 kappa)   (p porte l'incompressibilite)
  Active  : contrainte active le long des fibres  P_a = T_a F (f0 x f0)
  BC      : base fixee (Dirichlet u=0 sur le sous-espace deplacement)

Solveur : PETSc SNES 'newtonls' + line search 'bt', KSP direct LU/MUMPS
          (systeme point-selle indefini -> solveur direct obligatoire),
          continuation ADAPTATIVE de la tension active + garde-fou min(J).

Historique du fix (P15) :
  v1 exp(b*(tr(C)-3)) NON isochore + kappa faible + relaxation fixe
     -> volume x1e11, fausse convergence a haute charge.
  v2 isochore + kappa=1e6 + SNES/bt + GAMG -> min_J=1.0 mais GAMG diverge
     (reason=-3, penalite mal conditionnee).
  v3 KSP direct LU -> min_J repasse positif a faible charge (maillage sain),
     mais line search 'bt' echoue (reason=-6).
  v3b line search 'l2' -> min_J=-36 (pire) : ni bt ni l2 ne trouvent de
     direction admissible => LOCKING VOLUMETRIQUE du P1 confirme.
  v4 (cette version) : formulation MIXTE P2-P1 (Taylor-Hood). La pression
     est une inconnue a part entiere -> plus de locking. C'est la
     formulation standard cardiaque (fenicsx-pulse, Ambit).
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
                    load=round(result.load_fraction, 3),
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
        u, p = ufl.split(w)
        w_test = ufl.TestFunction(W)
        v, q = ufl.split(w_test)

        I = ufl.Identity(3)
        F_def = I + ufl.grad(u)
        C = F_def.T * F_def
        J = ufl.det(F_def)

        # --- Passif ISOCHORE ---
        a_Pa = params.a_kPa * 1000.0
        b = params.b
        I1_bar = J ** (-2.0 / 3.0) * ufl.tr(C)
        W_iso = (a_Pa / (2.0 * b)) * (ufl.exp(b * (I1_bar - 3.0)) - 1.0)

        # --- Terme volumetrique MIXTE : p porte l'incompressibilite ---
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

        # Residu = premiere variation du potentiel (u ET p) + contrib active (u)
        F_passive = ufl.derivative(Pi, w, w_test)
        F_active = T_act * ufl.inner(
            F_def * ufl.outer(f0, f0), ufl.grad(v)
        ) * ufl.dx
        F_form = F_passive + F_active
        dF = ufl.derivative(F_form, w)

        # --- BC : base fixee sur le SOUS-ESPACE deplacement W.sub(0) ---
        z_max = nodes[:, 2].max()
        z_threshold = z_max - params.base_bc_depth_mm
        W0, _ = W.sub(0).collapse()
        base_dofs = dolfinx.fem.locate_dofs_geometrical(
            (W.sub(0), W0), lambda x: x[2] > z_threshold
        )
        u_bc = dolfinx.fem.Function(W0)
        u_bc.x.array[:] = 0.0
        bc = dolfinx.fem.dirichletbc(u_bc, base_dofs, W.sub(0))

        # --- Wrapper SNES (assemblage monolithique de l'espace mixte) ---
        class _SNESProblem:
            def __init__(self, F, w, bc, Jform):
                self.L = dolfinx.fem.form(F)
                self.a = dolfinx.fem.form(Jform)
                self.bc = bc
                self.w = w

            def F(self, snes, x, b_vec):
                x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                              mode=PETSc.ScatterMode.FORWARD)
                x.copy(self.w.vector)
                self.w.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                          mode=PETSc.ScatterMode.FORWARD)
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

        pde = _SNESProblem(F_form, w, bc, dF)
        b_vec = dolfinx.fem.petsc.create_vector(pde.L)
        A_mat = dolfinx.fem.petsc.create_matrix(pde.a)

        snes = PETSc.SNES().create(msh.comm)
        snes.setOptionsPrefix("mech_")
        snes.setFunction(pde.F, b_vec)
        snes.setJacobian(pde.J, A_mat)
        snes.setType("newtonls")
        # Tolerances un peu relachees : le residu mixte melange deux echelles
        # (deplacement ~mm, pression ~1e5 Pa).
        snes.setTolerances(atol=1e-6, rtol=1e-6, stol=0.0, max_it=50)

        opts = PETSc.Options()
        prefix = snes.getOptionsPrefix()
        opts[f"{prefix}snes_linesearch_type"] = "bt"
        opts[f"{prefix}snes_linesearch_order"] = 3

        # KSP direct LU/MUMPS : le systeme mixte est un POINT-SELLE indefini,
        # GAMG (multigrille) ne s'y applique pas sans fieldsplit. MUMPS gere
        # nativement les systemes indefinis.
        ksp = snes.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        try:
            pc.setFactorSolverType("mumps")
        except Exception:
            logger.warning("mechanics.fenicsx.mumps_unavailable_fallback_petsc_lu")

        snes.setFromOptions()

        # --- Monitoring min(J) : J depend de u = split(w)[0] ---
        DG0s = dolfinx.fem.functionspace(msh, ("DG", 0))
        J_func = dolfinx.fem.Function(DG0s)
        J_expr = dolfinx.fem.Expression(J, DG0s.element.interpolation_points())

        # --- Continuation ADAPTATIVE ---
        T_target_Pa = float(T_act_kPa * 1000.0)
        lam = 0.0
        dlam = 1.0 / 30.0
        DLAM_MIN = 1.0e-3
        DLAM_MAX = 0.1
        J_MIN = 0.1
        MAX_STEPS = 400

        w_accepted = w.x.array.copy()
        n_its_total = 0
        n_steps = 0
        min_J_global = 1.0

        while lam < 1.0 - 1e-9 and n_steps < MAX_STEPS:
            n_steps += 1
            target = min(lam + dlam, 1.0)
            T_act.value = T_target_Pa * target

            try:
                snes.solve(None, w.vector)
            except Exception as e:
                logger.error("mechanics.fenicsx.snes_exception",
                             target=round(target, 4), error=str(e))
                w.x.array[:] = w_accepted
                w.x.scatter_forward()
                dlam *= 0.5
                if dlam < DLAM_MIN:
                    break
                continue

            w.x.scatter_forward()
            its = snes.getIterationNumber()
            reason = snes.getConvergedReason()
            finite = bool(np.isfinite(w.x.array).all())

            if finite:
                J_func.interpolate(J_expr)
                arr = J_func.x.array
                local_min = float(arr.min()) if arr.size else np.inf
                min_J = msh.comm.allreduce(local_min, op=MPI.MIN)
            else:
                min_J = float("-inf")

            accept = (reason > 0) and finite and (min_J > J_MIN)

            if accept:
                lam = target
                w_accepted = w.x.array.copy()
                n_its_total += its
                min_J_global = min_J
                logger.info("mechanics.fenicsx.load_step",
                            lam=round(lam, 4),
                            T_act_kPa=round(T_act.value / 1000.0, 2),
                            its=its, min_J=round(min_J, 4),
                            dlam=round(dlam, 4))
                if its < 5:
                    dlam = min(dlam * 1.5, DLAM_MAX)
            else:
                w.x.array[:] = w_accepted
                w.x.scatter_forward()
                dlam *= 0.5
                logger.warning("mechanics.fenicsx.load_step_reject",
                               target=round(target, 4), reason=int(reason),
                               min_J=(round(min_J, 4) if np.isfinite(min_J) else None),
                               finite=finite, new_dlam=round(dlam, 6))
                if dlam < DLAM_MIN:
                    logger.error("mechanics.fenicsx.continuation_stalled",
                                 lam=round(lam, 4))
                    break

        converged = lam >= 1.0 - 1e-9
        n_its = n_its_total

        # --- Post-traitement : extraire u (P2), re-interpoler sur P1 aux
        #     sommets pour que le remap coordonnee (base sur "nodes") marche.
        #     P2 a des DOF aux aretes que "nodes" ne connait pas. ---
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
            active_tension_kPa=T_act_kPa,
            endo_radial_disp_mm=endo_disp,
            epi_radial_disp_mm=epi_disp,
            solver_version=f"fenicsx-{dolfinx.__version__}-mixedP2P1",
            n_iterations=n_its,
            converged=converged,
            min_jacobian=float(min_J_global),
            load_fraction=float(lam),
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
