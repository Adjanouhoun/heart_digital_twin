"""
Solveur mecanique cardiaque — FEniCSx 0.7+

Formulation MIXTE quasi-incompressible (Taylor-Hood P2-P1), p DIRECT (Pa) :
  Deplacement u : Lagrange P2 vectoriel
  Pression     p : Lagrange P1 scalaire (PAS de substitution/adimensionnement)
  Passive : Holzapfel-Ogden orthotrope complet sur les invariants
            isochore I1_bar et structuraux I4f, I4s, I8fs.
  Volume  : Pi_vol = p*(J-1) - p^2/(2*kappa)   (p porte l'incompressibilite)
  Active  : contrainte active le long des fibres  P_a = T_a F (f0 x f0)
  BC      : encastrement basal historique, ou appui normal avec points
            anti-modes-rigides pour les diagnostics du Sprint 2.

Solveur : PETSc SNES 'newtonls' + line search 'bt' + garde-fou domaine
          (SNESSetFunctionDomainError, verif J avant assemblage complet),
          KSP direct LU/MUMPS, continuation ADAPTATIVE ultra-fine avec
          restauration immediate sur rejet + CHECKPOINT DISQUE (reprise
          automatique apres interruption, valide sur DoE / runs longs).

Historique complet du fix (P15), sessions 2026-07-06 -> 2026-07-09 :
  v1-v5 : voir CDT_JOURNAL_TECHNIQUE.md (isochore -> kappa -> mixte P2-P1
     -> bug p_hat -> retour p direct).
  v6 : p direct + garde-fou domaine + continuation adaptative -> palier 1
     en 3 iterations, PAS COMPLET (aucun backtracking). Committe 07-08.
  v7 (2026-07-09 nuit) : VALIDATION COMPLETE lam=0 -> lam=1.0 (135kPa),
     deux fois de suite, sur maillage reduit (patient001_coarse5, 5686
     tets) :
       - Run 1, fibres tangentielles simplifiees : 35 paliers, 0 rejet,
         2585.9s, min_J final=0.567, deplacements -17.6/+9.8mm.
       - Run 2, fibres LDRB reelles (app/fibers/ldrb.py) : 35 paliers,
         0 rejet, 3682.5s, min_J final=0.503.
     Goulot de vitesse identifie : assemblage FFCx (boucles C par
     cellule), PAS MUMPS/KSP (1 iteration, rapide) ni BLAS (teste 1 vs 8
     threads OpenBLAS, aucun effet). Scale avec le nb de tetraedres ->
     d'ou l'usage d'un maillage reduit pour la validation rapide.
     Anomalie observee (epi_radial_disp plus negatif que endo, persistante
     avec fibres tangentielles ET LDRB) tracee a une cause GEOMETRIQUE :
     seulement ~3-4 elements traversent radialement la paroi sur ce
     maillage reduit (span radial median par element = 31.9% de
     l'epaisseur pariétale totale) -> resolution insuffisante pour un
     gradient transmural fiable, INDEPENDANT de la formulation ou des
     fibres. Le maillage GROSSIER est valide pour dev/validation de
     stabilite, mais PAS pour des metriques dependant du gradient
     transmural (a reevaluer selon besoins DoE : maillage fin requis si
     ces metriques sont necessaires en aval).
     Ajout du CHECKPOINT DISQUE dans cette version (valide en pratique
     lors des runs de cette nuit) pour proteger tout run long futur.
"""
import time
import os
import traceback
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
    base_bc_mode: str = "fully_fixed"
    # Interface réservée au futur couplage. Toute valeur non nulle est refusée
    # tant que la porte qualité du maillage n'est pas franchie.
    p_endo_kPa: float = 0.0
    duration_ms: float = 500.0
    dt_ms: float = 1.0
    # --- Parametres de continuation (P15 v6/v7, valides en pratique) ---
    dlam_init: float = 1.0e-4
    dlam_min: float = 1.0e-6
    dlam_max: float = 0.05
    j_min_accept: float = 0.1
    # J=det(F) est cubique pour un déplacement P2. Le garde-fou l'évalue sur
    # la grille tétraédrique complète d'ordre 3 (20 points, sommets inclus),
    # au lieu d'un unique barycentre DG0.
    j_lattice_order: int = 3
    max_continuation_steps: int = 500
    easy_iteration_threshold: int = 4
    # --- Checkpoint disque (P15 v7) : reprise apres interruption. None =
    #     desactive (comportement identique aux versions precedentes). ---
    checkpoint_dir: Optional[str] = None


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
    resumed_from_checkpoint: bool = False


class FenicsxSolver:

    @staticmethod
    def _split_microstructure(microstructure, n_nodes):
        """Valide et separe un champ LDRB nodal ``[fibre, sheet]``.

        La mecanique orthotrope requiert six composantes par noeud. Refuser
        explicitement les anciens champs fibre-seule evite de fabriquer une
        direction sheet et de produire un resultat mecaniquement trompeur.
        """
        field = np.asarray(microstructure, dtype=np.float64)
        if field.shape != (n_nodes, 6):
            raise ValueError(
                "La mecanique Holzapfel-Ogden orthotrope exige un champ "
                f"LDRB (n_nodes, 6) [fibre+sheet]; recu {field.shape}."
            )
        if not np.isfinite(field).all():
            raise ValueError("Le champ LDRB contient des valeurs non finies.")

        fibers = field[:, :3].copy()
        sheets = field[:, 3:].copy()
        fiber_norms = np.linalg.norm(fibers, axis=1)
        sheet_norms = np.linalg.norm(sheets, axis=1)
        if np.any(fiber_norms <= 1.0e-8) or np.any(sheet_norms <= 1.0e-8):
            raise ValueError("Le champ LDRB contient un vecteur fibre/sheet nul.")
        fibers /= fiber_norms[:, None]
        sheets /= sheet_norms[:, None]
        max_abs_dot = float(np.max(np.abs(np.sum(fibers * sheets, axis=1))))
        if max_abs_dot > 1.0e-3:
            raise ValueError(
                "La base LDRB fibre/sheet n'est pas orthogonale "
                f"(max |f.s|={max_abs_dot:.6g})."
            )
        return fibers, sheets

    @staticmethod
    def _project_microstructure_to_elements(fibers, sheets, elements):
        """Projette des axes nodaux vers DG0 sans dépendre de leur signe."""
        elements = np.asarray(elements, dtype=np.int64)
        fiber_per_element = np.empty((len(elements), 3), dtype=np.float64)
        sheet_per_element = np.empty((len(elements), 3), dtype=np.float64)

        def principal_axis(vectors):
            tensor = np.einsum("ni,nj->ij", vectors, vectors) / len(vectors)
            values, axes = np.linalg.eigh(tensor)
            axis = axes[:, int(np.argmax(values))]
            if values.max() <= 1e-10 or not np.isfinite(axis).all():
                raise ValueError("Tenseur structural élémentaire dégénéré.")
            pivot = int(np.argmax(np.abs(axis)))
            return axis if axis[pivot] >= 0.0 else -axis

        for index, tet in enumerate(elements):
            fiber = principal_axis(fibers[tet])
            sheet_candidate = principal_axis(sheets[tet])
            sheet = sheet_candidate - np.dot(sheet_candidate, fiber) * fiber
            norm = np.linalg.norm(sheet)
            if norm <= 1e-8:
                raise ValueError(
                    f"Axes fibre/feuillet colinéaires dans l'élément {index}."
                )
            fiber_per_element[index] = fiber
            sheet_per_element[index] = sheet / norm
        return fiber_per_element, sheet_per_element

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

        if params.p_endo_kPa != 0.0:
            raise RuntimeError(
                "Pression endocardique bloquée : la porte qualité du maillage "
                "(min qualité > 0.1 et 0 tétraèdre < 0.05) n'est pas franchie."
            )

        if T_act_kPa is None:
            T_act_kPa = params.T_max_kPa

        if self._fenicsx_available:
            fibers, sheets = self._split_microstructure(fibers, len(nodes))
            result = self._run_fenicsx(params, nodes, elements, fibers,
                                        sheets, T_act_kPa, twin_id, job_id)
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
                    resumed=result.resumed_from_checkpoint,
                    duration_s=round(result.duration_seconds, 1))
        return result

    def _checkpoint_path(self, params, twin_id, job_id):
        if not params.checkpoint_dir:
            return None
        os.makedirs(params.checkpoint_dir, exist_ok=True)
        return os.path.join(params.checkpoint_dir, f"{twin_id}_{job_id}_mech.npz")

    def _run_fenicsx(self, params, nodes, elements, fibers, sheets,
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

        _tdim = msh.topology.dim

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

        # --- Microstructure LDRB : fibre + sheet, DG0 par element ---
        DG = dolfinx.fem.functionspace(msh, ("DG", 0, (3,)))
        f0_func = dolfinx.fem.Function(DG)
        s0_func = dolfinx.fem.Function(DG)
        fiber_per_elem, sheet_per_elem = self._project_microstructure_to_elements(
            fibers, sheets, elements
        )
        f0_func.x.array[:] = fiber_per_elem.flatten()
        s0_func.x.array[:] = sheet_per_elem.flatten()
        f0 = ufl.as_vector([f0_func[0], f0_func[1], f0_func[2]])
        s0 = ufl.as_vector([s0_func[0], s0_func[1], s0_func[2]])

        # --- Parametres orthotropes Holzapfel-Ogden ---
        a_f_Pa = dolfinx.fem.Constant(msh, PETSc.ScalarType(params.a_f_kPa * 1000.0))
        b_f_const = dolfinx.fem.Constant(msh, PETSc.ScalarType(params.b_f))
        a_s_Pa = dolfinx.fem.Constant(msh, PETSc.ScalarType(params.a_s_kPa * 1000.0))
        b_s_const = dolfinx.fem.Constant(msh, PETSc.ScalarType(params.b_s))
        a_fs_Pa = dolfinx.fem.Constant(msh, PETSc.ScalarType(params.a_fs_kPa * 1000.0))
        b_fs_const = dolfinx.fem.Constant(msh, PETSc.ScalarType(params.b_fs))

        # --- Passif ISOCHORE (Constant : evite la recompilation FFCx entre
        #     runs du DoE -- seule .value change, le .so compile est reutilise
        #     pour les 500 simulations au lieu d'etre recompile a chaque fois) ---
        a_Pa = dolfinx.fem.Constant(msh, PETSc.ScalarType(params.a_kPa * 1000.0))
        b_const = dolfinx.fem.Constant(msh, PETSc.ScalarType(params.b))
        J_reg = ufl.max_value(J, 1.0e-3)
        I1_bar = J_reg ** (-2.0 / 3.0) * ufl.tr(C)
        W_iso = (a_Pa / (2.0 * b_const)) * (ufl.exp(b_const * (I1_bar - 3.0)) - 1.0)

        # --- Terme volumetrique en p DIRECT (kappa aussi en Constant) ---
        kappa = dolfinx.fem.Constant(msh, PETSc.ScalarType(params.kappa_vol))
        Pi_vol = p * (J - 1.0) - 0.5 * p * p / kappa
        I4f = ufl.dot(f0, C * f0)
        I4s = ufl.dot(s0, C * s0)
        I8fs = ufl.dot(f0, C * s0)
        I4f_pos = ufl.max_value(I4f - 1.0, 0.0)  # partie positive : traction seule
        I4s_pos = ufl.max_value(I4s - 1.0, 0.0)
        W_f = (a_f_Pa / (2.0 * b_f_const)) * (ufl.exp(b_f_const * I4f_pos**2) - 1.0)
        W_s = (a_s_Pa / (2.0 * b_s_const)) * (ufl.exp(b_s_const * I4s_pos**2) - 1.0)
        W_fs = (a_fs_Pa / (2.0 * b_fs_const)) * (ufl.exp(b_fs_const * I8fs**2) - 1.0)
        Pi = (W_iso + W_f + W_s + W_fs + Pi_vol) * ufl.dx

        T_act = dolfinx.fem.Constant(msh, PETSc.ScalarType(0.0))
        F_passive = ufl.derivative(Pi, w, w_test)
        F_active = T_act * ufl.inner(
            F_def * ufl.outer(f0, f0), ufl.grad(v)
        ) * ufl.dx
        F_form = F_passive + F_active
        dF = ufl.derivative(F_form, w)

        # --- BC basales : mode historique par defaut, plus un mode
        #     diagnostique qui ne bloque que la normale et les modes rigides. ---
        z_max = nodes[:, 2].max()
        z_threshold = z_max - params.base_bc_depth_mm
        if params.base_bc_mode == "fully_fixed":
            W0, _ = W.sub(0).collapse()
            base_dofs = dolfinx.fem.locate_dofs_geometrical(
                (W.sub(0), W0), lambda x: x[2] > z_threshold
            )
            u_bc = dolfinx.fem.Function(W0)
            u_bc.x.array[:] = 0.0
            bcs = [dolfinx.fem.dirichletbc(u_bc, base_dofs, W.sub(0))]
        elif params.base_bc_mode == "normal_with_rigid_pins":
            base_nodes = nodes[nodes[:, 2] > z_threshold]
            if len(base_nodes) < 2:
                raise ValueError("Au moins deux noeuds basaux sont requis.")
            order = np.lexsort((base_nodes[:, 1], base_nodes[:, 0]))
            anchor_a = base_nodes[order[0]]
            distances = np.linalg.norm(base_nodes[:, :2] - anchor_a[:2], axis=1)
            anchor_b = base_nodes[int(np.argmax(distances))]

            def scalar_bc(component, marker):
                subspace = W.sub(0).sub(component)
                collapsed, _ = subspace.collapse()
                dofs = dolfinx.fem.locate_dofs_geometrical(
                    (subspace, collapsed), marker
                )
                zero = dolfinx.fem.Function(collapsed)
                zero.x.array[:] = 0.0
                return dolfinx.fem.dirichletbc(zero, dofs, subspace)

            def at_point(point):
                return lambda x: (
                    np.isclose(x[0], point[0])
                    & np.isclose(x[1], point[1])
                    & np.isclose(x[2], point[2])
                )

            bcs = [
                scalar_bc(2, lambda x: x[2] > z_threshold),
                scalar_bc(0, at_point(anchor_a)),
                scalar_bc(1, at_point(anchor_a)),
                scalar_bc(1, at_point(anchor_b)),
            ]
        else:
            raise ValueError(f"Mode de BC basale inconnu: {params.base_bc_mode}")

        # --- J multipoint pour le garde-fou domaine ---
        # Pour u P2, grad(u) est P1 et det(F) est cubique. La grille d'ordre 3
        # contient les 20 noeuds du tétraèdre cubique, dont les 4 sommets. Elle
        # détecte notamment les inversions que l'ancien échantillon DG0 au
        # seul barycentre pouvait manquer.
        import basix
        j_points = basix.create_lattice(
            basix.CellType.tetrahedron,
            params.j_lattice_order,
            basix.LatticeType.equispaced,
            True,
        )
        J_expr = dolfinx.fem.Expression(J, j_points)
        cell_map = msh.topology.index_map(_tdim)
        owned_cells = np.arange(cell_map.size_local, dtype=np.int32)

        # --- Wrapper SNES avec garde-fou domaine (evite l'assemblage
        #     complet quand J<=0, cf. SNESSetFunctionDomainError PETSc) ---
        class _SNESProblem:
            def __init__(self, F, w, bcs, Jform, msh, J_expr, cells, j_min):
                self.L = dolfinx.fem.form(F)
                self.a = dolfinx.fem.form(Jform)
                self.bcs = bcs
                self.w = w
                self.msh = msh
                self.J_expr = J_expr
                self.cells = cells
                self.j_min = j_min
                self.n_domain_errors = 0

            def min_jacobian(self):
                try:
                    values = self.J_expr.eval(self.msh, self.cells)
                    finite = bool(np.isfinite(values).all())
                    local_min = (
                        float(values.min()) if finite and values.size else np.inf
                    )
                except Exception:
                    finite, local_min = False, -np.inf
                global_finite = self.msh.comm.allreduce(finite, op=MPI.LAND)
                global_min = self.msh.comm.allreduce(local_min, op=MPI.MIN)
                return bool(global_finite), float(global_min)

            def F(self, snes, x, b_vec):
                x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                              mode=PETSc.ScatterMode.FORWARD)
                x.copy(self.w.vector)
                self.w.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                          mode=PETSc.ScatterMode.FORWARD)

                finite, min_j = self.min_jacobian()

                if (not finite) or (min_j <= self.j_min):
                    self.n_domain_errors += 1
                    if hasattr(snes, "setFunctionDomainError"):
                        snes.setFunctionDomainError()
                    else:
                        # petsc4py 3.20 n'expose pas encore
                        # SNESSetFunctionDomainError. Un résidu NaN force un
                        # motif de divergence négatif ; la continuation
                        # externe restaure alors l'état accepté et réduit le
                        # pas, sans jamais assembler sur J invalide.
                        b_vec.set(PETSc.ScalarType(np.nan))
                    return

                with b_vec.localForm() as b_local:
                    b_local.set(0.0)
                dolfinx.fem.petsc.assemble_vector(b_vec, self.L)
                dolfinx.fem.petsc.apply_lifting(
                    b_vec, [self.a], bcs=[self.bcs], x0=[x], scale=-1.0)
                b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                  mode=PETSc.ScatterMode.REVERSE)
                dolfinx.fem.petsc.set_bc(b_vec, self.bcs, x, -1.0)

            def J(self, snes, x, A, P):
                A.zeroEntries()
                dolfinx.fem.petsc.assemble_matrix(A, self.a, bcs=self.bcs)
                A.assemble()

        pde = _SNESProblem(F_form, w, bcs, dF, msh, J_expr, owned_cells,
                            params.j_min_accept)
        b_vec = dolfinx.fem.petsc.create_vector(pde.L)
        A_mat = dolfinx.fem.petsc.create_matrix(pde.a)

        snes = PETSc.SNES().create(msh.comm)
        snes.setOptionsPrefix("mech_")
        def _snes_function(snes_obj, x_vec, residual_vec):
            try:
                return pde.F(snes_obj, x_vec, residual_vec)
            except Exception:
                logger.error("mechanics.fenicsx.callback_exception",
                             callback="residual", traceback=traceback.format_exc())
                raise

        def _snes_jacobian(snes_obj, x_vec, jacobian, preconditioner):
            try:
                return pde.J(snes_obj, x_vec, jacobian, preconditioner)
            except Exception:
                logger.error("mechanics.fenicsx.callback_exception",
                             callback="jacobian", traceback=traceback.format_exc())
                raise

        snes.setFunction(_snes_function, b_vec)
        snes.setJacobian(_snes_jacobian, A_mat)
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

        # --- CHECKPOINT : reprise si un fichier existe deja ---
        ckpt_path = self._checkpoint_path(params, twin_id, job_id)
        T_target_Pa = float(T_act_kPa * 1000.0)
        lam = 0.0
        dlam = params.dlam_init
        dlam_min = params.dlam_min
        dlam_max = params.dlam_max
        j_min_accept = params.j_min_accept
        max_steps = params.max_continuation_steps
        n_steps_prev = 0
        resumed = False

        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = np.load(ckpt_path)
            w.x.array[:] = ckpt["w_array"]
            w.x.scatter_forward()
            lam = float(ckpt["lam"])
            dlam = float(ckpt["dlam"])
            n_steps_prev = int(ckpt["n_steps"])
            resumed = True
            logger.info("mechanics.fenicsx.checkpoint_resumed",
                        lam=round(lam, 6), n_steps_prev=n_steps_prev)

        w_accepted = w.x.array.copy()
        n_its_total = 0
        n_steps = n_steps_prev
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
                j_finite, min_j = pde.min_jacobian()
                finite = finite and j_finite
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
                consecutive_easy = (
                    consecutive_easy + 1
                    if its <= params.easy_iteration_threshold else 0
                )
                if consecutive_easy >= 2:
                    dlam = min(dlam * 1.5, dlam_max)

                if ckpt_path:
                    tmp_path = ckpt_path + ".tmp.npz"
                    np.savez(tmp_path, w_array=w_accepted, lam=lam, dlam=dlam,
                             n_steps=n_steps, min_J=min_J_global)
                    os.replace(tmp_path, ckpt_path)
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

        # Nettoyage du checkpoint une fois convergence complete atteinte
        # (evite de reprendre un run deja termine par erreur)
        if converged and ckpt_path and os.path.exists(ckpt_path):
            os.remove(ckpt_path)

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
            solver_version=f"fenicsx-{dolfinx.__version__}-HO-orthotropic-mixedP2P1-pdirect-ckpt",
            n_iterations=n_its,
            converged=converged,
            min_jacobian=float(min_J_global),
            load_fraction=float(lam),
            domain_errors_total=pde.n_domain_errors,
            resumed_from_checkpoint=resumed,
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
