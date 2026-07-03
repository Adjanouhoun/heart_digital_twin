import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FiberResult:
    fiber_vectors: np.ndarray
    sheet_vectors: np.ndarray
    normal_vectors: np.ndarray
    lon_bytes: bytes
    duration_seconds: float
    algorithm: str = "LDRB-Bayer2012"


class LDRBFiberGenerator:

    ALPHA_ENDO =  60.0
    ALPHA_EPI  = -60.0
    BETA_ENDO  =  0.0
    BETA_EPI   =  25.0

    def generate(self, nodes, elements, element_tags, apex_node_idx=None):
        t0 = time.time()
        N = len(nodes)
        logger.info("ldrb.start", num_nodes=N)

        if apex_node_idx is None:
            apex_node_idx = int(np.argmin(nodes[:, 2]))

        phi_ab  = self._solve_apex_base(nodes, elements, apex_node_idx)
        phi_epi = self._solve_endo_epi(nodes, elements, element_tags)
        grad_ab  = self._compute_gradient(nodes, elements, phi_ab)
        grad_epi = self._compute_gradient(nodes, elements, phi_epi)
        fiber_vecs, sheet_vecs, normal_vecs = self._apply_fiber_rules(nodes, phi_epi, phi_ab, grad_ab, grad_epi)
        lon_bytes = self._export_lon(fiber_vecs, sheet_vecs)

        duration = time.time() - t0
        logger.info("ldrb.complete", duration_s=round(duration, 2))
        return FiberResult(fiber_vectors=fiber_vecs, sheet_vectors=sheet_vecs,
                           normal_vectors=normal_vecs, lon_bytes=lon_bytes,
                           duration_seconds=round(duration, 3))

    def _solve_apex_base(self, nodes, elements, apex_idx):
        z = nodes[:, 2]
        z_min, z_max = z.min(), z.max()
        if z_max - z_min < 1e-10:
            return np.zeros(len(nodes))
        phi = (z - z_min) / (z_max - z_min)
        phi = np.abs(phi - phi[apex_idx])
        return (phi / (phi.max() + 1e-10)).astype(np.float64)

    def _solve_endo_epi(self, nodes, elements, element_tags):
        center = nodes.mean(axis=0)
        dist = np.linalg.norm(nodes - center, axis=1)
        d_min, d_max = dist.min(), dist.max()
        if d_max - d_min < 1e-10:
            return np.zeros(len(nodes))
        return ((dist - d_min) / (d_max - d_min)).astype(np.float64)

    def _compute_gradient(self, nodes, elements, phi):
        N = len(nodes)
        grad = np.zeros((N, 3))
        weights = np.zeros(N)
        for tet in elements:
            v = nodes[tet]
            p = phi[tet]
            B = np.array([v[1]-v[0], v[2]-v[0], v[3]-v[0]]).T
            try:
                B_inv = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                continue
            dp = np.array([p[1]-p[0], p[2]-p[0], p[3]-p[0]])
            g = B_inv.T @ dp
            vol = abs(np.linalg.det(B)) / 6.0
            for idx in tet:
                grad[idx] += g * vol
                weights[idx] += vol
        mask = weights > 1e-12
        grad[mask] /= weights[mask, np.newaxis]
        norms = np.linalg.norm(grad, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return grad / norms

    def _apply_fiber_rules(self, nodes, phi_epi, phi_ab, grad_ab, grad_epi):
        N = len(nodes)
        fiber_vecs  = np.zeros((N, 3))
        sheet_vecs  = np.zeros((N, 3))
        normal_vecs = np.zeros((N, 3))
        for i in range(N):
            t = phi_epi[i]
            alpha = np.radians(self.ALPHA_ENDO + t * (self.ALPHA_EPI - self.ALPHA_ENDO))
            beta  = np.radians(self.BETA_ENDO  + t * (self.BETA_EPI  - self.BETA_ENDO))
            e_long  = grad_ab[i]
            e_trans = grad_epi[i]
            e_circ = np.cross(e_long, e_trans)
            n = np.linalg.norm(e_circ)
            e_circ = e_circ / n if n > 1e-10 else np.array([1., 0., 0.])
            fiber = np.cos(alpha) * e_circ + np.sin(alpha) * np.cross(e_trans, e_circ)
            nf = np.linalg.norm(fiber)
            if nf > 1e-10: fiber /= nf
            sheet = np.cos(beta) * e_trans + np.sin(beta) * np.cross(fiber, e_trans)
            ns = np.linalg.norm(sheet)
            if ns > 1e-10: sheet /= ns
            fiber_vecs[i]  = fiber
            sheet_vecs[i]  = sheet
            normal_vecs[i] = np.cross(fiber, sheet)
        return fiber_vecs, sheet_vecs, normal_vecs

    def _export_lon(self, fiber_vecs, sheet_vecs):
        lines = ["2"]
        for f, s in zip(fiber_vecs, sheet_vecs):
            lines.append(f"{f[0]:.6f} {f[1]:.6f} {f[2]:.6f} {s[0]:.6f} {s[1]:.6f} {s[2]:.6f}")
        return "\n".join(lines).encode()
