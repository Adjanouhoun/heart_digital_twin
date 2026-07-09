"""
GP Surrogate Predictor — Remplacement temps reel des solveurs lents
Charge les GP entraines sur le DoE et predit en <10ms
"""
import os
import json
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import structlog

logger = structlog.get_logger(__name__)

DOE_PATH = os.environ.get("CDT_DOE_PATH", "/app/data/doe/doe_500_results.json")

PARAM_NAMES = ["sigma_l", "sigma_t", "T_max_kPa", "heart_rate_bpm",
               "a_kPa", "b", "R_p", "C_a"]

OUTPUT_NAMES = ["cv_ms", "apd90_ms", "ef_pct", "edv_mL", "esv_mL",
                "sv_mL", "p_systolic_mmHg", "p_diastolic_mmHg",
                "p_mean_mmHg", "dp_dt_max", "co_L_min"]


class GPPredictor:
    def __init__(self, doe_path=None):
        self.doe_path = doe_path or DOE_PATH
        self.models = {}
        self.X_mean = None
        self.X_std = None
        self.Y_stats = {}
        self._fitted = False
        self._fit()

    def _fit(self):
        if not os.path.exists(self.doe_path):
            logger.warning("gp_predictor.no_doe", path=self.doe_path)
            return

        with open(self.doe_path) as f:
            doe = json.load(f)

        X = np.array([[d[k] for k in PARAM_NAMES] for d in doe])
        self.X_mean = X.mean(0)
        self.X_std = X.std(0) + 1e-8
        X_norm = (X - self.X_mean) / self.X_std

        for name in OUTPUT_NAMES:
            y = np.array([d[name] for d in doe])
            y_mean = y.mean()
            y_std = y.std() + 1e-8
            y_norm = (y - y_mean) / y_std

            kernel = ConstantKernel() * RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel()
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=42)
            gp.fit(X_norm, y_norm)

            self.models[name] = gp
            self.Y_stats[name] = {"mean": y_mean, "std": y_std}

        self._fitted = True
        logger.info("gp_predictor.fitted", n_models=len(self.models), n_samples=len(doe))

    def predict(self, params):
        if not self._fitted:
            return {name: 0.0 for name in OUTPUT_NAMES}

        x = np.array([params.get(k, 0) for k in PARAM_NAMES]).reshape(1, -1)
        x_norm = (x - self.X_mean) / self.X_std

        results = {}
        for name in OUTPUT_NAMES:
            pred_norm, pred_std = self.models[name].predict(x_norm, return_std=True)
            pred = pred_norm[0] * self.Y_stats[name]["std"] + self.Y_stats[name]["mean"]
            uncertainty = pred_std[0] * self.Y_stats[name]["std"]
            results[name] = round(float(pred), 4)
            results[f"{name}_std"] = round(float(uncertainty), 4)

        return results

    def predict_with_uq(self, params, n_samples=100):
        if not self._fitted:
            return {}

        x = np.array([params.get(k, 0) for k in PARAM_NAMES]).reshape(1, -1)
        x_norm = (x - self.X_mean) / self.X_std

        results = {}
        for name in OUTPUT_NAMES:
            pred_norm, pred_std = self.models[name].predict(x_norm, return_std=True)
            pred = pred_norm[0] * self.Y_stats[name]["std"] + self.Y_stats[name]["mean"]
            std = pred_std[0] * self.Y_stats[name]["std"]
            samples = np.random.normal(pred, std, n_samples)
            results[name] = {
                "mean": round(float(pred), 4),
                "std": round(float(std), 4),
                "ci_5": round(float(np.percentile(samples, 5)), 4),
                "ci_95": round(float(np.percentile(samples, 95)), 4),
            }

        return results
