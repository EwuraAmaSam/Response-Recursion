from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

@dataclass
class RiskSurfaceModel:
    pca_dim: int = 10
    noise: float = 1e-2
    length_scale: float = 1.0

    pca: Optional[PCA] = None
    gp: Optional[GaussianProcessRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        p = min(self.pca_dim, X.shape[1])
        self.pca = PCA(n_components=p, random_state=0)
        Xp = self.pca.fit_transform(X)

        kernel = RBF(length_scale=self.length_scale) + WhiteKernel(noise_level=self.noise)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
        self.gp.fit(Xp, y)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.pca is None or self.gp is None:
            raise RuntimeError("RiskSurfaceModel not fitted.")
        Xp = self.pca.transform(np.asarray(X, dtype=np.float32))
        mean, std = self.gp.predict(Xp, return_std=True)
        return mean.astype(np.float32), std.astype(np.float32)

    def summary(self) -> Dict[str, Any]:
        return {
            "model": "GaussianProcessRegressor",
            "pca_dim": self.pca_dim,
            "kernel": str(self.gp.kernel_) if self.gp is not None else None,
        }