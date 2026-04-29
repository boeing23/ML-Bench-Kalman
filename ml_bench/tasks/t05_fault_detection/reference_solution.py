"""Reference solution for Task 5 (fault detection).

Hand-crafted features + gradient boosting. Trains at import time on
/workspace/data/train.npz, then exposes `predict(X)`.

Features per (window, channel):
  mean, std, range, slope (linear regression coef vs time), max abs deviation
  from mean, dominant FFT magnitude (excluding DC).
Across 6 channels → 36 features per window.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


_T = 64
_M = 6


def _features(X: np.ndarray) -> np.ndarray:
    # X: (N, T, M)
    N = X.shape[0]
    feats = np.empty((N, _M * 6), dtype=np.float32)
    t = np.arange(_T, dtype=np.float32)
    t_centered = t - t.mean()
    t_norm = (t_centered ** 2).sum()
    for ch in range(_M):
        x = X[:, :, ch]                                  # (N, T)
        mu = x.mean(axis=1)
        sd = x.std(axis=1)
        rng = x.max(axis=1) - x.min(axis=1)
        slope = ((x - mu[:, None]) * t_centered).sum(axis=1) / t_norm
        max_dev = np.abs(x - mu[:, None]).max(axis=1)
        spec = np.abs(np.fft.rfft(x - mu[:, None], axis=1))
        if spec.shape[1] > 1:
            dom = spec[:, 1:].max(axis=1) / (_T / 2)
        else:
            dom = np.zeros(N, dtype=np.float32)
        feats[:, ch * 6:ch * 6 + 6] = np.stack([mu, sd, rng, slope, max_dev, dom], axis=1)
    return feats


def _train() -> tuple[StandardScaler, GradientBoostingClassifier]:
    import os
    ws = Path(os.environ.get("MLBENCH_WORKSPACE", "/workspace"))
    data_path = ws / "data" / "train.npz"
    if not data_path.exists():
        # Fallback for direct invocation outside the sandbox
        data_path = Path(__file__).resolve().parent / "data" / "train.npz"
    blob = np.load(data_path)
    X_tr, y_tr = blob["X"], blob["y"]
    feats_tr = _features(X_tr)
    scaler = StandardScaler().fit(feats_tr)
    feats_tr = scaler.transform(feats_tr)
    clf = GradientBoostingClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.1, random_state=0
    )
    clf.fit(feats_tr, y_tr)
    return scaler, clf


_SCALER, _CLF = _train()


def predict(X: np.ndarray) -> np.ndarray:
    feats = _features(X.astype(np.float32))
    feats = _SCALER.transform(feats)
    proba = _CLF.predict_proba(feats)[:, 1]
    return proba.astype(np.float64)
