"""Reference Kalman filter for t01, v2 — incorporates Opus's suggested fixes.

Differences from v1:
  - Joseph-form covariance update:  P = (I-KC) P (I-KC)^T + K R K^T
    (guaranteed symmetric and PSD by construction — vs (I-KC) P which can
    lose symmetry under roundoff)
  - Symmetrize P after every update (belt-and-suspenders)
  - Use scipy.linalg.cho_solve since S = C P C^T + R is symmetric +ve definite,
    which is more numerically stable than np.linalg.solve.

The judge predicted these fixes would matter especially in the OOD case
(R is 3.3× larger). This file lets us test that prediction empirically.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def estimate(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    U: np.ndarray,
    Y: np.ndarray,
    x0_mean: np.ndarray,
    x0_cov: np.ndarray,
) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    x0_mean = np.asarray(x0_mean, dtype=np.float64).reshape(-1)
    x0_cov = np.asarray(x0_cov, dtype=np.float64)

    N, T, _ = Y.shape
    n = A.shape[0]
    I_n = np.eye(n)

    x_hat = np.zeros((N, T, n), dtype=np.float64)

    for i in range(N):
        x = x0_mean.copy()
        P = x0_cov.copy()
        for k in range(T):
            # --- Update with y[k] ---
            innov = Y[i, k] - C @ x
            S = C @ P @ C.T + R
            S = 0.5 * (S + S.T)                     # symmetrize before factoring
            cho = cho_factor(S, lower=True)
            # K = P C^T S^-1  =>  K.T = S^-1 (C P)  =>  solve via cho
            K = cho_solve(cho, C @ P).T
            x = x + K @ innov

            # Joseph form: P = (I - KC) P (I - KC)^T + K R K^T
            IKC = I_n - K @ C
            P = IKC @ P @ IKC.T + K @ R @ K.T
            P = 0.5 * (P + P.T)                     # belt-and-suspenders symmetrize

            x_hat[i, k] = x

            # --- Predict to k+1 ---
            x = A @ x + B @ U[i, k]
            P = A @ P @ A.T + Q
            P = 0.5 * (P + P.T)

    return x_hat
