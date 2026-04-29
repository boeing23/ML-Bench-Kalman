"""Reference Kalman filter for t01.

Standard predict/update recursion with Joseph-form-style covariance update.
Initializes from (x0_mean, x0_cov) as the prior for x[0|-1], updates with
y[0] to get x[0|0], stores it, predicts to x[1|0], etc.

Returns the FILTERED estimate x[k|k] for k = 0..T-1.
"""
from __future__ import annotations

import numpy as np


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
    p = C.shape[0]
    I_n = np.eye(n)

    x_hat = np.zeros((N, T, n), dtype=np.float64)

    for i in range(N):
        x = x0_mean.copy()
        P = x0_cov.copy()
        for k in range(T):
            # --- Update with y[k] ---
            innov = Y[i, k] - C @ x
            S = C @ P @ C.T + R
            K = np.linalg.solve(S.T, (P @ C.T).T).T   # K = P C^T S^{-1}
            x = x + K @ innov
            P = (I_n - K @ C) @ P
            x_hat[i, k] = x
            # --- Predict to k+1 ---
            x = A @ x + B @ U[i, k]
            P = A @ P @ A.T + Q

    return x_hat
