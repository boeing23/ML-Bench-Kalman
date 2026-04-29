"""Reference solution for Task 2 (system identification).

Multivariate ARX(na, nb) model fit by ordinary least squares:

    y[k] ≈ Σ_{i=1..na} A_i y[k-i]  +  Σ_{j=0..nb-1} B_j u[k-j]

The coefficient block is solved jointly across the two output channels via
`np.linalg.lstsq`. Forecast is rolled out open-loop from the warmup window:
predicted ŷ[k] is fed back as the regressor for ŷ[k+1].

Why this works: any controllable+observable LTI system of state dimension
n_state admits an exact ARX representation of order ≥ n_state. Here
n_state = 3, so ARX(4, 4) is comfortably sufficient. With 200 trajectories
× ~76 usable steps = 15.2k samples and a 16-dim regressor, OLS recovers
the coefficients tightly — well below the noise floor.
"""
from __future__ import annotations

import numpy as np


_NA = 4
_NB = 4


def _build_design_matrix(
    U: np.ndarray, Y: np.ndarray, na: int, nb: int
) -> tuple[np.ndarray, np.ndarray]:
    N, T, P = U.shape
    _, _, Q = Y.shape
    L = max(na, nb)
    n_per = T - L
    rows = N * n_per
    feat_dim = na * Q + nb * P
    Phi = np.empty((rows, feat_dim), dtype=np.float64)
    Tgt = np.empty((rows, Q), dtype=np.float64)
    r = 0
    for i in range(N):
        for k in range(L, T):
            past_y = Y[i, k - na:k][::-1].reshape(-1)
            past_u = U[i, k - nb + 1:k + 1][::-1].reshape(-1)
            Phi[r, :na * Q] = past_y
            Phi[r, na * Q:] = past_u
            Tgt[r] = Y[i, k]
            r += 1
    return Phi, Tgt


def _fit(U_train: np.ndarray, Y_train: np.ndarray) -> np.ndarray:
    Phi, Tgt = _build_design_matrix(
        U_train.astype(np.float64), Y_train.astype(np.float64), _NA, _NB
    )
    Theta, *_ = np.linalg.lstsq(Phi, Tgt, rcond=None)
    return Theta


def fit_and_predict(
    U_train: np.ndarray,
    Y_train: np.ndarray,
    U_test: np.ndarray,
    Y_test_warmup: np.ndarray,
) -> np.ndarray:
    Theta = _fit(U_train, Y_train)

    U_test = U_test.astype(np.float64)
    Y_warm = Y_test_warmup.astype(np.float64)
    N, T, P = U_test.shape
    _, T_warm, Q = Y_warm.shape
    horizon = T - T_warm
    Y_pred = np.empty((N, horizon, Q), dtype=np.float64)

    na, nb = _NA, _NB
    feat_dim = na * Q + nb * P
    phi = np.empty(feat_dim, dtype=np.float64)

    for i in range(N):
        y_hist = np.empty((T, Q), dtype=np.float64)
        y_hist[:T_warm] = Y_warm[i]
        for k in range(T_warm, T):
            past_y = y_hist[k - na:k][::-1].reshape(-1)
            past_u = U_test[i, k - nb + 1:k + 1][::-1].reshape(-1)
            phi[:na * Q] = past_y
            phi[na * Q:] = past_u
            y_next = phi @ Theta
            y_hist[k] = y_next
            Y_pred[i, k - T_warm] = y_next

    return Y_pred
