from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class EstimatorResult:
    support_hat: np.ndarray
    F_hat: np.ndarray
    screening_sets: List[np.ndarray]
    coefficients: np.ndarray
    condition_numbers: np.ndarray
    singular_rows: np.ndarray
    gram_min_eigenvalues: np.ndarray
    gram_max_eigenvalues: np.ndarray
    gram_ranks: np.ndarray
    gram_solve_success: np.ndarray
    gram_solver_used: List[str]
    ridge_used: np.ndarray
    ridge_values: np.ndarray
    tau: float
    Y: np.ndarray
    Z: np.ndarray


def build_bin_indicators(
    events_by_node: list[np.ndarray],
    n: int,
    h: float,
) -> np.ndarray:
    d = len(events_by_node)
    Y = np.zeros((n, d), dtype=float)
    upper = n * h
    for i, events in enumerate(events_by_node):
        ev = np.asarray(events, dtype=float)
        if ev.size == 0:
            continue
        ev = ev[(ev > 0.0) & (ev <= upper)]
        if ev.size == 0:
            continue
        idx = np.ceil(ev / h).astype(int) - 1
        idx = idx[(idx >= 0) & (idx < n)]
        Y[np.unique(idx), i] = 1.0
    return Y


def _top_m_indices(scores: np.ndarray, m: int) -> np.ndarray:
    d = scores.size
    m_eff = min(max(int(m), 0), d)
    if m_eff == 0:
        return np.empty(0, dtype=int)
    idx = np.argpartition(-scores, kth=m_eff - 1)[:m_eff]
    return idx[np.argsort(-scores[idx])]


def estimate_from_sufficient_statistics(
    n: int,
    sum_Z: np.ndarray,
    sum_Y: np.ndarray,
    sum_YZ: np.ndarray,
    sum_ZZ: np.ndarray,
    h: float,
    R: float,
    alpha: float,
    w_minus: float,
    k: int,
    m: int,
    ridge: Optional[float] = None,
    cond_threshold: float = 1e12,
    allow_ridge_fallback: bool = False,
) -> EstimatorResult:
    del k, R
    if n <= 0:
        raise ValueError("n must be positive")
    if h <= 0:
        raise ValueError("h must be positive")
    if alpha <= 0 or w_minus <= 0:
        raise ValueError("alpha and w_minus must be positive")

    sum_Z = np.asarray(sum_Z, dtype=float)
    sum_Y = np.asarray(sum_Y, dtype=float)
    sum_YZ = np.asarray(sum_YZ, dtype=float)
    sum_ZZ = np.asarray(sum_ZZ, dtype=float)
    d = sum_Z.size
    if sum_Y.shape != (d,) or sum_YZ.shape != (d, d) or sum_ZZ.shape != (d, d):
        raise ValueError("sufficient statistic shapes are inconsistent")

    mean_Z = sum_Z / float(n)
    mean_Y = sum_Y / float(n)
    F_hat = (sum_YZ / float(n)) - np.outer(mean_Y, mean_Z)
    Sigma = (sum_ZZ / float(n)) - np.outer(mean_Z, mean_Z)
    tau = float(alpha) * float(w_minus) * float(h) / 2.0

    support_hat = np.zeros((d, d), dtype=np.int8)
    coefficients = np.full((d, d), np.nan, dtype=float)
    condition_numbers = np.full(d, np.nan, dtype=float)
    singular_rows = np.zeros(d, dtype=bool)
    gram_min_eigenvalues = np.full(d, np.nan, dtype=float)
    gram_max_eigenvalues = np.full(d, np.nan, dtype=float)
    gram_ranks = np.zeros(d, dtype=int)
    gram_solve_success = np.zeros(d, dtype=bool)
    gram_solver_used = ["not_attempted"] * d
    ridge_used = np.zeros(d, dtype=bool)
    ridge_values = np.zeros(d, dtype=float)
    screening_sets: list[np.ndarray] = []

    use_ridge = ridge is not None and ridge > 0.0
    if use_ridge and not allow_ridge_fallback:
        raise ValueError("ridge requires allow_ridge_fallback=True")

    for i in range(d):
        C = _top_m_indices(F_hat[i], m=m)
        screening_sets.append(C)
        if C.size == 0:
            singular_rows[i] = True
            continue

        Sigma_C = Sigma[np.ix_(C, C)]
        g_C = F_hat[i, C]
        try:
            eigvals = np.linalg.eigvalsh(0.5 * (Sigma_C + Sigma_C.T))
            gram_min_eigenvalues[i] = float(np.min(eigvals))
            gram_max_eigenvalues[i] = float(np.max(eigvals))
            condition_numbers[i] = (
                float(gram_max_eigenvalues[i] / gram_min_eigenvalues[i])
                if gram_min_eigenvalues[i] > 0.0
                else np.inf
            )
            gram_ranks[i] = int(np.linalg.matrix_rank(Sigma_C))
        except np.linalg.LinAlgError:
            condition_numbers[i] = np.inf
            gram_ranks[i] = 0

        Sigma_solve = Sigma_C
        if use_ridge:
            Sigma_solve = Sigma_C + float(ridge) * np.eye(C.size)
            ridge_used[i] = True
            ridge_values[i] = float(ridge)
            gram_solver_used[i] = "ridge_solve"
        elif (
            not np.isfinite(condition_numbers[i])
            or condition_numbers[i] > cond_threshold
        ):
            singular_rows[i] = True
            continue
        else:
            gram_solver_used[i] = "solve"

        try:
            coef_C = np.linalg.solve(Sigma_solve, g_C)
        except np.linalg.LinAlgError:
            singular_rows[i] = True
            gram_solver_used[i] = "failed"
            continue

        gram_solve_success[i] = True
        coefficients[i, C] = coef_C
        support_hat[i, C[coef_C >= tau]] = 1

    empty = np.empty((0, d), dtype=float)
    return EstimatorResult(
        support_hat=support_hat,
        F_hat=F_hat,
        screening_sets=screening_sets,
        coefficients=coefficients,
        condition_numbers=condition_numbers,
        singular_rows=singular_rows,
        gram_min_eigenvalues=gram_min_eigenvalues,
        gram_max_eigenvalues=gram_max_eigenvalues,
        gram_ranks=gram_ranks,
        gram_solve_success=gram_solve_success,
        gram_solver_used=gram_solver_used,
        ridge_used=ridge_used,
        ridge_values=ridge_values,
        tau=tau,
        Y=empty,
        Z=empty,
    )


def bounded_block_screen_and_regress(
    events_by_node: list[np.ndarray],
    sampled_X: np.ndarray,
    grid_times: np.ndarray,
    h: float,
    R: float,
    alpha: float,
    w_minus: float,
    k: int,
    m: int,
    ridge: Optional[float] = None,
    cond_threshold: float = 1e12,
    allow_ridge_fallback: bool = False,
) -> EstimatorResult:
    del k  # k is part of the theorem parameters; m controls the local block size.
    if h <= 0:
        raise ValueError("h must be positive")
    if R <= 0:
        raise ValueError("R must be positive")
    if alpha <= 0 or w_minus <= 0:
        raise ValueError("alpha and w_minus must be positive")

    X = np.asarray(sampled_X, dtype=float)
    if X.ndim != 2:
        raise ValueError("sampled_X must have shape (n, d)")
    n, d = X.shape
    if n == 0:
        raise ValueError("sampled_X must contain at least one grid sample")
    if len(events_by_node) != d:
        raise ValueError("events_by_node length must match sampled_X dimension")
    if np.asarray(grid_times).shape[0] != n:
        raise ValueError("grid_times length must match sampled_X")

    Z = np.minimum(X, R)
    Y = build_bin_indicators(events_by_node, n=n, h=h)
    Z_tilde = Z - Z.mean(axis=0, keepdims=True)
    Y_tilde = Y - Y.mean(axis=0, keepdims=True)

    F_hat = (Y_tilde.T @ Z_tilde) / float(n)
    tau = float(alpha) * float(w_minus) * float(h) / 2.0

    support_hat = np.zeros((d, d), dtype=np.int8)
    coefficients = np.full((d, d), np.nan, dtype=float)
    condition_numbers = np.full(d, np.nan, dtype=float)
    singular_rows = np.zeros(d, dtype=bool)
    gram_min_eigenvalues = np.full(d, np.nan, dtype=float)
    gram_max_eigenvalues = np.full(d, np.nan, dtype=float)
    gram_ranks = np.zeros(d, dtype=int)
    gram_solve_success = np.zeros(d, dtype=bool)
    gram_solver_used = ["not_attempted"] * d
    ridge_used = np.zeros(d, dtype=bool)
    ridge_values = np.zeros(d, dtype=float)
    screening_sets: list[np.ndarray] = []

    use_ridge = ridge is not None and ridge > 0.0
    if use_ridge and not allow_ridge_fallback:
        raise ValueError("ridge requires allow_ridge_fallback=True")

    for i in range(d):
        C = _top_m_indices(F_hat[i], m=m)
        screening_sets.append(C)
        if C.size == 0:
            singular_rows[i] = True
            continue

        Z_C = Z_tilde[:, C]
        Sigma_C = (Z_C.T @ Z_C) / float(n)
        g_C = (Z_C.T @ Y_tilde[:, i]) / float(n)

        try:
            eigvals = np.linalg.eigvalsh(0.5 * (Sigma_C + Sigma_C.T))
            gram_min_eigenvalues[i] = float(np.min(eigvals))
            gram_max_eigenvalues[i] = float(np.max(eigvals))
            condition_numbers[i] = (
                float(gram_max_eigenvalues[i] / gram_min_eigenvalues[i])
                if gram_min_eigenvalues[i] > 0.0
                else np.inf
            )
            gram_ranks[i] = int(np.linalg.matrix_rank(Sigma_C))
        except np.linalg.LinAlgError:
            gram_min_eigenvalues[i] = np.nan
            gram_max_eigenvalues[i] = np.nan
            condition_numbers[i] = np.inf
            gram_ranks[i] = 0

        Sigma_solve = Sigma_C
        if use_ridge:
            Sigma_solve = Sigma_C + float(ridge) * np.eye(C.size)
            ridge_used[i] = True
            ridge_values[i] = float(ridge)
            gram_solver_used[i] = "ridge_solve"
        elif (
            not np.isfinite(condition_numbers[i])
            or condition_numbers[i] > cond_threshold
        ):
            singular_rows[i] = True
            continue
        else:
            gram_solver_used[i] = "solve"

        try:
            coef_C = np.linalg.solve(Sigma_solve, g_C)
        except np.linalg.LinAlgError:
            singular_rows[i] = True
            gram_solver_used[i] = "failed"
            continue

        gram_solve_success[i] = True
        coefficients[i, C] = coef_C
        support_hat[i, C[coef_C >= tau]] = 1

    return EstimatorResult(
        support_hat=support_hat,
        F_hat=F_hat,
        screening_sets=screening_sets,
        coefficients=coefficients,
        condition_numbers=condition_numbers,
        singular_rows=singular_rows,
        gram_min_eigenvalues=gram_min_eigenvalues,
        gram_max_eigenvalues=gram_max_eigenvalues,
        gram_ranks=gram_ranks,
        gram_solve_success=gram_solve_success,
        gram_solver_used=gram_solver_used,
        ridge_used=ridge_used,
        ridge_values=ridge_values,
        tau=tau,
        Y=Y,
        Z=Z,
    )
