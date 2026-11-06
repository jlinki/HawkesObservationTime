from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class HawkesSample:
    events_by_node: List[np.ndarray]
    grid_times: np.ndarray
    sampled_X: np.ndarray
    sampled_lambda: np.ndarray
    sampled_excitation: np.ndarray


def check_stability(k: int, alpha: float, w_plus: float, beta: float) -> float:
    """Check the paper's weak-coupling sufficient stability condition."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    gamma = float(k) * float(alpha) * float(w_plus) / float(beta)
    if gamma >= 1.0:
        raise ValueError(
            f"Stability check failed: gamma=k*alpha*w_plus/beta={gamma:.6g} >= 1"
        )
    return gamma


def stationary_mean_intensity(
    beta: float,
    mu: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Return E[lambda] = (I - Theta / beta)^(-1) mu."""
    mu_vec = np.asarray(mu, dtype=float)
    theta_mat = np.asarray(theta, dtype=float)
    if theta_mat.shape != (mu_vec.size, mu_vec.size):
        raise ValueError("theta shape must be (d, d), matching mu")
    return np.linalg.solve(np.eye(mu_vec.size) - theta_mat / beta, mu_vec)


def _validate_inputs(
    d: int,
    beta: float,
    mu: np.ndarray,
    theta: np.ndarray,
    T: float,
    burn_in: float,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    if d <= 0:
        raise ValueError("d must be positive")
    if beta <= 0:
        raise ValueError("beta must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be nonnegative")
    if h <= 0:
        raise ValueError("h must be positive")
    mu_vec = np.asarray(mu, dtype=float)
    theta_mat = np.asarray(theta, dtype=float)
    if mu_vec.shape != (d,):
        raise ValueError(f"mu must have shape ({d},)")
    if theta_mat.shape != (d, d):
        raise ValueError(f"theta must have shape ({d}, {d})")
    if np.any(mu_vec <= 0):
        raise ValueError("mu entries must be positive")
    if np.any(theta_mat < 0):
        raise ValueError("theta entries must be nonnegative")
    return mu_vec, theta_mat


def simulate_hawkes(
    d: int,
    beta: float,
    mu: np.ndarray,
    theta: np.ndarray,
    T: float,
    burn_in: float,
    h: float,
    seed: int,
    max_events: Optional[int] = None,
) -> HawkesSample:
    """
    Simulate a nonnegative exponential Hawkes process by exact Ogata thinning.

    The simulator evolves the source-filter basis
        U_j(t) = integral exp(-beta * (t-s)) dN_j(s)
    and computes the target excitation as X(t) = theta @ U(t). This is
    equivalent to dX_i = -beta X_i dt + sum_j theta[i, j] dN_j(t), and it is
    the representation needed by the screen-and-regress estimator because the
    local regression covariates are the source filters U_j.

    For compatibility with the estimator API, sampled_X contains sampled U.
    sampled_excitation contains theta @ U. Events before time zero are used as
    burn-in and discarded, while the state at zero is retained.
    """
    mu_vec, theta_mat = _validate_inputs(d, beta, mu, theta, T, burn_in, h)
    rng = np.random.default_rng(seed)

    n = int(np.floor(T / h))
    grid_times = np.arange(n, dtype=float) * h
    sampled_X = np.zeros((n, d), dtype=float)
    sampled_excitation = np.zeros((n, d), dtype=float)
    sampled_lambda = np.zeros((n, d), dtype=float)
    events: list[list[float]] = [[] for _ in range(d)]

    t = -float(burn_in)
    u = np.zeros(d, dtype=float)
    grid_idx = 0
    event_count = 0
    event_limit = max_events if max_events is not None else max(100_000, int(1000 * d * (T + burn_in)))

    while t < T:
        lam = mu_vec + theta_mat @ u
        total_upper = float(lam.sum())
        if total_upper <= 0.0 or not np.isfinite(total_upper):
            raise RuntimeError("invalid total intensity during simulation")

        wait = float(rng.exponential(1.0 / total_upper))
        t_prop = t + wait

        while grid_idx < n and grid_times[grid_idx] <= min(t_prop, T):
            dt_grid = grid_times[grid_idx] - t
            if dt_grid < -1e-12:
                raise RuntimeError("internal grid bookkeeping error")
            u_grid = u * np.exp(-beta * max(dt_grid, 0.0))
            excitation_grid = theta_mat @ u_grid
            sampled_X[grid_idx] = u_grid
            sampled_excitation[grid_idx] = excitation_grid
            sampled_lambda[grid_idx] = mu_vec + excitation_grid
            grid_idx += 1

        if t_prop > T:
            break

        u *= np.exp(-beta * wait)
        t = t_prop
        lam_prop = mu_vec + theta_mat @ u
        total_prop = float(lam_prop.sum())

        if rng.random() * total_upper <= total_prop:
            probs = lam_prop / total_prop
            node = int(rng.choice(d, p=probs))
            if t >= 0.0:
                events[node].append(t)
            u[node] += 1.0
            event_count += 1
            if event_count > event_limit:
                raise RuntimeError(
                    "simulation exceeded max_events; check stability and parameters"
                )

    while grid_idx < n:
        dt_grid = grid_times[grid_idx] - t
        u_grid = u * np.exp(-beta * max(dt_grid, 0.0))
        excitation_grid = theta_mat @ u_grid
        sampled_X[grid_idx] = u_grid
        sampled_excitation[grid_idx] = excitation_grid
        sampled_lambda[grid_idx] = mu_vec + excitation_grid
        grid_idx += 1

    return HawkesSample(
        events_by_node=[np.asarray(e, dtype=float) for e in events],
        grid_times=grid_times,
        sampled_X=sampled_X,
        sampled_lambda=sampled_lambda,
        sampled_excitation=sampled_excitation,
    )
