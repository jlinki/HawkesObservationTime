from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from estimator import EstimatorResult, estimate_from_sufficient_statistics
from hawkes_sim import _validate_inputs


@dataclass(frozen=True)
class StreamingHawkesStats:
    n: int
    sum_Z: np.ndarray
    sum_Y: np.ndarray
    sum_YZ: np.ndarray
    sum_ZZ: np.ndarray
    event_counts: np.ndarray
    total_events: int


def simulate_hawkes_sufficient_statistics(
    d: int,
    beta: float,
    mu: np.ndarray,
    theta: np.ndarray,
    T: float,
    burn_in: float,
    h: float,
    R: float,
    seed: int,
    max_events: Optional[int] = None,
) -> StreamingHawkesStats:
    """
    Simulate Hawkes events and accumulate screen-and-regress sufficient stats.

    This avoids materializing sampled_X with shape (floor(T/h), d). It uses the
    same source-filter convention as simulate_hawkes: U_j jumps by one when node
    j fires, and lambda = mu + theta @ U.
    """
    mu_vec, theta_mat = _validate_inputs(d, beta, mu, theta, T, burn_in, h)
    if R <= 0:
        raise ValueError("R must be positive")
    rng = np.random.default_rng(seed)

    n = int(np.floor(T / h))
    if n <= 0:
        raise ValueError("floor(T / h) must be positive")

    sum_Z = np.zeros(d, dtype=float)
    sum_Y = np.zeros(d, dtype=float)
    sum_YZ = np.zeros((d, d), dtype=float)
    sum_ZZ = np.zeros((d, d), dtype=float)
    event_counts = np.zeros(d, dtype=np.int64)

    t = -float(burn_in)
    u = np.zeros(d, dtype=float)
    bin_idx = 0
    current_Y = np.zeros(d, dtype=float)
    current_Z = np.zeros(d, dtype=float)
    current_bin_started = False
    event_count = 0
    event_limit = (
        max_events
        if max_events is not None
        else max(100_000, int(1000 * d * (T + burn_in)))
    )

    def start_bin(idx: int) -> None:
        nonlocal current_Z, current_Y, current_bin_started
        grid_time = idx * h
        dt_grid = grid_time - t
        if dt_grid < -1e-12:
            raise RuntimeError("internal grid bookkeeping error")
        u_grid = u * np.exp(-beta * max(dt_grid, 0.0))
        current_Z = np.minimum(u_grid, R)
        current_Y = np.zeros(d, dtype=float)
        current_bin_started = True

    def close_bin() -> None:
        nonlocal current_bin_started
        sum_Z[:] += current_Z
        sum_Y[:] += current_Y
        sum_YZ[:] += np.outer(current_Y, current_Z)
        sum_ZZ[:] += np.outer(current_Z, current_Z)
        current_bin_started = False

    while t < T:
        lam = mu_vec + theta_mat @ u
        total_upper = float(lam.sum())
        if total_upper <= 0.0 or not np.isfinite(total_upper):
            raise RuntimeError("invalid total intensity during simulation")

        wait = float(rng.exponential(1.0 / total_upper))
        t_prop = t + wait

        while bin_idx < n and bin_idx * h <= min(t_prop, T):
            if not current_bin_started:
                start_bin(bin_idx)
            next_boundary = (bin_idx + 1) * h
            if next_boundary <= min(t_prop, T):
                close_bin()
                bin_idx += 1
            else:
                break

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
                idx = int(np.ceil(t / h) - 1)
                if 0 <= idx < n:
                    while bin_idx < idx:
                        if not current_bin_started:
                            start_bin(bin_idx)
                        close_bin()
                        bin_idx += 1
                    if bin_idx == idx and not current_bin_started:
                        start_bin(bin_idx)
                    if bin_idx == idx:
                        current_Y[node] = 1.0
                event_counts[node] += 1
            u[node] += 1.0
            event_count += 1
            if event_count > event_limit:
                raise RuntimeError(
                    "simulation exceeded max_events; check stability and parameters"
                )

    while bin_idx < n:
        if not current_bin_started:
            start_bin(bin_idx)
        close_bin()
        bin_idx += 1

    return StreamingHawkesStats(
        n=n,
        sum_Z=sum_Z,
        sum_Y=sum_Y,
        sum_YZ=sum_YZ,
        sum_ZZ=sum_ZZ,
        event_counts=event_counts,
        total_events=int(event_count),
    )


def simulate_hawkes_sufficient_statistics_snapshots(
    d: int,
    beta: float,
    mu: np.ndarray,
    theta: np.ndarray,
    T_max: float,
    burn_in: float,
    h: float,
    R: float,
    seed: int,
    n_targets: list[int],
    max_events: Optional[int] = None,
) -> dict[int, StreamingHawkesStats]:
    """
    Simulate once up to T_max and snapshot cumulative sufficient stats.

    Snapshots are keyed by number of closed bins. A snapshot at n contains the
    same statistics that simulate_hawkes_sufficient_statistics would have
    accumulated over bins 0, ..., n-1 for the same sample path.
    """
    mu_vec, theta_mat = _validate_inputs(d, beta, mu, theta, T_max, burn_in, h)
    if R <= 0:
        raise ValueError("R must be positive")
    targets = sorted({int(n) for n in n_targets})
    if not targets or targets[0] <= 0:
        raise ValueError("n_targets must contain positive integers")
    n_max = targets[-1]
    if n_max > int(np.floor(T_max / h)):
        raise ValueError("largest n_target exceeds floor(T_max / h)")

    rng = np.random.default_rng(seed)
    sum_Z = np.zeros(d, dtype=float)
    sum_Y = np.zeros(d, dtype=float)
    sum_YZ = np.zeros((d, d), dtype=float)
    sum_ZZ = np.zeros((d, d), dtype=float)
    event_counts = np.zeros(d, dtype=np.int64)
    snapshots: dict[int, StreamingHawkesStats] = {}

    t = -float(burn_in)
    u = np.zeros(d, dtype=float)
    bin_idx = 0
    current_Y = np.zeros(d, dtype=float)
    current_Z = np.zeros(d, dtype=float)
    current_bin_started = False
    event_count = 0
    event_limit = (
        max_events
        if max_events is not None
        else max(100_000, int(1000 * d * (T_max + burn_in)))
    )
    target_set = set(targets)

    def snapshot_if_needed() -> None:
        if bin_idx in target_set and bin_idx not in snapshots:
            snapshots[bin_idx] = StreamingHawkesStats(
                n=bin_idx,
                sum_Z=sum_Z.copy(),
                sum_Y=sum_Y.copy(),
                sum_YZ=sum_YZ.copy(),
                sum_ZZ=sum_ZZ.copy(),
                event_counts=event_counts.copy(),
                total_events=int(event_counts.sum()),
            )

    def start_bin(idx: int) -> None:
        nonlocal current_Z, current_Y, current_bin_started
        grid_time = idx * h
        dt_grid = grid_time - t
        if dt_grid < -1e-12:
            raise RuntimeError("internal grid bookkeeping error")
        u_grid = u * np.exp(-beta * max(dt_grid, 0.0))
        current_Z = np.minimum(u_grid, R)
        current_Y = np.zeros(d, dtype=float)
        current_bin_started = True

    def close_bin() -> None:
        nonlocal current_bin_started
        sum_Z[:] += current_Z
        sum_Y[:] += current_Y
        sum_YZ[:] += np.outer(current_Y, current_Z)
        sum_ZZ[:] += np.outer(current_Z, current_Z)
        current_bin_started = False

    while t < T_max and bin_idx < n_max:
        lam = mu_vec + theta_mat @ u
        total_upper = float(lam.sum())
        if total_upper <= 0.0 or not np.isfinite(total_upper):
            raise RuntimeError("invalid total intensity during simulation")
        wait = float(rng.exponential(1.0 / total_upper))
        t_prop = t + wait

        while bin_idx < n_max and bin_idx * h <= min(t_prop, T_max):
            if not current_bin_started:
                start_bin(bin_idx)
            next_boundary = (bin_idx + 1) * h
            if next_boundary <= min(t_prop, T_max):
                close_bin()
                bin_idx += 1
                snapshot_if_needed()
            else:
                break

        if t_prop > T_max:
            break

        u *= np.exp(-beta * wait)
        t = t_prop
        lam_prop = mu_vec + theta_mat @ u
        total_prop = float(lam_prop.sum())
        if rng.random() * total_upper <= total_prop:
            probs = lam_prop / total_prop
            node = int(rng.choice(d, p=probs))
            if t >= 0.0:
                idx = int(np.ceil(t / h) - 1)
                if 0 <= idx < n_max:
                    while bin_idx < idx:
                        if not current_bin_started:
                            start_bin(bin_idx)
                        close_bin()
                        bin_idx += 1
                        snapshot_if_needed()
                    if bin_idx == idx and not current_bin_started:
                        start_bin(bin_idx)
                    if bin_idx == idx:
                        current_Y[node] = 1.0
                event_counts[node] += 1
            u[node] += 1.0
            event_count += 1
            if event_count > event_limit:
                raise RuntimeError(
                    "simulation exceeded max_events; check stability and parameters"
                )

    while bin_idx < n_max:
        if not current_bin_started:
            start_bin(bin_idx)
        close_bin()
        bin_idx += 1
        snapshot_if_needed()

    missing = [n for n in targets if n not in snapshots]
    if missing:
        raise RuntimeError(f"failed to create snapshots for n={missing}")
    return snapshots


def streaming_screen_and_regress(
    d: int,
    beta: float,
    mu: np.ndarray,
    theta: np.ndarray,
    T: float,
    burn_in: float,
    h: float,
    R: float,
    alpha: float,
    w_minus: float,
    k: int,
    m: int,
    seed: int,
    ridge: Optional[float] = None,
    cond_threshold: float = 1e12,
    allow_ridge_fallback: bool = False,
) -> tuple[EstimatorResult, StreamingHawkesStats]:
    stats = simulate_hawkes_sufficient_statistics(
        d=d,
        beta=beta,
        mu=mu,
        theta=theta,
        T=T,
        burn_in=burn_in,
        h=h,
        R=R,
        seed=seed,
    )
    result = estimate_from_sufficient_statistics(
        n=stats.n,
        sum_Z=stats.sum_Z,
        sum_Y=stats.sum_Y,
        sum_YZ=stats.sum_YZ,
        sum_ZZ=stats.sum_ZZ,
        h=h,
        R=R,
        alpha=alpha,
        w_minus=w_minus,
        k=k,
        m=m,
        ridge=ridge,
        cond_threshold=cond_threshold,
        allow_ridge_fallback=allow_ridge_fallback,
    )
    return result, stats
