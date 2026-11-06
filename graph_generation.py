from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from hawkes_sim import check_stability


@dataclass(frozen=True)
class GraphSpec:
    B: np.ndarray
    theta: np.ndarray
    support_true: np.ndarray
    gamma: float


def generate_row_sparse_B(
    d: int,
    k: int,
    w_minus: float,
    w_plus: float,
    exact_k: bool = True,
    no_self_edges: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    if d <= 0:
        raise ValueError("d must be positive")
    if k < 0:
        raise ValueError("k must be nonnegative")
    if w_minus <= 0 or w_plus <= 0:
        raise ValueError("w_minus and w_plus must be positive")
    if w_minus > w_plus:
        raise ValueError("w_minus must be <= w_plus")

    max_parents = d - 1 if no_self_edges else d
    if k > max_parents:
        raise ValueError(f"k must be <= {max_parents}")

    rng = np.random.default_rng(seed)
    B = np.zeros((d, d), dtype=float)

    for i in range(d):
        candidates = np.arange(d)
        if no_self_edges:
            candidates = candidates[candidates != i]

        if exact_k:
            parents = rng.choice(candidates, size=k, replace=False)
        else:
            num_parents = rng.integers(1, k+1)
            parents = rng.choice(candidates, size=num_parents, replace=False)

        if parents.size == 0:
            continue
        if w_minus == w_plus:
            weights = np.full(parents.size, w_minus, dtype=float)
        else:
            weights = rng.uniform(w_minus, w_plus, size=parents.size)
        B[i, parents] = weights

    return B


def generate_graph(
    d: int,
    k: int,
    alpha: float,
    beta: float,
    w_minus: float,
    w_plus: float,
    exact_k: bool = True,
    no_self_edges: bool = True,
    seed: Optional[int] = None,
) -> GraphSpec:
    gamma = check_stability(k=k, alpha=alpha, w_plus=w_plus, beta=beta)
    B = generate_row_sparse_B(
        d=d,
        k=k,
        w_minus=w_minus,
        w_plus=w_plus,
        exact_k=exact_k,
        no_self_edges=no_self_edges,
        seed=seed,
    )
    theta = float(alpha) * B
    support_true = (theta != 0.0).astype(np.int8)
    return GraphSpec(B=B, theta=theta, support_true=support_true, gamma=gamma)
