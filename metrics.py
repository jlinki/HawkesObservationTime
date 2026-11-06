from __future__ import annotations

from typing import Dict, List

import numpy as np

from estimator import EstimatorResult


def support_metrics(
    support_hat: np.ndarray,
    support_true: np.ndarray,
) -> Dict[str, float]:
    hat = np.asarray(support_hat, dtype=bool)
    truth = np.asarray(support_true, dtype=bool)
    if hat.shape != truth.shape:
        raise ValueError("support_hat and support_true shapes must match")

    tp = float(np.logical_and(hat, truth).sum())
    fp = float(np.logical_and(hat, ~truth).sum())
    fn = float(np.logical_and(~hat, truth).sum())
    true_edges = float(truth.sum())
    inactive_edges = float((~truth).sum())

    row_success = np.all(hat == truth, axis=1)
    exact_success = bool(np.all(hat == truth))
    has_false_positive = bool(fp > 0.0)
    has_false_negative = bool(fn > 0.0)
    if exact_success != (not has_false_positive and not has_false_negative):
        raise AssertionError("exact_success must equal zero false positives and false negatives")
    return {
        "exact_success": float(exact_success),
        "row_success_rate": float(row_success.mean()),
        "edge_true_positive_rate": tp / true_edges if true_edges > 0 else np.nan,
        "edge_false_positive_rate": fp / inactive_edges if inactive_edges > 0 else np.nan,
        "edge_false_negative_rate": fn / true_edges if true_edges > 0 else np.nan,
        "n_true_edges": true_edges,
        "n_false_positives": fp,
        "n_false_negatives": fn,
        "has_false_positive": float(has_false_positive),
        "has_false_negative": float(has_false_negative),
    }


def screening_metrics(
    result: EstimatorResult,
    support_true: np.ndarray,
    no_self_edges: bool = True,
) -> Dict[str, float]:
    truth = np.asarray(support_true, dtype=bool)
    row_success: list[bool] = []
    gaps: list[float] = []

    for i, C in enumerate(result.screening_sets):
        active = np.flatnonzero(truth[i])
        C_set = set(int(x) for x in C)
        row_success.append(all(int(j) in C_set for j in active))

        inactive_mask = ~truth[i].copy()
        if no_self_edges:
            inactive_mask[i] = False
        inactive = np.flatnonzero(inactive_mask)
        if active.size > 0 and inactive.size > 0:
            gaps.append(
                float(np.min(result.F_hat[i, active]) - np.max(result.F_hat[i, inactive]))
            )

    gap_arr = np.asarray(gaps, dtype=float)
    return {
        "screening_success_all_rows": float(all(row_success)),
        "screening_success_row_rate": float(np.mean(row_success)),
        "mean_screening_gap": float(np.nanmean(gap_arr)) if gap_arr.size else np.nan,
        "median_screening_gap": float(np.nanmedian(gap_arr)) if gap_arr.size else np.nan,
        "q10_screening_gap": float(np.nanquantile(gap_arr, 0.10)) if gap_arr.size else np.nan,
        "q90_screening_gap": float(np.nanquantile(gap_arr, 0.90)) if gap_arr.size else np.nan,
    }


def screening_gap_rows(
    result: EstimatorResult,
    support_true: np.ndarray,
    no_self_edges: bool,
) -> List[Dict[str, float]]:
    truth = np.asarray(support_true, dtype=bool)
    rows: list[dict[str, float]] = []
    for i, C in enumerate(result.screening_sets):
        active = np.flatnonzero(truth[i])
        C_set = set(int(x) for x in C)
        all_true_screened = all(int(j) in C_set for j in active)
        inactive_mask = ~truth[i].copy()
        if no_self_edges:
            inactive_mask[i] = False
        inactive = np.flatnonzero(inactive_mask)

        min_true = (
            float(np.min(result.F_hat[i, active])) if active.size > 0 else np.nan
        )
        max_nonparent = (
            float(np.max(result.F_hat[i, inactive])) if inactive.size > 0 else np.nan
        )
        gap = (
            float(min_true - max_nonparent)
            if np.isfinite(min_true) and np.isfinite(max_nonparent)
            else np.nan
        )
        rows.append(
            {
                "row_i": i,
                "screening_gap_i": gap,
                "row_screening_success": float(all_true_screened),
                "min_true_parent_score": min_true,
                "max_nonparent_score": max_nonparent,
                "selected_candidate_set_size": int(len(C)),
                "true_parent_count": int(active.size),
                "all_true_parents_screened": float(all_true_screened),
            }
        )
    return rows


def screening_gap_summary(screening_rows: List[Dict[str, float]]) -> Dict[str, float]:
    gaps = np.asarray([row["screening_gap_i"] for row in screening_rows], dtype=float)
    finite = gaps[np.isfinite(gaps)]
    successes = [bool(row["all_true_parents_screened"]) for row in screening_rows]
    return {
        "min_screening_gap_over_rows": float(np.min(finite)) if finite.size else np.nan,
        "median_screening_gap_over_rows": float(np.median(finite)) if finite.size else np.nan,
        "mean_screening_gap_over_rows": float(np.mean(finite)) if finite.size else np.nan,
        "screening_success_all_rows": float(all(successes)),
        "screening_success_row_rate": float(np.mean(successes)) if successes else np.nan,
    }


def ols_separation_rows(
    result: EstimatorResult,
    support_true: np.ndarray,
) -> List[Dict[str, float]]:
    truth = np.asarray(support_true, dtype=bool)
    rows: list[dict[str, float]] = []

    for i, C in enumerate(result.screening_sets):
        active = np.flatnonzero(truth[i])
        C_set = set(int(x) for x in C)
        all_active_screened = all(int(j) in C_set for j in active)

        if all_active_screened and active.size > 0:
            active_coef = result.coefficients[i, active]
            finite_active = active_coef[np.isfinite(active_coef)]
            min_active = float(np.min(finite_active)) if finite_active.size else np.nan
        else:
            min_active = np.nan

        inactive_in_C = [int(j) for j in C if not truth[i, int(j)]]
        if inactive_in_C:
            inactive_coef = result.coefficients[i, inactive_in_C]
            finite_inactive = inactive_coef[np.isfinite(inactive_coef)]
            max_inactive = float(np.max(finite_inactive)) if finite_inactive.size else np.nan
        else:
            max_inactive = np.nan

        rows.append(
            {
                "row": i,
                "row_i": i,
                "all_active_screened": float(all_active_screened),
                "min_active_coef": min_active,
                "max_inactive_coef": max_inactive,
                "tau": float(result.tau),
                "condition_number": float(result.condition_numbers[i]),
                "singular_row": float(result.singular_rows[i]),
            }
        )

    return rows


def gram_diagnostic_rows(result: EstimatorResult) -> List[Dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for i, C in enumerate(result.screening_sets):
        rows.append(
            {
                "row_i": i,
                "candidate_set_size": int(len(C)),
                "gram_min_eigenvalue": float(result.gram_min_eigenvalues[i]),
                "gram_max_eigenvalue": float(result.gram_max_eigenvalues[i]),
                "gram_condition_number": float(result.condition_numbers[i]),
                "gram_rank": int(result.gram_ranks[i]),
                "gram_solve_success": float(result.gram_solve_success[i]),
                "gram_solver_used": result.gram_solver_used[i],
                "ridge_used": float(result.ridge_used[i]),
                "ridge_value": float(result.ridge_values[i]),
            }
        )
    return rows


def worst_case_ols_diagnostics(
    result: EstimatorResult,
    support_true: np.ndarray,
) -> Dict[str, float]:
    truth = np.asarray(support_true, dtype=bool)
    threshold = float(result.tau)
    screened_active_coefs: list[float] = []
    screened_inactive_coefs: list[float] = []
    all_active_edges_screened = True
    all_screened_active_finite = True
    all_screened_inactive_finite = True

    for i, C in enumerate(result.screening_sets):
        C_list = [int(j) for j in C]
        C_set = set(C_list)
        active = np.flatnonzero(truth[i])
        for j in active:
            if int(j) not in C_set:
                all_active_edges_screened = False
            else:
                coef = float(result.coefficients[i, int(j)])
                if np.isfinite(coef):
                    screened_active_coefs.append(coef)
                else:
                    all_screened_active_finite = False
        for j in C_list:
            if not truth[i, j]:
                coef = float(result.coefficients[i, j])
                if np.isfinite(coef):
                    screened_inactive_coefs.append(coef)
                else:
                    all_screened_inactive_finite = False

    global_min_active = (
        float(np.min(screened_active_coefs))
        if all_active_edges_screened and all_screened_active_finite and screened_active_coefs
        else np.nan
    )
    global_max_inactive = (
        float(np.max(screened_inactive_coefs))
        if all_screened_inactive_finite and screened_inactive_coefs
        else np.nan
    )
    if np.isfinite(global_min_active) and np.isfinite(global_max_inactive):
        margin = float(global_min_active - global_max_inactive)
    else:
        margin = np.nan

    active_above = bool(
        all_active_edges_screened
        and all_screened_active_finite
        and screened_active_coefs
        and all(coef >= threshold for coef in screened_active_coefs)
    )
    inactive_below = bool(
        all_screened_inactive_finite
        and
        all(coef < threshold for coef in screened_inactive_coefs)
    )
    ols_threshold_success = bool(active_above and inactive_below)

    # Explicit validation required by the diagnostic spec.
    expected_active = bool(
        all_active_edges_screened
        and all_screened_active_finite
        and screened_active_coefs
        and all(coef >= threshold for coef in screened_active_coefs)
    )
    expected_inactive = bool(
        all_screened_inactive_finite
        and all(coef < threshold for coef in screened_inactive_coefs)
    )
    if active_above != expected_active or inactive_below != expected_inactive:
        raise AssertionError("OLS threshold success flags are inconsistent")

    return {
        "global_min_active_ols": global_min_active,
        "global_max_inactive_ols": global_max_inactive,
        "global_ols_margin": margin,
        "active_above_threshold_all": float(active_above),
        "inactive_below_threshold_all": float(inactive_below),
        "ols_threshold_success": float(ols_threshold_success),
        "screening_failed": float(not all_active_edges_screened),
    }


def classify_failure_mode(
    graph_metrics: Dict[str, float],
    result: EstimatorResult,
) -> Dict[str, float | str]:
    exact_success = bool(graph_metrics["exact_success"])
    screening_success_all_rows = bool(graph_metrics["screening_success_all_rows"])
    ols_threshold_success = bool(graph_metrics["ols_threshold_success"])
    gram_success = bool(np.all(result.gram_solve_success))
    has_false_positive = bool(graph_metrics["has_false_positive"])
    has_false_negative = bool(graph_metrics["has_false_negative"])

    if exact_success:
        failure_mode = "success"
    elif not gram_success:
        failure_mode = "gram_singularity_or_numerical_failure"
    elif not screening_success_all_rows:
        failure_mode = "screening_failure"
    elif has_false_negative and not has_false_positive:
        failure_mode = "ols_false_negative_after_successful_screening"
    elif has_false_positive and not has_false_negative:
        failure_mode = "ols_false_positive_after_successful_screening"
    elif has_false_positive and has_false_negative:
        failure_mode = "ols_mixed_false_positive_and_false_negative_after_successful_screening"
    else:
        failure_mode = "unknown_failure"

    return {
        "gram_numerical_success_all_rows": float(gram_success),
        "failure_mode": failure_mode,
    }


def failed_edge_rows(
    result: EstimatorResult,
    support_true: np.ndarray,
) -> List[Dict[str, float | str]]:
    hat = np.asarray(result.support_hat, dtype=bool)
    truth = np.asarray(support_true, dtype=bool)
    rows: list[dict[str, float | str]] = []
    fp = np.argwhere(np.logical_and(hat, ~truth))
    fn = np.argwhere(np.logical_and(~hat, truth))
    screened = [set(int(x) for x in C) for C in result.screening_sets]
    for edge_type, edges in (("false_positive", fp), ("false_negative", fn)):
        for i, j in edges:
            i_int = int(i)
            j_int = int(j)
            coef = float(result.coefficients[i_int, j_int])
            rows.append(
                {
                    "row_i": i_int,
                    "parent_j": j_int,
                    "edge_type": edge_type,
                    "was_screened": float(j_int in screened[i_int]),
                    "ols_coefficient": coef if np.isfinite(coef) else np.nan,
                    "threshold": float(result.tau),
                }
            )
    return rows


def validate_replicate_diagnostics(
    result: EstimatorResult,
    support_true: np.ndarray,
    graph_metrics: Dict[str, float | str],
    allow_ridge_fallback: bool,
) -> None:
    hat = np.asarray(result.support_hat, dtype=bool)
    truth = np.asarray(support_true, dtype=bool)
    fp = int(np.logical_and(hat, ~truth).sum())
    fn = int(np.logical_and(~hat, truth).sum())
    exact = bool(graph_metrics["exact_success"])
    if exact != (fp == 0 and fn == 0):
        raise AssertionError("exact_success is not equivalent to zero FP and zero FN")

    row_screening = []
    for i, C in enumerate(result.screening_sets):
        C_set = set(int(j) for j in C)
        active = np.flatnonzero(truth[i])
        row_screening.append(all(int(j) in C_set for j in active))
    if bool(graph_metrics["screening_success_all_rows"]) != all(row_screening):
        raise AssertionError("screening_success_all_rows is inconsistent with candidate sets")

    threshold = float(result.tau)
    active_ok = True
    inactive_ok = True
    for i, C in enumerate(result.screening_sets):
        C_set = set(int(j) for j in C)
        for j in np.flatnonzero(truth[i]):
            if int(j) not in C_set:
                active_ok = False
            else:
                coef = float(result.coefficients[i, int(j)])
                if not np.isfinite(coef) or coef < threshold:
                    active_ok = False
        for j in C:
            j_int = int(j)
            if not truth[i, j_int]:
                coef = float(result.coefficients[i, j_int])
                if np.isfinite(coef) and coef >= threshold:
                    inactive_ok = False
                if not np.isfinite(coef):
                    inactive_ok = False
    expected_ols = active_ok and inactive_ok
    if bool(graph_metrics["ols_threshold_success"]) != expected_ols:
        raise AssertionError("ols_threshold_success is inconsistent with coefficients")

    if np.any(result.ridge_used) and not allow_ridge_fallback:
        raise AssertionError("ridge was used without allow_ridge_fallback")


def aggregate_condition_numbers(condition_numbers: np.ndarray) -> Dict[str, float]:
    cond = np.asarray(condition_numbers, dtype=float)
    finite = cond[np.isfinite(cond)]
    return {
        "mean_condition_number": float(np.mean(finite)) if finite.size else np.inf,
        "max_condition_number": float(np.max(finite)) if finite.size else np.inf,
        "n_singular_rows": float(np.sum(~np.isfinite(cond))),
    }


def empirical_mean_intensity(events_by_node: list[np.ndarray], T: float) -> np.ndarray:
    return np.asarray([len(ev) / float(T) for ev in events_by_node], dtype=float)
