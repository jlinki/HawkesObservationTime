from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from estimator import bounded_block_screen_and_regress, estimate_from_sufficient_statistics
from graph_generation import generate_graph
from hawkes_sim import simulate_hawkes
from metrics import (
    aggregate_condition_numbers,
    classify_failure_mode,
    failed_edge_rows,
    gram_diagnostic_rows,
    ols_separation_rows,
    screening_gap_rows,
    screening_gap_summary,
    screening_metrics,
    support_metrics,
    validate_replicate_diagnostics,
    worst_case_ols_diagnostics,
)
from streaming import (
    simulate_hawkes_sufficient_statistics_snapshots,
    streaming_screen_and_regress,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "name": "scaling",
    "beta": 1.0,
    "mu": 1.0,
    "w_minus": 1.0,
    "w_plus": 1.0,
    "k": 5,
    "alpha": 0.10,
    "m_multiplier": 2,
    "dimensions": [10, 50, 100],
    "c_values": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500],
    "horizon_scaling": "logd",
    "n_reps": 100,
    "burn_in_multiplier": 50.0,
    "A_h": 2.0,
    "A_R": 8.0,
    "h": None,
    "R": None,
    "exact_k": False,
    "no_self_edges": False,
    "base_seed": 12345,
    "output_prefix": "scaling",
    "ridge": None,
    "cond_threshold": 1e12,
    "stream_results": True,
    "n_jobs": 100,
    "allow_ridge_fallback": False,
    "sweep_parameter": None,
    "sweep_values": None,
    "estimation_mode": "streaming",
    "reuse_c_paths": True,
}


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    with open(path, "r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    cfg.update(loaded)
    return cfg


def resolve_h_R(cfg: dict[str, Any], alpha: float) -> tuple[float, float]:
    h = cfg["h"] if cfg.get("h") is not None else float(cfg["A_h"]) * alpha**2
    R = cfg["R"] if cfg.get("R") is not None else float(cfg["A_R"]) / alpha
    return float(h), float(R)


def horizon_scale(d: int, scaling: str) -> float:
    normalized = str(scaling).strip().lower().replace("-", "_")
    if normalized in {"logd", "log_d", "log"}:
        return math.log(float(d))
    if normalized in {"d", "linear", "dimension"}:
        return float(d)
    if normalized in {"sqrt_d", "sqrtd", "sqrt"}:
        return math.sqrt(float(d))
    if normalized in {"constant", "const", "one", "1"}:
        return 1.0
    raise ValueError(
        "horizon_scaling must be one of: logd, d, sqrt_d, constant"
    )


def horizon_scaling_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("horizon_scaling", "logd")).strip().lower().replace("-", "_")


def alpha_values(cfg: dict[str, Any]) -> list[float]:
    alpha = cfg.get("alpha", 0.10)
    if isinstance(alpha, list):
        return [float(a) for a in alpha]
    return [float(alpha)]


def _sweep_variants(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    parameter = cfg.get("sweep_parameter")
    values = cfg.get("sweep_values")
    if parameter is None or values is None:
        variant = cfg.copy()
        variant["sweep_parameter"] = ""
        variant["sweep_value"] = ""
        variant["_variant_index"] = 0
        return [variant]
    variants: list[dict[str, Any]] = []
    for idx, value in enumerate(values):
        variant = cfg.copy()
        variant[str(parameter)] = value
        variant["sweep_parameter"] = str(parameter)
        variant["sweep_value"] = value
        variant["_variant_index"] = idx
        variants.append(variant)
    return variants


def build_tasks(cfg: dict[str, Any]) -> list[tuple[dict[str, Any], int, float, float, int]]:
    tasks: list[tuple[dict[str, Any], int, float, float, int]] = []
    for variant in _sweep_variants(cfg):
        for alpha in alpha_values(variant):
            for d in [int(x) for x in variant["dimensions"]]:
                for c_value in [float(x) for x in variant["c_values"]]:
                    for rep in range(int(variant["n_reps"])):
                        tasks.append((variant, d, c_value, alpha, rep))
    return tasks


def build_path_tasks(cfg: dict[str, Any]) -> list[tuple[dict[str, Any], int, float, int]]:
    tasks: list[tuple[dict[str, Any], int, float, int]] = []
    for variant in _sweep_variants(cfg):
        for alpha in alpha_values(variant):
            for d in [int(x) for x in variant["dimensions"]]:
                for rep in range(int(variant["n_reps"])):
                    tasks.append((variant, d, alpha, rep))
    return tasks


def run_task(
    task: tuple[dict[str, Any], int, float, float, int],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    cfg, d, c_value, alpha, rep = task
    return run_one(cfg, d=d, c_value=c_value, alpha=alpha, rep=rep)


def run_path_task(
    task: tuple[dict[str, Any], int, float, int],
) -> list[
    tuple[
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]
]:
    cfg, d, alpha, rep = task
    return run_one_path_all_c(cfg, d=d, alpha=alpha, rep=rep)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_for_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def _summary_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
        return out if np.isfinite(out) else None
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered == "true":
        return 1.0
    if lowered == "false":
        return 0.0
    try:
        out = float(text)
    except ValueError:
        return None
    return out if np.isfinite(out) else None


SUMMARY_GROUP_KEYS = [
    "experiment_id",
    "d",
    "c",
    "T",
    "log_d",
    "horizon_scaling",
    "horizon_scale",
    "alpha",
    "beta",
    "k",
    "m",
    "h",
    "R",
    "w_minus",
    "w_plus",
    "gamma",
    "m_multiplier",
    "sweep_parameter",
    "sweep_value",
    "estimation_mode",
]

SUMMARY_NON_METRIC_KEYS = {
    *SUMMARY_GROUP_KEYS,
    "replicate_id",
    "rep",
    "seed",
    "graph_seed",
    "row",
    "row_i",
    "parent_j",
    "edge_type",
    "failure_mode",
    "gram_solver_used",
}


def summarize_numeric_metrics(
    rows: list[dict[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    present_group_keys = [key for key in SUMMARY_GROUP_KEYS if key in rows[0]]
    metric_keys: list[str] = []
    seen_metrics: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in SUMMARY_NON_METRIC_KEYS or key in seen_metrics:
                continue
            if _summary_float(value) is not None:
                seen_metrics.add(key)
                metric_keys.append(key)

    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        group_tuple = tuple(row.get(key, "") for key in present_group_keys)
        bucket = grouped.setdefault(
            group_tuple,
            {
                "group_values": {key: row.get(key, "") for key in present_group_keys},
                "replicate_ids": set(),
                "values": {metric: [] for metric in metric_keys},
            },
        )
        replicate_id = str(row.get("replicate_id", ""))
        if replicate_id:
            bucket["replicate_ids"].add(replicate_id)
        for metric in metric_keys:
            value = _summary_float(row.get(metric))
            if value is not None:
                bucket["values"][metric].append(value)

    summary_rows: list[dict[str, Any]] = []
    for bucket in grouped.values():
        for metric in metric_keys:
            values = bucket["values"][metric]
            if not values:
                continue
            arr = np.asarray(values, dtype=float)
            summary_rows.append(
                {
                    "source": source,
                    **bucket["group_values"],
                    "metric": metric,
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                    "min": float(np.min(arr)),
                    "q10": float(np.quantile(arr, 0.10)),
                    "median": float(np.median(arr)),
                    "q90": float(np.quantile(arr, 0.90)),
                    "max": float(np.max(arr)),
                    "n_observations": int(arr.size),
                    "n_replicates_used": int(len(bucket["replicate_ids"])),
                }
            )
    return sorted(
        summary_rows,
        key=lambda row: (
            str(row.get("experiment_id", "")),
            str(row.get("source", "")),
            str(row.get("metric", "")),
        ),
    )


def write_metric_summary(path: Path, source: str, rows: list[dict[str, Any]]) -> None:
    summary = summarize_numeric_metrics(rows, source=source)
    write_csv(path, summary)


def _format_c_column(c_value: float) -> str:
    text = f"{c_value:g}".replace("-", "m").replace(".", "p")
    return f"c_{text}"


def _nonempty_string(value: Any) -> str:
    text = str(value).strip()
    return "" if text in {"", "nan", "None"} else text


def _format_pivot_value(value: Any) -> Any:
    parsed = _summary_float(value)
    if parsed is None:
        return _nonempty_string(value)
    return int(parsed) if parsed.is_integer() else parsed


def exact_recovery_pivot_tables(
    graph_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    has_sweep = any(_nonempty_string(row.get("sweep_parameter", "")) for row in graph_rows)
    buckets: dict[tuple[tuple[Any, ...], float], list[float]] = {}
    row_meta: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in graph_rows:
        d_value = _summary_float(row.get("d"))
        c_value = _summary_float(row.get("c"))
        exact_success = _summary_float(row.get("exact_success"))
        if d_value is None or c_value is None or exact_success is None:
            continue
        d_out = int(d_value) if d_value.is_integer() else d_value
        if has_sweep:
            sweep_parameter = _nonempty_string(row.get("sweep_parameter", ""))
            sweep_value = _format_pivot_value(row.get("sweep_value", ""))
            row_key = (sweep_parameter, str(sweep_value), d_value)
            row_meta[row_key] = {
                "sweep_parameter": sweep_parameter,
                "sweep_value": sweep_value,
                "d": d_out,
            }
        else:
            row_key = (d_value,)
            row_meta[row_key] = {"d": d_out}
        buckets.setdefault((row_key, c_value), []).append(exact_success)

    def row_sort_key(key: tuple[Any, ...]) -> tuple[str, float, str, float]:
        sweep_numeric = _summary_float(row_meta[key].get("sweep_value"))
        return (
            str(row_meta[key].get("sweep_parameter", "")),
            sweep_numeric if sweep_numeric is not None else math.inf,
            str(row_meta[key].get("sweep_value", "")),
            float(row_meta[key]["d"]),
        )

    row_keys = sorted(row_meta, key=row_sort_key)
    cs = sorted({c for _, c in buckets})
    mean_rows: list[dict[str, Any]] = []
    std_rows: list[dict[str, Any]] = []
    wilson_rows: list[dict[str, Any]] = []
    for row_key in row_keys:
        mean_row: dict[str, Any] = dict(row_meta[row_key])
        std_row: dict[str, Any] = dict(row_meta[row_key])
        wilson_row: dict[str, Any] = dict(row_meta[row_key])
        for c_value in cs:
            column = _format_c_column(c_value)
            values = buckets.get((row_key, c_value), [])
            if values:
                arr = np.asarray(values, dtype=float)
                mean_row[column] = float(np.mean(arr))
                std_row[column] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
                lower, upper = wilson_interval(successes=float(np.sum(arr)), n=int(arr.size))
                wilson_row[f"{column}_lower"] = lower
                wilson_row[f"{column}_upper"] = upper
                wilson_row[f"{column}_n"] = int(arr.size)
            else:
                mean_row[column] = ""
                std_row[column] = ""
                wilson_row[f"{column}_lower"] = ""
                wilson_row[f"{column}_upper"] = ""
                wilson_row[f"{column}_n"] = ""
        mean_rows.append(mean_row)
        std_rows.append(std_row)
        wilson_rows.append(wilson_row)
    return mean_rows, std_rows, wilson_rows


def wilson_interval(successes: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    phat = successes / float(n)
    denom = 1.0 + z**2 / float(n)
    center = (phat + z**2 / (2.0 * n)) / denom
    half_width = (
        z
        * math.sqrt((phat * (1.0 - phat) / n) + (z**2 / (4.0 * n**2)))
        / denom
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(seconds, 60.0)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"


class StreamingCsvWriter:
    def __init__(self, path: Path, fieldnames: list[str] | None = None):
        self.path = path
        self.fieldnames: list[str] | None = fieldnames
        if self.fieldnames is not None:
            with open(self.path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.fieldnames)
                writer.writeheader()

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        if self.fieldnames is None:
            fieldnames: list[str] = []
            seen: set[str] = set()
            for row in rows:
                for key in row:
                    if key not in seen:
                        seen.add(key)
                        fieldnames.append(key)
            self.fieldnames = fieldnames
            with open(self.path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            return

        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writerows(rows)


def format_replicate_outputs(
    cfg: dict[str, Any],
    graph,
    estimate,
    d: int,
    c_value: float,
    alpha: float,
    beta: float,
    k: int,
    m: int,
    h: float,
    R: float,
    T: float,
    rep: int,
    graph_seed: int,
    sim_seed: int,
    runtime: float,
    total_events: int,
    estimation_mode: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    w_minus = float(cfg["w_minus"])
    w_plus = float(cfg["w_plus"])
    sweep_parameter = str(cfg.get("sweep_parameter", "") or "")
    sweep_value = cfg.get("sweep_value", "")
    m_multiplier = float(cfg["m_multiplier"])
    scaling = horizon_scaling_name(cfg)
    scale_factor = horizon_scale(d, scaling)
    experiment_id = (
        f"d={d}|scale={scaling}|c={float(c_value):g}|alpha={alpha:g}|k={k}|mu={float(cfg['mu']):g}|"
        f"beta={beta:g}|m_multiplier={m_multiplier:g}|A_h={float(cfg['A_h']):g}|"
        f"A_R={float(cfg['A_R']):g}|h={h:g}|R={R:g}"
    )
    if sweep_parameter:
        experiment_id = f"{experiment_id}|{sweep_parameter}={sweep_value}"
    replicate_id = f"{experiment_id}|rep={rep}"

    screening_rows_raw = screening_gap_rows(
        estimate,
        graph.support_true,
        no_self_edges=bool(cfg["no_self_edges"]),
    )
    screening_summary = screening_gap_summary(screening_rows_raw)
    support_summary = support_metrics(estimate.support_hat, graph.support_true)
    ols_summary = worst_case_ols_diagnostics(estimate, graph.support_true)

    graph_row: dict[str, Any] = {
        "experiment_id": experiment_id,
        "replicate_id": replicate_id,
        "d": d,
        "c": float(c_value),
        "T": T,
        "log_d": math.log(float(d)),
        "horizon_scaling": scaling,
        "horizon_scale": scale_factor,
        "alpha": alpha,
        "beta": beta,
        "k": k,
        "m": m,
        "h": h,
        "R": R,
        "w_minus": w_minus,
        "w_plus": w_plus,
        "gamma": graph.gamma,
        "rep": rep,
        "graph_seed": graph_seed,
        "seed": sim_seed,
        "runtime_seconds": runtime,
        "threshold": estimate.tau,
        "estimation_mode": estimation_mode,
        "total_events": total_events,
        "m_multiplier": m_multiplier,
        "sweep_parameter": sweep_parameter,
        "sweep_value": sweep_value,
    }
    graph_row.update(support_summary)
    graph_row.update(screening_metrics(
        estimate,
        graph.support_true,
        no_self_edges=bool(cfg["no_self_edges"]),
    ))
    graph_row.update(screening_summary)
    graph_row.update(ols_summary)
    graph_row.update(aggregate_condition_numbers(estimate.condition_numbers))
    graph_row.update(classify_failure_mode(graph_row, estimate))
    validate_replicate_diagnostics(
        estimate,
        graph.support_true,
        graph_row,
        allow_ridge_fallback=bool(cfg.get("allow_ridge_fallback", False)),
    )

    row_records: list[dict[str, Any]] = []
    row_success = np.all(estimate.support_hat == graph.support_true, axis=1)
    for row_diag in ols_separation_rows(estimate, graph.support_true):
        i = int(row_diag["row"])
        record = {
            "experiment_id": experiment_id,
            "replicate_id": replicate_id,
            "d": d,
            "c": float(c_value),
            "T": T,
            "log_d": math.log(float(d)),
            "horizon_scaling": scaling,
            "horizon_scale": scale_factor,
            "alpha": alpha,
            "beta": beta,
            "k": k,
            "m": m,
            "m_multiplier": m_multiplier,
            "h": h,
            "R": R,
            "w_minus": w_minus,
            "w_plus": w_plus,
            "gamma": graph.gamma,
            "sweep_parameter": sweep_parameter,
            "sweep_value": sweep_value,
            "rep": rep,
            "graph_seed": graph_seed,
            "seed": sim_seed,
            "row_success": float(row_success[i]),
            "n_true_parents": int(graph.support_true[i].sum()),
        }
        record.update(row_diag)
        row_records.append(record)

    common = {
        "experiment_id": experiment_id,
        "replicate_id": replicate_id,
        "seed": sim_seed,
        "d": d,
        "T": T,
        "c": float(c_value),
        "log_d": math.log(float(d)),
        "horizon_scaling": scaling,
        "horizon_scale": scale_factor,
        "alpha": alpha,
        "beta": beta,
        "k": k,
        "h": h,
        "R": R,
        "m": m,
        "w_minus": w_minus,
        "w_plus": w_plus,
        "gamma": graph.gamma,
        "m_multiplier": m_multiplier,
        "sweep_parameter": sweep_parameter,
        "sweep_value": sweep_value,
    }
    screening_rows_out = [dict(common, **row) for row in screening_rows_raw]
    gram_rows_out = [dict(common, **row) for row in gram_diagnostic_rows(estimate)]
    failed_edges_out = [dict(common, **row) for row in failed_edge_rows(estimate, graph.support_true)]

    return graph_row, row_records, screening_rows_out, gram_rows_out, failed_edges_out


def run_one(
    cfg: dict[str, Any],
    d: int,
    c_value: float,
    alpha: float,
    rep: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    beta = float(cfg["beta"])
    k = int(cfg["k"])
    w_minus = float(cfg["w_minus"])
    w_plus = float(cfg["w_plus"])
    m_cfg = cfg.get("m")
    m = int(m_cfg) if m_cfg is not None else int(cfg["m_multiplier"]) * k
    h, R = resolve_h_R(cfg, alpha)
    scaling = horizon_scaling_name(cfg)
    scale_factor = horizon_scale(d, scaling)
    T = float(c_value) * scale_factor
    burn_in = float(cfg["burn_in_multiplier"]) / beta
    mu = np.full(d, float(cfg["mu"]), dtype=float)

    base_seed = int(cfg["base_seed"])
    graph_seed = base_seed + 1_000_000 * rep + 10_000 * d + int(round(1000 * c_value))
    sim_seed = graph_seed + 137

    start = time.perf_counter()
    graph = generate_graph(
        d=d,
        k=k,
        alpha=alpha,
        beta=beta,
        w_minus=w_minus,
        w_plus=w_plus,
        exact_k=bool(cfg["exact_k"]),
        no_self_edges=bool(cfg["no_self_edges"]),
        seed=graph_seed,
    )
    estimation_mode = str(cfg.get("estimation_mode", "streaming"))
    if estimation_mode == "streaming":
        estimate, stream_stats = streaming_screen_and_regress(
            d=d,
            beta=beta,
            mu=mu,
            theta=graph.theta,
            T=T,
            burn_in=burn_in,
            h=h,
            R=R,
            alpha=alpha,
            w_minus=w_minus,
            k=k,
            m=m,
            seed=sim_seed,
            ridge=cfg.get("ridge"),
            cond_threshold=float(cfg["cond_threshold"]),
            allow_ridge_fallback=bool(cfg.get("allow_ridge_fallback", False)),
        )
        total_events = stream_stats.total_events
    elif estimation_mode == "full_sample":
        sample = simulate_hawkes(
            d=d,
            beta=beta,
            mu=mu,
            theta=graph.theta,
            T=T,
            burn_in=burn_in,
            h=h,
            seed=sim_seed,
        )
        estimate = bounded_block_screen_and_regress(
            events_by_node=sample.events_by_node,
            sampled_X=sample.sampled_X,
            grid_times=sample.grid_times,
            h=h,
            R=R,
            alpha=alpha,
            w_minus=w_minus,
            k=k,
            m=m,
            ridge=cfg.get("ridge"),
            cond_threshold=float(cfg["cond_threshold"]),
            allow_ridge_fallback=bool(cfg.get("allow_ridge_fallback", False)),
        )
        total_events = sum(len(events) for events in sample.events_by_node)
    else:
        raise ValueError("estimation_mode must be 'streaming' or 'full_sample'")
    runtime = time.perf_counter() - start
    sweep_parameter = str(cfg.get("sweep_parameter", "") or "")
    sweep_value = cfg.get("sweep_value", "")
    m_multiplier = float(cfg["m_multiplier"])
    experiment_id = (
        f"d={d}|scale={scaling}|c={float(c_value):g}|alpha={alpha:g}|k={k}|mu={float(cfg['mu']):g}|"
        f"beta={beta:g}|m_multiplier={m_multiplier:g}|A_h={float(cfg['A_h']):g}|"
        f"A_R={float(cfg['A_R']):g}|h={h:g}|R={R:g}"
    )
    if sweep_parameter:
        experiment_id = f"{experiment_id}|{sweep_parameter}={sweep_value}"
    replicate_id = f"{experiment_id}|rep={rep}"
    screening_rows_raw = screening_gap_rows(
        estimate,
        graph.support_true,
        no_self_edges=bool(cfg["no_self_edges"]),
    )
    screening_summary = screening_gap_summary(screening_rows_raw)
    support_summary = support_metrics(estimate.support_hat, graph.support_true)
    ols_summary = worst_case_ols_diagnostics(estimate, graph.support_true)

    graph_row: dict[str, Any] = {
        "experiment_id": experiment_id,
        "replicate_id": replicate_id,
        "d": d,
        "c": float(c_value),
        "T": T,
        "log_d": math.log(float(d)),
        "horizon_scaling": scaling,
        "horizon_scale": scale_factor,
        "alpha": alpha,
        "beta": beta,
        "k": k,
        "m": m,
        "h": h,
        "R": R,
        "w_minus": w_minus,
        "w_plus": w_plus,
        "gamma": graph.gamma,
        "rep": rep,
        "graph_seed": graph_seed,
        "seed": sim_seed,
        "runtime_seconds": runtime,
        "threshold": estimate.tau,
        "estimation_mode": estimation_mode,
        "total_events": total_events,
        "m_multiplier": m_multiplier,
        "sweep_parameter": sweep_parameter,
        "sweep_value": sweep_value,
    }
    graph_row.update(support_summary)
    graph_row.update(screening_metrics(
        estimate,
        graph.support_true,
        no_self_edges=bool(cfg["no_self_edges"]),
    ))
    graph_row.update(screening_summary)
    graph_row.update(ols_summary)
    graph_row.update(aggregate_condition_numbers(estimate.condition_numbers))
    graph_row.update(classify_failure_mode(graph_row, estimate))
    validate_replicate_diagnostics(
        estimate,
        graph.support_true,
        graph_row,
        allow_ridge_fallback=bool(cfg.get("allow_ridge_fallback", False)),
    )

    row_records: list[dict[str, Any]] = []
    row_success = np.all(estimate.support_hat == graph.support_true, axis=1)
    for row_diag in ols_separation_rows(estimate, graph.support_true):
        i = int(row_diag["row"])
        record = {
            "experiment_id": experiment_id,
            "replicate_id": replicate_id,
            "d": d,
            "c": float(c_value),
            "T": T,
            "log_d": math.log(float(d)),
            "horizon_scaling": scaling,
            "horizon_scale": scale_factor,
            "alpha": alpha,
            "beta": beta,
            "k": k,
            "m": m,
            "m_multiplier": m_multiplier,
            "h": h,
            "R": R,
            "w_minus": w_minus,
            "w_plus": w_plus,
            "gamma": graph.gamma,
            "sweep_parameter": sweep_parameter,
            "sweep_value": sweep_value,
            "rep": rep,
            "graph_seed": graph_seed,
            "seed": sim_seed,
            "row_success": float(row_success[i]),
            "n_true_parents": int(graph.support_true[i].sum()),
        }
        record.update(row_diag)
        row_records.append(record)

    common = {
        "experiment_id": experiment_id,
        "replicate_id": replicate_id,
        "seed": sim_seed,
        "d": d,
        "T": T,
        "c": float(c_value),
        "log_d": math.log(float(d)),
        "horizon_scaling": scaling,
        "horizon_scale": scale_factor,
        "alpha": alpha,
        "beta": beta,
        "k": k,
        "h": h,
        "R": R,
        "m": m,
        "w_minus": w_minus,
        "w_plus": w_plus,
        "gamma": graph.gamma,
        "m_multiplier": m_multiplier,
        "sweep_parameter": sweep_parameter,
        "sweep_value": sweep_value,
    }
    screening_rows_out = [dict(common, **row) for row in screening_rows_raw]
    gram_rows_out = [dict(common, **row) for row in gram_diagnostic_rows(estimate)]
    failed_edges_out = [dict(common, **row) for row in failed_edge_rows(estimate, graph.support_true)]

    return graph_row, row_records, screening_rows_out, gram_rows_out, failed_edges_out


def run_one_path_all_c(
    cfg: dict[str, Any],
    d: int,
    alpha: float,
    rep: int,
) -> list[
    tuple[
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]
]:
    beta = float(cfg["beta"])
    k = int(cfg["k"])
    w_minus = float(cfg["w_minus"])
    w_plus = float(cfg["w_plus"])
    m_cfg = cfg.get("m")
    m = int(m_cfg) if m_cfg is not None else int(cfg["m_multiplier"]) * k
    h, R = resolve_h_R(cfg, alpha)
    burn_in = float(cfg["burn_in_multiplier"]) / beta
    mu = np.full(d, float(cfg["mu"]), dtype=float)
    c_values = sorted(float(x) for x in cfg["c_values"])
    scaling = horizon_scaling_name(cfg)
    scale_factor = horizon_scale(d, scaling)
    T_by_c = {c: c * scale_factor for c in c_values}
    n_by_c = {c: int(math.floor(T_by_c[c] / h)) for c in c_values}
    if any(n <= 0 for n in n_by_c.values()):
        raise ValueError("all c values must satisfy floor(T / h) > 0")

    base_seed = int(cfg["base_seed"])
    variant_index = int(cfg.get("_variant_index", 0))
    graph_seed = base_seed + 100_000_000 * variant_index + 1_000_000 * rep + 10_000 * d
    sim_seed = graph_seed + 137

    start = time.perf_counter()
    graph = generate_graph(
        d=d,
        k=k,
        alpha=alpha,
        beta=beta,
        w_minus=w_minus,
        w_plus=w_plus,
        exact_k=bool(cfg["exact_k"]),
        no_self_edges=bool(cfg["no_self_edges"]),
        seed=graph_seed,
    )
    snapshots = simulate_hawkes_sufficient_statistics_snapshots(
        d=d,
        beta=beta,
        mu=mu,
        theta=graph.theta,
        T_max=max(T_by_c.values()),
        burn_in=burn_in,
        h=h,
        R=R,
        seed=sim_seed,
        n_targets=list(n_by_c.values()),
    )
    path_runtime = time.perf_counter() - start

    outputs = []
    for c_value in c_values:
        stats = snapshots[n_by_c[c_value]]
        estimate = estimate_from_sufficient_statistics(
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
            ridge=cfg.get("ridge"),
            cond_threshold=float(cfg["cond_threshold"]),
            allow_ridge_fallback=bool(cfg.get("allow_ridge_fallback", False)),
        )
        outputs.append(
            format_replicate_outputs(
                cfg=cfg,
                graph=graph,
                estimate=estimate,
                d=d,
                c_value=c_value,
                alpha=alpha,
                beta=beta,
                k=k,
                m=m,
                h=h,
                R=R,
                T=T_by_c[c_value],
                rep=rep,
                graph_seed=graph_seed,
                sim_seed=sim_seed,
                runtime=path_runtime,
                total_events=stats.total_events,
                estimation_mode="streaming_reuse_c",
            )
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/scaling_logd.yaml",
        help="YAML config path",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--name", default=None, help="Override config name for output folder.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel worker processes. Overrides config n_jobs.",
    )
    parser.add_argument(
        "--allow-ridge-fallback",
        action="store_true",
        help="Allow diagnostic ridge fallback when config ridge is positive.",
    )
    args = parser.parse_args()
    experiment_start = time.perf_counter()

    cfg = load_config(args.config)
    if args.n_jobs is not None:
        cfg["n_jobs"] = int(args.n_jobs)
    if args.allow_ridge_fallback:
        cfg["allow_ridge_fallback"] = True
    experiment_name = str(args.name or cfg.get("name") or Path(args.config).stem)
    results_dir = Path(args.results_dir) / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    graph_rows: list[dict[str, Any]] = []
    row_rows: list[dict[str, Any]] = []
    screening_rows: list[dict[str, Any]] = []
    gram_rows: list[dict[str, Any]] = []
    failed_edge_rows_all: list[dict[str, Any]] = []
    prefix = str(cfg["output_prefix"])
    graph_path = results_dir / f"{prefix}_graph.csv"
    rows_path = results_dir / f"{prefix}_rows.csv"
    screening_rows_path = results_dir / f"{prefix}_screening_rows.csv"
    gram_rows_path = results_dir / f"{prefix}_gram_rows.csv"
    failed_edges_path = results_dir / f"{prefix}_failed_edges.csv"
    stream_results = bool(cfg.get("stream_results", True))
    graph_writer = StreamingCsvWriter(graph_path)
    row_writer = StreamingCsvWriter(rows_path)
    screening_writer = StreamingCsvWriter(screening_rows_path)
    gram_writer = StreamingCsvWriter(gram_rows_path)
    failed_edge_writer = StreamingCsvWriter(
        failed_edges_path,
        fieldnames=[
            "experiment_id",
            "replicate_id",
            "seed",
            "d",
            "T",
            "c",
            "log_d",
            "horizon_scaling",
            "horizon_scale",
            "alpha",
            "beta",
            "k",
            "h",
            "R",
            "m",
            "w_minus",
            "w_plus",
            "gamma",
            "m_multiplier",
            "sweep_parameter",
            "sweep_value",
            "row_i",
            "parent_j",
            "edge_type",
            "was_screened",
            "ols_coefficient",
            "threshold",
        ],
    )
    reuse_c_paths = (
        bool(cfg.get("reuse_c_paths", True))
        and str(cfg.get("estimation_mode", "streaming")) == "streaming"
    )
    tasks = build_path_tasks(cfg) if reuse_c_paths else build_tasks(cfg)
    n_jobs = int(cfg.get("n_jobs", 1))
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive")
    workers = min(n_jobs, max(1, len(tasks)))

    for alpha in alpha_values(cfg):
        h, R = resolve_h_R(cfg, alpha)
        print(f"alpha={alpha:g}, h={h:g}, R={R:g}")
    print(f"horizon_scaling={horizon_scaling_name(cfg)}")

    task_label = "(variant, d, alpha, rep) path tasks reusing all c_values" if reuse_c_paths else "(variant, d, c, alpha, rep) tasks"
    print(f"running {len(tasks)} {task_label} with n_jobs={workers}", flush=True)
    completed = 0

    def record_result(
        graph_row: dict[str, Any],
        row_records: list[dict[str, Any]],
        screening_records: list[dict[str, Any]],
        gram_records: list[dict[str, Any]],
        failed_edge_records: list[dict[str, Any]],
    ) -> None:
        if stream_results:
            graph_writer.write([graph_row])
            row_writer.write(row_records)
            screening_writer.write(screening_records)
            gram_writer.write(gram_records)
            failed_edge_writer.write(failed_edge_records)
        else:
            graph_rows.append(graph_row)
            row_rows.extend(row_records)
            screening_rows.extend(screening_records)
            gram_rows.extend(gram_records)
            failed_edge_rows_all.extend(failed_edge_records)

    def run_sequential(start_completed: int = 0) -> int:
        done = start_completed
        for task in tasks:
            if reuse_c_paths:
                _, d, alpha, rep = task
                print(f"running d={d}, all c_values, alpha={alpha:g}, rep={rep}", flush=True)
                results = run_path_task(task)
            else:
                _, d, c_value, alpha, rep = task
                print(f"running d={d}, c={c_value:g}, alpha={alpha:g}, rep={rep}", flush=True)
                results = [run_task(task)]
            done += 1
            for graph_row, row_records, screening_records, gram_records, failed_edge_records in results:
                record_result(
                    graph_row,
                    row_records,
                    screening_records,
                    gram_records,
                    failed_edge_records,
                )
            print(f"completed {done}/{len(tasks)}", flush=True)
        return done

    if workers == 1:
        completed = run_sequential()
    else:
        pool: ProcessPoolExecutor | None = None
        try:
            pool = ProcessPoolExecutor(max_workers=workers)
            task_runner = run_path_task if reuse_c_paths else run_task
            futures = {pool.submit(task_runner, task): task for task in tasks}
        except (OSError, PermissionError) as exc:
            if pool is not None:
                pool.shutdown(cancel_futures=True)
            print(
                "process pool unavailable in this environment; "
                f"falling back to n_jobs=1 ({exc})",
                flush=True,
            )
            completed = run_sequential()
        else:
            try:
                for future in as_completed(futures):
                    task = futures[future]
                    if reuse_c_paths:
                        _, d, alpha, rep = task
                        results = future.result()
                    else:
                        _, d, c_value, alpha, rep = task
                        results = [future.result()]
                    completed += 1
                    for graph_row, row_records, screening_records, gram_records, failed_edge_records in results:
                        record_result(
                            graph_row,
                            row_records,
                            screening_records,
                            gram_records,
                            failed_edge_records,
                        )
                    c_text = "all c_values" if reuse_c_paths else f"c={c_value:g}"
                    print(
                        f"completed {completed}/{len(tasks)} "
                        f"(d={d}, {c_text}, alpha={alpha:g}, rep={rep})",
                        flush=True,
                    )
            finally:
                pool.shutdown()

    if not stream_results:
        write_csv(graph_path, graph_rows)
        write_csv(rows_path, row_rows)
        write_csv(screening_rows_path, screening_rows)
        write_csv(gram_rows_path, gram_rows)
        write_csv(failed_edges_path, failed_edge_rows_all)

    metric_summary_rows: list[dict[str, Any]] = []
    for source, path in [
        ("graph", graph_path),
        ("rows", rows_path),
        ("screening_rows", screening_rows_path),
        ("gram_rows", gram_rows_path),
    ]:
        source_rows = read_csv_for_summary(path)
        summary_rows = summarize_numeric_metrics(source_rows, source=source)
        write_csv(results_dir / f"{prefix}_{source}_metric_summary.csv", summary_rows)
        metric_summary_rows.extend(summary_rows)
    metric_summary_path = results_dir / f"{prefix}_metric_summary.csv"
    write_csv(metric_summary_path, metric_summary_rows)
    exact_pivot_mean, exact_pivot_std, exact_pivot_wilson = exact_recovery_pivot_tables(
        read_csv_for_summary(graph_path)
    )
    exact_mean_path = results_dir / f"{prefix}_exact_recovery_mean_by_d_c.csv"
    exact_std_path = results_dir / f"{prefix}_exact_recovery_std_by_d_c.csv"
    exact_wilson_path = results_dir / f"{prefix}_exact_recovery_wilson95_by_d_c.csv"
    write_csv(exact_mean_path, exact_pivot_mean)
    write_csv(exact_std_path, exact_pivot_std)
    write_csv(exact_wilson_path, exact_pivot_wilson)

    print(f"wrote {graph_path}")
    print(f"wrote {rows_path}")
    print(f"wrote {screening_rows_path}")
    print(f"wrote {gram_rows_path}")
    print(f"wrote {failed_edges_path}")
    print(f"wrote {metric_summary_path}")
    print(f"wrote {exact_mean_path}")
    print(f"wrote {exact_std_path}")
    print(f"wrote {exact_wilson_path}")
    elapsed = time.perf_counter() - experiment_start
    metadata = {
        "experiment_name": experiment_name,
        "config": str(args.config),
        "n_tasks": len(tasks),
        "n_jobs_requested": n_jobs,
        "n_jobs_used": workers,
        "reuse_c_paths": reuse_c_paths,
        "estimation_mode": str(cfg.get("estimation_mode", "streaming")),
        "horizon_scaling": horizon_scaling_name(cfg),
        "elapsed_seconds": float(elapsed),
        "elapsed_human": format_seconds(elapsed),
    }
    with open(results_dir / "run_metadata.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(metadata, fh, sort_keys=True)
    print(f"total experiment wall time: {format_seconds(elapsed)} ({elapsed:.2f} seconds)")


if __name__ == "__main__":
    main()
