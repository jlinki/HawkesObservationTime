from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

try:
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/hawkesscaling_cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/hawkesscaling_matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - environment issue
    raise SystemExit(
        "plot_results.py requires a complete matplotlib installation. "
        "Install it with `pip install matplotlib` in the active environment."
    ) from exc


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            converted: dict[str, float | str] = {}
            for key, value in row.items():
                if value == "":
                    converted[key] = math.nan
                    continue
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value
            rows.append(converted)
    return rows


def resolve_named_results_dir(results_dir: Path, name: str | None, config: str | None) -> Path:
    resolved_name = name
    if resolved_name is None and config is not None:
        with open(config, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        resolved_name = str(cfg.get("name") or Path(config).stem)
    if resolved_name:
        return results_dir / resolved_name
    return results_dir


def write_csv_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
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


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def grouped_mean(
    rows: list[dict[str, float | str]],
    keys: tuple[str, ...],
    value_key: str,
) -> list[dict[str, float]]:
    buckets: dict[tuple[float, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[value_key])
        if np.isfinite(value):
            buckets[tuple(float(row[k]) for k in keys)].append(value)
    out: list[dict[str, float]] = []
    for key_values, values in buckets.items():
        if not values:
            continue
        item = {k: v for k, v in zip(keys, key_values)}
        item[value_key] = float(np.mean(values))
        out.append(item)
    return sorted(out, key=lambda x: tuple(x[k] for k in keys))


def grouped_mean_std(
    rows: list[dict[str, float | str]],
    keys: tuple[str, ...],
    value_key: str,
) -> list[dict[str, float]]:
    buckets: dict[tuple[float, ...], list[float]] = defaultdict(list)
    for row in rows:
        try:
            value = float(row[value_key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            buckets[tuple(float(row[k]) for k in keys)].append(value)
    out: list[dict[str, float]] = []
    for key_values, values in buckets.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        item = {k: v for k, v in zip(keys, key_values)}
        item[f"{value_key}_mean"] = float(np.mean(arr))
        item[f"{value_key}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        item[f"{value_key}_n"] = float(arr.size)
        out.append(item)
    return sorted(out, key=lambda x: tuple(x[k] for k in keys))


def grouped_quantiles(
    rows: list[dict[str, float | str]],
    keys: tuple[str, ...],
    value_key: str,
    quantile_low: float,
    quantile_high: float,
) -> list[dict[str, float]]:
    buckets: dict[tuple[float, ...], list[float]] = defaultdict(list)
    for row in rows:
        try:
            value = float(row[value_key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            buckets[tuple(float(row[k]) for k in keys)].append(value)
    out: list[dict[str, float]] = []
    for key_values, values in buckets.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        item = {k: v for k, v in zip(keys, key_values)}
        item[f"{value_key}_q_low"] = float(np.quantile(arr, quantile_low))
        item[f"{value_key}_median"] = float(np.median(arr))
        item[f"{value_key}_q_high"] = float(np.quantile(arr, quantile_high))
        item[f"{value_key}_mean"] = float(np.mean(arr))
        item[f"{value_key}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        item[f"{value_key}_n"] = float(arr.size)
        out.append(item)
    return sorted(out, key=lambda x: tuple(x[k] for k in keys))


def grouped_median(
    rows: list[dict[str, float | str]],
    key: str,
    value_keys: tuple[str, ...],
) -> list[dict[str, float]]:
    buckets: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        k = float(row[key])
        for value_key in value_keys:
            value = float(row[value_key])
            if np.isfinite(value):
                buckets[k][value_key].append(value)
    out: list[dict[str, float]] = []
    for k, values_by_name in buckets.items():
        item = {key: k}
        for value_key in value_keys:
            vals = values_by_name.get(value_key, [])
            item[value_key] = float(np.median(vals)) if vals else math.nan
        out.append(item)
    return sorted(out, key=lambda x: x[key])


def _row_experiment_id(row: dict[str, float | str]) -> str:
    if "experiment_id" in row and not isinstance(row["experiment_id"], float):
        return str(row["experiment_id"])
    return (
        f"alpha={float(row.get('alpha', math.nan)):g}|"
        f"d={int(float(row.get('d', math.nan)))}|"
        f"c={float(row.get('c', math.nan)):g}"
    )


def _finite_values(rows: list[dict[str, float | str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def horizon_scaling_from_rows(rows: list[dict[str, float | str]]) -> str:
    values = {
        str(row.get("horizon_scaling", "")).strip().lower().replace("-", "_")
        for row in rows
        if str(row.get("horizon_scaling", "")).strip() not in {"", "nan"}
    }
    if len(values) == 1:
        return next(iter(values))
    if not values:
        return "logd"
    return "mixed"


def horizon_axis_label(rows: list[dict[str, float | str]]) -> str:
    scaling = horizon_scaling_from_rows(rows)
    labels = {
        "logd": "c = T / log(d)",
        "log_d": "c = T / log(d)",
        "log": "c = T / log(d)",
        "d": "c = T / d",
        "linear": "c = T / d",
        "dimension": "c = T / d",
        "sqrt_d": "c = T / sqrt(d)",
        "sqrtd": "c = T / sqrt(d)",
        "sqrt": "c = T / sqrt(d)",
        "constant": "c = T",
        "const": "c = T",
        "one": "c = T",
        "1": "c = T",
    }
    return labels.get(scaling, "c = normalized observation horizon")


def horizon_filename_stem(rows: list[dict[str, float | str]]) -> str:
    scaling = horizon_scaling_from_rows(rows)
    aliases = {
        "log_d": "logd",
        "log": "logd",
        "linear": "d",
        "dimension": "d",
        "sqrtd": "sqrt_d",
        "sqrt": "sqrt_d",
        "const": "constant",
        "one": "constant",
        "1": "constant",
    }
    return aliases.get(scaling, scaling)


def plot_mean_std_by_dimension(
    ax,
    rows: list[dict[str, float | str]],
    value_key: str,
    line_label_template: str,
    show_std: bool = True,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> None:
    agg = grouped_mean_std(rows, ("d", "c"), value_key)
    by_d: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in agg:
        by_d[row["d"]].append(row)
    for d in sorted(by_d):
        part = sorted(by_d[d], key=lambda x: x["c"])
        xs = np.asarray([x["c"] for x in part], dtype=float)
        means = np.asarray([x[f"{value_key}_mean"] for x in part], dtype=float)
        stds = np.asarray([x[f"{value_key}_std"] for x in part], dtype=float)
        line = ax.plot(
            xs,
            means,
            marker="o",
            label=line_label_template.format(d=int(d)),
        )[0]
        if show_std:
            lower = means - stds
            upper = means + stds
            if clip_min is not None:
                lower = np.maximum(lower, clip_min)
            if clip_max is not None:
                upper = np.minimum(upper, clip_max)
            ax.fill_between(xs, lower, upper, color=line.get_color(), alpha=0.14)


def plot_mean_std_by_dimension_x(
    ax,
    rows: list[dict[str, float | str]],
    x_key: str,
    value_key: str,
    line_label_template: str,
    show_std: bool = True,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> None:
    agg = grouped_mean_std(rows, ("d", x_key), value_key)
    by_d: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in agg:
        by_d[row["d"]].append(row)
    for d in sorted(by_d):
        part = sorted(by_d[d], key=lambda x: x[x_key])
        xs = np.asarray([x[x_key] for x in part], dtype=float)
        means = np.asarray([x[f"{value_key}_mean"] for x in part], dtype=float)
        stds = np.asarray([x[f"{value_key}_std"] for x in part], dtype=float)
        line = ax.plot(
            xs,
            means,
            marker="o",
            label=line_label_template.format(d=int(d)),
        )[0]
        if show_std:
            lower = means - stds
            upper = means + stds
            if clip_min is not None:
                lower = np.maximum(lower, clip_min)
            if clip_max is not None:
                upper = np.minimum(upper, clip_max)
            ax.fill_between(xs, lower, upper, color=line.get_color(), alpha=0.14)


def plot_exact_recovery(
    graph_rows: list[dict[str, float | str]],
    figures_dir: Path,
    show_std: bool,
) -> None:
    x_label = horizon_axis_label(graph_rows)
    x_stem = horizon_filename_stem(graph_rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_mean_std_by_dimension(
        ax,
        graph_rows,
        "exact_success",
        "d={d}",
        show_std=show_std,
        clip_min=0.0,
        clip_max=1.0,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Empirical recovery probability")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(title="dimension d", ncol=1, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / f"exact_recovery_vs_{x_stem}.png", dpi=200)
    fig.savefig(figures_dir / f"exact_recovery_vs_{x_stem}.pdf")
    plt.close(fig)


def plot_screening_success(
    graph_rows: list[dict[str, float | str]],
    figures_dir: Path,
    show_std: bool,
) -> None:
    x_label = horizon_axis_label(graph_rows)
    x_stem = horizon_filename_stem(graph_rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_mean_std_by_dimension(
        ax,
        graph_rows,
        "screening_success_all_rows",
        "d={d}",
        show_std=show_std,
        clip_min=0.0,
        clip_max=1.0,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Empirical screening success probability")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(title="dimension d", ncol=1, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / f"screening_success_vs_{x_stem}.png", dpi=200)
    fig.savefig(figures_dir / f"screening_success_vs_{x_stem}.pdf")
    plt.close(fig)


def plot_exact_heatmap(graph_rows: list[dict[str, float | str]], figures_dir: Path) -> None:
    x_label = horizon_axis_label(graph_rows)
    agg = grouped_mean(graph_rows, ("d", "c"), "exact_success")
    ds = sorted({row["d"] for row in agg})
    cs = sorted({row["c"] for row in agg})
    values = np.full((len(ds), len(cs)), np.nan, dtype=float)
    d_index = {d: idx for idx, d in enumerate(ds)}
    c_index = {c: idx for idx, c in enumerate(cs)}
    for row in agg:
        values[d_index[row["d"]], c_index[row["c"]]] = row["exact_success"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(values, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(cs)))
    ax.set_xticklabels([f"{c:g}" for c in cs], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ds)))
    ax.set_yticklabels([str(int(d)) for d in ds])
    ax.set_xlabel(x_label)
    ax.set_ylabel("d")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("exact recovery probability")
    fig.tight_layout()
    fig.savefig(figures_dir / "exact_recovery_heatmap.png", dpi=200)
    fig.savefig(figures_dir / "exact_recovery_heatmap.pdf")
    plt.close(fig)


def summarize_ols_separation(
    row_rows: list[dict[str, float | str]],
    quantile_low: float,
    quantile_high: float,
) -> list[dict[str, float | str]]:
    needed = {"c", "min_active_coef", "max_inactive_coef", "tau"}
    if not row_rows or not needed.issubset(row_rows[0].keys()):
        return []

    buckets: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in row_rows:
        buckets[_row_experiment_id(row)].append(row)

    summary: list[dict[str, float | str]] = []
    for experiment_id, rows in buckets.items():
        first = rows[0]
        active = _finite_values(rows, "min_active_coef")
        inactive = _finite_values(rows, "max_inactive_coef")
        replicate_ids = {
            str(row.get("replicate_id", row.get("rep", "")))
            for row in rows
            if str(row.get("replicate_id", row.get("rep", ""))) != ""
        }
        summary.append(
            {
                "experiment_id": experiment_id,
                "d": float(first.get("d", math.nan)),
                "T": float(first.get("T", math.nan)),
                "c": float(first.get("c", math.nan)),
                "horizon_scaling": str(first.get("horizon_scaling", "logd")),
                "horizon_scale": float(first.get("horizon_scale", math.nan)),
                "alpha": float(first.get("alpha", math.nan)),
                "h": float(first.get("h", math.nan)),
                "R": float(first.get("R", math.nan)),
                "m": float(first.get("m", math.nan)),
                "threshold": float(first.get("tau", math.nan)),
                "active_q_low": float(np.quantile(active, quantile_low)) if active else math.nan,
                "active_median": float(np.median(active)) if active else math.nan,
                "active_q_high": float(np.quantile(active, quantile_high)) if active else math.nan,
                "inactive_q_low": float(np.quantile(inactive, quantile_low)) if inactive else math.nan,
                "inactive_median": float(np.median(inactive)) if inactive else math.nan,
                "inactive_q_high": float(np.quantile(inactive, quantile_high)) if inactive else math.nan,
                "n_replicates_used": float(len(replicate_ids)) if replicate_ids else math.nan,
            }
        )
    return sorted(summary, key=lambda x: (float(x["d"]), float(x["c"])))


def plot_ols_separation(
    row_rows: list[dict[str, float | str]],
    figures_dir: Path,
    results_dir: Path,
    quantile_low: float,
    quantile_high: float,
    show_bands: bool,
) -> None:
    agg = summarize_ols_separation(row_rows, quantile_low, quantile_high)
    if not agg:
        return
    write_csv_rows(results_dir / "ols_separation_summary.csv", agg)
    x_label = horizon_axis_label(row_rows)

    by_d: dict[float, list[dict[str, float | str]]] = defaultdict(list)
    for row in agg:
        by_d[float(row["d"])].append(row)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d in sorted(by_d):
        part = sorted(by_d[d], key=lambda x: float(x["c"]))
        xs = np.asarray([float(x["c"]) for x in part], dtype=float)
        active_line = ax.plot(
            xs,
            [float(x["active_median"]) for x in part],
            marker="o",
            linestyle="-",
            label=f"d={int(d)} active",
        )[0]
        ax.fill_between(
            xs,
            [float(x["active_q_low"]) for x in part],
            [float(x["active_q_high"]) for x in part],
            color=active_line.get_color(),
            alpha=0.14 if show_bands else 0.0,
            label=f"d={int(d)} active band"
            if show_bands
            else None,
        )
        ax.plot(
            xs,
            [float(x["inactive_median"]) for x in part],
            marker="s",
            linestyle="--",
            color=active_line.get_color(),
            label=f"d={int(d)} inactive",
        )
        if show_bands:
            ax.fill_between(
                xs,
                [float(x["inactive_q_low"]) for x in part],
                [float(x["inactive_q_high"]) for x in part],
                color=active_line.get_color(),
                alpha=0.08,
                label=f"d={int(d)} inactive band",
            )

    thresholds = sorted({float(x["threshold"]) for x in agg if np.isfinite(float(x["threshold"]))})
    for idx, threshold in enumerate(thresholds):
        ax.axhline(
            threshold,
            linestyle=":",
            color="black",
            linewidth=1,
            label="OLS threshold tau" if idx == 0 else None,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("local OLS coefficient summary over rows and replicates")
    ax.grid(True, alpha=0.25)
    ax.legend(title="dimension d / coefficient group", fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "ols_coefficient_separation.png", dpi=200)
    fig.savefig(figures_dir / "ols_coefficient_separation.pdf")
    plt.close(fig)


def plot_concentration(conc_rows: list[dict[str, float | str]], figures_dir: Path) -> None:
    if not conc_rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    scale_key = "theoretical_scale" if "theoretical_scale" in conc_rows[0] else "rate"
    dev_key = (
        "max_abs_F_deviation"
        if "max_abs_F_deviation" in conc_rows[0]
        else "max_F_deviation"
    )
    rates = np.asarray([float(row[scale_key]) for row in conc_rows], dtype=float)
    devs = np.asarray([float(row[dev_key]) for row in conc_rows], dtype=float)
    distinct = len({round(float(x), 12) for x in rates if np.isfinite(x)})
    print(f"concentration plot: {distinct} distinct theoretical_scale values")
    if distinct < 5:
        print("warning: fewer than 5 distinct concentration x-values")
    by_d_points: dict[float, list[int]] = defaultdict(list)
    for idx, row in enumerate(conc_rows):
        by_d_points[float(row.get("d", math.nan))].append(idx)
    for d in sorted(by_d_points):
        idxs = by_d_points[d]
        ax.scatter(
            rates[idxs],
            devs[idxs],
            alpha=0.65,
            label=f"d={int(d)} replicates",
        )
    by_scale: dict[float, list[float]] = defaultdict(list)
    by_d_scale: dict[tuple[float, float], list[float]] = defaultdict(list)
    for rate, dev in zip(rates, devs):
        if np.isfinite(rate) and np.isfinite(dev):
            by_scale[float(rate)].append(float(dev))
    for row, rate, dev in zip(conc_rows, rates, devs):
        d = float(row.get("d", math.nan))
        if np.isfinite(d) and np.isfinite(rate) and np.isfinite(dev):
            by_d_scale[(d, float(rate))].append(float(dev))
    for d in sorted({key[0] for key in by_d_scale}):
        xs_d = sorted(rate for d_key, rate in by_d_scale if d_key == d)
        ax.plot(
            xs_d,
            [float(np.median(by_d_scale[(d, x)])) for x in xs_d],
            marker="o",
            linewidth=1,
            label=f"d={int(d)} median",
        )
    xs_trend = sorted(by_scale)
    if xs_trend:
        ax.plot(
            xs_trend,
            [float(np.median(by_scale[x])) for x in xs_trend],
            marker="o",
            color="black",
            linewidth=1,
            label="pooled median trend",
        )
    mask = np.isfinite(rates) & np.isfinite(devs)
    if int(mask.sum()) >= 2:
        coef = np.polyfit(rates[mask], devs[mask], deg=1)
        xs = np.linspace(rates[mask].min(), rates[mask].max(), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="tab:red", linewidth=1, label="pooled linear fit")
    ax.set_xlabel("theoretical deviation scale")
    ax.set_ylabel("empirical max |F_hat - mean F_hat|")
    ax.grid(True, alpha=0.25)
    ax.legend(title="dimension d", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "screening_concentration_rate.png", dpi=200)
    fig.savefig(figures_dir / "screening_concentration_rate.pdf")
    plt.close(fig)


def plot_global_ols_margin(
    graph_rows: list[dict[str, float | str]],
    figures_dir: Path,
    show_std: bool,
) -> None:
    if not graph_rows or "global_ols_margin" not in graph_rows[0]:
        return
    x_label = horizon_axis_label(graph_rows)
    x_stem = horizon_filename_stem(graph_rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_mean_std_by_dimension(
        ax,
        graph_rows,
        "global_ols_margin",
        "d={d}",
        show_std=show_std,
    )
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel("mean global OLS margin over replicates")
    ax.grid(True, alpha=0.25)
    ax.legend(title="dimension d", ncol=1, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / f"global_ols_margin_vs_{x_stem}.png", dpi=200)
    fig.savefig(figures_dir / f"global_ols_margin_vs_{x_stem}.pdf")
    plt.close(fig)


def summarize_screening_gaps(
    screening_rows: list[dict[str, float | str]],
    quantile_low: float,
    quantile_high: float,
) -> list[dict[str, float | str]]:
    buckets: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in screening_rows:
        buckets[_row_experiment_id(row)].append(row)
    summary: list[dict[str, float | str]] = []
    for experiment_id, rows in buckets.items():
        first = rows[0]
        gaps = _finite_values(rows, "screening_gap_i")
        replicate_ids = {
            str(row.get("replicate_id", row.get("rep", "")))
            for row in rows
            if str(row.get("replicate_id", row.get("rep", ""))) != ""
        }
        summary.append(
            {
                "experiment_id": experiment_id,
                "d": float(first.get("d", math.nan)),
                "T": float(first.get("T", math.nan)),
                "c": float(first.get("c", math.nan)),
                "horizon_scaling": str(first.get("horizon_scaling", "logd")),
                "horizon_scale": float(first.get("horizon_scale", math.nan)),
                "alpha": float(first.get("alpha", math.nan)),
                "h": float(first.get("h", math.nan)),
                "R": float(first.get("R", math.nan)),
                "screening_gap_q_low": float(np.quantile(gaps, quantile_low)) if gaps else math.nan,
                "screening_gap_median": float(np.median(gaps)) if gaps else math.nan,
                "screening_gap_q_high": float(np.quantile(gaps, quantile_high)) if gaps else math.nan,
                "n_replicates_used": float(len(replicate_ids)) if replicate_ids else math.nan,
            }
        )
    return sorted(summary, key=lambda x: (float(x["d"]), float(x["c"])))


def plot_screening_gap_diagnostics(
    graph_rows: list[dict[str, float | str]],
    screening_rows: list[dict[str, float | str]],
    figures_dir: Path,
    results_dir: Path,
    quantile_low: float,
    quantile_high: float,
    show_bands: bool,
    show_std: bool,
) -> None:
    if not screening_rows:
        return
    x_label = horizon_axis_label(screening_rows)
    x_stem = horizon_filename_stem(screening_rows)
    summary = summarize_screening_gaps(screening_rows, quantile_low, quantile_high)
    write_csv_rows(results_dir / "screening_gap_summary.csv", summary)

    by_d: dict[float, list[dict[str, float | str]]] = defaultdict(list)
    for row in summary:
        by_d[float(row["d"])].append(row)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d in sorted(by_d):
        part = sorted(by_d[d], key=lambda x: float(x["c"]))
        xs = np.asarray([float(x["c"]) for x in part], dtype=float)
        ax.plot(
            xs,
            [float(x["screening_gap_median"]) for x in part],
            marker="o",
            label=f"d={int(d)}",
        )
        if show_bands:
            ax.fill_between(
                xs,
                [float(x["screening_gap_q_low"]) for x in part],
                [float(x["screening_gap_q_high"]) for x in part],
                alpha=0.15,
            )
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel("screening gap")
    ax.grid(True, alpha=0.25)
    ax.legend(title="dimension d", ncol=1, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / f"screening_gap_median_vs_{x_stem}.png", dpi=200)
    fig.savefig(figures_dir / f"screening_gap_median_vs_{x_stem}.pdf")
    plt.close(fig)

    if graph_rows and "min_screening_gap_over_rows" in graph_rows[0]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_mean_std_by_dimension(
            ax,
            graph_rows,
            "min_screening_gap_over_rows",
            "d={d}",
            show_std=show_std,
        )
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel(x_label)
        ax.set_ylabel("mean graph-level minimum screening gap over replicates")
        ax.grid(True, alpha=0.25)
        ax.legend(title="dimension d", ncol=1, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / f"screening_min_gap_vs_{x_stem}.png", dpi=200)
        fig.savefig(figures_dir / f"screening_min_gap_vs_{x_stem}.pdf")
        plt.close(fig)


def plot_gram_diagnostics(
    gram_rows: list[dict[str, float | str]],
    figures_dir: Path,
    results_dir: Path,
    quantile_low: float,
    quantile_high: float,
    show_bands: bool,
    show_std: bool,
) -> None:
    if not gram_rows:
        return
    x_label = horizon_axis_label(gram_rows)
    x_stem = horizon_filename_stem(gram_rows)
    success = grouped_mean_std(gram_rows, ("d", "c"), "gram_solve_success")
    eig = grouped_quantiles(
        gram_rows,
        ("d", "c"),
        "gram_min_eigenvalue",
        quantile_low,
        quantile_high,
    )
    cond = grouped_quantiles(
        gram_rows,
        ("d", "c"),
        "gram_condition_number",
        quantile_low,
        quantile_high,
    )
    write_csv_rows(results_dir / "gram_diagnostics_summary.csv", success + eig + cond)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    by_d: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in success:
        by_d[row["d"]].append(row)
    for d in sorted(by_d):
        part = sorted(by_d[d], key=lambda x: x["c"])
        xs = np.asarray([x["c"] for x in part], dtype=float)
        means = np.asarray([x["gram_solve_success_mean"] for x in part], dtype=float)
        stds = np.asarray([x["gram_solve_success_std"] for x in part], dtype=float)
        line = ax.plot(
            xs,
            means,
            marker="o",
            label=f"d={int(d)}",
        )[0]
        if show_std:
            ax.fill_between(
                xs,
                np.maximum(means - stds, 0.0),
                np.minimum(means + stds, 1.0),
                color=line.get_color(),
                alpha=0.14,
            )
    ax.set_xlabel(x_label)
    ax.set_ylabel("mean Gram solve success probability over row OLS problems")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(title="dimension d", ncol=1, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / f"gram_solve_success_vs_{x_stem}.png", dpi=200)
    fig.savefig(figures_dir / f"gram_solve_success_vs_{x_stem}.pdf")
    plt.close(fig)

    if eig:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        by_d_eig: dict[float, list[dict[str, float]]] = defaultdict(list)
        for row in eig:
            by_d_eig[row["d"]].append(row)
        for d in sorted(by_d_eig):
            part = sorted(by_d_eig[d], key=lambda x: x["c"])
            xs = np.asarray([x["c"] for x in part], dtype=float)
            line = ax.plot(
                xs,
                [x["gram_min_eigenvalue_median"] for x in part],
                marker="o",
                label=f"d={int(d)}",
            )[0]
            if show_bands:
                ax.fill_between(
                    xs,
                    [x["gram_min_eigenvalue_q_low"] for x in part],
                    [x["gram_min_eigenvalue_q_high"] for x in part],
                    color=line.get_color(),
                    alpha=0.14,
                )
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel(x_label)
        ax.set_ylabel("smallest Gram eigenvalue over row OLS problems")
        ax.grid(True, alpha=0.25)
        ax.legend(title="dimension d", ncol=1, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / f"gram_min_eigenvalue_vs_{x_stem}.png", dpi=200)
        fig.savefig(figures_dir / f"gram_min_eigenvalue_vs_{x_stem}.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        by_d_cond: dict[float, list[dict[str, float]]] = defaultdict(list)
        for row in cond:
            by_d_cond[row["d"]].append(row)
        for d in sorted(by_d_cond):
            part = sorted(by_d_cond[d], key=lambda x: x["c"])
            xs = np.asarray([x["c"] for x in part], dtype=float)
            line = ax.plot(
                xs,
                [x["gram_condition_number_median"] for x in part],
                marker="o",
                label=f"d={int(d)}",
            )[0]
            if show_bands:
                ax.fill_between(
                    xs,
                    [x["gram_condition_number_q_low"] for x in part],
                    [x["gram_condition_number_q_high"] for x in part],
                    color=line.get_color(),
                    alpha=0.14,
                )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Gram condition number over row OLS problems")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(title="dimension d", ncol=1, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / f"gram_condition_number_vs_{x_stem}.png", dpi=200)
        fig.savefig(figures_dir / f"gram_condition_number_vs_{x_stem}.pdf")
        plt.close(fig)


def plot_failure_modes(
    graph_rows: list[dict[str, float | str]],
    figures_dir: Path,
    results_dir: Path,
) -> None:
    if not graph_rows or "failure_mode" not in graph_rows[0]:
        return
    x_label = horizon_axis_label(graph_rows)
    modes = sorted({str(row["failure_mode"]) for row in graph_rows})
    buckets: dict[tuple[float, float], list[str]] = defaultdict(list)
    for row in graph_rows:
        buckets[(float(row["d"]), float(row["c"]))].append(str(row["failure_mode"]))
    summary: list[dict[str, float | str]] = []
    for (d, c), values in buckets.items():
        total = len(values)
        for mode in modes:
            summary.append(
                {
                    "d": d,
                    "c": c,
                    "horizon_scaling": horizon_scaling_from_rows(graph_rows),
                    "failure_mode": mode,
                    "fraction": values.count(mode) / float(total),
                    "n_replicates_used": total,
                }
            )
    write_csv_rows(results_dir / "failure_mode_summary.csv", summary)

    for d in sorted({float(row["d"]) for row in graph_rows}):
        cs = sorted({float(row["c"]) for row in graph_rows if float(row["d"]) == d})
        bottom = np.zeros(len(cs), dtype=float)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for mode in modes:
            vals = []
            for c in cs:
                match = [
                    row for row in summary
                    if float(row["d"]) == d and float(row["c"]) == c and row["failure_mode"] == mode
                ]
                vals.append(float(match[0]["fraction"]) if match else 0.0)
            ax.bar([f"{c:g}" for c in cs], vals, bottom=bottom, label=mode)
            bottom += np.asarray(vals, dtype=float)
        ax.set_xlabel(x_label)
        ax.set_ylabel("fraction of replicates")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"d={int(d)}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures_dir / f"failure_modes_d{int(d)}.png", dpi=200)
        fig.savefig(figures_dir / f"failure_modes_d{int(d)}.pdf")
        plt.close(fig)


def plot_parameter_sweeps(
    graph_rows: list[dict[str, float | str]],
    figures_dir: Path,
    results_dir: Path,
    show_std: bool,
) -> None:
    if not graph_rows or "sweep_parameter" not in graph_rows[0]:
        return
    x_label = horizon_axis_label(graph_rows)
    sweep_params = sorted(
        {
            str(row["sweep_parameter"])
            for row in graph_rows
            if str(row.get("sweep_parameter", "")) not in {"", "nan"}
        }
    )
    if not sweep_params:
        return

    summary_rows: list[dict[str, float | str]] = []
    metrics = [
        ("exact_success", "mean exact recovery probability"),
        ("screening_success_all_rows", "mean screening success probability"),
        ("global_ols_margin", "mean global OLS margin"),
    ]
    for sweep_param in sweep_params:
        rows_for_param = [
            row for row in graph_rows if str(row.get("sweep_parameter", "")) == sweep_param
        ]
        for d in sorted({float(row["d"]) for row in rows_for_param}):
            d_rows = [row for row in rows_for_param if float(row["d"]) == d]
            for metric, ylabel in metrics:
                if metric not in d_rows[0]:
                    continue
                buckets: dict[tuple[float, float], list[float]] = defaultdict(list)
                for row in d_rows:
                    value = float(row[metric])
                    if np.isfinite(value):
                        buckets[(float(row["sweep_value"]), float(row["c"]))].append(value)
                if not buckets:
                    continue
                fig, ax = plt.subplots(figsize=(7, 4.5))
                for sweep_value in sorted({key[0] for key in buckets}):
                    cs = sorted(c for value, c in buckets if value == sweep_value)
                    means = []
                    stds = []
                    ns = []
                    for c in cs:
                        arr = np.asarray(buckets[(sweep_value, c)], dtype=float)
                        means.append(float(np.mean(arr)))
                        stds.append(float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0)
                        ns.append(int(arr.size))
                        summary_rows.append(
                            {
                                "sweep_parameter": sweep_param,
                                "sweep_value": sweep_value,
                                "d": d,
                                "c": c,
                                "horizon_scaling": horizon_scaling_from_rows(graph_rows),
                                "metric": metric,
                                "mean": means[-1],
                                "std": stds[-1],
                                "n_replicates_used": ns[-1],
                            }
                        )
                    xs = np.asarray(cs, dtype=float)
                    means_arr = np.asarray(means, dtype=float)
                    stds_arr = np.asarray(stds, dtype=float)
                    line = ax.plot(
                        xs,
                        means_arr,
                        marker="o",
                        label=f"{sweep_param}={sweep_value:g}",
                    )[0]
                    if show_std:
                        lower = means_arr - stds_arr
                        upper = means_arr + stds_arr
                        if metric.endswith("success"):
                            lower = np.maximum(lower, 0.0)
                            upper = np.minimum(upper, 1.0)
                        ax.fill_between(xs, lower, upper, color=line.get_color(), alpha=0.14)
                ax.set_xlabel(x_label)
                ax.set_ylabel(ylabel)
                ax.set_title(f"{sweep_param} sweep, d={int(d)}")
                if metric.endswith("success"):
                    ax.set_ylim(-0.03, 1.03)
                ax.grid(True, alpha=0.25)
                ax.legend(title=f"Fixed d={int(d)} sweep values", fontsize=8)
                fig.tight_layout()
                stem = f"sweep_{sweep_param}_{metric}_d{int(d)}"
                fig.savefig(figures_dir / f"{stem}.png", dpi=200)
                fig.savefig(figures_dir / f"{stem}.pdf")
                plt.close(fig)
    write_csv_rows(results_dir / "parameter_sweep_summary.csv", summary_rows)


def plot_diagnostic_t_sweep(
    diag_rows: list[dict[str, float | str]],
    figures_dir: Path,
    show_std: bool,
) -> None:
    t_rows = [row for row in diag_rows if row.get("diagnostic") == "T_sweep"]
    if not t_rows:
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot_mean_std_by_dimension_x(
        ax,
        t_rows,
        "T",
        "exact_success",
        "d={d} exact",
        show_std=show_std,
        clip_min=0.0,
        clip_max=1.0,
    )
    plot_mean_std_by_dimension_x(
        ax,
        t_rows,
        "T",
        "row_success_rate",
        "d={d} row",
        show_std=show_std,
        clip_min=0.0,
        clip_max=1.0,
    )
    ax.set_xlabel("T")
    ax.set_ylabel("mean recovery probability over replicates")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(title="dimension d / metric", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "diagnostic_recovery_vs_T.png", dpi=200)
    fig.savefig(figures_dir / "diagnostic_recovery_vs_T.pdf")
    plt.close(fig)


def plot_intensity_diagnostic(
    diag_rows: list[dict[str, float | str]],
    figures_dir: Path,
    show_std: bool,
) -> None:
    if not diag_rows or "stationary_intensity_rel_l2_error" not in diag_rows[0]:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot_mean_std_by_dimension_x(
        ax,
        diag_rows,
        "T",
        "stationary_intensity_rel_l2_error",
        "d={d}",
        show_std=show_std,
    )
    ax.set_xlabel("T")
    ax.set_ylabel("mean relative L2 error in stationary intensity over replicates")
    ax.grid(True, alpha=0.25)
    ax.legend(title="dimension d", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "diagnostic_stationary_intensity.png", dpi=200)
    fig.savefig(figures_dir / "diagnostic_stationary_intensity.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--config", default=None, help="Optional config path; uses its name field.")
    parser.add_argument("--name", default=None, help="Named results subfolder to plot.")
    parser.add_argument("--graph-csv", default="scaling_graph.csv")
    parser.add_argument("--rows-csv", default="scaling_rows.csv")
    parser.add_argument("--screening-rows-csv", default="scaling_screening_rows.csv")
    parser.add_argument("--gram-rows-csv", default="scaling_gram_rows.csv")
    parser.add_argument("--concentration-csv", default="concentration_replicates.csv")
    parser.add_argument("--quantile-low", type=float, default=0.10)
    parser.add_argument("--quantile-high", type=float, default=0.90)
    parser.add_argument("--show-bands", type=parse_bool, default=True)
    parser.add_argument("--show-std", type=parse_bool, default=True)
    args = parser.parse_args()

    results_dir = resolve_named_results_dir(Path(args.results_dir), args.name, args.config)
    figures_dir = Path(args.figures_dir) / results_dir.name if (args.name or args.config) else Path(args.figures_dir)
    _ensure_dir(figures_dir)

    graph_rows = read_csv_rows(results_dir / args.graph_csv)
    row_rows = read_csv_rows(results_dir / args.rows_csv)
    plot_exact_recovery(graph_rows, figures_dir, show_std=False)
    plot_screening_success(graph_rows, figures_dir, show_std=False)
    plot_exact_heatmap(graph_rows, figures_dir)
    plot_ols_separation(
        row_rows,
        figures_dir,
        results_dir,
        quantile_low=args.quantile_low,
        quantile_high=args.quantile_high,
        show_bands=args.show_bands,
    )
    plot_global_ols_margin(graph_rows, figures_dir, show_std=args.show_std)
    plot_failure_modes(graph_rows, figures_dir, results_dir)
    plot_parameter_sweeps(graph_rows, figures_dir, results_dir, show_std=args.show_std)

    screening_path = results_dir / args.screening_rows_csv
    if screening_path.exists():
        plot_screening_gap_diagnostics(
            graph_rows,
            read_csv_rows(screening_path),
            figures_dir,
            results_dir,
            quantile_low=args.quantile_low,
            quantile_high=args.quantile_high,
            show_bands=args.show_bands,
            show_std=args.show_std,
        )

    gram_path = results_dir / args.gram_rows_csv
    if gram_path.exists():
        plot_gram_diagnostics(
            read_csv_rows(gram_path),
            figures_dir,
            results_dir,
            quantile_low=args.quantile_low,
            quantile_high=args.quantile_high,
            show_bands=args.show_bands,
            show_std=args.show_std,
        )

    conc_path = results_dir / args.concentration_csv
    if conc_path.exists():
        plot_concentration(read_csv_rows(conc_path), figures_dir)
    else:
        legacy_conc_path = results_dir / "diagnostics_concentration.csv"
        if legacy_conc_path.exists():
            plot_concentration(read_csv_rows(legacy_conc_path), figures_dir)

    diag_path = results_dir / "diagnostics_graph.csv"
    if diag_path.exists():
        diag_rows = read_csv_rows(diag_path)
        plot_diagnostic_t_sweep(diag_rows, figures_dir, show_std=args.show_std)
        plot_intensity_diagnostic(diag_rows, figures_dir, show_std=args.show_std)

    print(f"wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
