from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def resolve_named_results_dir(results_dir: Path, name: str | None, config: str | None) -> Path:
    resolved_name = name
    if resolved_name is None and config is not None:
        with open(config, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        resolved_name = str(cfg.get("name") or Path(config).stem)
    if resolved_name:
        return results_dir / resolved_name
    return results_dir


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write for {path}")
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


def parse_float(value: Any, column: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"column {column!r} contains a non-numeric value: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"column {column!r} contains a non-finite value: {value!r}")
    return parsed


def parse_optional_float(value: Any) -> float | None:
    text = str(value).strip()
    if text in {"", "nan", "None"}:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def nonempty_string(value: Any) -> str:
    text = str(value).strip()
    return "" if text in {"", "nan", "None"} else text


def format_pivot_value(value: Any) -> Any:
    parsed = parse_optional_float(value)
    if parsed is None:
        return nonempty_string(value)
    return int(parsed) if parsed.is_integer() else parsed


def c_column_name(c_value: float) -> str:
    return f"c_{c_value:g}".replace("-", "m").replace(".", "p")


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


def exact_recovery_tables(
    graph_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    required = {"d", "c", "exact_success"}
    if not graph_rows:
        raise ValueError("graph CSV is empty")
    missing = required.difference(graph_rows[0])
    if missing:
        raise ValueError(f"graph CSV is missing required columns: {sorted(missing)}")

    has_sweep = any(nonempty_string(row.get("sweep_parameter", "")) for row in graph_rows)
    buckets: dict[tuple[tuple[Any, ...], float], list[float]] = {}
    row_meta: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in graph_rows:
        d_value = parse_float(row["d"], "d")
        c_value = parse_float(row["c"], "c")
        exact_success = parse_float(row["exact_success"], "exact_success")
        d_out: int | float = int(d_value) if d_value.is_integer() else d_value
        if has_sweep:
            sweep_parameter = nonempty_string(row.get("sweep_parameter", ""))
            sweep_value = format_pivot_value(row.get("sweep_value", ""))
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
        sweep_numeric = parse_optional_float(row_meta[key].get("sweep_value"))
        return (
            str(row_meta[key].get("sweep_parameter", "")),
            sweep_numeric if sweep_numeric is not None else math.inf,
            str(row_meta[key].get("sweep_value", "")),
            float(row_meta[key]["d"]),
        )

    row_keys = sorted(row_meta, key=row_sort_key)
    c_values = sorted({c for _, c in buckets})
    mean_rows: list[dict[str, Any]] = []
    std_rows: list[dict[str, Any]] = []
    wilson_rows: list[dict[str, Any]] = []

    for row_key in row_keys:
        mean_row: dict[str, Any] = dict(row_meta[row_key])
        std_row: dict[str, Any] = dict(row_meta[row_key])
        wilson_row: dict[str, Any] = dict(row_meta[row_key])
        for c_value in c_values:
            column = c_column_name(c_value)
            values = buckets.get((row_key, c_value))
            if values is None:
                mean_row[column] = ""
                std_row[column] = ""
                wilson_row[f"{column}_lower"] = ""
                wilson_row[f"{column}_upper"] = ""
                wilson_row[f"{column}_n"] = ""
                continue
            arr = np.asarray(values, dtype=float)
            mean_row[column] = float(np.mean(arr))
            std_row[column] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            lower, upper = wilson_interval(successes=float(np.sum(arr)), n=int(arr.size))
            wilson_row[f"{column}_lower"] = lower
            wilson_row[f"{column}_upper"] = upper
            wilson_row[f"{column}_n"] = int(arr.size)
        mean_rows.append(mean_row)
        std_rows.append(std_row)
        wilson_rows.append(wilson_row)

    return mean_rows, std_rows, wilson_rows


def resolve_graph_csv(results_dir: Path, graph_csv: str | None) -> Path:
    if graph_csv is not None:
        path = results_dir / graph_csv
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    default = results_dir / "scaling_graph.csv"
    if default.exists():
        return default

    candidates = sorted(results_dir.glob("*_graph.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"no *_graph.csv file found in {results_dir}")
    names = ", ".join(path.name for path in candidates)
    raise ValueError(f"multiple graph CSVs found in {results_dir}: {names}. Pass --graph-csv.")


def infer_prefix(graph_path: Path, output_prefix: str | None) -> str:
    if output_prefix:
        return output_prefix
    stem = graph_path.stem
    return stem[:-6] if stem.endswith("_graph") else stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export exact-recovery mean/std pivot tables from a scaling graph CSV."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default=None, help="Optional config path; uses its name field.")
    parser.add_argument("--name", default=None, help="Named results subfolder to read.")
    parser.add_argument(
        "--graph-csv",
        default=None,
        help="Graph-level CSV inside the results folder. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output filename prefix. Defaults to the graph CSV prefix.",
    )
    parser.add_argument("--mean-csv", default=None, help="Override mean table filename.")
    parser.add_argument("--std-csv", default=None, help="Override std table filename.")
    parser.add_argument("--wilson-csv", default=None, help="Override Wilson 95% interval table filename.")
    args = parser.parse_args()

    results_dir = resolve_named_results_dir(Path(args.results_dir), args.name, args.config)
    graph_path = resolve_graph_csv(results_dir, args.graph_csv)
    prefix = infer_prefix(graph_path, args.output_prefix)

    mean_path = results_dir / (args.mean_csv or f"{prefix}_exact_recovery_mean_by_d_c.csv")
    std_path = results_dir / (args.std_csv or f"{prefix}_exact_recovery_std_by_d_c.csv")
    wilson_path = results_dir / (
        args.wilson_csv or f"{prefix}_exact_recovery_wilson95_by_d_c.csv"
    )

    mean_rows, std_rows, wilson_rows = exact_recovery_tables(read_csv_rows(graph_path))
    write_csv_rows(mean_path, mean_rows)
    write_csv_rows(std_path, std_rows)
    write_csv_rows(wilson_path, wilson_rows)

    print(f"read {graph_path}")
    print(f"wrote {mean_path}")
    print(f"wrote {std_path}")
    print(f"wrote {wilson_path}")


if __name__ == "__main__":
    main()
