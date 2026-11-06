# On Minimal Observation Time for Recovering Latent Hawkes Networks

This repository implements the illustration from the paper [On Minimal Observation Time for Recovering Latent Hawkes Networks](https://arxiv.org/abs/2605.08400). It illustrates the $\log d$ observation time scaling in a reproducible numerical way using the bounded-block screen-and-regress support-recovery method. It uses heavy parallelization across CPU threads.

## Requirements

Run the scripts from the repository root with a Python environment containing:

```bash
numpy matplotlib pyyaml
```

## $\log d$ Scaling Experiment

Run the main experiment:

```bash
python run_scaling_experiment.py --config configs/scaling_logd.yaml
```


Each config has a `name` field. Results are written under:

```text
results/<name>/
```

For example, `scaling_logd.yaml` writes to `results/scaling_logd/`.

Raw outputs for the default `scaling_logd` run:

- `results/scaling_logd/scaling_graph.csv`
- `results/scaling_logd/scaling_rows.csv`
- `results/scaling_logd/scaling_screening_rows.csv`
- `results/scaling_logd/scaling_gram_rows.csv`
- `results/scaling_logd/scaling_failed_edges.csv`
- `results/scaling_logd/scaling_metric_summary.csv`
- `results/scaling_logd/scaling_exact_recovery_mean_by_d_c.csv`
- `results/scaling_logd/scaling_exact_recovery_std_by_d_c.csv`
- `results/scaling_logd/scaling_exact_recovery_wilson95_by_d_c.csv`

Every replicate records its graph seed and simulation seed explicitly.
For each graph repetition, the support is sampled row-wise, independently conditional on the graph seed.

The default is `n_jobs: 100`; override it in YAML or on the command line:

```bash
python run_scaling_experiment.py \
  --config configs/scaling_logd.yaml \
  --n-jobs 32
```

For faster exploratory runs, copy the YAML and reduce `dimensions`, `c_values`,
or `n_reps`. 

For a quick execution check, including multiprocessing and CSV streaming, run:

```bash
python run_scaling_experiment.py \
  --config configs/scaling_smoke.yaml
```
## Figures

After running the scaling experiment, generate the figures with:

```bash
python plot_results.py
```

To plot a named run, either pass the config or the name:

```bash
python plot_results.py --config configs/scaling_logd.yaml
```

```bash
python plot_results.py --name scaling_logd
```

Figures are written under `figures/<name>/` for named runs.

## Tables
To generate only the exact-recovery tables from an existing results folder without making plots:

```bash
python export_exact_recovery_tables.py --config configs/scaling_logd.yaml
```

This writes the mean table, standard-deviation table, and 95% Wilson binomial interval table.

If a result folder contains a non-default graph CSV name, pass it explicitly:

```bash
python export_exact_recovery_tables.py \
  --name scaling_smoke \
  --graph-csv scaling_smoke_graph.csv
```
## Paper
```bash
@inproceedings{
anonymous2026on,
title={On Minimal Observation Time for Recovering Latent Hawkes Networks},
author={Jonas Linkerhägner and Michele Bortolasi and Lorenzo Baldassari and Maarten V. de Hoop and Ivan Dokmanić},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=tZwHSbbaof}
}
```