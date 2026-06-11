# MS-LIF-TCN-HAR

Reproducible PyTorch implementation for:

**MS-LIF-TCN: Causal Cross-Window Spiking Temporal Modeling for Wearable Human Activity Recognition**

This repository provides the public code release for spiking temporal-context
modeling for wearable human activity recognition (HAR). The main model,
`MS-LIF-TCN`, encodes
each inertial window with a multi-scale leaky integrate-and-fire (LIF) spiking
encoder and applies a causal window-level temporal convolutional network (TCN)
over spike-rate window representations.

Repository URL:

```text
https://github.com/YinDaYe258/MS-LIF-TCN-HAR
```

Public code release tag:

```text
v1.0-code-release
```

## What Is Included

- PyTorch implementations of the HAR baselines and MS-LIF-TCN.
- Dataset loaders and reproducible subject-disjoint split handling.
- Training, diagnostic, and analysis scripts.
- Forward-shape and analysis tests.

The manuscript text, LaTeX source, submission PDFs, paper figures, and paper
tables are intentionally not included in this public code repository. Raw
third-party datasets are also not redistributed. This repository contains the
code, configuration files, and scripts needed to reproduce the experiments once
the public datasets are downloaded from their original providers.

## Main Evaluation Protocol

The main protocol evaluates six model families on four public wearable HAR
datasets:

```text
datasets: ucihar, hapt6, pamap2, mhealth
models: cnn1d, window_gru, ms_lif_snn, ms_lif_snn_wide, ms_ann_tcn, ms_lif_tcn
seeds: 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
context_len: 8
target_mode: last
```

The sequence input and final-window objective are:

```text
x: [B, K, T, C]
y: [B, K]
loss: CE(logits[:, -1, :], y[:, -1]) + spike_reg_lambda * spike_rate
```

Subject-disjoint splits are used whenever subject IDs are available. Random
window-level train/test splitting is not used for the main evidence package.

## Repository Layout

```text
configs/                         Experiment configuration files.
data/                            Local dataset root; raw data are not tracked.
results/                         Local output directory for generated experiment files.
scripts/                         Training, diagnostics, and analysis entry points.
src/                             Dataset, model, training, and analysis code.
tests/                           Forward-shape and analysis tests.
```

## Installation

```bash
pip install -r requirements.txt
```

## Basic Checks

Run the test suite:

```bash
python -m pytest -q
```

Run a smoke test:

```bash
python scripts/run_ucihar_baselines.py --config configs/ucihar_smoke.yaml --model cmg_lif --smoke_test
```

Smoke tests are code-path checks only. Any synthetic or smoke-test rows are
marked in the output CSVs and are not used as real experimental evidence.

## Reproducing Experiments

After downloading the datasets locally, the main experiment runner can be used as
follows:

```bash
python scripts/run_final_paper_v3.py --datasets ucihar --models main --seeds 42 43 44 45 46 47 48 49 50 51
```

Repeat the command for `hapt6`, `pamap2`, and `mhealth`.
Generated CSV files, figures, checkpoints, logs, and submission materials are
local runtime artifacts and are not tracked in this public code release.

## Scope of the Reported Results

The manuscript reports algorithmic resource descriptors, including parameter
count, dense MAC proxies, spike rates, SynOps proxies, model-size proxies, and
activation-memory proxies. It does not report direct hardware energy
measurements.

## Citation

A formal citation and release DOI will be added after journal acceptance or
archive release.
