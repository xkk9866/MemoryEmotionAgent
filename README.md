# MemoryEmotionAgent / RPI

Code for session-level support-strategy adaptation from previously observed
actions, with residual, structured and count-based variants.

## Code map

- `cate/rla.py`: residual logit correction underlying RPI; the internal module
  retains the RLA name used by the original experiment runners.
- `cate/spire.py`: structured session adaptation.
- `cate/hrpi.py`: historical-count and residual adaptation.
- `cate/crossfit.py`: cross-fitted backbone predictions.
- `cate/session_specificity.py`, `cate/resplit*.py`: session and partition controls.
- Other `cate/` modules: classifiers, baselines, evaluation and graph-memory code.
- `scripts/`: batch jobs and optional scheduling utilities.

## Environment and data

Use Python 3.10 or newer and a suitable PyTorch/CUDA installation:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt
```

Set `CATE_DATA` to the local data/artifact directory and `HF_HOME` to the
Hugging Face cache. The official-split loader expects:

- `$CATE_DATA/ext/EmoDynamiX-v2/data/` for ESConv and AnnoMI.
- `$CATE_DATA/ext/p4g/300_dialog.xlsx` for PersuasionForGood.

Obtain the [EmoDynamiX-v2](https://github.com/cw-wan/EmoDynamiX-v2) data layout
and other datasets from their original providers. Data and model weights are
not bundled.

## Main entry points

```bash
python -m cate.crossfit --help
python -m cate.rla_analysis --help
python -m cate.spire --help
```

A single backbone fold, after data preparation:

```bash
python -m cate.crossfit --dataset esconv --encoder roberta-base --tag roberta-base --seed 42 --fold 0
```

Generate the complete fold/seed collection expected by each downstream runner
before running adaptation summaries. RPI consumes independently observed past
strategy labels, ordered within a session.

## Optional remote scheduler

`scripts/dispatch.py` reads connection details from `REMOTE_HOST`,
`REMOTE_PORT`, `REMOTE_USER`, and optionally `REMOTE_PASSWORD`.
Without a password it can use Paramiko's normal SSH key authentication.
Set `REMOTE_PROJECT_DIR` and `REMOTE_PYTHON` for the remote installation.
No server address or password is included in this public copy.

## Release contents

This repository contains the project source code, training and evaluation
entry points, baseline implementations, and available tests. Experiment and
metric-analysis code is retained. Manuscripts, downloaded reference papers,
editorial notes, publication figures and LaTeX-table exporters are not included.

Datasets, model weights, local prediction caches, logs, and machine credentials
are not distributed with this release. Obtain datasets and model weights from
their original providers and follow their respective access and usage terms.
Configuration defaults in some experimental scripts refer to the original
Windows or Linux workspace; use the documented environment variables and
command-line options, and review path constants before a full run.

The release was checked for Python syntax and accidental inclusion of manuscript
files and embedded credentials. This publication check does not rerun GPU
training or certify that every experimental entry point is turnkey.

## Repository status

Public code repository: https://github.com/xkk9866/MemoryEmotionAgent

The public release is a code snapshot. Dataset and checkpoint availability is
separate from source-code availability.
