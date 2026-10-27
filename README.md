# MemoryEmotionAgent / RPI

RPI adapts a strategy classifier to the current dialogue using previously observed actions. A regularized correction updates the classifier logits as session history becomes available. This repository contains the backbone training pipeline, RPI, structured adaptation with SPIRE and historical-count variants.

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

The shell workers use `python` from the active environment. Set `PYTHON_BIN`
to select a different interpreter.

## Optional remote scheduler

`scripts/dispatch.py` reads connection details from `REMOTE_HOST`,
`REMOTE_PORT`, `REMOTE_USER`, and optionally `REMOTE_PASSWORD`.
Without a password it can use Paramiko's normal SSH key authentication.
Set `REMOTE_PROJECT_DIR` and `REMOTE_PYTHON` for the remote installation.
Supply these values in your local environment. Keep connection settings and
SSH keys outside version control. The scheduler is optional; the commands above
run locally.

## Working with session predictions

Cross-fitting saves feature and prediction arrays for each dataset, encoder,
split, seed and fold. Build the complete collection required by the selected
adaptation runner before evaluating it.

Preserve dialogue order and use past observed strategy labels as the adaptation
history. The session-transfer and repartitioning scripts provide separate
controls for the source of that history and the choice of train/test partition.

## Data and configuration

Download benchmark datasets and pretrained models from their original providers
and follow their usage terms. Data, checkpoints and generated predictions are
stored locally and are not included in this repository.

Set the paths described above before running experiments. Some auxiliary scripts
use fixed local path defaults; adjust those paths for your installation.
The command examples assume execution from the repository root.
