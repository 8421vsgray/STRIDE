# NEWSTRIDE

NEWSTRIDE is a clean, YAML-only entry point for the two-view STRIDE protocol.
It is a new project directory and does not modify the historical `STRIDE`
project or any of its configurations.

## Data contract

The loader follows PLCI's correspondence-noise procedure:

1. load the two selected views;
2. normalize the clean views;
3. with `RandomState(1234)`, select `int(m_ratio * N)` rows;
4. shuffle only the second input view within that selected subset;
5. update the second-view labels and IDs with the same source permutation.

The fixed/disturbed view mapping is explicit in each YAML file:

| Dataset | fixed input | disturbed input |
|---|---|---|
| Scene15 | original `X[1]` (59-D) | original `X[0]` (20-D) |
| Handwritten | original `X[0]` Pixel (240-D) | original `X[1]` Fourier (76-D) |
| CCV20 | original `X[1]` | original `X[2]` |
| UMPC-Food101 | `img_features` (768-D) | `txt_features` (768-D) |

`data_X[0]` is always the fixed anchor and `data_X[1]` is always the
disturbed view.  This removes the old ambiguity between MATLAB indices and
input positions.

## Run

All model and experiment choices live in YAML.  The only command-line
override is the seed list, so one YAML remains the single source of truth.

```powershell
python main_train.py --config_file config/scene15.yaml
python main_train.py --config_file config/handwritten.yaml
python main_train.py --config_file config/ccv20.yaml
python main_train.py --config_file config/food101.yaml

# One seed: uses the seed declared in YAML (seed: 1).
python main_train.py --config_file config/scene15.yaml

# Multi-seed validation: the same YAML is reused for every seed.
python main_train.py --config_file config/scene15.yaml --seeds 1 2 3 4 5
```

For a single seed, results are written to the YAML `output_dir`.  For a
multi-seed run, each seed gets an isolated directory below `output_dir/`, and
the parent directory receives `summary.csv` with the ACC/NMI/ARI/CAR rows.
The old `train_time` YAML field is intentionally unsupported; use `--seeds`
for a multi-seed run.

Put the matching `.mat` files under `NEWSTRIDE/data/`, or change only the
`data_path` field in the selected YAML file.

The protocol is two-stage: diagonal point CE for the first 20 epochs, then
detached dense Euclidean-GW point supervision.  Edge-CE and Gaussian-mixture
guidance are disabled in these migrated configurations.

The relation-flow contract is explicit: the source graph is built from the
online decoder output (P), while the target graph is built from the EMA
encoder output (Z_t).  There is no decoder-EMA branch in NEWSTRIDE.

Food101 expects `umpc_food101.mat` with `img_features`, `txt_features`, and
`labels` fields.

## Project boundary

The dense-GW model/solver is copied into this project so experiments are
self-contained.  The original `STRIDE` directory is not imported at runtime
and is not modified by NEWSTRIDE commands.


## Transport fallback policy

The prior-preserving gate uses 0.1% positive off-diagonal mass as the
minimum useful diffusion threshold. If either direction in the first
post-warmup batch of an epoch falls below this threshold, that epoch is
marked as prior-preserving and all later batches return the diagonal teacher.
The numerical projector remains in transport_solver.py; model.py only
delegates relation construction and consumes the returned teacher.
