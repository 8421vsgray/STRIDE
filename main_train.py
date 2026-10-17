"""YAML-only NEWSTRIDE training entry point."""

import argparse
import copy
import csv
import datetime
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

import utils
from data_loader import load_dataset
from engine_train import train_one_epoch
from model import OursModel


PROJECT_ROOT = Path(__file__).resolve().parent


DEFAULTS = {
    "n_views": 2, "epochs": 200, "warmup_epochs": 20,
    "batch_size": 512,
    "threads": 8, "num_workers": 0, "print_freq": 5,
    "data_norm": "standard", "weight_decay": 0.0,
    "momentum": 0.98,
    "temperature": 0.5, "drop_rate": 0.2, "decoder_drop_rate": 0.0,
    "blr": 5.0e-4, "min_blr": 5.0e-5,
    # Public transport controls. Numerical solver settings are owned by
    # transport_projector.py and are not valid YAML knobs.
    "init": "diagonal", "steps": 5, "step_size": 0.02,
    "structure_diagonal_only_teacher": False,
    "structure_oracle_true_diagonal": False,
    "structure_oracle_tii_threshold": 0.1,
    "structure_oracle_quality_weighted": False,
    "structure_debug_true_quality": False,
    "qce_enabled": False, "qce_nu": 2.0, "qce_normalize": False,
    "qce_reduction": "batch_mean", "qce_probability_mode": "global",
    "qce_identity_target": False, "qce_disable_structure": False,
    "lambda_intra_qce": 1.0, "lambda_inter_qce": 1.0,
    "m_ratio": 0.0, "c_ratio": 0.0, "device": "cuda:0",
    "pin_mem": True, "save_ckpt": False, "output_dir": "output",
}


def load_config(config_file):
    path = Path(config_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / "config" / path
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError("The YAML root must be a mapping")
    if "train_time" in values:
        raise ValueError(
            "train_time is no longer a YAML option; use "
            "--seeds 1 2 3 4 5 for multi-seed runs"
        )
    unknown = sorted(set(values) - set(DEFAULTS) - {
        "dataset", "data_path", "encoder_dim", "n_classes", "seed",
        "view_indices", "experiment_id", "config_name",
    })
    if unknown:
        raise ValueError(f"Unknown YAML keys: {unknown}")
    merged = dict(DEFAULTS)
    merged.update(values)
    for key in ("dataset", "data_path", "encoder_dim", "n_classes", "seed"):
        if key not in merged or merged[key] is None:
            raise ValueError(f"Missing required YAML key: {key}")
    merged["data_path"] = str((PROJECT_ROOT / merged["data_path"]).resolve()) if not Path(merged["data_path"]).is_absolute() else str(Path(merged["data_path"]).resolve())
    merged["output_dir"] = str((PROJECT_ROOT / merged["output_dir"]).resolve()) if not Path(merged["output_dir"]).is_absolute() else str(Path(merged["output_dir"]).resolve())
    merged["config_file"] = str(path.resolve())
    return SimpleNamespace(**merged)


def _prepare_runtime_args(args):
    """Resolve values derived from the YAML once per training invocation."""
    args.embed_dim = args.encoder_dim[0][-1]
    args.lr = args.blr * args.batch_size / 256.0
    args.min_lr = args.min_blr * args.batch_size / 256.0
    if args.min_lr > args.lr:
        raise ValueError("min_blr must not exceed blr")


def _seed_output_dir(base_output, seed, multi_seed):
    base_output = Path(base_output)
    return base_output / f"seed{seed}" if multi_seed else base_output


def _write_summary(summary_path, rows):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["seed", "epoch", "acc", "nmi", "ari", "car"]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_one_time(args, logger, train_id=0):
    utils.fix_random_seeds(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(device)

    train_set, test_set = load_dataset(args)
    args.n_sample_test = len(test_set)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=min(args.batch_size, len(train_set)),
        shuffle=True, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=min(512, len(test_set)), shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
    )

    model = OursModel(
        n_views=args.n_views, layer_dims=args.encoder_dim,
        temperature=args.temperature, n_classes=args.n_classes,
        drop_rate=args.drop_rate, decoder_drop_rate=args.decoder_drop_rate,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )
    scheduler = utils.Ratioschedule(
        args.warmup_epochs, args.epochs, args.lr, args.min_lr,
        schedule_type="exp", gamma=0.98,
    )

    logger.write(f"Data source: {args.data_source}")
    logger.write(f"Effective config: {vars(args)}")
    states = {}
    for epoch in range(args.epochs):
        args.print_this_epoch = (epoch + 1) % args.print_freq == 0 or epoch + 1 == args.epochs
        state = train_one_epoch(
            model, train_loader, test_loader, optimizer, device, epoch,
            scheduler, args,
        )
        if state is not None:
            states[epoch] = state
        if args.print_this_epoch and state is not None:
            result = state["k-means"]
            logger.write(
                f"Epoch {epoch} --------------------------------\n"
                f"K-means  : ACC_k = {result['acc']:.4f} "
                f"NMI_k = {result['nmi']:.4f} ARI_k = {result['ari']:.4f} "
                f"F_k = {result['f']:.4f} CAR = {result['car']:.4f}"
            )
    if args.save_ckpt:
        torch.save(model.state_dict(), Path(args.output_dir) / f"checkpoint_{args.epochs}.pt")
    return states


def main(args, seeds=None):
    seeds = [int(args.seed)] if seeds is None else [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds are not allowed: {seeds}")

    _prepare_runtime_args(args)
    base_output = Path(args.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    multi_seed = len(seeds) > 1
    summary_rows = []
    started = time.time()

    for seed in seeds:
        run_args = copy.deepcopy(args)
        run_args.seed = seed
        run_args.output_dir = str(_seed_output_dir(base_output, seed, multi_seed))
        run_args.visual_dir = str(Path(run_args.output_dir) / "visualize")
        Path(run_args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(run_args.visual_dir).mkdir(parents=True, exist_ok=True)

        logger = utils.FileLogger(str(Path(run_args.output_dir) / "log_train.txt"))
        logger.write(f"NEWSTRIDE config: {run_args.config_file}")
        logger.write(f"Seed: {seed}")
        logger.write(f"Start time: {datetime.datetime.now().isoformat(timespec='seconds')}")
        seed_started = time.time()
        states = train_one_time(run_args, logger, train_id=0)
        for epoch, state in sorted(states.items()):
            result = state["k-means"]
            summary_rows.append({
                "seed": seed,
                "epoch": epoch,
                "acc": f"{result['acc'] * 100:.4f}",
                "nmi": f"{result['nmi'] * 100:.4f}",
                "ari": f"{result['ari'] * 100:.4f}",
                "car": f"{result['car'] * 100:.4f}",
            })
        logger.write(
            f"Seed {seed} training time: "
            f"{datetime.timedelta(seconds=int(time.time() - seed_started))}"
        )

    _write_summary(base_output / "summary.csv", summary_rows)
    print(
        f"Completed seeds={seeds}; summary={base_output / 'summary.csv'}; "
        f"total_time={datetime.timedelta(seconds=int(time.time() - started))}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEWSTRIDE: YAML-only training")
    parser.add_argument("--config_file", required=True)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Seeds to run. Omit for the single seed declared in YAML.",
    )
    cli = parser.parse_args()
    main(load_config(cli.config_file), seeds=cli.seeds)
