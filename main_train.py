"""YAML-only NEWSTRIDE training entry point."""

import argparse
import copy
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
    "n_views": 2,
    "epochs": 200,
    "warmup_epochs": 20,
    "diagonal_epochs": 20,
    "batch_size": 512,
    "train_time": 1,
    "threads": 8,
    "num_workers": 0,
    "print_freq": 5,
    "data_norm": "standard",
    "weight_decay": 0.0,
    "momentum": 0.98,
    "temperature": 0.5,
    "drop_rate": 0.2,
    "decoder_drop_rate": 0.0,
    "blr": 5.0e-4,
    "min_blr": 5.0e-4,
    "init": "diagonal",
    "steps": 5,
    "step_size": 0.02,
    "m_ratio": 0.0,
    "device": "cuda:0",
    "pin_mem": True,
    "save_ckpt": False,
    "output_dir": "output",
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


def train_one_time(args, logger, train_id=0):
    utils.fix_random_seeds(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(device)

    train_set, test_set = load_dataset(args)
    args.n_sample_test = len(test_set)
    # Match CorreGen exactly: both loaders enumerate the complete dataset
    # once per pass through RandomSampler.  This is a random permutation of
    # all samples, not random-subset evaluation.
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_test = torch.utils.data.RandomSampler(test_set)
    train_loader = torch.utils.data.DataLoader(
        train_set, sampler=sampler_train,
        batch_size=min(args.batch_size, len(train_set)),
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=min(512, len(test_set)), sampler=sampler_test,
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


def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.visual_dir = str(Path(args.output_dir) / "visualize")
    Path(args.visual_dir).mkdir(parents=True, exist_ok=True)
    args.embed_dim = args.encoder_dim[0][-1]
    # Treat blr/min_blr as the actual learning-rate values.  The previous
    # linear batch-size scaling silently doubled the configured rate for the
    # batch sizes used by the current experiments.
    args.lr = args.blr
    args.min_lr = args.min_blr
    if args.min_lr > args.lr:
        raise ValueError("min_blr must not exceed blr")

    logger = utils.FileLogger(str(Path(args.output_dir) / "log_train.txt"))
    logger.write(f"NEWSTRIDE config: {args.config_file}")
    logger.write(f"Start time: {datetime.datetime.now().isoformat(timespec='seconds')}")
    started = time.time()

    result_rows = {}
    for train_id in range(args.train_time):
        states = train_one_time(args, logger, train_id)
        for epoch, state in states.items():
            result_rows.setdefault(epoch, []).append(state["k-means"])
        args.seed += 1

    for epoch, rows in sorted(result_rows.items()):
        logger.write(
            f"Epoch {epoch}: title,acc,nmi,ari,car-------------\n"
            "Average K-means Result   : "
            f"ACC = {np.mean([r['acc'] for r in rows])*100:.2f} "
            f"NMI = {np.mean([r['nmi'] for r in rows])*100:.2f} "
            f"ARI = {np.mean([r['ari'] for r in rows])*100:.2f} "
            f"CAR = {np.mean([r['car'] for r in rows])*100:.2f}"
        )
    logger.write(f"Total training time: {datetime.timedelta(seconds=int(time.time()-started))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEWSTRIDE: YAML-only training")
    parser.add_argument("--config_file", required=True)
    cli = parser.parse_args()
    main(load_config(cli.config_file))
