"""Training loop for the two-view structural transport protocol."""

import math
import sys

import torch
from torch.utils.data import DataLoader

from eval import evaluate
from model import OursModel
from utils import MetricLogger, SmoothedValue


def _new_projection_totals():
    return {
        "newton": 0,
        "fallback": 0,
        "projection_failure": 0,
        "warm_started": 0,
        "active_fraction_min": 1.0,
        "active_fraction_max": 0.0,
        "off_diagonal_mass_min": 1.0,
        "off_diagonal_mass_max": 0.0,
        "worst_relative_residual": 0.0,
        "max_row_residual": 0.0,
        "max_col_residual": 0.0,
        "total_mass_error": 0.0,
        "gradient_norm": 0.0,
    }


def _accumulate_projection_stats(total, current):
    total["newton"] += int(current.get("newton", 0))
    total["fallback"] += int(current.get("fallback", 0))
    total["projection_failure"] += int(current.get("projection_failure", 0))
    total["warm_started"] += int(current.get("warm_started", 0))
    total["active_fraction_min"] = min(
        total["active_fraction_min"], float(current.get("active_fraction_min", 1.0))
    )
    total["active_fraction_max"] = max(
        total["active_fraction_max"], float(current.get("active_fraction_max", 0.0))
    )
    total["off_diagonal_mass_min"] = min(
        total["off_diagonal_mass_min"],
        float(current.get("off_diagonal_mass_min", 1.0)),
    )
    total["off_diagonal_mass_max"] = max(
        total["off_diagonal_mass_max"],
        float(current.get("off_diagonal_mass_max", 0.0)),
    )
    for key in ("worst_relative_residual", "max_row_residual", "max_col_residual", "total_mass_error", "gradient_norm"):
        total[key] = max(total[key], float(current.get(key, 0.0)))


def train_one_epoch(
    model: OursModel,
    data_loader_train: DataLoader,
    data_loader_test: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scheduler=None,
    args=None,
):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    iterator = (
        metric_logger.log_every(data_loader_train, args.print_freq, header)
        if args.print_this_epoch
        else data_loader_train
    )
    diagonal_phase = epoch < args.diagonal_epochs
    tag = {"diagonal_warmup": diagonal_phase}
    param = {
        "mmt": float(args.momentum),
        "init": args.init,
        "steps": int(args.steps),
        "step_size": float(args.step_size),
    }

    model.train(True)
    model.set_target_dropout_eval()
    optimizer.zero_grad(set_to_none=True)
    projection_totals = _new_projection_totals()

    for data_iter_step, batch in enumerate(iterator):
        _, samples, _label1, _label2, _id1, _id2, _label_align = batch
        if len(samples[0]) < math.ceil(args.batch_size / 2):
            continue
        smooth_epoch = epoch + (data_iter_step + 1) / max(len(data_loader_train), 1)
        if scheduler is not None:
            lr = scheduler(optimizer, smooth_epoch)
        for view_idx in range(args.n_views):
            samples[view_idx] = samples[view_idx].to(device, non_blocking=True)

        with torch.autocast("cuda", enabled=False):
            loss = model(samples, param, tag)
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if not diagonal_phase:
            _accumulate_projection_stats(projection_totals, model.last_projection_stats)
        if args.print_this_epoch:
            metric_logger.update(lr=lr, loss=loss_value)

    if args.print_this_epoch and not diagonal_phase:
        print(f"Projection diagnostics: {projection_totals}")
    return evaluate(model, data_loader_test, device, args) if args.print_this_epoch else None
