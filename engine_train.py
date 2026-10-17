import math
import sys

import torch
from torch.utils.data import DataLoader

from eval import evaluate
from transport_projector import build_projection_params
from model import OursModel
from utils import MetricLogger, SmoothedValue


@torch.no_grad()
def _pseudo_label_quality(transport, source_ids, target_ids, source_labels, target_labels):
    """Evaluation-only quality of a batch coupling against known IDs/classes."""
    source_ids = source_ids.to(transport.device)
    target_ids = target_ids.to(transport.device)
    source_labels = source_labels.to(transport.device)
    target_labels = target_labels.to(transport.device)
    true_pair = source_ids[:, None].eq(target_ids[None, :])
    covered = true_pair.any(dim=1)
    if not bool(covered.any()):
        return None
    rows = torch.arange(transport.shape[0], device=transport.device)
    top1_index = transport.argmax(dim=1)
    top1 = true_pair[rows, top1_index][covered].float().mean()
    topk = min(5, transport.shape[1])
    topk_index = transport.topk(topk, dim=1).indices
    top5 = true_pair.gather(1, topk_index).any(dim=1)[covered].float().mean()
    true_mass = (transport * true_pair.to(dtype=transport.dtype)).sum(dim=1)[covered].mean()
    offdiag = ~torch.eye(transport.shape[0], dtype=torch.bool, device=transport.device)
    offdiag_mass_per_row = (transport * offdiag.to(dtype=transport.dtype)).sum(dim=1)
    offdiag_true_mass_per_row = (
        transport * true_pair.to(dtype=transport.dtype) * offdiag.to(dtype=transport.dtype)
    ).sum(dim=1)
    offdiag_mass = offdiag_mass_per_row[covered].mean()
    offdiag_true_mass = offdiag_true_mass_per_row[covered].mean()
    offdiag_precision = (offdiag_true_mass_per_row[covered].sum()
                         / offdiag_mass_per_row[covered].sum().clamp_min(1.0e-12))
    diagonal = torch.eye(transport.shape[0], dtype=torch.bool, device=transport.device)
    diagonal_mass_per_row = (transport * diagonal.to(dtype=transport.dtype)).sum(dim=1)
    diagonal_true_mass_per_row = (
        transport * true_pair.to(dtype=transport.dtype) * diagonal.to(dtype=transport.dtype)
    ).sum(dim=1)
    diagonal_precision = (
        diagonal_true_mass_per_row[covered].sum()
        / diagonal_mass_per_row[covered].sum().clamp_min(1.0e-12)
    )
    covered_diagonal_mass = diagonal_mass_per_row[covered]
    diagonal_mass_threshold_stats = {}
    for threshold in (0.1, 0.2, 0.3, 0.5):
        selected = covered_diagonal_mass >= threshold
        diagonal_mass_threshold_stats[str(threshold)] = {
            "row_fraction": float(selected.float().mean().detach().cpu()),
            "mass_share": float(
                covered_diagonal_mass[selected].sum()
                / covered_diagonal_mass.sum().clamp_min(1.0e-12)
            ),
        }
    diagonal_is_true = torch.diag(true_pair)
    same_class = source_labels[:, None].eq(target_labels[None, :])
    same_class_mass = (transport * same_class.to(dtype=transport.dtype)).sum(dim=1).mean()
    row_prob = transport / transport.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    entropy = -(row_prob.clamp_min(1.0e-12) * row_prob.clamp_min(1.0e-12).log()).sum(dim=1).mean()
    return {
        "coverage": float(covered.float().mean().detach().cpu()),
        "top1": float(top1.detach().cpu()),
        "top5": float(top5.detach().cpu()),
        "true_mass": float(true_mass.detach().cpu()),
        "offdiag_mass": float(offdiag_mass.detach().cpu()),
        "offdiag_true_mass": float(offdiag_true_mass.detach().cpu()),
        "offdiag_precision": float(offdiag_precision.detach().cpu()),
        "diagonal_mass": float(diagonal_mass_per_row[covered].mean().detach().cpu()),
        "diagonal_true_mass": float(diagonal_true_mass_per_row[covered].mean().detach().cpu()),
        "diagonal_precision": float(diagonal_precision.detach().cpu()),
        "diagonal_mass_threshold_stats": diagonal_mass_threshold_stats,
        # Evaluation-only row-level values used to test whether diagonal
        # transport mass separates clean and noisy observed pairs.
        "diagonal_mass_true_values": diagonal_mass_per_row[covered & diagonal_is_true].detach().cpu(),
        "diagonal_mass_false_values": diagonal_mass_per_row[covered & ~diagonal_is_true].detach().cpu(),
        "same_class_mass": float(same_class_mass.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
    }


def train_one_epoch(model: OursModel, data_loader_train: DataLoader, data_loader_test: DataLoader,
                    optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
                    scheduler=None, args=None):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    data_loader = (enumerate(metric_logger.log_every(data_loader_train, args.print_freq, header))
                   if args.print_this_epoch else enumerate(data_loader_train))
    # The only public phase control is the generic warmup length.  The
    # training method itself does not expose a method-specific switch.
    tag = {"diagonal_warmup": epoch < args.warmup_epochs}
    target_momentum = float(args.momentum)
    param = build_projection_params(args)
    model.begin_structure_epoch(epoch)
    param.update({
        "target_momentum": target_momentum,
        "qce_enabled": args.qce_enabled,
        "qce_nu": args.qce_nu,
        "qce_normalize": args.qce_normalize,
        "qce_reduction": args.qce_reduction,
        "qce_probability_mode": args.qce_probability_mode,
        "qce_identity_target": args.qce_identity_target,
        "qce_disable_structure": args.qce_disable_structure,
        "lambda_intra_qce": args.lambda_intra_qce,
        "lambda_inter_qce": args.lambda_inter_qce,
        "structure_diagonal_only_teacher": args.structure_diagonal_only_teacher,
        "structure_oracle_true_diagonal": args.structure_oracle_true_diagonal,
        "structure_oracle_tii_threshold": args.structure_oracle_tii_threshold,
        "structure_oracle_quality_weighted": args.structure_oracle_quality_weighted,
    })
    model.train(True)
    model.set_target_dropout_eval()
    optimizer.zero_grad()
    projection_totals = {
        "newton": 0,
        "newton_ridge_retry": 0,
        "qrot_bcd": 0,
        "qrot_warmup": 0,
        "qrot_direct": 0,
        "affine": 0,
        "projection_failure": 0,
        "warm_started": 0,
        "active_fraction_min": 1.0,
        "active_fraction_max": 0.0,
        "off_diagonal_mass_min": 1.0,
        "off_diagonal_mass_max": 0.0,
        "worst_relative_residual": 0.0,
        "componentwise_newton": 0,
        "prior_gate": 0,
        "prior_precheck": 0,
        "epoch_prior_fallback": 0,
        "min_transport": float("inf"),
        "max_row_residual": 0.0,
        "max_col_residual": 0.0,
        "total_mass_error": 0.0,
        "structure_energy_damped": 0,
        "structure_energy_rejected": 0,
        "structure_kept_old_plan": 0,
        "structure_gamma_min": 1.0,
        "structure_first_order_min": float("inf"),
        "structure_first_order_max": float("-inf"),
        "structure_delta_min_gamma_max": float("-inf"),
        "structure_descent_violations": 0,
        "structure_projection_optimality_max": float("-inf"),
        "structure_projection_optimality_violations": 0,
        "projection_refined": 0,
        "structure_update_mix_min": 1.0,
        "point_gradient_norm": 0.0,
        "structure_gradient_norm": 0.0,
        "mixed_gradient_norm": 0.0,
        "d1_asymmetry_max": 0.0,
        "query_target_gap_max": 0.0,
    }
    quality_totals = {"coverage": 0.0, "top1": 0.0, "top5": 0.0, "true_mass": 0.0,
                      "offdiag_mass": 0.0, "offdiag_true_mass": 0.0, "offdiag_precision": 0.0,
                      "diagonal_mass": 0.0, "diagonal_true_mass": 0.0,
                      "diagonal_precision": 0.0,
                      "same_class_mass": 0.0, "entropy": 0.0, "count": 0}
    diagonal_mass_buckets = {"true": [], "false": []}
    diagonal_mass_threshold_totals = {
        str(threshold): {"row_fraction": 0.0, "mass_share": 0.0}
        for threshold in (0.1, 0.2, 0.3, 0.5)
    }
    for data_iter_step, (_, samples, label1, label2, id1, id2, _label_align) in data_loader:
        if len(samples[0]) < math.ceil(args.batch_size / 2):
            continue
        smooth_epoch = epoch + (data_iter_step + 1) / len(data_loader_train)
        if scheduler is not None:
            lr = scheduler(optimizer, smooth_epoch)
        for view_idx in range(args.n_views):
            samples[view_idx] = samples[view_idx].to(device, non_blocking=True)
        if args.structure_oracle_true_diagonal:
            # A supervised diagnostic only: true correspondence is determined
            # by the dataset IDs, not by the class labels.
            param["structure_oracle_correct_diagonal"] = id1.to(device).eq(id2.to(device))
        with torch.autocast("cuda", enabled=False):
            loss = model(samples, param, tag)
        if args.structure_debug_true_quality and not tag["diagonal_warmup"]:
            quality_specs = (
                (model.last_pseudo_labels[0], id1, id2, label1, label2),
                (model.last_pseudo_labels[1], id2, id1, label2, label1),
            )
            for quality_spec in quality_specs:
                quality = _pseudo_label_quality(*quality_spec)
                if quality is not None:
                    for key in ("coverage", "top1", "top5", "true_mass", "offdiag_mass",
                                "offdiag_true_mass", "offdiag_precision", "diagonal_mass",
                                "diagonal_true_mass", "diagonal_precision", "same_class_mass",
                                "entropy"):
                        quality_totals[key] += quality[key]
                    diagonal_mass_buckets["true"].append(quality["diagonal_mass_true_values"])
                    diagonal_mass_buckets["false"].append(quality["diagonal_mass_false_values"])
                    for threshold, values in quality["diagonal_mass_threshold_stats"].items():
                        diagonal_mass_threshold_totals[threshold]["row_fraction"] += values["row_fraction"]
                        diagonal_mass_threshold_totals[threshold]["mass_share"] += values["mass_share"]
                    quality_totals["count"] += 1
        if not tag["diagonal_warmup"]:
            for key in ("newton", "newton_ridge_retry", "qrot_bcd", "qrot_warmup", "qrot_direct", "affine", "projection_failure", "warm_started", "componentwise_newton", "prior_gate", "epoch_prior_fallback"):
                projection_totals[key] += int(model.last_projection_stats.get(key, 0))
            for key in ("active_fraction_min", "active_fraction_max"):
                value = model.last_projection_stats.get(key)
                if value is not None:
                    if key.endswith("min"):
                        projection_totals[key] = min(projection_totals[key], value)
                    else:
                        projection_totals[key] = max(projection_totals[key], value)
            off_diagonal_mass = model.last_projection_stats.get("off_diagonal_mass")
            if off_diagonal_mass is not None:
                projection_totals["off_diagonal_mass_min"] = min(
                    projection_totals["off_diagonal_mass_min"], off_diagonal_mass
                )
                projection_totals["off_diagonal_mass_max"] = max(
                    projection_totals["off_diagonal_mass_max"], off_diagonal_mass
                )
            projection_totals["worst_relative_residual"] = max(
                projection_totals["worst_relative_residual"],
                float(model.last_projection_stats.get("worst_relative_residual", 0.0)),
            )
            projection_totals["min_transport"] = min(
                projection_totals["min_transport"],
                float(model.last_projection_stats.get("min_transport", float("inf"))),
            )
            for key in ("max_row_residual", "max_col_residual", "total_mass_error"):
                projection_totals[key] = max(
                    projection_totals[key],
                    float(model.last_projection_stats.get(key, 0.0)),
                )
            for key in ("structure_energy_damped", "structure_energy_rejected", "structure_kept_old_plan"):
                projection_totals[key] += int(model.last_projection_stats.get(key, 0))
            projection_totals["structure_gamma_min"] = min(
                projection_totals["structure_gamma_min"],
                float(model.last_projection_stats.get("structure_gamma_min", 1.0)),
            )
            projection_totals["structure_first_order_min"] = min(
                projection_totals["structure_first_order_min"],
                float(model.last_projection_stats.get("structure_first_order_min", float("inf"))),
            )
            projection_totals["structure_first_order_max"] = max(
                projection_totals["structure_first_order_max"],
                float(model.last_projection_stats.get("structure_first_order_max", float("-inf"))),
            )
            projection_totals["structure_delta_min_gamma_max"] = max(
                projection_totals["structure_delta_min_gamma_max"],
                float(model.last_projection_stats.get("structure_delta_min_gamma_max", float("-inf"))),
            )
            projection_totals["structure_descent_violations"] += int(
                model.last_projection_stats.get("structure_descent_violations", 0)
            )
            projection_totals["structure_projection_optimality_max"] = max(
                projection_totals["structure_projection_optimality_max"],
                float(model.last_projection_stats.get("structure_projection_optimality_max", float("-inf"))),
            )
            projection_totals["structure_projection_optimality_violations"] += int(
                model.last_projection_stats.get("structure_projection_optimality_violations", 0)
            )
            projection_totals["projection_refined"] += int(
                model.last_projection_stats.get("projection_refined", 0)
            )
            projection_totals["structure_update_mix_min"] = min(
                projection_totals["structure_update_mix_min"],
                float(model.last_projection_stats.get("structure_update_mix_min", 1.0)),
            )
            for key in ("point_gradient_norm", "structure_gradient_norm", "mixed_gradient_norm"):
                projection_totals[key] = max(
                    projection_totals[key],
                    float(model.last_projection_stats.get(key, 0.0)),
                )
            projection_totals["d1_asymmetry_max"] = max(
                projection_totals["d1_asymmetry_max"],
                float(model.last_projection_stats.get("d1_asymmetry", 0.0)),
            )
            projection_totals["query_target_gap_max"] = max(
                projection_totals["query_target_gap_max"],
                float(model.last_projection_stats.get("query_target_gap", 0.0)),
            )
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if args.print_this_epoch:
            metric_logger.update(lr=lr)
            metric_logger.update(loss=loss_value)
    if args.print_this_epoch and not tag["diagonal_warmup"]:
        print(f"Projection diagnostics: {projection_totals}")
        if args.structure_debug_true_quality and quality_totals["count"]:
            count = float(quality_totals.pop("count"))
            print("Pseudo-label diagnostics:", {
                key: value / count for key, value in quality_totals.items()
            })
            print("Diagonal mass share by threshold:", {
                threshold: {
                    "row_fraction": values["row_fraction"] / count,
                    "mass_share": values["mass_share"] / count,
                }
                for threshold, values in diagonal_mass_threshold_totals.items()
            })
            diagonal_summary = {}
            for bucket_name, values in diagonal_mass_buckets.items():
                if values:
                    flat = torch.cat(values).to(dtype=torch.float64)
                    diagonal_summary[bucket_name] = {
                        "count": int(flat.numel()),
                        "min": float(flat.min()),
                        "q25": float(torch.quantile(flat, 0.25)),
                        "median": float(torch.quantile(flat, 0.50)),
                        "q75": float(torch.quantile(flat, 0.75)),
                        "max": float(flat.max()),
                    }
            print("Diagonal mass by correctness:", diagonal_summary)
            all_values = []
            all_labels = []
            if diagonal_mass_buckets["true"]:
                all_values.append(torch.cat(diagonal_mass_buckets["true"]))
                all_labels.append(torch.ones(all_values[-1].numel(), dtype=torch.bool))
            if diagonal_mass_buckets["false"]:
                all_values.append(torch.cat(diagonal_mass_buckets["false"]))
                all_labels.append(torch.zeros(all_values[-1].numel(), dtype=torch.bool))
            if all_values:
                values = torch.cat(all_values).to(dtype=torch.float64)
                labels = torch.cat(all_labels)
                threshold_table = []
                for threshold in (0.0, 1.0e-4, 1.0e-3, 5.0e-3, 1.0e-2,
                                  2.0e-2, 5.0e-2, 1.0e-1, 2.0e-1, 3.0e-1, 5.0e-1):
                    selected = values >= threshold
                    selected_count = int(selected.sum())
                    if selected_count:
                        threshold_table.append({
                            "threshold": threshold,
                            "coverage": selected_count / float(values.numel()),
                            "precision": float(labels[selected].double().mean()),
                            "selected": selected_count,
                        })
                print("Diagonal mass threshold scan:", threshold_table)
    return evaluate(model, data_loader_test, device, args) if args.print_this_epoch else None
