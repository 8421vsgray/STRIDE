"""NEWSTRIDE model: diagonal warm-up followed by structural point supervision.

The project deliberately contains no GMM guide and no Edge-CE branch.  The
only learned cross-view loss is the point-level likelihood whose detached
teacher is either the observed diagonal pairing (warm-up) or a finite
Euclidean transport coupling.  The model has one cross-view point-level
transport target path.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from projection import EuclideanProjector


L2norm = nn.functional.normalize


class OursModel(nn.Module):
    def __init__(
        self,
        n_views,
        layer_dims,
        temperature,
        n_classes,
        drop_rate=0.5,
        decoder_drop_rate=0.0,
    ):
        super().__init__()
        if n_views != 2:
            raise ValueError("STRIDE currently defines the two-view protocol only")

        self.n_views = n_views
        self.n_classes = n_classes
        self.temperature = temperature

        self.online_encoder = nn.ModuleList(
            [FCN(layer_dims[i], drop_out=drop_rate) for i in range(n_views)]
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.cross_view_decoder = nn.ModuleList(
            [MLP(layer_dims[i][-1], layer_dims[i][-1], drop_out=decoder_drop_rate)
             for i in range(n_views)]
        )
        for module in (self.target_encoder,):
            for parameter in module.parameters():
                parameter.requires_grad = False

        self.cl = MaxLikelihoodLoss(temperature)
        self.feature_dim = [layer_dims[i][-1] for i in range(n_views)]
        self.last_projection_stats = {}
        self.last_pseudo_labels = []
        self.projector = EuclideanProjector()

    @torch.no_grad()
    def set_target_dropout_eval(self):
        """Keep EMA targets deterministic while the online branch trains."""
        for target_module in (self.target_encoder,):
            for module in target_module.modules():
                if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                    module.eval()

    @torch.no_grad()
    def _ema_update_module(self, online_module, target_module, momentum):
        for online_parameter, target_parameter in zip(
            online_module.parameters(), target_module.parameters()
        ):
            target_parameter.data.mul_(momentum).add_(
                online_parameter.data, alpha=1.0 - momentum
            )

    @torch.no_grad()
    def _update_target_branch(self, momentum):
        for view_idx in range(self.n_views):
            self._ema_update_module(
                self.online_encoder[view_idx], self.target_encoder[view_idx], momentum
            )
        self.set_target_dropout_eval()












    @torch.no_grad()
    def _make_transport_labels(self, query, key, param):
        """Create detached row-stochastic labels by finite structural diffusion."""
        query = L2norm(query)
        key = L2norm(key).detach()
        if query.shape[0] != key.shape[0]:
            raise ValueError("Transport labels require equal batch dimensions")
        n = query.shape[0]
        marginal = 1.0 / float(n)
        d1 = (1.0 - query @ query.t()).clamp(0.0, 2.0)
        d2 = (1.0 - key @ key.t()).clamp(0.0, 2.0)
        d1 = 0.5 * (d1 + d1.t())
        d2 = 0.5 * (d2 + d2.t())

        init = str(param.get("init", "diagonal")).lower()
        if init != "diagonal":
            raise ValueError("NEWSTRIDE currently supports init=diagonal only")
        transport = torch.eye(n, dtype=query.dtype, device=query.device) * marginal
        dual_state = None
        active_state = None
        stats = {
            "solver": "none", "newton": 0, "fallback": 0,
            "projection_failure": 0, "warm_started": 0,
            "active_fraction_min": 1.0, "active_fraction_max": 0.0,
            "off_diagonal_mass_min": 1.0, "off_diagonal_mass_max": 0.0,
            "worst_relative_residual": 0.0, "max_row_residual": 0.0,
            "max_col_residual": 0.0, "total_mass_error": 0.0,
            "step_count": int(param.get("steps", 0)),
            "step_size": float(param.get("step_size", 0.0)),
            "gradient_norm": 0.0,
        }

        for _ in range(max(int(param.get("steps", 0)), 0)):
            gradient = -2.0 * (
                d1 @ transport @ d2.t() + d1.t() @ transport @ d2
            )
            stats["gradient_norm"] = max(
                stats["gradient_norm"], float(gradient.norm().detach().cpu())
            )
            candidate = transport - float(param.get("step_size", 0.0)) * gradient
            previous = transport
            transport, projection_stats = self.projector.project(
                candidate,
                marginal,
                max_iter=50,
                relative_tol=1.0e-3,
                schur_ridge=1.0e-6,
                fallback_iters=100,
                retry_with_stronger_ridge=True,
                dual_warm_start=dual_state,
                fast_full_step=True,
                # The reference drift protocol uses the dense Schur solve.
                # The sparse CG path is retained in projection.py for
                # experiments, but is not the canonical training path.
                sparse_schur=False,
                sparse_schur_iters=24,
                adaptive_ridge=False,
                active_warm_start=active_state,
                active_set_margin=0.0,
                dual_merit_guard=False,
                dual_merit_tol=1.0e-6,
            )
            solver = str(projection_stats.get("solver", "unknown"))
            stats["solver"] = solver
            if solver.startswith("newton") or solver == "affine":
                stats["newton"] += 1
            elif solver == "projection_failure":
                stats["projection_failure"] += 1
                transport = previous
            else:
                stats["fallback"] += 1
            if projection_stats.get("warm_started", False):
                stats["warm_started"] += 1
            for key_name in (
                "active_fraction", "relative_residual", "max_row_residual",
                "max_col_residual", "total_mass_error",
            ):
                if key_name in projection_stats:
                    if key_name == "active_fraction":
                        stats["active_fraction_min"] = min(
                            stats["active_fraction_min"], projection_stats[key_name]
                        )
                        stats["active_fraction_max"] = max(
                            stats["active_fraction_max"], projection_stats[key_name]
                        )
                    elif key_name == "relative_residual":
                        stats["worst_relative_residual"] = max(
                            stats["worst_relative_residual"], projection_stats[key_name]
                        )
                    else:
                        stats[key_name] = max(stats[key_name], projection_stats[key_name])
            dual_state = projection_stats.get("dual_state")
            active_state = projection_stats.get("active_state")

        row_stochastic = transport * float(n)
        diagonal_mass = row_stochastic.diagonal().sum() / float(n)
        diagonal_mass = diagonal_mass.clamp(0.0, 1.0)
        offdiag_mass = (1.0 - diagonal_mass).clamp(0.0, 1.0)
        stats["diagonal_mass"] = float(diagonal_mass.detach().cpu())
        stats["off_diagonal_mass"] = float(offdiag_mass.detach().cpu())
        stats["off_diagonal_mass_min"] = min(stats["off_diagonal_mass_min"], stats["off_diagonal_mass"])
        stats["off_diagonal_mass_max"] = max(stats["off_diagonal_mass_max"], stats["off_diagonal_mass"])
        self.last_projection_stats = stats
        return row_stochastic



    @torch.no_grad()
    def realign(self, z_t, p):
        n_samples = z_t[0].shape[0]
        z_t_normalized = [L2norm(z_t[i]) for i in range(self.n_views)]
        p_normalized = [L2norm(p[i]) for i in range(self.n_views)]
        indices = torch.zeros((self.n_views - 1, n_samples), dtype=torch.long)
        features = [z_t_normalized[0].detach()]
        for view_idx in range(1, self.n_views):
            match = (z_t_normalized[0] @ p_normalized[view_idx].t()).argmax(dim=1)
            indices[view_idx - 1] = match.cpu()
            features.append(F.one_hot(match, num_classes=n_samples).float() @ z_t_normalized[view_idx])
        return features, indices

    @torch.no_grad()
    def extract_feature(self, data):
        z_t = [self.target_encoder[i](data[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z_t[i]) for i in range(self.n_views)]
        features, indices = self.realign(z_t, p)
        return features, indices, [L2norm(value) for value in z_t]

    @torch.no_grad()
    def forward_features(self, data):
        z_t = [self.target_encoder[i](data[i]) for i in range(self.n_views)]
        return [L2norm(self.cross_view_decoder[i](z_t[i])) for i in range(self.n_views)]

    def forward(self, data, param, tag):
        self._update_target_branch(param["mmt"])
        z = [self.online_encoder[i](data[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        z_t = [self.target_encoder[i](data[i]) for i in range(self.n_views)]

        batch_size = z[0].shape[0]
        diagonal = torch.eye(batch_size, dtype=z[0].dtype, device=z[0].device)
        if tag["diagonal_warmup"]:
            labels = [diagonal, diagonal]
            self.last_projection_stats = {"phase": "diagonal_warmup"}
        else:
            labels = [
                self._make_transport_labels(p[0], z_t[1], param),
                self._make_transport_labels(p[1], z_t[0], param),
            ]
        self.last_pseudo_labels = [value.detach() for value in labels]
        self.last_training_targets = [value.detach() for value in labels]

        # Intra-view supervision remains diagonal for every epoch.
        l_intra = (
            self.cl(z[0], z_t[0], diagonal)
            + self.cl(z[1], z_t[1], diagonal)
        ) / 2.0
        l_inter = (
            self.cl(p[0], z_t[1], labels[0])
            + self.cl(p[1], z_t[0], labels[1])
        ) / 2.0
        return l_intra + l_inter


class FCN(nn.Module):
    def __init__(self, dim_layer=None, norm_layer=None, act_layer=None, drop_out=0.0, norm_last_layer=True):
        super().__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm1d
        layers = []
        for layer_idx in range(1, len(dim_layer) - 1):
            layers.extend([nn.Linear(dim_layer[layer_idx - 1], dim_layer[layer_idx], bias=False), norm_layer(dim_layer[layer_idx]), act_layer()])
            if drop_out != 0.0 and layer_idx != len(dim_layer) - 2:
                layers.append(nn.Dropout(drop_out))
        if norm_last_layer:
            layers.extend([nn.Linear(dim_layer[-2], dim_layer[-1], bias=False), nn.BatchNorm1d(dim_layer[-1], affine=False)])
        else:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=True))
        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        return self.ffn(x)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out=None, hidden_ratio=4.0, act_layer=None, drop_out=0.0):
        super().__init__()
        dim_out = dim_out or dim_in
        hidden_dim = int(dim_in * hidden_ratio)
        act_layer = act_layer or nn.ReLU
        layers = [nn.Linear(dim_in, hidden_dim), act_layer()]
        if drop_out != 0.0:
            layers.append(nn.Dropout(drop_out))
        layers.append(nn.Linear(hidden_dim, dim_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MaxLikelihoodLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, x_q, x_k, target=None):
        x_q = L2norm(x_q)
        x_k = L2norm(x_k).detach()
        n_samples = x_q.shape[0]
        if target is None:
            target = torch.eye(n_samples, dtype=x_q.dtype, device=x_q.device)
        else:
            target = target.to(device=x_q.device, dtype=x_q.dtype)
        similarity = (x_q @ x_k.t()) / self.temperature
        probability = torch.softmax(similarity.reshape(-1), dim=0).mul(n_samples)
        probability = probability.reshape(n_samples, n_samples)
        nll = -torch.log(probability.clamp_min(torch.finfo(probability.dtype).tiny))
        return (nll * target.detach()).mean()
