"""STRIDE model: diagonal warm-up followed by Euclidean transport point supervision.

The project deliberately contains no GMM guide and no Edge-CE branch.  The
only learned cross-view loss is the point-level likelihood whose detached
teacher is either the observed diagonal pairing (warm-up) or a dense
relation-guided transport coupling.  The historical SAKE/CorreGen global QCE
weighting is available as an opt-in loss mode; it is deliberately disabled
by default so existing STRIDE protocols are unchanged.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from transport_solver import TransportProjector


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
        self.transport_projector = TransportProjector()
        self.last_projection_stats = {}
        self.last_pseudo_labels = []

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
    def begin_structure_epoch(self, epoch):
        self.transport_projector.begin_epoch(epoch)
        self.last_projection_stats = {}

    @torch.no_grad()
    def _structure_transport(
        self, query, key, param, query_target=None,
        precomputed_d1=None, precomputed_d2=None
    ):
        transport = self.transport_projector._structure_transport(
            query, key, param,
            query_target=query_target,
            precomputed_d1=precomputed_d1,
            precomputed_d2=precomputed_d2,
        )
        self.last_projection_stats = self.transport_projector.last_projection_stats
        return transport

    @staticmethod
    @torch.no_grad()
    def _diagonal_only_teacher(transport):
        """Keep only diagonal confidence, with the original target scale."""
        n = transport.shape[0]
        diagonal = torch.diag(transport.diagonal())
        return diagonal * (float(n) / diagonal.sum().clamp_min(1.0e-12))

    @staticmethod
    @torch.no_grad()
    def _oracle_true_diagonal_teacher(
        transport, correct_diagonal, threshold=0.1, quality_weighted=True
    ):
        """Oracle diagnostic target using only correct observed diagonal pairs.

        This is intentionally supervised and must never be enabled for a
        canonical unsupervised result.  ``transport`` is row-stochastic at the
        CE boundary, so its diagonal entries are the quality scores.  We keep
        only entries whose source/target IDs are truly equal and whose score
        passes ``threshold``.  The selected weights are rescaled to total
        ``n`` so the MaxLikelihoodLoss remains on the same scale as the usual
        identity or dense row-stochastic teacher.
        """
        n = transport.shape[0]
        scores = transport.diagonal().clamp_min(0.0)
        selected = correct_diagonal.to(dtype=torch.bool) & (scores >= float(threshold))
        weights = scores if quality_weighted else torch.ones_like(scores)
        weights = torch.where(selected, weights, torch.zeros_like(weights))
        total = weights.sum()
        if float(total.detach().cpu()) <= 1.0e-12:
            # Keep the supervised protocol well-defined on a rare batch with
            # no selected pair; a zero cross-view target contributes no
            # gradient rather than inventing a false correspondence.
            return torch.zeros_like(transport)
        return torch.diag(weights * (float(n) / total))

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
        self._update_target_branch(param["target_momentum"])
        z = [self.online_encoder[i](data[i]) for i in range(self.n_views)]
        p = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        z_t = [self.target_encoder[i](data[i]) for i in range(self.n_views)]

        batch_size = z[0].shape[0]
        diagonal = torch.eye(batch_size, dtype=z[0].dtype, device=z[0].device)
        qce_enabled = bool(param.get("qce_enabled", False))
        qce_identity_target = bool(param.get("qce_identity_target", False))
        qce_disable_structure = bool(param.get("qce_disable_structure", False))
        if tag["diagonal_warmup"] or (
            qce_enabled and qce_identity_target and qce_disable_structure
        ):
            mp_inter = [diagonal, diagonal]
            self.last_projection_stats = {
                "phase": "diagonal_warmup" if tag["diagonal_warmup"] else "qce_identity"
            }
        else:
            p_relation = p
            p_relation_target = [None for _ in range(self.n_views)]
            if param.get("structure_batch_relations", False):
                # Optional speed experiment. Batched GEMM can use a different
                # reduction/TF32 path, so it is not part of the canonical
                # protocol unless explicitly enabled.
                with torch.no_grad():
                    relation_sources = torch.stack(
                        [L2norm(p_relation[0]), L2norm(p_relation[1])], dim=0
                    )
                    relation_targets = torch.stack(
                        [L2norm(z_t[1]).detach(), L2norm(z_t[0]).detach()], dim=0
                    )
                    view_relations = 1.0 - torch.bmm(
                        relation_sources, relation_sources.transpose(1, 2)
                    )
                    target_relations = 1.0 - torch.bmm(
                        relation_targets, relation_targets.transpose(1, 2)
                    )
                    view_relations = 0.5 * (
                        view_relations + view_relations.transpose(1, 2)
                    )
                    target_relations = 0.5 * (
                        target_relations + target_relations.transpose(1, 2)
                    )
                mp_inter = [
                    self._structure_transport(
                        p_relation[0], z_t[1], param,
                        query_target=p_relation_target[0],
                        precomputed_d1=view_relations[0],
                        precomputed_d2=target_relations[0],
                    ),
                    self._structure_transport(
                        p_relation[1], z_t[0], param,
                        query_target=p_relation_target[1],
                        precomputed_d1=view_relations[1],
                        precomputed_d2=target_relations[1],
                    ),
                ]
            else:
                mp_inter = [
                    self._structure_transport(
                        p_relation[0], z_t[1], param,
                        query_target=p_relation_target[0],
                    ),
                    self._structure_transport(
                        p_relation[1], z_t[0], param,
                        query_target=p_relation_target[1],
                    ),
                ]
        raw_mp_inter = [value.detach() for value in mp_inter]
        if (
            not tag["diagonal_warmup"]
            and param.get("structure_oracle_true_diagonal", False)
        ):
            correct_diagonal = param["structure_oracle_correct_diagonal"]
            threshold = param.get("structure_oracle_tii_threshold", 0.1)
            quality_weighted = param.get("structure_oracle_quality_weighted", True)
            mp_inter = [
                self._oracle_true_diagonal_teacher(
                    value, correct_diagonal, threshold, quality_weighted
                )
                for value in mp_inter
            ]
        if not tag["diagonal_warmup"] and param.get("structure_diagonal_only_teacher", False):
            mp_inter = [self._diagonal_only_teacher(value) for value in mp_inter]
        # Keep raw transport labels for mechanism diagnostics; the training target is
        # tracked separately so oracle leakage cannot make quality metrics look
        # artificially perfect.
        self.last_pseudo_labels = raw_mp_inter
        self.last_training_targets = [value.detach() for value in mp_inter]

        # Intra-view supervision is intentionally diagonal for every epoch.
        if qce_enabled:
            qce_normalize = bool(param.get("qce_normalize", False))
            qce_reduction = str(param.get("qce_reduction", "batch_mean"))
            qce_nu = float(param.get("qce_nu", 2.0))
            qce_probability_mode = str(
                param.get("qce_probability_mode", "global")
            ).lower()
            if qce_probability_mode != "global":
                raise ValueError(
                    "STRIDE QCE reproduction only supports the historical "
                    "global B^2 probability mode; use qce_probability_mode=global"
                )
            qce_inter_targets = (
                [diagonal, diagonal] if qce_identity_target else mp_inter
            )
            l_intra = (
                self.cl(
                    z[0], z_t[0], diagonal, use_q=True,
                    qce_nu=qce_nu, qce_normalize=qce_normalize,
                    qce_reduction=qce_reduction,
                )
                + self.cl(
                    z[1], z_t[1], diagonal, use_q=True,
                    qce_nu=qce_nu, qce_normalize=qce_normalize,
                    qce_reduction=qce_reduction,
                )
            ) / 2.0
            l_inter = (
                self.cl(
                    p[0], z_t[1], qce_inter_targets[0], use_q=True,
                    qce_nu=qce_nu, qce_normalize=qce_normalize,
                    qce_reduction=qce_reduction,
                )
                + self.cl(
                    p[1], z_t[0], qce_inter_targets[1], use_q=True,
                    qce_nu=qce_nu, qce_normalize=qce_normalize,
                    qce_reduction=qce_reduction,
                )
            ) / 2.0
        else:
            l_intra = (self.cl(z[0], z_t[0], diagonal) + self.cl(z[1], z_t[1], diagonal)) / 2.0
            l_inter = (self.cl(p[0], z_t[1], mp_inter[0]) + self.cl(p[1], z_t[0], mp_inter[1])) / 2.0
        cross_weight = float(param.get("structure_cross_weight", 1.0))
        intra_weight = float(param.get("lambda_intra_qce", 1.0)) if qce_enabled else 1.0
        inter_weight = float(param.get("lambda_inter_qce", cross_weight)) if qce_enabled else cross_weight
        return intra_weight * l_intra + inter_weight * l_inter


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

    def forward(
        self,
        x_q,
        x_k,
        target=None,
        use_q=False,
        qce_nu=2.0,
        qce_normalize=False,
        qce_reduction="batch_mean",
    ):
        x_q = L2norm(x_q)
        x_k = L2norm(x_k).detach()
        n_samples = x_q.shape[0]
        if target is None:
            target = torch.eye(n_samples, dtype=x_q.dtype, device=x_q.device)
        else:
            target = target.to(device=x_q.device, dtype=x_q.dtype)
        similarity = (x_q @ x_k.t()) / self.temperature
        # Preserve CorreGen's global batch normalization convention:
        # p_ij = B * softmax(vec(similarity))_ij, so sum_ij p_ij = B.
        probability = torch.softmax(similarity.reshape(-1), dim=0).mul(n_samples)
        probability = probability.reshape(n_samples, n_samples)
        nll = -torch.log(probability.clamp_min(torch.finfo(probability.dtype).tiny))
        target = target.detach()
        if not use_q:
            return (nll * target).mean()

        # Historical QCE: q is a detached self-paced weight.  The identity
        # protocol uses only q_ii; for a dense transport target, the historical
        # implementation retains its off-diagonal mass and reweights only the
        # observed diagonal entries.
        q = (
            probability.detach()
            / torch.sqrt(
                torch.as_tensor(float(qce_nu), device=probability.device, dtype=probability.dtype)
                + probability.detach().square()
            )
        )
        weighted_target = target.clone()
        weighted_target.diagonal().mul_(q.diagonal())
        if qce_normalize:
            denominator = weighted_target.sum().clamp_min(1.0e-12)
        elif str(qce_reduction).lower() == "global_mean":
            denominator = float(n_samples * n_samples)
        else:
            # SAKE historical convention: average selected diagonal target mass
            # by B, matching -(1/B) sum_i q(p_ii) log p_ii.
            denominator = float(n_samples)
        return (nll * weighted_target).sum() / denominator
