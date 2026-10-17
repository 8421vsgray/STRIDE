"""Standalone transport pseudo-label solver.

The neural model supplies relation matrices and consumes the returned
row-stochastic teacher. All transport projection, diffusion, gating, and
solver diagnostics live here so the model file contains no numerical
projection implementation.
"""

import torch

L2norm = torch.nn.functional.normalize


class TransportProjector:
    def __init__(self):
        self.last_projection_stats = {}
        self._epoch_id = None
        self._epoch_transport_calls = 0
        self._epoch_first_batch_decided = False
        self._epoch_fallback = False

    @torch.no_grad()
    def begin_epoch(self, epoch):
        self._epoch_id = int(epoch)
        self._epoch_transport_calls = 0
        self._epoch_first_batch_decided = False
        self._epoch_fallback = False
        self.last_projection_stats = {}

    @staticmethod
    @torch.no_grad()
    def _qrot_bcd_projection(
        matrix,
        marginal,
        max_iter=50,
        relative_tol=1.0e-3,
        dual_warm_start=None,
    ):
        """QROT block-coordinate projection candidate.

        The Euclidean projection is equivalent, up to a constant, to the
        quadratic-regularized OT problem

            min_T < -X, T > + 0.5 ||T||_F^2
            s.t. T >= 0, T 1 = a, T^T 1 = b.

        With dual potentials ``u`` and ``v``, the primal plan is
        ``[X-u[:, None]-v[None, :]]_+``.  Holding one potential fixed, each
        row/column potential has an exact water-filling solution.  This is a
        GPU-friendly QROT BCD baseline inspired by RegOT's QROT solvers; it
        deliberately remains separate from the canonical Newton projector.
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("QROT BCD requires a square transport matrix")
        n = matrix.shape[0]
        if n == 1:
            return matrix.new_full((1, 1), float(marginal)), {
                "solver": "qrot_bcd", "iterations": 0,
                "relative_residual": 0.0, "warm_started": False,
                "dual_state": (matrix.new_zeros(1), matrix.new_zeros(1)),
                "active_state": torch.ones_like(matrix, dtype=torch.bool),
            }

        x = matrix
        target = torch.as_tensor(marginal, dtype=x.dtype, device=x.device)
        tol = target.abs().clamp_min(1.0e-12) * float(relative_tol)

        def waterfill(values, dim):
            # Find theta such that sum(max(values - theta, 0)) = target.
            ordered, _ = torch.sort(values, dim=dim, descending=True)
            if dim == 1:
                ranks = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)[None, :]
                cumulative = ordered.cumsum(dim=1)
                feasible = ordered - (cumulative - target) / ranks > 0.0
                rho = feasible.sum(dim=1).clamp_min(1).to(torch.long) - 1
                thresholds = (cumulative - target) / ranks
                theta = thresholds.gather(1, rho[:, None]).squeeze(1)
            else:
                ranks = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)[:, None]
                cumulative = ordered.cumsum(dim=0)
                feasible = ordered - (cumulative - target) / ranks > 0.0
                rho = feasible.sum(dim=0).clamp_min(1).to(torch.long) - 1
                thresholds = (cumulative - target) / ranks
                theta = thresholds.gather(0, rho[None, :]).squeeze(0)
            return theta

        warm_started = False
        if (
            dual_warm_start is not None
            and isinstance(dual_warm_start, (tuple, list))
            and len(dual_warm_start) == 2
            and tuple(dual_warm_start[0].shape) == (n,)
            and tuple(dual_warm_start[1].shape) == (n,)
            and dual_warm_start[0].device == x.device
            and dual_warm_start[1].device == x.device
            and torch.isfinite(dual_warm_start[0]).all()
            and torch.isfinite(dual_warm_start[1]).all()
        ):
            u = dual_warm_start[0].to(dtype=x.dtype).clone()
            v = dual_warm_start[1].to(dtype=x.dtype).clone()
            warm_started = True
        else:
            u = torch.zeros(n, dtype=x.dtype, device=x.device)
            v = torch.zeros(n, dtype=x.dtype, device=x.device)

        plan = torch.zeros_like(x)
        residual = torch.full((), float("inf"), dtype=x.dtype, device=x.device)
        for iteration in range(max(int(max_iter), 1)):
            # Exact row block minimization for fixed v.
            u = waterfill(x - v[None, :], dim=1)
            # Exact column block minimization for fixed u.
            v = waterfill(x - u[:, None], dim=0)
            plan = (x - u[:, None] - v[None, :]).clamp_min(0.0)
            residual = torch.maximum(
                (plan.sum(dim=1) - target).abs().amax(),
                (plan.sum(dim=0) - target).abs().amax(),
            )
            if residual <= tol:
                return plan, {
                    "solver": "qrot_bcd", "iterations": iteration + 1,
                    "relative_residual": float((residual / target).detach().cpu()),
                    "warm_started": warm_started,
                    "active_fraction": float((plan > 0.0).to(dtype=x.dtype).mean().detach().cpu()),
                    "dual_state": (u.detach(), v.detach()),
                    "active_state": (plan > 0.0).detach(),
                }

        return plan, {
            "solver": "qrot_bcd_inexact", "iterations": max(int(max_iter), 1),
            "relative_residual": float((residual / target).detach().cpu()),
            "warm_started": warm_started,
            "active_fraction": float((plan > 0.0).to(dtype=x.dtype).mean().detach().cpu()),
            "dual_state": (u.detach(), v.detach()),
            "active_state": (plan > 0.0).detach(),
        }

    @staticmethod
    @torch.no_grad()
    def _entropy_euclidean_projection(
        matrix,
        marginal,
        entropy_eps,
        max_iter=20,
        relative_tol=1.0e-3,
        dual_warm_start=None,
    ):
        """Entropy-smoothed Euclidean projection onto the transport polytope.

        Solves the actual subproblem

            min_T 0.5 ||T - matrix||_F^2
                   + entropy_eps * sum(T * log(T))
            s.t. T >= 0, T 1 = a, T^T 1 = a.

        This is the requested ``0.5||T-X||^2 - eps H(T)`` projection.  The
        entropy term makes the KKT plan strictly positive, so no active-set
        truncation is used.  We solve the smooth dual with Newton/Schur.  The
        scalar primal map is evaluated in log-space by a short Newton solve:

            t + eps * log(t) = s.

        ``dual_warm_start`` is kept separate from the nonsmooth Euclidean
        projector because its Jacobian is W=t/(t+eps), not an active mask.
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("entropy projection requires a square matrix")
        if entropy_eps <= 0.0:
            raise ValueError("entropy_eps must be positive")
        n = matrix.shape[0]
        if n == 1:
            return matrix.new_full((1, 1), float(marginal)), {
                "solver": "entropy_newton", "iterations": 0,
                "relative_residual": 0.0, "warm_started": False,
                "dual_state": (matrix.new_zeros(1), matrix.new_zeros(1)),
                "active_state": torch.ones_like(matrix, dtype=torch.bool),
                "active_fraction": 1.0,
            }

        x = matrix
        eps = torch.as_tensor(float(entropy_eps), dtype=x.dtype, device=x.device)
        target = torch.as_tensor(float(marginal), dtype=x.dtype, device=x.device)
        tol = target.abs().clamp_min(1.0e-12) * float(relative_tol)

        def primal_map(score):
            # Solve exp(y) + eps*y = score, with t=exp(y).  The log-domain
            # solve remains positive and avoids a Lambert-W dependency.
            y = torch.where(
                score > eps,
                torch.log(score.clamp_min(eps)),
                score / eps,
            )
            for _ in range(7):
                ey = y.exp()
                y = y - (ey + eps * y - score) / (ey + eps)
                y = y.clamp(min=-80.0, max=20.0)
            return y.exp()

        warm_started = False
        if (
            dual_warm_start is not None
            and isinstance(dual_warm_start, (tuple, list))
            and len(dual_warm_start) == 2
            and tuple(dual_warm_start[0].shape) == (n,)
            and tuple(dual_warm_start[1].shape) == (n,)
            and dual_warm_start[0].device == x.device
            and dual_warm_start[1].device == x.device
            and torch.isfinite(dual_warm_start[0]).all()
            and torch.isfinite(dual_warm_start[1]).all()
        ):
            u = dual_warm_start[0].to(dtype=x.dtype).clone()
            v = dual_warm_start[1].to(dtype=x.dtype).clone()
            warm_started = True
        else:
            u = torch.zeros(n, dtype=x.dtype, device=x.device)
            v = torch.zeros(n, dtype=x.dtype, device=x.device)

        plan = primal_map(x - u[:, None] - v[None, :])
        residual = torch.full((), float("inf"), dtype=x.dtype, device=x.device)
        iterations = max(int(max_iter), 1)
        for iteration in range(iterations):
            plan = primal_map(x - u[:, None] - v[None, :])
            weight = plan / (plan + eps)
            row_residual = plan.sum(dim=1) - target
            col_residual = plan.sum(dim=0) - target
            residual = torch.maximum(row_residual.abs().amax(), col_residual.abs().amax())
            if residual <= tol:
                break

            row_degree = weight.sum(dim=1).clamp_min(torch.finfo(x.dtype).eps)
            inv_row_degree = row_degree.reciprocal()
            block = weight[:, :-1]
            col_degree = weight.sum(dim=0)
            schur = torch.diag(col_degree[:-1]) - (
                block.t() * inv_row_degree[None, :]
            ) @ block
            rhs = col_residual[:-1] - block.t() @ (inv_row_degree * row_residual)
            # A tiny ridge handles finite-precision near-gauge directions;
            # unlike the nonsmooth path, the entropy Hessian is positive.
            ridge = torch.finfo(x.dtype).eps * max(1.0, float(n))
            schur = schur + ridge * torch.eye(n - 1, dtype=x.dtype, device=x.device)
            chol, info = torch.linalg.cholesky_ex(schur, check_errors=False)
            if not bool((info == 0).all()):
                ridge = ridge * 1.0e3
                schur = schur + ridge * torch.eye(n - 1, dtype=x.dtype, device=x.device)
                chol, info = torch.linalg.cholesky_ex(schur, check_errors=False)
            if not bool((info == 0).all()):
                break
            delta_v = torch.linalg.solve_triangular(
                chol.t(),
                torch.linalg.solve_triangular(chol, rhs[:, None], upper=False),
                upper=True,
            ).squeeze(1)
            delta_u = inv_row_degree * (row_residual - block @ delta_v)
            # A damped Newton step prevents overflow when starting from the
            # diagonal plan and keeps the smooth dual in a finite range.
            step = 1.0
            old_residual = residual
            for _ in range(4):
                trial_u = u + step * delta_u
                trial_v = v.clone()
                trial_v[:-1] = trial_v[:-1] + step * delta_v
                trial_plan = primal_map(x - trial_u[:, None] - trial_v[None, :])
                trial_residual = torch.maximum(
                    (trial_plan.sum(dim=1) - target).abs().amax(),
                    (trial_plan.sum(dim=0) - target).abs().amax(),
                )
                if trial_residual <= old_residual:
                    u, v, plan, residual = trial_u, trial_v, trial_plan, trial_residual
                    break
                step *= 0.5
            else:
                u = u + 0.125 * delta_u
                v[:-1] = v[:-1] + 0.125 * delta_v

        plan = primal_map(x - u[:, None] - v[None, :])
        residual = torch.maximum(
            (plan.sum(dim=1) - target).abs().amax(),
            (plan.sum(dim=0) - target).abs().amax(),
        )
        return plan, {
            "solver": "entropy_newton",
            "iterations": iteration + 1,
            "relative_residual": float((residual / target).detach().cpu()),
            "warm_started": warm_started,
            "active_fraction": float((plan > 0.0).to(dtype=x.dtype).mean().detach().cpu()),
            "dual_state": (u.detach(), v.detach()),
            "active_state": (plan > 0.0).detach(),
        }

    @staticmethod
    @torch.no_grad()
    def _entropy_euclidean_projection_bcd(
        matrix,
        marginal,
        entropy_eps,
        max_iter=20,
        relative_tol=1.0e-3,
        dual_warm_start=None,
    ):
        """Alternating dual Newton for the entropy-smoothed projection.

        This follows the generalized Sinkhorn form directly.  With

            T_ij = phi_eps(X_ij - u_i - v_j),
            phi_eps(s) + eps*log(phi_eps(s)) = s,

        each row/column potential is updated by its one-dimensional Newton
        correction.  Unlike the Euclidean projector, every entry remains
        strictly positive (up to floating-point underflow).
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("entropy BCD projection requires a square matrix")
        if entropy_eps <= 0.0:
            raise ValueError("entropy_eps must be positive")
        n = matrix.shape[0]
        x = matrix
        eps = torch.as_tensor(float(entropy_eps), dtype=x.dtype, device=x.device)
        target = torch.as_tensor(float(marginal), dtype=x.dtype, device=x.device)
        tol = target.abs().clamp_min(1.0e-12) * float(relative_tol)

        def primal_map(score):
            y = torch.where(
                score > eps,
                torch.log(score.clamp_min(eps)),
                score / eps,
            )
            for _ in range(7):
                ey = y.exp()
                y = y - (ey + eps * y - score) / (ey + eps)
                y = y.clamp(min=-80.0, max=20.0)
            return y.exp()

        warm_started = False
        if (
            dual_warm_start is not None
            and isinstance(dual_warm_start, (tuple, list))
            and len(dual_warm_start) == 2
            and tuple(dual_warm_start[0].shape) == (n,)
            and tuple(dual_warm_start[1].shape) == (n,)
            and dual_warm_start[0].device == x.device
            and dual_warm_start[1].device == x.device
            and torch.isfinite(dual_warm_start[0]).all()
            and torch.isfinite(dual_warm_start[1]).all()
        ):
            u = dual_warm_start[0].to(dtype=x.dtype).clone()
            v = dual_warm_start[1].to(dtype=x.dtype).clone()
            warm_started = True
        else:
            u = torch.zeros(n, dtype=x.dtype, device=x.device)
            v = torch.zeros(n, dtype=x.dtype, device=x.device)

        plan = primal_map(x - u[:, None] - v[None, :])
        residual = torch.full((), float("inf"), dtype=x.dtype, device=x.device)
        for iteration in range(max(int(max_iter), 1)):
            # Row dual Newton: dT/du = -T/(T+eps).
            plan = primal_map(x - u[:, None] - v[None, :])
            weight = plan / (plan + eps)
            row_residual = plan.sum(dim=1) - target
            row_slope = weight.sum(dim=1).clamp_min(torch.finfo(x.dtype).eps)
            u = u + row_residual / row_slope

            # Column dual Newton: dT/dv = -T/(T+eps).
            plan = primal_map(x - u[:, None] - v[None, :])
            weight = plan / (plan + eps)
            col_residual = plan.sum(dim=0) - target
            col_slope = weight.sum(dim=0).clamp_min(torch.finfo(x.dtype).eps)
            v = v + col_residual / col_slope

            # Fix the additive gauge without changing X-u-v.
            gauge = v[-1].clone()
            v = v - gauge
            u = u + gauge

            plan = primal_map(x - u[:, None] - v[None, :])
            residual = torch.maximum(
                (plan.sum(dim=1) - target).abs().amax(),
                (plan.sum(dim=0) - target).abs().amax(),
            )
            if residual <= tol:
                break

        return plan, {
            "solver": "entropy_bcd",
            "iterations": iteration + 1,
            "relative_residual": float((residual / target).detach().cpu()),
            "warm_started": warm_started,
            "active_fraction": float((plan > 0.0).to(dtype=x.dtype).mean().detach().cpu()),
            "dual_state": (u.detach(), v.detach()),
            "active_state": (plan > 0.0).detach(),
        }

    @staticmethod
    @torch.no_grad()
    def _componentwise_newton_direction(
        x, target, u, v, active, row_residual, col_residual
    ):
        """Solve the Newton equations after fixing one gauge per support component.

        The ordinary reduced Schur system fixes only one global column dual.
        If the positive active bipartite graph is disconnected, every connected
        component has its own dual gauge and the global system is singular or
        badly conditioned.  This slow path is invoked only after the global
        direction fails to decrease the primal marginal residual.
        """
        n = x.shape[0]
        active_cpu = active.to(dtype=torch.bool).detach().cpu()
        row_to_cols = [torch.nonzero(active_cpu[i], as_tuple=False).flatten().tolist()
                       for i in range(n)]
        col_to_rows = [torch.nonzero(active_cpu[:, j], as_tuple=False).flatten().tolist()
                       for j in range(n)]
        seen_rows = [False] * n
        seen_cols = [False] * n
        components = []
        for start_row in range(n):
            if seen_rows[start_row] or not row_to_cols[start_row]:
                continue
            rows = []
            cols = []
            row_queue = [start_row]
            seen_rows[start_row] = True
            while row_queue:
                i = row_queue.pop()
                rows.append(i)
                for j in row_to_cols[i]:
                    if not seen_cols[j]:
                        seen_cols[j] = True
                        cols.append(j)
                        for next_i in col_to_rows[j]:
                            if not seen_rows[next_i]:
                                seen_rows[next_i] = True
                                row_queue.append(next_i)
            components.append((rows, cols))
        if len(components) <= 1 or any(len(r) == 0 or len(c) == 0 for r, c in components):
            return None

        delta_u = torch.zeros_like(u)
        delta_v = torch.zeros_like(v)
        for rows, cols in components:
            ridx = torch.as_tensor(rows, dtype=torch.long, device=x.device)
            cidx = torch.as_tensor(cols, dtype=torch.long, device=x.device)
            fixed_col = cidx[-1:]
            free_col = cidx[:-1]
            row_degree = active.index_select(0, ridx).sum(dim=1)
            inv_row_degree = row_degree.reciprocal()
            if free_col.numel() > 0:
                block = active.index_select(0, ridx).index_select(1, free_col)
                col_degree = active.index_select(0, ridx).index_select(1, cidx).sum(dim=0)
                schur = torch.diag(col_degree[:-1]) - (
                    block.t() * inv_row_degree[None, :]
                ) @ block
                rhs = col_residual.index_select(0, free_col) - block.t() @ (
                    inv_row_degree * row_residual.index_select(0, ridx)
                )
                chol, info = torch.linalg.cholesky_ex(schur, check_errors=False)
                if not bool((info == 0).all()):
                    return None
                dv = torch.linalg.solve_triangular(
                    chol.t(),
                    torch.linalg.solve_triangular(chol, rhs[:, None], upper=False),
                    upper=True,
                ).squeeze(1)
                delta_v[free_col] = dv
                delta_u[ridx] = inv_row_degree * (
                    row_residual.index_select(0, ridx) - block @ dv
                )
            else:
                delta_u[ridx] = inv_row_degree * row_residual.index_select(0, ridx)
            # fixed_col remains zero, selecting one gauge per component
            del fixed_col
        return delta_u, delta_v

    @staticmethod
    @torch.no_grad()
    def _active_schur_cg(
        active,
        row_degree,
        col_degree,
        inverse_row_degree,
        rhs,
        ridge,
        max_iter=24,
    ):
        """Matrix-free CG solve for the reduced active-set Schur system.

        RegOT's sparse Newton path avoids materializing a dense Hessian when
        the positive KKT support is small.  This is the same reduced operator
        used by the dense Schur path, evaluated with edge-index scatter-adds:

            S x = D_c x - A^T D_r^{-1} A x + ridge*x.

        It is an experimental fast path.  The caller still certifies the
        resulting primal plan and can fall back to the dense solver.
        """
        n = active.shape[0]
        m = n - 1
        if m <= 0:
            return rhs.new_zeros((0,))
        edge_rows, edge_cols = torch.nonzero(active[:, :-1] > 0.0, as_tuple=True)
        edge_cols = edge_cols.to(dtype=torch.long)
        edge_rows = edge_rows.to(dtype=torch.long)

        def matvec(value):
            row_acc = torch.zeros(n, dtype=value.dtype, device=value.device)
            row_acc.scatter_add_(0, edge_rows, value.index_select(0, edge_cols))
            weighted = inverse_row_degree * row_acc
            col_acc = torch.zeros(m, dtype=value.dtype, device=value.device)
            col_acc.scatter_add_(0, edge_cols, weighted.index_select(0, edge_rows))
            return col_degree[:-1] * value - col_acc + float(ridge) * value

        result = torch.zeros_like(rhs)
        residual = rhs - matvec(result)
        direction = residual.clone()
        denom = torch.dot(direction, direction).clamp_min(torch.finfo(rhs.dtype).eps)
        for _ in range(max(int(max_iter), 1)):
            hess_direction = matvec(direction)
            step = denom / torch.dot(direction, hess_direction).clamp_min(
                torch.finfo(rhs.dtype).eps
            )
            result = result + step * direction
            residual = residual - step * hess_direction
            next_denom = torch.dot(residual, residual)
            beta = next_denom / denom.clamp_min(torch.finfo(rhs.dtype).eps)
            direction = residual + beta * direction
            denom = next_denom
        return result

    @staticmethod
    @torch.no_grad()
    def _projection_dual_merit(plan, u, v, target):
        """Convex dual merit for the Euclidean transport projection.

        Up to an X-only constant, minimizing this quantity is equivalent to
        maximizing the projection dual.  It is used only as an optional
        globalization guard; primal feasibility remains the final certificate.
        """
        return 0.5 * plan.square().sum() + target * (u.sum() + v.sum())

    @staticmethod
    @torch.no_grad()
    def _record_transport_diagnostics(stats, transport, marginal):
        """Record primal feasibility without changing the solver path."""
        row_residual = (transport.sum(dim=1) - marginal).abs().amax()
        col_residual = (transport.sum(dim=0) - marginal).abs().amax()
        total_mass_error = (transport.sum() - 1.0).abs()
        stats["min_transport"] = min(
            stats.get("min_transport", float("inf")),
            float(transport.amin().detach().cpu()),
        )
        stats["max_row_residual"] = max(
            stats.get("max_row_residual", 0.0),
            float(row_residual.detach().cpu()),
        )
        stats["max_col_residual"] = max(
            stats.get("max_col_residual", 0.0),
            float(col_residual.detach().cpu()),
        )
        stats["total_mass_error"] = max(
            stats.get("total_mass_error", 0.0),
            float(total_mass_error.detach().cpu()),
        )

    @staticmethod
    @torch.no_grad()
    def _structure_energy(d1, d2, transport):
        """Variable part of the relation-alignment objective for a feasible T."""
        pushed_d2 = transport @ d2 @ transport.t()
        return -2.0 * torch.sum(d1 * pushed_d2)

    @staticmethod
    @torch.no_grad()
    def _structure_point_energy(d1, d2, point_cost, transport, alpha):
        """Variable mixed point-relation energy used by the optional line-search safeguard.

        The point term is linear in the coupling and the structural term is
        the variable part of relation-alignment.  ``alpha`` is the structural weight:
        alpha=1 reproduces the original method relation objective.
        """
        return ((1.0 - alpha) * torch.sum(point_cost * transport)
                + alpha * TransportProjector._structure_energy(d1, d2, transport))

    @staticmethod
    @torch.no_grad()
    def _euclidean_bistochastic_projection(
        matrix,
        marginal,
        max_iter=50,
        relative_tol=1.0e-3,
        schur_ridge=1.0e-6,
        retry_with_stronger_ridge=True,
        dual_warm_start=None,
        fast_full_step=False,
        sparse_schur=False,
        sparse_schur_iters=24,
        adaptive_ridge=False,
        active_warm_start=None,
        active_set_margin=0.0,
        dual_merit_guard=False,
        dual_merit_tol=1.0e-6,
    ):
        """GPU semismooth-Newton projection onto the uniform transport polytope.

        It solves ``min 0.5 ||T-X||_F^2`` subject to non-negativity and both
        marginals being ``marginal``.  The dual KKT form is
        ``T = [X-u1^T-1v^T]_+``.  A fixed gauge ``v[-1]=0`` gives an
        ``(n-1)`` Schur system.  A failure returns a safe feasible fallback plan.
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Euclidean transport requires a square transport matrix")
        n = matrix.shape[0]
        if n == 1:
            return matrix.new_full((1, 1), float(marginal)), {
                "solver": "trivial", "iterations": 0, "relative_residual": 0.0,
                "active_state": torch.ones_like(matrix, dtype=torch.bool),
            }

        x = matrix
        target = torch.as_tensor(marginal, dtype=x.dtype, device=x.device)
        total_target = target * float(n)
        # Exact affine projection: include the total-mass correction.
        row_error = x.sum(dim=1) - target
        col_error = x.sum(dim=0) - target
        total_error = x.sum() - total_target
        affine = (
            x
            - row_error[:, None] / float(n)
            - col_error[None, :] / float(n)
            + total_error / float(n * n)
        )
        affine_residual = torch.maximum(
            (affine.sum(dim=1) - target).abs().amax(),
            (affine.sum(dim=0) - target).abs().amax(),
        )
        tol = target.abs().clamp_min(1.0e-12) * float(relative_tol)
        if affine.amin() >= -tol and affine_residual <= tol:
            return affine.clamp_min(0.0), {
                "solver": "affine", "iterations": 0,
                "relative_residual": float((affine_residual / target).detach().cpu()),
                "active_state": (affine > 0.0).detach(),
            }

        # RegOT-style dual warm start: consecutive transport candidates are close,
        # so reusing the previous potentials is cheaper than rebuilding them
        # from the affine projection.  The gauge is fixed without changing
        # the primal quantity u_i + v_j.
        warm_started = False
        if (
            dual_warm_start is not None
            and isinstance(dual_warm_start, (tuple, list))
            and len(dual_warm_start) == 2
            and tuple(dual_warm_start[0].shape) == (n,)
            and tuple(dual_warm_start[1].shape) == (n,)
            and dual_warm_start[0].device == x.device
            and dual_warm_start[1].device == x.device
        ):
            u = dual_warm_start[0].to(dtype=x.dtype).clone()
            v = dual_warm_start[1].to(dtype=x.dtype).clone()
            if torch.isfinite(u).all() and torch.isfinite(v).all():
                warm_started = True
            else:
                warm_started = False

        if not warm_started:
            # X-u-v equals the affine solution for this dual initialization.
            u = row_error / float(n)
            v = col_error / float(n) - total_error / float(n * n)

        gauge_shift = v[-1].clone()
        u = u + gauge_shift
        v = v - gauge_shift
        eye = (
            torch.eye(n - 1, dtype=x.dtype, device=x.device)
            if not sparse_schur else None
        )
        accepted = False
        final_plan = None
        final_residual = None
        componentwise_used = False
        permutation_fast_path = 0
        active_predictor = None
        if (
            active_warm_start is not None
            and tuple(active_warm_start.shape) == tuple(x.shape)
            and active_warm_start.device == x.device
            and active_warm_start.dtype == torch.bool
        ):
            active_predictor = active_warm_start

        for iteration in range(max(int(max_iter), 1)):
            y = x - u[:, None] - v[None, :]
            plan = y.clamp_min(0.0)
            if not torch.isfinite(plan).all():
                break
            row_residual = plan.sum(dim=1) - target
            col_residual = plan.sum(dim=0) - target
            residual = torch.maximum(row_residual.abs().amax(), col_residual.abs().amax())
            final_plan, final_residual = plan, residual
            if residual <= tol:
                accepted = True
                break

            # DROT-style boundary handling: the primal plan is always the
            # positive part of the KKT expression.  Do not feed negative
            # boundary values into the generalized Hessian.
            active_bool = y > 0.0
            if active_predictor is not None and iteration == 0:
                # Only retain old edges that are within a small KKT boundary
                # band.  Large negative values are never forced into support.
                active_bool = active_bool | (
                    active_predictor & (y >= -float(active_set_margin))
                )
            active = active_bool.to(dtype=x.dtype)
            row_degree = active.sum(dim=1)
            col_degree = active.sum(dim=0)
            # An empty active row/column has no valid generalized Jacobian.
            if row_degree.amin() <= 0.0 or col_degree.amin() <= 0.0:
                break

            # Exact fast path for a permutation active support.  In this
            # case every connected support component contains one row and one
            # column, so the Newton correction is simply the row residual.
            # The next loop still rechecks the positive support and KKT
            # residual; this is not a relaxed projection.
            if bool(((row_degree == 1.0).all() & (col_degree == 1.0).all())):
                u = u + row_residual
                permutation_fast_path += 1
                continue

            active_short = active[:, :-1]
            inverse_row_degree = row_degree.reciprocal()
            rhs = col_residual[:-1] - active_short.t() @ (
                inverse_row_degree * row_residual
            )

            # A disconnected active support has more than one dual gauge.
            # Globalize this Newton direction with a finite primal-residual
            # line search.  If this trajectory stalls, the outer one-time
            # stronger-ridge restart below handles the degenerate support.
            current_norm = row_residual.square().sum() + col_residual.square().sum()
            best_u, best_v, best_norm = u, v, current_norm
            current_dual = TransportProjector._projection_dual_merit(plan, u, v, target)
            dual_limit = current_dual + float(dual_merit_tol) * (
                1.0 + current_dual.abs()
            )
            try:
                # The regularized Schur matrix is symmetric positive
                # definite on the valid active support.  Cholesky avoids the
                # pivoting/extra work of a generic LU solve; retain LU as a
                # boundary-safe fallback for degenerate supports.
                if sparse_schur:
                    delta_v_short = TransportProjector._active_schur_cg(
                        active,
                        row_degree,
                        col_degree,
                        inverse_row_degree,
                        rhs,
                        schur_ridge,
                        max_iter=sparse_schur_iters,
                    )
                else:
                    schur_core = torch.diag(col_degree[:-1]) - (
                        active_short.t() * inverse_row_degree[None, :]
                    ) @ active_short
                    schur = schur_core + float(schur_ridge) * eye
                    chol, chol_info = torch.linalg.cholesky_ex(schur, check_errors=False)
                    if bool((chol_info == 0).all()):
                        delta_v_short = torch.linalg.solve_triangular(
                            chol.t(),
                            torch.linalg.solve_triangular(chol, rhs[:, None], upper=False),
                            upper=True,
                        ).squeeze(1)
                    else:
                        delta_v_short = torch.linalg.solve(schur, rhs)
            except RuntimeError:
                break
            delta_v = torch.zeros_like(v)
            delta_v[:-1] = delta_v_short
            delta_u = inverse_row_degree * (
                row_residual - active_short @ delta_v_short
            )
            if fast_full_step:
                full_trial = (
                    x
                    - (u + delta_u)[:, None]
                    - (v + delta_v)[None, :]
                ).clamp_min(0.0)
                full_row = full_trial.sum(dim=1) - target
                full_col = full_trial.sum(dim=0) - target
                full_norm = full_row.square().sum() + full_col.square().sum()
                full_dual = TransportProjector._projection_dual_merit(
                    full_trial, u + delta_u, v + delta_v, target
                )
                full_ok = full_norm < current_norm
                if dual_merit_guard:
                    full_ok = full_ok & (full_dual <= dual_limit)
                if full_ok:
                    u = u + delta_u
                    v = v + delta_v
                    continue
            # Evaluate the finite globalization candidates in one batched
            # GPU expression instead of launching five sequential matrix
            # expressions.  This preserves the original acceptance rule.
            alphas = torch.as_tensor(
                (1.0, 0.5, 0.25, 0.125, 1.0 / 64.0),
                dtype=x.dtype,
                device=x.device,
            )
            trial_u = u[None, :] + alphas[:, None] * delta_u[None, :]
            trial_v = v[None, :] + alphas[:, None] * delta_v[None, :]
            trial = (
                x[None, :, :]
                - trial_u[:, :, None]
                - trial_v[:, None, :]
            ).clamp_min(0.0)
            trial_row = trial.sum(dim=2) - target
            trial_col = trial.sum(dim=1) - target
            trial_norms = trial_row.square().sum(dim=1) + trial_col.square().sum(dim=1)
            trial_dual = 0.5 * trial.square().sum(dim=(1, 2)) + target * (
                trial_u.sum(dim=1) + trial_v.sum(dim=1)
            )
            acceptable = trial_norms < current_norm
            if dual_merit_guard:
                acceptable = acceptable & (trial_dual <= dual_limit)
            ranked_norms = torch.where(
                acceptable, trial_norms, torch.full_like(trial_norms, float("inf"))
            )
            best_index = ranked_norms.argmin()
            candidate_norm = ranked_norms[best_index]
            if candidate_norm < best_norm:
                best_u = trial_u[best_index]
                best_v = trial_v[best_index]
                best_norm = candidate_norm
            if best_norm >= current_norm:
                component_direction = TransportProjector._componentwise_newton_direction(
                    x, target, u, v, active, row_residual, col_residual
                )
                if component_direction is None:
                    break
                component_du, component_dv = component_direction
                comp_trial_u = u[None, :] + alphas[:, None] * component_du[None, :]
                comp_trial_v = v[None, :] + alphas[:, None] * component_dv[None, :]
                comp_trial = (
                    x[None, :, :]
                    - comp_trial_u[:, :, None]
                    - comp_trial_v[:, None, :]
                ).clamp_min(0.0)
                comp_row = comp_trial.sum(dim=2) - target
                comp_col = comp_trial.sum(dim=1) - target
                comp_norms = comp_row.square().sum(dim=1) + comp_col.square().sum(dim=1)
                comp_dual = 0.5 * comp_trial.square().sum(dim=(1, 2)) + target * (
                    comp_trial_u.sum(dim=1) + comp_trial_v.sum(dim=1)
                )
                comp_acceptable = comp_norms < current_norm
                if dual_merit_guard:
                    comp_acceptable = comp_acceptable & (comp_dual <= dual_limit)
                comp_ranked = torch.where(
                    comp_acceptable, comp_norms,
                    torch.full_like(comp_norms, float("inf"))
                )
                comp_index = comp_ranked.argmin()
                if not torch.isfinite(comp_ranked[comp_index]):
                    break
                u = comp_trial_u[comp_index]
                v = comp_trial_v[comp_index]
                componentwise_used = True
            else:
                u, v = best_u, best_v

        if accepted:
            return final_plan, {
            "solver": "newton", "iterations": iteration + 1,
                "relative_residual": float((final_residual / target).detach().cpu()),
                "warm_started": warm_started,
                "active_fraction": float((plan > 0.0).to(dtype=x.dtype).mean().detach().cpu()),
                "componentwise_newton": componentwise_used,
                "permutation_fast_path": permutation_fast_path,
                "dual_state": (u.detach(), v.detach()),
                "active_state": (plan > 0.0).detach(),
            }

        # An early small-ridge trajectory can lock into a poor disconnected
        # support.  Restarting once from the affine dual initialization with
        # a stronger ridge is still a Newton solve and is preferable to
        # silently falling through to the slower fallback.
        if retry_with_stronger_ridge and float(schur_ridge) < 1.0e-2:
            if adaptive_ridge:
                ridge_candidates = [
                    max(float(schur_ridge) * 10.0, 1.0e-7),
                    max(float(schur_ridge) * 100.0, 1.0e-6),
                    1.0e-2,
                ]
                ridge_candidates = list(dict.fromkeys(ridge_candidates))
            else:
                ridge_candidates = [1.0e-2]
            for retry_ridge in ridge_candidates:
                retry_plan, retry_stats = TransportProjector._euclidean_bistochastic_projection(
                    matrix,
                    marginal,
                    max_iter=max_iter,
                    relative_tol=relative_tol,
                    schur_ridge=retry_ridge,
                    retry_with_stronger_ridge=False,
                    dual_warm_start=None,
                    fast_full_step=fast_full_step,
                    sparse_schur=False,
                    sparse_schur_iters=sparse_schur_iters,
                    adaptive_ridge=False,
                )
                retry_is_feasible = (
                    retry_stats["solver"] not in {"projection_failure"}
                    and float(retry_stats.get("relative_residual", float("inf")))
                    <= float(relative_tol)
                )
                if retry_is_feasible:
                    retry_stats["solver"] = "newton_ridge_retry"
                    retry_stats["ridge_used"] = retry_ridge
                    return retry_plan, retry_stats

        safe = torch.full_like(matrix, float(marginal) / float(n))
        residual = torch.maximum(
            (safe.sum(dim=1) - target).abs().amax(),
            (safe.sum(dim=0) - target).abs().amax(),
        )
        return safe, {
            "solver": "projection_failure", "iterations": int(max_iter),
            "relative_residual": float((residual / target).detach().cpu()),
            "warm_started": warm_started,
            "active_fraction": float((safe > 0.0).to(dtype=x.dtype).mean().detach().cpu()),
            "active_state": (safe > 0.0).detach(),
        }

    @torch.no_grad()
    def _structure_transport(
        self, query, key, param, query_target=None,
        precomputed_d1=None, precomputed_d2=None
    ):
        """Return detached row-stochastic dense Euclidean transport pseudo labels.

        ``query`` is online decoder prediction P and ``key`` is the EMA
        encoder Z_t.  Thus D1=1-P P^T and D2=1-Z_t Z_t^T.  Both relations are
        symmetric, although the general relation-alignment gradient is retained.
        """
        call_index = self._epoch_transport_calls
        self._epoch_transport_calls += 1
        first_batch = (
            not self._epoch_first_batch_decided and call_index < 2
        )
        if call_index >= 1:
            self._epoch_first_batch_decided = True

        structure_alpha = float(param.get("structure_alpha", 1.0))
        if not 0.0 <= structure_alpha <= 1.0:
            raise ValueError("structure_structure_alpha must be in [0, 1]")
        if precomputed_d1 is None or precomputed_d2 is None:
            query = L2norm(query)
            key = L2norm(key).detach()
            n = query.shape[0]
            if key.shape[0] != n:
                raise ValueError("STRIDE dense transport expects equal batch dimensions")
            if query_target is None:
                d1 = (1.0 - query @ query.t()).clamp(0.0, 2.0)
            else:
                query_target = L2norm(query_target).detach()
                if query_target.shape != query.shape:
                    raise ValueError("STRIDE dense transport query_target must match query shape")
                d1 = (1.0 - query @ query_target.t()).clamp(0.0, 2.0)
            d2 = (1.0 - key @ key.t()).clamp(0.0, 2.0)
            d2 = 0.5 * (d2 + d2.t())
        else:
            d1 = precomputed_d1
            d2 = precomputed_d2
            n = d1.shape[0]
            if d1.ndim != 2 or d2.shape != d1.shape or d1.shape[1] != n:
                raise ValueError("precomputed relations must be matching square matrices")
            if structure_alpha < 1.0 or bool(param.get("structure_energy_guard", False)):
                query = L2norm(query)
                key = L2norm(key).detach()

        marginal = 1.0 / float(n)
        # The point term is only needed for an actual mixed point-relation mixture or the
        # optional energy guard.  The canonical setting is alpha=1,
        # so computing this matrix would be pure overhead.
        use_point_cost = (
            structure_alpha < 1.0
            or bool(param.get("structure_energy_guard", False))
        )
        point_cost = (
            (1.0 - query @ key.t()).clamp(0.0, 2.0)
            if use_point_cost else None
        )
        init_mode = str(param.get("structure_init", "diagonal")).lower()
        if init_mode == "diagonal":
            transport = torch.eye(n, dtype=query.dtype, device=query.device) * marginal
        elif init_mode == "uniform":
            # `transport` is stored in mass convention: every row and column
            # must sum to 1/n, hence a uniform plan has entries 1/n^2.
            transport = torch.full(
                (n, n), marginal / float(n),
                dtype=query.dtype, device=query.device,
            )
        else:
            raise ValueError(
                "structure_init must be either 'diagonal' or 'uniform', "
                f"got {init_mode!r}"
            )
        initial_transport = transport.clone()
        dual_state = None
        entropy_dual_state = None
        active_state = None

        stats = {
            "newton": 0,
            "entropy_newton": 0,
            "entropy_bcd": 0,
            "newton_ridge_retry": 0,
            "qrot_bcd": 0,
            "qrot_warmup": 0,
            "qrot_direct": 0,
            "affine": 0,
            "projection_failure": 0,
            "warm_started": 0,
            "active_fraction_min": 1.0,
            "active_fraction_max": 0.0,
            "componentwise_newton": 0,
            "prior_gate": 0,
            "prior_precheck": 0,
            "epoch_prior_fallback": 0,
            "worst_relative_residual": 0.0,
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
            "fast_accepts": 0,
            "fast_cert_failures": 0,
            "strict_repairs": 0,
            "structure_update_mix_min": 1.0,
            "structure_alpha": structure_alpha,
            "structure_init": init_mode,
            "point_cost_mean": (
                float(point_cost.mean().detach().cpu())
                if point_cost is not None else 0.0
            ),
            "point_gradient_norm": 0.0,
            "structure_gradient_norm": 0.0,
            "mixed_gradient_norm": 0.0,
            # Relation-chain audit for the canonical online-P source relation.
            "d1_asymmetry": float((d1 - d1.t()).abs().mean().detach().cpu()),
            "query_target_gap": (
                float((L2norm(query) - L2norm(query_target)).abs().mean().detach().cpu())
                if query_target is not None else 0.0
            ),
        }
        if self._epoch_fallback:
            stats["prior_gate"] = 1
            stats["prior_precheck"] = 1
            stats["epoch_prior_fallback"] = 1
            stats["diagonal_mass"] = 1.0
            stats["off_diagonal_mass"] = 0.0
            self._record_transport_diagnostics(stats, initial_transport, marginal)
            self.last_projection_stats = stats
            return initial_transport * float(n)

        for _ in range(max(int(param["structure_steps"]), 0)):
            # Always use the general relation-alignment gradient.  The shortcut
            # -4*D1@T@D2 is algebraically valid only when both relations are
            # exactly symmetric.  In this training path D1 is formed by a
            # finite-precision GEMM (and may also be precomputed from an
            # online/target relation), so using the general expression keeps
            # the energy test and the actual update direction consistent.
            structure_gradient = -2.0 * (
                d1 @ transport @ d2.t() + d1.t() @ transport @ d2
            )
            gradient = (
                structure_gradient
                if structure_alpha >= 1.0
                else (1.0 - structure_alpha) * point_cost + structure_alpha * structure_gradient
            )
            if point_cost is None:
                stats["point_gradient_norm"] = 0.0
            else:
                stats["point_gradient_norm"] = float(
                    ((1.0 - structure_alpha) * point_cost).norm().detach().cpu()
                )
            stats["structure_gradient_norm"] = float(
                (structure_alpha * structure_gradient).norm().detach().cpu()
            )
            stats["mixed_gradient_norm"] = float(gradient.norm().detach().cpu())
            candidate = transport - float(param["structure_step_size"]) * gradient
            previous_transport = transport
            if bool(param.get("structure_prior_precheck", True)) and bool(
                param.get("structure_prior_gate", False)
            ):
                positive_candidate = candidate.clamp_min(0.0)
                positive_total = positive_candidate.sum().clamp_min(1.0e-12)
                positive_offdiag = (
                    positive_total - positive_candidate.diagonal().sum()
                ) / positive_total
                if positive_offdiag <= float(
                    param.get("structure_prior_gate_offdiag_tol", 1.0e-3)
                ):
                    if first_batch:
                        self._epoch_fallback = True
                    stats["prior_gate"] += 1
                    stats["prior_precheck"] += 1
                    stats["epoch_prior_fallback"] = int(self._epoch_fallback)
                    transport = initial_transport.clone()
                    self._record_transport_diagnostics(stats, transport, marginal)
                    stats["diagonal_mass"] = 1.0
                    stats["off_diagonal_mass"] = 0.0
                    self.last_projection_stats = stats
                    return transport * float(n)
            projection_solver = param.get("structure_projection_solver", "newton")
            entropy_eps = float(param.get("structure_entropy_projection_eps", 0.0))
            effective_entropy_eps = float(param.get("structure_entropy_effective_eps", 0.0))
            if effective_entropy_eps > 0.0:
                entropy_eps = effective_entropy_eps / float(n)
            if entropy_eps > 0.0:
                entropy_method = param.get("structure_entropy_projection_method", "schur")
                entropy_solver = (
                    self._entropy_euclidean_projection_bcd
                    if entropy_method == "bcd"
                    else self._entropy_euclidean_projection
                )
                transport, projection_stats = entropy_solver(
                    candidate, marginal, entropy_eps=entropy_eps,
                    max_iter=int(param["structure_newton_iters"]),
                    relative_tol=float(param["structure_projection_rel_tol"]),
                    dual_warm_start=entropy_dual_state,
                )
            elif projection_solver == "qrot_bcd":
                transport, projection_stats = self._qrot_bcd_projection(
                    candidate,
                    marginal,
                    max_iter=int(param.get("structure_qrot_bcd_iters", 50)),
                    relative_tol=float(param["structure_projection_rel_tol"]),
                    dual_warm_start=dual_state,
                )
            elif projection_solver == "qrot_hybrid":
                # Use QROT/BCD as a cheap dual predictor, never as an
                # unconditional approximate label.  This borrows the dual
                # warm-start idea while retaining the Euclidean projection
                # certification of the Newton/Schur path.
                qrot_plan, qrot_stats = self._qrot_bcd_projection(
                    candidate,
                    marginal,
                    max_iter=int(param.get("structure_qrot_warmup_iters", 10)),
                    relative_tol=float(param.get(
                        "structure_qrot_warmup_rel_tol",
                        param["structure_projection_rel_tol"],
                    )),
                    dual_warm_start=dual_state,
                )
                stats["qrot_warmup"] += 1
                if (
                    qrot_stats["solver"] == "qrot_bcd"
                    and bool(param.get("structure_qrot_hybrid_direct", False))
                ):
                    transport, projection_stats = qrot_plan, qrot_stats
                    stats["qrot_direct"] += 1
                else:
                    transport, projection_stats = self._euclidean_bistochastic_projection(
                        candidate,
                        marginal,
                        max_iter=int(param["structure_newton_iters"]),
                        relative_tol=float(param["structure_projection_rel_tol"]),
                        schur_ridge=float(param["structure_schur_regularization"]),
                        dual_warm_start=qrot_stats.get("dual_state"),
                        fast_full_step=bool(param.get("structure_newton_fast_full_step", False)),
                        adaptive_ridge=bool(param.get("structure_adaptive_ridge", False)),
                        active_warm_start=(
                            active_state if param.get("structure_active_set_warm_start", False) else None
                        ),
                        active_set_margin=float(param.get("structure_active_set_margin", 0.0)),
                        dual_merit_guard=bool(param.get("structure_dual_merit_guard", False)),
                        dual_merit_tol=float(param.get("structure_dual_merit_tol", 1.0e-6)),
                    )
                    projection_stats["qrot_warmup_iterations"] = qrot_stats.get("iterations", 0)
            elif projection_solver == "newton_sparse":
                transport, projection_stats = self._euclidean_bistochastic_projection(
                    candidate,
                    marginal,
                    max_iter=int(param["structure_newton_iters"]),
                    relative_tol=float(param["structure_projection_rel_tol"]),
                    schur_ridge=float(param["structure_schur_regularization"]),
                    dual_warm_start=dual_state,
                    fast_full_step=bool(param.get("structure_newton_fast_full_step", False)),
                    sparse_schur=True,
                    sparse_schur_iters=int(param.get("structure_sparse_schur_iters", 24)),
                    adaptive_ridge=bool(param.get("structure_adaptive_ridge", False)),
                    active_warm_start=(
                        active_state if param.get("structure_active_set_warm_start", False) else None
                    ),
                    active_set_margin=float(param.get("structure_active_set_margin", 0.0)),
                    dual_merit_guard=bool(param.get("structure_dual_merit_guard", False)),
                    dual_merit_tol=float(param.get("structure_dual_merit_tol", 1.0e-6)),
                )
            else:
                transport, projection_stats = self._euclidean_bistochastic_projection(
                    candidate,
                    marginal,
                    max_iter=int(param["structure_newton_iters"]),
                    relative_tol=float(param["structure_projection_rel_tol"]),
                    schur_ridge=float(param["structure_schur_regularization"]),
                    dual_warm_start=dual_state,
                    fast_full_step=bool(param.get("structure_newton_fast_full_step", False)),
                    adaptive_ridge=bool(param.get("structure_adaptive_ridge", False)),
                    active_warm_start=(
                        active_state if param.get("structure_active_set_warm_start", False) else None
                    ),
                    active_set_margin=float(param.get("structure_active_set_margin", 0.0)),
                    dual_merit_guard=bool(param.get("structure_dual_merit_guard", False)),
                    dual_merit_tol=float(param.get("structure_dual_merit_tol", 1.0e-6)),
                )
            # The fast solver is accepted on a final primal marginal check.
            # Strict repair is deliberately not triggered by a projected-
            # gradient optimality heuristic: that heuristic caused almost
            # every normal projection to be solved twice.  The fast/strict
            # split is a numerical certificate, not a second optimization
            # objective.
            if (
                param.get("structure_stabilize_updates", False)
                and projection_solver == "newton"
            ):
                fast_tol = float(param.get(
                    "structure_fast_accept_rel_tol",
                    param["structure_projection_rel_tol"],
                ))
                strict_tol = float(param.get(
                    "structure_strict_repair_rel_tol",
                    param.get("structure_stabilize_projection_rel_tol", 1.0e-5),
                ))
                actual_residual = torch.maximum(
                    (transport.sum(dim=1) - marginal).abs().amax(),
                    (transport.sum(dim=0) - marginal).abs().amax(),
                ) / max(abs(float(marginal)), 1.0e-12)
                actual_residual_value = float(actual_residual.detach().cpu())
                if (
                    torch.isfinite(transport).all()
                    and actual_residual_value <= fast_tol
                ):
                    stats["fast_accepts"] += 1
                else:
                    stats["fast_cert_failures"] += 1
                    transport, projection_stats = self._euclidean_bistochastic_projection(
                        candidate,
                        marginal,
                        max_iter=int(param["structure_newton_iters"]),
                        relative_tol=strict_tol,
                        schur_ridge=float(param["structure_schur_regularization"]),
                        dual_warm_start=projection_stats.get("dual_state", dual_state),
                        fast_full_step=False,
                        adaptive_ridge=bool(param.get("structure_adaptive_ridge", False)),
                        active_warm_start=projection_stats.get("active_state", active_state),
                        active_set_margin=float(param.get("structure_active_set_margin", 0.0)),
                        dual_merit_guard=bool(param.get("structure_dual_merit_guard", False)),
                        dual_merit_tol=float(param.get("structure_dual_merit_tol", 1.0e-6)),
                    )
                    stats["projection_refined"] += 1
                    stats["strict_repairs"] += 1

            if projection_stats["solver"] in {"projection_failure", "qrot_bcd_inexact"}:
                # Preserve the last valid transport instead of injecting a
                # numerically infeasible pseudo-label into the CE target.
                transport = transport.detach() * 0.0 + previous_transport
                dual_state = None
                active_state = None
                stats["projection_failure"] = stats.get("projection_failure", 0) + 1
                self._record_transport_diagnostics(stats, transport, marginal)
                continue
            if projection_stats.get("solver") in {"entropy_newton", "entropy_bcd"}:
                entropy_dual_state = projection_stats.get("dual_state")
            else:
                dual_state = projection_stats.get("dual_state")
            active_state = projection_stats.get("active_state")
            if projection_stats.get("warm_started", False):
                stats["warm_started"] += 1
            if projection_stats.get("componentwise_newton", False):
                stats["componentwise_newton"] += 1
            if "active_fraction" in projection_stats:
                stats["active_fraction_min"] = min(
                    stats["active_fraction_min"], projection_stats["active_fraction"]
                )
                stats["active_fraction_max"] = max(
                    stats["active_fraction_max"], projection_stats["active_fraction"]
                )
            stats["worst_relative_residual"] = max(
                stats["worst_relative_residual"], projection_stats["relative_residual"]
            )
            self._record_transport_diagnostics(stats, transport, marginal)
            # Count the solver that produced this candidate before applying
            # the semantic prior gate. A gated candidate is still a valid
            # projection attempt and must remain visible in diagnostics.
            if projection_stats["solver"] == "newton":
                stats["newton"] += 1
            elif projection_stats["solver"] == "entropy_newton":
                stats["entropy_newton"] += 1
            elif projection_stats["solver"] == "entropy_bcd":
                stats["entropy_bcd"] += 1
            elif projection_stats["solver"] == "newton_ridge_retry":
                stats["newton_ridge_retry"] += 1
            elif projection_stats["solver"] == "affine":
                stats["affine"] += 1
            elif projection_stats["solver"].startswith("qrot_bcd"):
                stats["qrot_bcd"] = stats.get("qrot_bcd", 0) + 1
            if param.get("structure_stabilize_updates", False):
                # Preserve the Git-version update semantics.  The projected
                # endpoint is only a proposal; the actual transport update
                # is selected from a short, discrete damping grid.  If no
                # candidate satisfies the quadratic energy test, retain the
                # previous feasible plan instead of injecting the proposal.
                direction = transport - previous_transport
                first_order = torch.sum(gradient * direction)
                projection_optimality = first_order + direction.square().sum() / max(
                    float(param["structure_step_size"]), 1.0e-30
                )
                projection_optimality_value = float(
                    projection_optimality.detach().cpu()
                )
                stats["structure_projection_optimality_max"] = max(
                    stats["structure_projection_optimality_max"],
                    projection_optimality_value,
                )
                if projection_optimality_value > 1.0e-7:
                    stats["structure_projection_optimality_violations"] += 1
                quadratic = -2.0 * structure_alpha * torch.sum(
                    d1 * (direction @ d2 @ direction.t())
                )
                first_order_value = float(first_order.detach().cpu())
                min_gamma_delta = (
                    (1.0 / 64.0) * first_order
                    + (1.0 / 64.0) ** 2 * quadratic
                )
                min_gamma_delta_value = float(min_gamma_delta.detach().cpu())
                stats["structure_first_order_min"] = min(
                    stats["structure_first_order_min"], first_order_value
                )
                stats["structure_first_order_max"] = max(
                    stats["structure_first_order_max"], first_order_value
                )
                stats["structure_delta_min_gamma_max"] = max(
                    stats["structure_delta_min_gamma_max"], min_gamma_delta_value
                )
                if first_order_value > 1.0e-8:
                    stats["structure_descent_violations"] += 1
                energy_tolerance = float(
                    param.get("structure_energy_tolerance", 1.0e-7)
                )
                damping_values = (1.0, 0.5, 0.25, 0.125, 0.015625)
                gamma_value = None
                for value in damping_values:
                    value_tensor = torch.as_tensor(
                        value, device=transport.device, dtype=transport.dtype
                    )
                    delta = (
                        value_tensor * first_order
                        + value_tensor.square() * quadratic
                    )
                    if bool(torch.isfinite(delta).item()) and float(
                        delta.detach().cpu()
                    ) <= energy_tolerance:
                        gamma_value = value
                        break
                if gamma_value is None:
                    stats["structure_energy_rejected"] += 1
                    stats["structure_kept_old_plan"] += 1
                    stats["structure_gamma_min"] = min(stats["structure_gamma_min"], 0.0)
                    transport = previous_transport
                    dual_state = None
                    active_state = None
                    self._record_transport_diagnostics(stats, transport, marginal)
                    continue
                gamma = torch.as_tensor(
                    gamma_value, device=transport.device, dtype=transport.dtype
                )
                stats["structure_gamma_min"] = min(stats["structure_gamma_min"], gamma_value)
                if gamma_value < 1.0:
                    stats["structure_energy_damped"] += 1
                    stats["structure_energy_rejected"] += 1
                transport = previous_transport + gamma * direction
                # The projected endpoint's dual potentials are not an exact
                # certificate for the interior damped point.  Force the next
                # projection to reinitialize its dual state.
                dual_state = None if gamma_value < 1.0 else dual_state
            elif param.get("structure_energy_guard", False):
                # The projected candidate is feasible, but the structural
                # objective is non-convex and the projector is approximate.
                # Minimize the exact mixed mixed point-relation quadratic on the feasible
                # segment between the previous and candidate couplings.
                direction = transport - previous_transport
                base_energy = self._structure_point_energy(
                    d1, d2, point_cost, previous_transport, structure_alpha
                )
                candidate_energy = self._structure_point_energy(
                    d1, d2, point_cost, transport, structure_alpha
                )
                structure_first_order = -2.0 * (
                    torch.sum(d1 * (direction @ d2 @ previous_transport.t()))
                    + torch.sum(d1 * (previous_transport @ d2 @ direction.t()))
                )
                first_order = ((1.0 - structure_alpha) * torch.sum(point_cost * direction)
                               + structure_alpha * structure_first_order)
                quadratic = candidate_energy - base_energy - first_order
                if float(quadratic.detach().cpu()) > 1.0e-10:
                    gamma = (-first_order / (2.0 * quadratic)).clamp(0.0, 1.0)
                else:
                    energy_tol = float(param.get("structure_energy_tol", 1.0e-7))
                    gamma = torch.ones_like(quadratic)
                    if float(candidate_energy.detach().cpu()) > float(base_energy.detach().cpu()) + energy_tol:
                        gamma = torch.zeros_like(quadratic)
                gamma_value = float(gamma.detach().cpu())
                stats["structure_gamma_min"] = min(stats["structure_gamma_min"], gamma_value)
                if gamma_value < 1.0 - 1.0e-6:
                    stats["structure_energy_damped"] += 1
                    dual_state = None
                    transport = previous_transport + gamma * direction
                if gamma_value <= 1.0e-6:
                    stats["structure_energy_rejected"] += 1
            update_mix = float(param.get("structure_update_mix", 1.0))
            if not 0.0 < update_mix <= 1.0:
                raise ValueError("structure_update_mix must be in (0, 1]")
            if update_mix < 1.0:
                transport = previous_transport + update_mix * (transport - previous_transport)
                # The dual potentials belong to the projected endpoint, not
                # to this interior trust-region point.
                dual_state = None
                stats["structure_update_mix_min"] = min(stats["structure_update_mix_min"], update_mix)
            projected_offdiag = float(
                (1.0 - transport.diagonal().sum()).clamp(0.0, 1.0).detach().cpu()
            )
            if param.get("structure_prior_gate", False):
                offdiag_tol = float(param.get("structure_prior_gate_offdiag_tol", 1.0e-3))
                # Only gate an actually uninformative near-diagonal proposal.
                # A relative drop from the previous transport is not enough:
                # it can reject a useful structural correction late in training.
                if projected_offdiag <= offdiag_tol:
                    # Do not inject a numerically valid but semantically
                    # uninformative near-diagonal step.  Keep the last valid
                    # coupling and stop this finite diffusion trajectory.
                    if first_batch:
                        self._epoch_fallback = True
                    transport = initial_transport.clone()
                    dual_state = None
                    stats["prior_gate"] += 1
                    stats["epoch_prior_fallback"] = int(self._epoch_fallback)
                    self._record_transport_diagnostics(stats, transport, marginal)
                    stats["diagonal_mass"] = 1.0
                    stats["off_diagonal_mass"] = 0.0
                    break
        self.last_projection_stats = stats
        row_stochastic = transport * float(n)
        diagonal_mass = row_stochastic.diagonal().sum() / float(n)
        # The scalar diagnostic is formed from a float32 reduction; clamp the
        # reported mass to its physical range so roundoff cannot show a tiny
        # negative off-diagonal mass even when the plan itself is nonnegative.
        diagonal_mass = diagonal_mass.clamp(0.0, 1.0)
        stats["diagonal_mass"] = float(diagonal_mass.detach().cpu())
        stats["off_diagonal_mass"] = float((1.0 - diagonal_mass).detach().cpu())
        return row_stochastic


