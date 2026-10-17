"""GPU Euclidean projection onto the uniform transport polytope.

The neural model calls only ``EuclideanProjector.project``.  Solver
tolerances, active-set handling, and fallback policy are implementation
details rather than YAML training modes.
"""

import torch


class EuclideanProjector:
    @staticmethod
    @torch.no_grad()
    def _fallback_projection(matrix, marginal, max_iter, tol):
        """Finite robust fallback for the Euclidean transport projection."""
        x = matrix
        n = x.shape[0]
        corr_row = torch.zeros_like(x)
        corr_col = torch.zeros_like(x)
        corr_nonneg = torch.zeros_like(x)
        target = torch.as_tensor(marginal, dtype=x.dtype, device=x.device)
        for _ in range(max(int(max_iter), 1)):
            y = x + corr_row
            x_next = y + ((target - y.sum(dim=1)) / float(n))[:, None]
            corr_row = y - x_next

            y = x_next + corr_col
            x_next = y + ((target - y.sum(dim=0)) / float(n))[None, :]
            corr_col = y - x_next

            y = x_next + corr_nonneg
            x = y.clamp_min(0.0)
            corr_nonneg = y - x
            residual = torch.maximum(
                (x.sum(dim=1) - target).abs().amax(),
                (x.sum(dim=0) - target).abs().amax(),
            )
            if residual <= tol:
                break
        return x

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
    def record_diagnostics(stats, transport, marginal):
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
    def project(
        matrix,
        marginal,
        max_iter=50,
        relative_tol=1.0e-3,
        schur_ridge=1.0e-6,
        fallback_iters=100,
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
        ``(n-1)`` Schur system.  The alternating correction path is only a
        finite failure fallback.
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Euclidean transport projection requires a square matrix")
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

        # Dual warm start: consecutive candidates are close,
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
            current_dual = EuclideanProjector._projection_dual_merit(plan, u, v, target)
            dual_limit = current_dual + float(dual_merit_tol) * (
                1.0 + current_dual.abs()
            )
            try:
                # The regularized Schur matrix is symmetric positive
                # definite on the valid active support.  Cholesky avoids the
                # pivoting/extra work of a generic LU solve; retain LU as a
                # boundary-safe fallback for degenerate supports.
                if sparse_schur:
                    delta_v_short = EuclideanProjector._active_schur_cg(
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
                full_dual = EuclideanProjector._projection_dual_merit(
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
                component_direction = EuclideanProjector._componentwise_newton_direction(
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
                retry_plan, retry_stats = EuclideanProjector.project(
                    matrix,
                    marginal,
                    max_iter=max_iter,
                    relative_tol=relative_tol,
                    schur_ridge=retry_ridge,
                    fallback_iters=fallback_iters,
                    retry_with_stronger_ridge=False,
                    dual_warm_start=None,
                    fast_full_step=fast_full_step,
                    sparse_schur=False,
                    sparse_schur_iters=sparse_schur_iters,
                    adaptive_ridge=False,
                )
                retry_is_feasible = (
                    retry_stats["solver"] not in {"fallback", "projection_failure"}
                    and float(retry_stats.get("relative_residual", float("inf")))
                    <= float(relative_tol)
                )
                if retry_is_feasible:
                    retry_stats["solver"] = "newton_ridge_retry"
                    retry_stats["ridge_used"] = retry_ridge
                    return retry_plan, retry_stats

        fallback = EuclideanProjector._fallback_projection(
            matrix, target, fallback_iters, tol
        )
        fallback_residual = torch.maximum(
            (fallback.sum(dim=1) - target).abs().amax(),
            (fallback.sum(dim=0) - target).abs().amax(),
        )
        # Never expose an incompletely converged fallback as a transport
        # label.  In the diagonal/near-diagonal regime the fallback can stagnate
        # on a disconnected support; returning that intermediate iterate can
        # violate the marginal contract by a large amount.  The caller will
        # retain its previous feasible coupling and try no further diffusion
        # step.  The uniform feasible plan is returned only as a defensive
        # value for callers that use this projector directly.
        if not torch.isfinite(fallback_residual) or fallback_residual > tol:
            safe = torch.full_like(matrix, float(marginal) / float(n))
            return safe, {
                "solver": "projection_failure", "iterations": int(max_iter),
                "relative_residual": float((fallback_residual / target).detach().cpu()),
                "warm_started": warm_started,
                "active_fraction": float((fallback > 0.0).to(dtype=x.dtype).mean().detach().cpu()),
                "active_state": (fallback > 0.0).detach(),
            }
        return fallback, {
            "solver": "fallback", "iterations": int(max_iter),
            "relative_residual": float((fallback_residual / target).detach().cpu()),
            "warm_started": warm_started,
            "active_fraction": float((fallback > 0.0).to(dtype=x.dtype).mean().detach().cpu()),
            "active_state": (fallback > 0.0).detach(),
        }
