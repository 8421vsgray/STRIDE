"""Internal Euclidean transport projector configuration.

The training YAML exposes only the three experiment-facing diffusion controls:
``init``, ``steps`` and ``step_size``.  Numerical solver details live here so
that changing the projector implementation does not require editing every
dataset configuration.
"""

from copy import deepcopy


# These values are the current transport defaults.  Keeping them in one place
# preserves the existing solver protocol while removing implementation details
# from dataset YAML files.
PROJECTION_DEFAULTS = {
    "structure_projector": "gpu_newton",
    "structure_projection_precision": "float32",
    "structure_projection_tol": 1.0e-5,
    "structure_projection_fallback_to_float64": True,
    "structure_newton_iters": 50,
    "structure_projection_solver": "newton",
    "structure_qrot_bcd_iters": 50,
    "structure_qrot_warmup_iters": 10,
    "structure_qrot_warmup_rel_tol": 1.0e-3,
    "structure_qrot_hybrid_direct": False,
    "structure_sparse_schur_iters": 24,
    "structure_adaptive_ridge": False,
    "structure_batch_relations": False,
    "structure_active_set_warm_start": True,
    "structure_active_set_margin": 0.0,
    "structure_dual_merit_guard": False,
    "structure_dual_merit_tol": 1.0e-6,
    "structure_newton_fast_full_step": True,
    "structure_projection_rel_tol": 1.0e-3,
    "structure_fast_accept_rel_tol": 1.0e-3,
    "structure_strict_repair_rel_tol": 1.0e-5,
    "structure_entropy_projection_eps": 0.0,
    "structure_entropy_projection_method": "schur",
    "structure_entropy_effective_eps": 0.0,
    "structure_schur_regularization": 1.0e-6,
    "structure_stabilize_updates": True,
    "structure_energy_tolerance": 1.0e-7,
    "structure_transport_assert_tol": 5.0e-3,
    "structure_newton_alpha_min": 1.0 / 64.0,
    "structure_newton_schur_regularization": 1.0e-6,
    "structure_newton_schur_regularization_candidates": [1.0e-6, 1.0e-5],
    "structure_newton_fixed_schur": False,
    "structure_stabilize_projection_rel_tol": 1.0e-5,
    "structure_projection_optimality_tol": 1.0e-7,
    "structure_update_mix": 1.0,
    "structure_cross_weight": 1.0,
    "structure_alpha": 1.0,
    "structure_prior_gate": True,
    "structure_prior_gate_offdiag_tol": 1.0e-3,
    "structure_prior_precheck": True,
    "structure_prior_lock": False,
    "structure_prior_lock_after_gates": 4,
}


# Food101 historically disabled the optional energy/stabilization guard in its
# canonical YAML.  Preserve that behavior without exposing a solver knob in
# the dataset configuration.
DATASET_OVERRIDES = {
    "Food101": {"structure_stabilize_updates": False},
}


def build_projection_params(args):
    """Return the internal parameter mapping consumed by ``model.py``."""

    params = deepcopy(PROJECTION_DEFAULTS)
    params.update(DATASET_OVERRIDES.get(str(args.dataset), {}))
    params.update(
        {
            "structure_init": str(getattr(args, "init", "diagonal")).lower(),
            "structure_steps": int(getattr(args, "steps", 5)),
            "structure_step_size": float(getattr(args, "step_size", 0.02)),
            # The model owns the single online/target relation contract.
        }
    )
    return params
