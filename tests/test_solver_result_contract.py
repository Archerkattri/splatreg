"""Regression tests for the builtin LM result contract."""

import pytest
import torch

from splatreg.solvers.lm import run_lm


class TranslationResidual:
    """One-row residual with a known returned-pose diagnostic."""

    weight = 1.0

    def __init__(self):
        self.seen_T = []

    def residual(self, T, target, source):
        self.seen_T.append(T.detach().clone())
        return T[0, 3:4]

    def jacobian(self, T, target, source):
        J = torch.zeros(1, 6, dtype=T.dtype, device=T.device)
        J[0, 0] = 1.0
        return J


def test_zero_iterations_have_defined_public_failure():
    with pytest.raises(ValueError, match="positive integer"):
        run_lm(torch.eye(4), [TranslationResidual()], None, None, n_iters=0)


def test_one_iteration_diagnostics_match_returned_transform():
    residual = TranslationResidual()
    result = run_lm(
        torch.eye(4),
        [residual],
        None,
        None,
        n_iters=1,
        convergence_tol=0.0,
        max_trans_step=0.1,
    )

    assert torch.allclose(residual.seen_T[-1], result.T)
    final_residual = float(result.T[0, 3].abs())
    assert result.info["diagnostics_at"] == "returned_transform"
    assert result.info["rmse"] == pytest.approx(final_residual)
    assert result.info["cost"] == pytest.approx(0.5 * final_residual**2)
