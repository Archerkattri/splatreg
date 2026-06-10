"""Solver backend ABC.

Two pluggability axes (kept separate):
  * a ``Solver`` swaps the *numerical step* inside splatreg's builtin LM loop (LM/GN, damping,
    Cholesky/QR), see ``splatreg.solvers.lm``;
  * a ``backend=`` string on ``register`` hands the whole assembled problem to an external engine
    (pypose / theseus / gtsam). Every backend consumes one ``LinearizedProblem`` and returns an
    ``SE3Update`` (a tangent step), so a new backend never touches the residual plugins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.types import LinearizedProblem, SE3Update


class Solver(ABC):
    @abstractmethod
    def solve(self, problem: LinearizedProblem) -> SE3Update:
        """Solve the (damped) normal equations ``(JᵀWJ + λD) δ = −JᵀW r`` and return the step."""
