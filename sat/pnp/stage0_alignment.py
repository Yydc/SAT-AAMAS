"""Stage-0 alignment for plug-and-play agent upgrades.

Given a pretrained replacement policy ``pi_pre`` and the incumbent
``pi_cur``, the manuscript (Algorithm 1 + eq. (18)-(19)) initialises the
new agent as the per-state KL projection of ``pi_pre`` onto the
trust-region ball around ``pi_cur``:

    pi_new(a|s) = normalise( pi_pre(a|s)^{1/(1+lam(s))}
                             pi_cur(a|s)^{lam(s)/(1+lam(s))} )

where ``lam(s) >= 0`` is chosen by binary search so that the per-state
KL satisfies ``KL(pi_new || pi_cur) <= delta0(s)`` with equality when
``pi_pre`` would otherwise violate the constraint. ``lam = 0`` recovers
``pi_pre`` verbatim; ``lam -> inf`` recovers ``pi_cur``.

The utility works on categorical distributions represented as logits or
probabilities, so it can be applied directly to next-token distributions
of Hugging Face LMs by slicing the output at a chosen position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Stage0Aligner:
    """KL projection of a pretrained token distribution onto a trust region.

    Args:
        delta0: per-state KL budget (KL_max).
        max_lambda: cap for binary search; beyond this we stop and keep
            ``pi_cur`` as the answer (it always satisfies the budget).
        n_iter: number of bisection steps; 30 gives ~1e-9 resolution on
            ``lam`` in the usual [0, max_lambda] window.
        eps: numerical floor for probabilities before taking logs.
    """

    delta0: float = 0.01
    max_lambda: float = 1e4
    n_iter: int = 30
    eps: float = 1e-12

    def project_logits(self, logits_pre: np.ndarray, logits_cur: np.ndarray) -> np.ndarray:
        """Project a single categorical distribution given raw logits."""
        probs_pre = _softmax(logits_pre)
        probs_cur = _softmax(logits_cur)
        return self.project(probs_pre, probs_cur)

    def project(self, pi_pre: np.ndarray, pi_cur: np.ndarray) -> np.ndarray:
        """Project a single categorical distribution given probabilities.

        Returns the projected ``pi_new`` as a probability vector. If the
        pretrained distribution already lies inside the trust region, it
        is returned unchanged.
        """
        pi_pre = np.clip(np.asarray(pi_pre, dtype=np.float64), self.eps, 1.0)
        pi_cur = np.clip(np.asarray(pi_cur, dtype=np.float64), self.eps, 1.0)

        if _kl(pi_pre, pi_cur) <= self.delta0:
            return pi_pre / pi_pre.sum()

        lo, hi = 0.0, self.max_lambda
        for _ in range(self.n_iter):
            mid = 0.5 * (lo + hi)
            pi_new = _geometric_mixture(pi_pre, pi_cur, mid)
            if _kl(pi_new, pi_cur) > self.delta0:
                lo = mid
            else:
                hi = mid
        return _geometric_mixture(pi_pre, pi_cur, hi)

    def project_batch(
        self, pi_pre_batch: Iterable[np.ndarray], pi_cur_batch: Iterable[np.ndarray]
    ) -> list:
        """Apply projection to a batch of categorical distributions."""
        return [
            self.project(pre, cur)
            for pre, cur in zip(pi_pre_batch, pi_cur_batch)
        ]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * (np.log(p) - np.log(q))))


def _geometric_mixture(pi_pre: np.ndarray, pi_cur: np.ndarray, lam: float) -> np.ndarray:
    w_pre = 1.0 / (1.0 + lam)
    w_cur = lam / (1.0 + lam)
    log_mix = w_pre * np.log(pi_pre) + w_cur * np.log(pi_cur)
    log_mix -= log_mix.max()
    mix = np.exp(log_mix)
    return mix / mix.sum()
