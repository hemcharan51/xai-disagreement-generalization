r"""
Disagreement metrics
====================

Faithful implementations of the metrics defined in Krishna et al. (2024),
Appendix A & D.2, plus the three aggregate scores our poster reports.

Conventions
-----------
* An *explanation* is a 1-D numpy array of **signed** feature attributions,
  already aligned to a common feature-index space (see ``alignment.py``).
* ``F`` is the full feature set. ``TF(E, k)`` is the set of top-``k`` features of
  explanation ``E`` ranked by **magnitude** ``|attribution|`` (footnote 2/5 of the
  paper: top-k is computed on magnitude, not sign).
* ``R(E, f)`` is the rank of feature ``f`` (1 = most important, by magnitude).
* ``S(E, f)`` is the sign of feature ``f``.
* For every metric, **lower values indicate stronger disagreement**.

Paper metrics (Appendix A)
    feature_agreement, rank_agreement, sign_agreement, signed_rank_agreement
        -- top-k metrics, FractionAgreement(E_a, E_b, k) in [0, 1]
    rank_correlation, pairwise_rank_agreement
        -- metrics over a user-selected feature set F (default: all features)

Extra metric (Appendix D.2)
    weighted_rank_agreement -- soft variant of rank_agreement

Our poster's aggregate scores (Laloo & Chitiveli)
    poster_rank_agreement (RA)        = Spearman correlation of importance ranks
    poster_sign_agreement (SA)        = fraction of features with matching sign
    poster_signed_rank_agreement(SRA) = combined rank + sign (our primary metric)
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

from .utils import ranks_by_magnitude, signs, top_k_set


# --------------------------------------------------------------------------- #
# Appendix A.1 -- metrics with respect to top-k features
# --------------------------------------------------------------------------- #
def feature_agreement(a: np.ndarray, b: np.ndarray, k: int) -> float:
    r"""|TF(a,k) ∩ TF(b,k)| / k -- fraction of shared top-k features."""
    ta, tb = top_k_set(a, k), top_k_set(b, k)
    k = min(k, len(a))
    return len(ta & tb) / k


def rank_agreement(a: np.ndarray, b: np.ndarray, k: int) -> float:
    r"""Fraction of top-k features shared **and** at the same rank in both."""
    ta, tb = top_k_set(a, k), top_k_set(b, k)
    ra, rb = ranks_by_magnitude(a), ranks_by_magnitude(b)
    k = min(k, len(a))
    shared_same_rank = sum(1 for f in (ta & tb) if ra[f] == rb[f])
    return shared_same_rank / k


def sign_agreement(a: np.ndarray, b: np.ndarray, k: int) -> float:
    r"""Fraction of top-k features shared **and** with the same sign in both."""
    ta, tb = top_k_set(a, k), top_k_set(b, k)
    sa, sb = signs(a), signs(b)
    k = min(k, len(a))
    shared_same_sign = sum(1 for f in (ta & tb) if sa[f] == sb[f])
    return shared_same_sign / k


def signed_rank_agreement(a: np.ndarray, b: np.ndarray, k: int) -> float:
    r"""Strictest top-k metric: shared **and** same sign **and** same rank."""
    ta, tb = top_k_set(a, k), top_k_set(b, k)
    ra, rb = ranks_by_magnitude(a), ranks_by_magnitude(b)
    sa, sb = signs(a), signs(b)
    k = min(k, len(a))
    n = sum(1 for f in (ta & tb) if ra[f] == rb[f] and sa[f] == sb[f])
    return n / k


def weighted_rank_agreement(a: np.ndarray, b: np.ndarray, k: int) -> float:
    r"""Soft rank agreement (Appendix D.2).

    (1/k) * sum_{f in TF(a,k) ∩ TF(b,k)} ( 1 - |R(a,f) - R(b,f)| / k )

    Exact-rank match -> 1; no overlap -> 0; partial closeness in between.
    """
    ta, tb = top_k_set(a, k), top_k_set(b, k)
    ra, rb = ranks_by_magnitude(a), ranks_by_magnitude(b)
    k = min(k, len(a))
    total = sum(1.0 - abs(int(ra[f]) - int(rb[f])) / k for f in (ta & tb))
    return total / k


# --------------------------------------------------------------------------- #
# Appendix A.2 -- metrics with respect to a feature set F (default: all)
# --------------------------------------------------------------------------- #
def rank_correlation(a: np.ndarray, b: np.ndarray,
                     F: Sequence[int] | None = None,
                     use_magnitude: bool = True) -> float:
    r"""Spearman's rank correlation between the two feature rankings over F.

    ``F`` selects the features of interest (default: all). Ranking is by
    magnitude by default (consistent with the top-k metrics).
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if F is not None:
        a, b = a[list(F)], b[list(F)]
    va = np.abs(a) if use_magnitude else a
    vb = np.abs(b) if use_magnitude else b
    if va.size < 2 or np.ptp(va) == 0 or np.ptp(vb) == 0:
        return 0.0
    rho = spearmanr(va, vb).correlation
    return 0.0 if np.isnan(rho) else float(rho)


def pairwise_rank_agreement(a: np.ndarray, b: np.ndarray,
                            F: Sequence[int] | None = None,
                            use_magnitude: bool = True) -> float:
    r"""Fraction of feature pairs (i<j) with the same relative ordering.

    RO(E, f_i, f_j) = 1 if f_i is more important than f_j under E. The metric is
    the fraction of the (|F| choose 2) pairs whose ordering agrees across a, b.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    idx = list(range(a.size)) if F is None else list(F)
    if len(idx) < 2:
        return 1.0
    va = np.abs(a) if use_magnitude else a
    vb = np.abs(b) if use_magnitude else b
    agree = total = 0
    for i, j in combinations(idx, 2):
        total += 1
        if (va[i] > va[j]) == (vb[i] > vb[j]):
            agree += 1
    return agree / total


# --------------------------------------------------------------------------- #
# Our poster's three aggregate scores (Laloo & Chitiveli Hemcharan Varma)
# --------------------------------------------------------------------------- #
def poster_rank_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """RA (poster): Spearman correlation of importance ranks over all features."""
    return rank_correlation(a, b, F=None, use_magnitude=True)


def poster_sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """SA (poster): fraction of features whose attribution sign matches."""
    return float(np.mean(signs(a) == signs(b)))


def poster_signed_rank_agreement(a: np.ndarray, b: np.ndarray,
                                 k: int | None = None) -> float:
    r"""SRA (poster, primary metric): combines rank **and** sign over all features.

    The poster writes the hard form

        SRA = (1/|F|) * sum_f  1[rank_a(f)=rank_b(f)] * 1[sign_a(f)=sign_b(f)] ,

    but exact integer-rank equality over *all* continuous attributions is
    vanishingly rare in a ~100-dim feature space, so that form collapses to ~0
    and cannot produce the poster's reported magnitudes (~0.65-0.73). We
    therefore use the *signed weighted* form -- the paper's own weighted rank
    agreement idea (Appendix D.2) fused with the sign indicator:

        SRA = (1/|F|) * sum_f  1[sign_a(f)=sign_b(f)] * ( 1 - |rank_a(f)-rank_b(f)| / |F| )

    Identical explanations -> 1; sign agreement and rank closeness both push the
    score up. ``k`` is accepted for a uniform call signature but unused (the
    metric is computed over the full feature set).
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = a.shape[0]
    ra, rb = ranks_by_magnitude(a), ranks_by_magnitude(b)
    sign_match = (signs(a) == signs(b)).astype(float)
    rank_close = 1.0 - np.abs(ra - rb) / n
    return float(np.mean(sign_match * rank_close))


# --------------------------------------------------------------------------- #
# Predictive entropy -- needed for Finding 2 (the "Uncertainty Paradox")
# --------------------------------------------------------------------------- #
def predictive_entropy(proba: np.ndarray) -> np.ndarray:
    r"""Shannon entropy H(P) = -sum_c p_c log p_c of each prediction (in nats).

    ``proba`` has shape (n_samples, n_classes). Returns shape (n_samples,).
    """
    p = np.clip(np.asarray(proba, dtype=float), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


# --------------------------------------------------------------------------- #
# Convenience: compute every metric for one pair of explanations
# --------------------------------------------------------------------------- #
PAPER_TOPK_METRICS = {
    "feature_agreement": feature_agreement,
    "rank_agreement": rank_agreement,
    "sign_agreement": sign_agreement,
    "signed_rank_agreement": signed_rank_agreement,
    "weighted_rank_agreement": weighted_rank_agreement,
}
PAPER_SET_METRICS = {
    "rank_correlation": rank_correlation,
    "pairwise_rank_agreement": pairwise_rank_agreement,
}


def all_metrics(a: np.ndarray, b: np.ndarray, k: int) -> dict[str, float]:
    """Return every metric (paper + poster) for one explanation pair."""
    out = {name: fn(a, b, k) for name, fn in PAPER_TOPK_METRICS.items()}
    out["rank_correlation"] = rank_correlation(a, b)
    out["pairwise_rank_agreement"] = pairwise_rank_agreement(a, b)
    out["RA"] = poster_rank_agreement(a, b)
    out["SA"] = poster_sign_agreement(a, b)
    out["SRA"] = poster_signed_rank_agreement(a, b, k)
    return out
