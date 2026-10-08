#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Minimal stop-dont-drift stub for the stop-triage-cliff join.

Faithful subset of Sun & Guo, arXiv:2609.24101 (stop-dont-drift):
evidence stream is source; principled stop (KL + quality-peak) is the
compile; full-budget retrieval is the inject that drifts.

Copied from stop-cliff-compile / stop-dont-drift so the join is
self-contained for GitHub. No numpy, no network, no LLM — pure
Python + math + random.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Labels and defaults
# ---------------------------------------------------------------------------
LABELS_3 = ("Beneficial", "NoEffect", "Harmful")
LABELS_2 = ("Beneficial", "NoEffect")

SEED = 42
DEFAULT_B = 0.15
DEFAULT_DELTA = 0.02
DEFAULT_ALPHA = 0.15
DEFAULT_BUDGET = 20


# ---------------------------------------------------------------------------
# Publication-bias model & vote-counting
# ---------------------------------------------------------------------------
def positive_rate(b: float) -> float:
    """P(report positive) = 1/2 + b under the paper's bias model."""
    if not (0.0 < b < 0.5):
        raise ValueError(f"b must be in (0, 1/2); got {b}")
    return 0.5 + b


def sigma_sq(b: float) -> float:
    """σ² = (1/2+b)(1/2−b) from Theorem 1."""
    return (0.5 + b) * (0.5 - b)


def sigma(b: float) -> float:
    return math.sqrt(sigma_sq(b))


def vote_label(p_hat_pos: float) -> str:
    """Vote-counting aggregator: Beneficial iff fraction_positive > 1/2."""
    return "Beneficial" if p_hat_pos > 0.5 else "NoEffect"


def phi(x: float) -> float:
    """Standard normal CDF Φ via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def clt_envelope(b: float, t: int) -> float:
    """CLT envelope Φ(b √t / σ). Strictly increasing in t for fixed b∈(0,1/2)."""
    if t <= 0:
        raise ValueError("t must be positive")
    return phi(b * math.sqrt(t) / sigma(b))


def drift_risk_clt(b: float, t: int) -> float:
    """D(t) ≈ Φ(b √t / σ) for true-NoEffect under vote-counting."""
    return clt_envelope(b, t)


# ---------------------------------------------------------------------------
# Exact binomial false-positive helper
# ---------------------------------------------------------------------------
def _binom_pmf(k: int, n: int, p: float) -> float:
    """P(K=k) for Binomial(n,p); stdlib only."""
    if k < 0 or k > n:
        return 0.0
    # C(n,k) * p^k * (1-p)^{n-k}
    log_c = 0.0
    for i in range(k):
        log_c += math.log(n - i) - math.log(i + 1)
    return math.exp(log_c + k * math.log(p) + (n - k) * math.log(1.0 - p))


def exact_false_positive(b: float, t: int) -> float:
    """Exact binomial D(t) = P(ˆp_t⁺ > 1/2) = P(K > t/2) for K~Bin(t, 1/2+b)."""
    p = positive_rate(b)
    # ˆp > 1/2 iff K > t/2 iff K >= floor(t/2)+1
    thresh = (t // 2) + 1
    return sum(_binom_pmf(k, t, p) for k in range(thresh, t + 1))


# ---------------------------------------------------------------------------
# Discrete label posteriors & KL
# ---------------------------------------------------------------------------
def uniform_posterior(labels: Sequence[str] = LABELS_3) -> Dict[str, float]:
    n = len(labels)
    return {lab: 1.0 / n for lab in labels}


def normalize(q: Dict[str, float]) -> Dict[str, float]:
    s = sum(q.values())
    if s <= 0:
        raise ValueError("posterior mass must be positive")
    return {k: v / s for k, v in q.items()}


def kl_divergence(q: Dict[str, float], p: Dict[str, float], eps: float = 1e-12) -> float:
    """KL(q || p) over shared discrete support."""
    keys = set(q) | set(p)
    total = 0.0
    for k in keys:
        qk = max(q.get(k, 0.0), eps)
        pk = max(p.get(k, 0.0), eps)
        total += qk * math.log(qk / pk)
    return total


def argmax_label(q: Dict[str, float]) -> str:
    return max(q.items(), key=lambda kv: (kv[1], kv[0]))[0]


def update_posterior_from_vote(
    q_prev: Dict[str, float],
    positive: bool,
    *,
    labels: Sequence[str] = LABELS_3,
    lr: float = 0.35,
) -> Dict[str, float]:
    """Toy Bayesian-ish update: positive report boosts Beneficial, else NoEffect.

    Harmful mass slowly decays under a null-query stream (no negative reports
    in the bias model). Keeps a proper distribution for KL monitoring.
    """
    q = dict(q_prev)
    for lab in labels:
        q.setdefault(lab, 0.0)
    if positive:
        q["Beneficial"] = q.get("Beneficial", 0.0) + lr
        # slight leak from others so mass remains finite
        for lab in labels:
            if lab != "Beneficial":
                q[lab] = max(eps_floor(q.get(lab, 0.0)) * (1.0 - 0.05 * lr), 1e-9)
    else:
        q["NoEffect"] = q.get("NoEffect", 0.0) + lr
        for lab in labels:
            if lab != "NoEffect":
                q[lab] = max(eps_floor(q.get(lab, 0.0)) * (1.0 - 0.05 * lr), 1e-9)
    return normalize(q)


def eps_floor(x: float) -> float:
    return max(x, 1e-12)


# ---------------------------------------------------------------------------
# Toy process-reward (quality) stand-in for Layer-2 decline
# ---------------------------------------------------------------------------
def toy_reward(
    t: int,
    p_hat: float,
    q: Dict[str, float],
    *,
    peak_at: int = 4,
) -> float:
    """Scalar quality proxy: peaks early then declines as bias accumulates.

    Mimics the paper's PRM decline layer without an MLP: reward rises while
    evidence is thin (NoEffect still plausible), then falls as ˆp drifts above
    1/2 and Beneficial posterior mass grows on a null query.
    """
    # early-evidence bonus that decays after peak_at
    freshness = math.exp(-0.15 * max(0, t - peak_at))
    # prefer staying near the NoEffect / 0.5 boundary early
    balance = 1.0 - abs(p_hat - 0.5)
    # penalise Beneficial dominance on a stream we treat as null-query
    ben = q.get("Beneficial", 0.0)
    noe = q.get("NoEffect", 0.0)
    correctness_proxy = noe - ben
    return freshness * (0.5 * balance + 0.5 * (0.5 + 0.5 * correctness_proxy))


# ---------------------------------------------------------------------------
# Stopping policies
# ---------------------------------------------------------------------------
def decide_stop(
    t: int,
    budget: int,
    kl: float,
    reward: float,
    r_max: float,
    *,
    delta: float = DEFAULT_DELTA,
    alpha: float = DEFAULT_ALPHA,
    policy: str = "principled",
) -> bool:
    """Return True iff the policy stops at this step.

    policy:
      - 'principled': Layer1 KL < δ OR Layer2 reward decline
      - 'full_budget': never stop until t == budget (caller handles)
      - 'kl_only': only Layer1
      - 'decline_only': only Layer2
    """
    if policy == "full_budget":
        return t >= budget
    layer1 = kl < delta and t >= 2
    layer2 = reward < r_max - alpha and t >= 2
    if policy == "kl_only":
        return layer1 or t >= budget
    if policy == "decline_only":
        return layer2 or t >= budget
    # principled = both layers OR budget
    return layer1 or layer2 or t >= budget


# ---------------------------------------------------------------------------
# Trajectory simulation
# ---------------------------------------------------------------------------
def simulate_null_query(
    b: float,
    budget: int,
    *,
    seed: int = SEED,
    delta: float = DEFAULT_DELTA,
    alpha: float = DEFAULT_ALPHA,
    policy: str = "principled",
    reports: Optional[Sequence[bool]] = None,
    labels: Sequence[str] = LABELS_3,
    peak_at: int = 4,
) -> Dict:
    """Simulate one true-NoEffect query under publication bias.

    Returns stopped step, final vote/posterior labels, KL/reward traces,
    and whether a drift event occurred along the way.
    """
    rng = random.Random(seed)
    p = positive_rate(b)
    q = uniform_posterior(labels)
    pos_count = 0
    history_labels: List[str] = []
    kl_trace: List[float] = []
    reward_trace: List[float] = []
    p_hat_trace: List[float] = []
    r_max = float("-inf")
    stopped_at = budget
    stop_reason = "budget"

    for t in range(1, budget + 1):
        if reports is not None:
            positive = bool(reports[t - 1])
        else:
            positive = rng.random() < p
        if positive:
            pos_count += 1
        p_hat = pos_count / t
        p_hat_trace.append(p_hat)

        q_prev = q
        q = update_posterior_from_vote(q_prev, positive, labels=labels)
        kl = kl_divergence(q, q_prev) if t >= 2 else float("inf")
        kl_trace.append(kl)

        r = toy_reward(t, p_hat, q, peak_at=peak_at)
        reward_trace.append(r)
        r_max = max(r_max, r)

        vote = vote_label(p_hat)
        post_lab = argmax_label(q)
        history_labels.append(vote)

        if decide_stop(
            t, budget, kl, r, r_max, delta=delta, alpha=alpha, policy=policy
        ):
            stopped_at = t
            if t >= budget:
                stop_reason = "budget"
            elif kl < delta and t >= 2 and policy in ("principled", "kl_only"):
                stop_reason = "kl"
            elif r < r_max - alpha:
                stop_reason = "decline"
            else:
                stop_reason = "stop"
            break

    # Drift event: some earlier step had correct NoEffect, later Beneficial
    drift_event = False
    first_correct = None
    for s, lab in enumerate(history_labels, start=1):
        if lab == "NoEffect" and first_correct is None:
            first_correct = s
        if first_correct is not None and lab == "Beneficial" and s > first_correct:
            drift_event = True
            break

    final_p = p_hat_trace[stopped_at - 1]
    return {
        "stopped_at": stopped_at,
        "stop_reason": stop_reason,
        "budget": budget,
        "policy": policy,
        "final_p_hat": final_p,
        "final_vote": vote_label(final_p),
        "final_posterior_label": argmax_label(q),
        "final_posterior": dict(q),
        "kl_trace": kl_trace,
        "reward_trace": reward_trace,
        "p_hat_trace": p_hat_trace,
        "label_trace": history_labels,
        "drift_event": drift_event,
        "r_max": r_max,
        "b": b,
    }


def false_beneficial_rate(
    b: float,
    budget: int,
    policy: str,
    *,
    n_trials: int = 200,
    seed: int = SEED,
    delta: float = DEFAULT_DELTA,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Fraction of null-query trials ending as Beneficial under a policy."""
    bad = 0
    for i in range(n_trials):
        out = simulate_null_query(
            b,
            budget,
            seed=seed + i,
            delta=delta,
            alpha=alpha,
            policy=policy,
        )
        if out["final_vote"] == "Beneficial":
            bad += 1
    return bad / n_trials


def construct_drifted_stable_trajectory(
    b: float = 0.2,
    *,
    stabilize_after: int = 12,
    total: int = 18,
) -> Tuple[List[bool], Dict]:
    """Synthetic reports: early mix → Beneficial lock-in → tiny successive KL.

    Demonstrates A5: KL can be small after *wrong* convergence.
    """
    # Force many positives to push past 1/2, then keep positives so posterior
    # stabilises on Beneficial.
    reports: List[bool] = []
    # first half: mostly positive to drift
    for i in range(stabilize_after):
        reports.append(True if i % 5 != 0 else False)  # 80% positive-ish
    # second half: all positive → posterior barely moves (tiny KL)
    while len(reports) < total:
        reports.append(True)

    # Run with a very tight delta so we don't stop early; we want the
    # post-drift KL values themselves.
    out = simulate_null_query(
        b,
        total,
        reports=reports,
        policy="full_budget",
        delta=1e-9,
        alpha=1e9,  # disable decline
    )
    return reports, out


def has_drift_event(label_trace: Sequence[str]) -> bool:
    """Definition 1 (finite): ∃ s<t with ˆL_s = NoEffect and ˆL_t = Beneficial."""
    saw_correct = False
    for lab in label_trace:
        if lab == "NoEffect":
            saw_correct = True
        elif saw_correct and lab == "Beneficial":
            return True
    return False


def expected_p_hat(b: float) -> float:
    """E[ˆp⁺] = 1/2 + b."""
    return positive_rate(b)
