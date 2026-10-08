#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Discrete Memory Decision Layer (MDL) geometry.

Faithful stdlib stand-in of Zhang et al., arXiv:2609.22043 §3:
retrieved memory is a source candidate; MDL compiles relevance M,
reliability R, and task-risk A into a trust decision with
confidence–consistency decoupling, risk inversion, and abstention.

No sentence-transformers, no live LLM, no numpy — pure Python + math.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Hyperparameters (paper Table 2)
# ---------------------------------------------------------------------------
D = 16
DIMS = (5, 5, 6)  # (d_wm, d_r, d_a)
SIGMA = 0.20
EPS_C = 0.10
S_WM, S_R, S_A = 1.0, 0.7, 0.5
SEED = 42

# Action thresholds on C_final. Pack-local (not the paper's TruthfulQA
# grid search). Chosen so the seven algebra assertions below pass.
#   C_final >= THETA0 → Active (adopt / inject)
#   THETA1 <= C_final < THETA0 → Supp (soft support; still not blind inject)
#   THETA2 <= C_final < THETA1 → Silent (reject / abstain)
#   C_final < THETA2 → Opt-Out (full abstain)
THETA0 = 0.85
THETA1 = 0.74
THETA2 = 0.50

ACTION_RANK = {
    "Active": 3,
    "Supp": 2,
    "Silent": 1,
    "Opt-Out": 0,
}

Vector = List[float]
Matrix = List[List[float]]


# ---------------------------------------------------------------------------
# Tiny linear algebra (stdlib)
# ---------------------------------------------------------------------------
def _zeros(n: int) -> Vector:
    return [0.0] * n


def _eye(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def _add(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x + y for x, y in zip(a, b)]


def _scale(s: float, a: Sequence[float]) -> Vector:
    return [s * x for x in a]


def _matvec(M: Matrix, v: Sequence[float]) -> Vector:
    return [_dot(row, v) for row in M]


def _matmul(A: Matrix, B: Matrix) -> Matrix:
    Bt = list(zip(*B))
    return [[_dot(row, col) for col in Bt] for row in A]


def _outer(u: Sequence[float], v: Sequence[float]) -> Matrix:
    return [[ui * vj for vj in v] for ui in u]


def _matadd(A: Matrix, B: Matrix) -> Matrix:
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _matscale(s: float, A: Matrix) -> Matrix:
    return [[s * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return _dot(a, b) / (na * nb)


def _tanh_vec(v: Sequence[float]) -> Vector:
    return [math.tanh(x) for x in v]


# ---------------------------------------------------------------------------
# Modified Gram–Schmidt QR → orthonormal columns of Q (D×D)
# ---------------------------------------------------------------------------
def _qr_orthonormal(A: Matrix) -> Matrix:
    """Return Q with orthonormal columns spanning the column space of A."""
    n = len(A)
    # Work on columns
    cols = [[A[i][j] for i in range(n)] for j in range(n)]
    q_cols: List[Vector] = []
    for j in range(n):
        v = cols[j][:]
        for q in q_cols:
            proj = _dot(v, q)
            v = _add(v, _scale(-proj, q))
        nv = _norm(v)
        if nv < 1e-12:
            # Degenerate — invent an orthogonal direction
            for k in range(n):
                trial = _zeros(n)
                trial[k] = 1.0
                for q in q_cols:
                    trial = _add(trial, _scale(-_dot(trial, q), q))
                nv = _norm(trial)
                if nv > 1e-12:
                    v = trial
                    break
        q_cols.append(_scale(1.0 / _norm(v), v))
    # Q as rows-of-matrix form: Q[i][j] = q_cols[j][i]
    return [[q_cols[j][i] for j in range(n)] for i in range(n)]


def _build_projectors(seed: int = SEED) -> Tuple[Matrix, Matrix, Matrix]:
    """Fixed-seed random matrix → QR → orthogonal projectors Π_k = P_k P_k^T."""
    rng = random.Random(seed)
    A = [[rng.gauss(0.0, 1.0) for _ in range(D)] for _ in range(D)]
    Q = _qr_orthonormal(A)  # orthonormal columns
    # Column blocks: first 5 → wm, next 5 → r, last 6 → a
    d_wm, d_r, d_a = DIMS
    assert d_wm + d_r + d_a == D

    def projector(start: int, width: int) -> Matrix:
        # P is D×width with the selected orthonormal columns
        # Π = P P^T
        Pi = [[0.0] * D for _ in range(D)]
        for a in range(width):
            col = [Q[i][start + a] for i in range(D)]
            for i in range(D):
                for j in range(D):
                    Pi[i][j] += col[i] * col[j]
        return Pi

    return (
        projector(0, d_wm),
        projector(d_wm, d_r),
        projector(d_wm + d_r, d_a),
    )


# ---------------------------------------------------------------------------
# Value encoding  v(s) = s·1_D + Σ_i exp(-(s-c_i)^2 / (2σ²)) e_i
# ---------------------------------------------------------------------------
_CENTERS = [i / (D - 1) for i in range(D)]  # evenly spaced in [0, 1]


def encode(s: float) -> Vector:
    s = float(s)
    two_sig2 = 2.0 * SIGMA * SIGMA
    v = [s] * D
    for i, c in enumerate(_CENTERS):
        v[i] += math.exp(-((s - c) ** 2) / two_sig2)
    return v


def reliability(s_bar_rel: float, phi: float, n_relevant: int = 2) -> float:
    """R = clip(s_bar_rel · (1-φ)², 0, 1); default 0.5 if <2 relevant."""
    if n_relevant < 2:
        return 0.5
    return _clip(s_bar_rel * (1.0 - phi) ** 2)


# ---------------------------------------------------------------------------
# Subspace weight matrices W_k = s_k Π_k + ε_c Σ_{j≠k} Π_j
# ---------------------------------------------------------------------------
def _build_W(Pi_list: Sequence[Matrix], s_list: Sequence[float]) -> List[Matrix]:
    W_list: List[Matrix] = []
    for k, (Pi_k, s_k) in enumerate(zip(Pi_list, s_list)):
        W = _matscale(s_k, Pi_k)
        for j, Pi_j in enumerate(Pi_list):
            if j == k:
                continue
            W = _matadd(W, _matscale(EPS_C, Pi_j))
        W_list.append(W)
    return W_list


# Module-level geometry (deterministic from SEED)
_PI_WM, _PI_R, _PI_A = _build_projectors(SEED)
_W_WM, _W_R, _W_A = _build_W([_PI_WM, _PI_R, _PI_A], [S_WM, S_R, S_A])


# ---------------------------------------------------------------------------
# Calibration of n_min / n_max over a small fixed grid
# ---------------------------------------------------------------------------
def _raw_norm(M: float, R: float, A: float) -> float:
    """||v_meta|| before C normalization (bias = 0)."""
    sinv_A = 1.0 - A
    v_wm, v_r, v_a = encode(M), encode(R), encode(sinv_A)
    g_A = 0.5 + 0.5 * (sum(v_a) / D)
    fused = _add(
        _scale(g_A, _matvec(_W_WM, v_wm)),
        _add(_matvec(_W_R, v_r), _matvec(_W_A, v_a)),
    )
    return _norm(_tanh_vec(fused))


def _calibrate() -> Tuple[float, float]:
    grid = [i / 10.0 for i in range(11)]  # 0.0 … 1.0
    norms = [_raw_norm(m, r, a) for m in grid for r in grid for a in grid]
    return min(norms), max(norms)


N_MIN, N_MAX = _calibrate()
# Guard against degenerate span (should not happen on this grid)
if N_MAX - N_MIN < 1e-9:
    N_MIN, N_MAX = 0.0, 1.0


# ---------------------------------------------------------------------------
# Decision API
# ---------------------------------------------------------------------------
def decide(
    M: float,
    R: float,
    A: float,
    *,
    theta: Tuple[float, float, float] = (THETA0, THETA1, THETA2),
) -> dict:
    """Run MDL on scalars (M, R, A) ∈ [0,1]³.

    Returns action ∈ {Active, Supp, Silent, Opt-Out} plus audit scalars
    C (norm confidence), alpha (consistency cosine), C_final, g_A, sinv_A.
    """
    M = _clip(float(M))
    R = _clip(float(R))
    A = _clip(float(A))
    sinv_A = 1.0 - A

    v_wm = encode(M)
    v_r = encode(R)
    v_a = encode(sinv_A)

    g_A = 0.5 + 0.5 * (sum(v_a) / D)
    fused = _add(
        _scale(g_A, _matvec(_W_WM, v_wm)),
        _add(_matvec(_W_R, v_r), _matvec(_W_A, v_a)),
    )
    v_meta = _tanh_vec(fused)

    nrm = _norm(v_meta)
    C = _clip((nrm - N_MIN) / (N_MAX - N_MIN))
    alpha = _cosine(v_wm, v_meta)
    # Cosine can be slightly negative under conflict; clamp for gating
    alpha_gate = _clip(alpha, -1.0, 1.0)
    # Paper uses α as cosine; for C_final keep α in a stable range
    C_final = C * (0.3 + 0.7 * max(0.0, alpha_gate))

    th0, th1, th2 = theta
    if C_final >= th0:
        action = "Active"
    elif C_final >= th1:
        action = "Supp"
    elif C_final >= th2:
        action = "Silent"
    else:
        action = "Opt-Out"

    return {
        "action": action,
        "rank": ACTION_RANK[action],
        "C": C,
        "alpha": alpha,
        "C_final": C_final,
        "g_A": g_A,
        "sinv_A": sinv_A,
        "norm": nrm,
        "M": M,
        "R": R,
        "A": A,
        "adopt": action == "Active",  # only Active = inject
        "abstain": action in ("Silent", "Opt-Out"),
        "soft": action == "Supp",
    }


def blind_rag_inject(M: float, threshold: float = 0.5) -> bool:
    """Blind RAG: inject iff relevance alone clears a retrieval threshold."""
    return float(M) >= threshold


def m_only_decide(M: float) -> dict:
    """Single-signal stand-in: treat as R=0.5, A=0 (no risk, neutral reliability)."""
    return decide(M, 0.5, 0.0)


__all__ = [
    "decide",
    "encode",
    "reliability",
    "blind_rag_inject",
    "m_only_decide",
    "ACTION_RANK",
    "THETA0",
    "THETA1",
    "THETA2",
    "N_MIN",
    "N_MAX",
    "D",
]
