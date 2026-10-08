#!/usr/bin/env python3
"""Refuse to certify.

Discrete analog of Sun and Yang, RouteGuard (arXiv:2608.07583).
Routing gain factors as G = π Δ_E. The design objective is the
gate's informativeness Φ, not AUC. Complementarity is not enough:
a finite-sample Bernstein bracket certifies a lower bound on G
or refuses. Predictions are hash-frozen before outcomes open.

Stdlib only. Run: python3 experiment.py
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

F = Fraction

# Primary is advisor 0. Correctness is 0/1. Gate T is an integer cell.
# Row: (T, C_primary, C_competitor)
Row = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Exact functionals (paper Prop. 1, Thm. 1, Lemma 1)
# ---------------------------------------------------------------------------

def mean_f(xs: Sequence[int]) -> F:
    if not xs:
        return F(0)
    return F(sum(xs), len(xs))


def decomp(rows: Sequence[Row], route: Sequence[int]) -> Tuple[F, F, F, Tuple[int, ...]]:
    """G, π, Δ_E, and the increment vector Z = C_R - C_1."""
    assert len(rows) == len(route)
    z: List[int] = []
    routed: List[int] = []
    for (t, c1, c2), r in zip(rows, route):
        del t
        cr = c1 if r == 1 else c2
        zi = cr - c1
        z.append(zi)
        if r != 1:
            routed.append(zi)
    m = len(rows)
    g = F(sum(z), m)
    pi = F(sum(1 for r in route if r != 1), m)
    d_e = F(sum(routed), len(routed)) if routed else F(0)
    if pi == 0:
        d_e = F(0)
    return g, pi, d_e, tuple(z)


def phi(rows: Sequence[Row]) -> F:
    """Φ = E[max_j η_j(T) - η_1(T)]."""
    buckets: Dict[int, List[Row]] = {}
    for row in rows:
        buckets.setdefault(row[0], []).append(row)
    total = F(0)
    m = len(rows)
    for group in buckets.values():
        eta1 = mean_f([r[1] for r in group])
        eta2 = mean_f([r[2] for r in group])
        total += F(len(group), m) * (max(eta1, eta2) - eta1)
    return total


def auc_gate_vs_primary(rows: Sequence[Row]) -> F:
    """AUC(T; C_1), Mann–Whitney with ties at 1/2. Exact Fraction."""
    pos = [r[0] for r in rows if r[1] == 1]
    neg = [r[0] for r in rows if r[1] == 0]
    if not pos or not neg:
        return F(1, 2)
    wins = 0
    ties = 0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return F(2 * wins + ties, 2 * len(pos) * len(neg))


def marginals(rows: Sequence[Row]) -> Tuple[F, F, F]:
    p1 = mean_f([r[1] for r in rows])
    p2 = mean_f([r[2] for r in rows])
    a_star = mean_f([max(r[1], r[2]) for r in rows])
    return p1, p2, a_star


def excess_over_independence(rows: Sequence[Row]) -> F:
    p1, p2, a_star = marginals(rows)
    h_ind = 1 - (1 - p1) * (1 - p2)
    return a_star - h_ind


def greedy_route(rows: Sequence[Row]) -> List[int]:
    """Empirical Bayes selector; ties break toward the primary."""
    buckets: Dict[int, List[Row]] = {}
    for row in rows:
        buckets.setdefault(row[0], []).append(row)
    pick: Dict[int, int] = {}
    for t, group in buckets.items():
        eta1 = mean_f([r[1] for r in group])
        eta2 = mean_f([r[2] for r in group])
        pick[t] = 1 if eta1 >= eta2 else 2
    return [pick[r[0]] for r in rows]


def stay_route(rows: Sequence[Row]) -> List[int]:
    return [1] * len(rows)


def route_on_t(rows: Sequence[Row], t_away: Iterable[int]) -> List[int]:
    away = set(t_away)
    return [2 if r[0] in away else 1 for r in rows]


# ---------------------------------------------------------------------------
# Bernstein certificate (paper §III-F, M = 2, δ = 0.05)
# ---------------------------------------------------------------------------

DELTA = 0.05
M_BOUND = 2  # |Z - EZ| ≤ 2


def routing_sigma2(z: Sequence[int], route: Sequence[int]) -> float:
    """Plug-in σ² = π σ_E² + π(1-π) Δ_E². Equals Var(Z)."""
    m = len(z)
    pi = sum(1 for r in route if r != 1) / m
    routed = [zi for zi, r in zip(z, route) if r != 1]
    if not routed or pi == 0.0:
        return 0.0
    d_e = sum(routed) / len(routed)
    sig_e = sum((zi - d_e) ** 2 for zi in routed) / len(routed)
    return pi * sig_e + pi * (1.0 - pi) * d_e * d_e


def routing_sigma(z: Sequence[int], route: Sequence[int]) -> float:
    return math.sqrt(routing_sigma2(z, route))


def bernstein(sigma: float, m: int, delta: float = DELTA, bound: int = M_BOUND) -> float:
    if m <= 0:
        return float("inf")
    lg = math.log(1.0 / delta)
    return sigma * math.sqrt(2.0 * lg / m) + (2.0 * bound * lg) / (3.0 * m)


def gamma_hat(g: float, sigma: float, m: int, rho: float = 0.0) -> float:
    return g - bernstein(sigma, m) - 2.0 * rho


def certifies(g: float, sigma: float, m: int, rho: float = 0.0) -> bool:
    return gamma_hat(g, sigma, m, rho) > 0.0


def m_star(g: float, sigma: float, m_cap: int = 2000) -> Optional[int]:
    """Smallest m at which the frozen Ĝ would certify (ρ = 0)."""
    for m in range(1, m_cap + 1):
        if certifies(g, sigma, m):
            return m
    return None


def icc_oneway(cells: Sequence[Sequence[int]]) -> float:
    """One-way ICC of increments. Constant cells ⇒ ICC = 1."""
    k = len(cells)
    ns = [len(c) for c in cells]
    n_bar = sum(ns) / k
    flat = [z for c in cells for z in c]
    grand = sum(flat) / len(flat)
    ssb = 0.0
    ssw = 0.0
    for c in cells:
        mu = sum(c) / len(c)
        ssb += len(c) * (mu - grand) ** 2
        ssw += sum((zi - mu) ** 2 for zi in c)
    dfb = k - 1
    dfw = len(flat) - k
    msb = ssb / dfb if dfb else 0.0
    msw = ssw / dfw if dfw else 0.0
    if msw == 0.0 and msb > 0.0:
        return 1.0
    denom = msb + (n_bar - 1.0) * msw
    if denom == 0.0:
        return 0.0
    return (msb - msw) / denom


def deff(n_bar: float, icc: float) -> float:
    return 1.0 + (n_bar - 1.0) * icc


# ---------------------------------------------------------------------------
# Hash-freeze protocol (paper §IV-E)
# ---------------------------------------------------------------------------

def freeze_digest(choices: Sequence[int]) -> str:
    blob = json.dumps(list(choices), separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


@dataclass
class FreezeProtocol:
    digest: Optional[str] = None
    opened: bool = False

    def freeze(self, choices: Sequence[int]) -> str:
        if self.opened:
            return "refuse:opened-before-freeze"
        self.digest = freeze_digest(choices)
        return self.digest

    def open(self, choices: Sequence[int]) -> str:
        if self.digest is None:
            return "refuse:no-freeze"
        if freeze_digest(choices) != self.digest:
            return "failed-certificate"
        self.opened = True
        return "opened"


def mark_opened_first(proto: FreezeProtocol) -> None:
    """Simulate reading outcomes before any freeze."""
    proto.opened = True


# ---------------------------------------------------------------------------
# Toy pools (not RouterBench, not OpenRCA)
# ---------------------------------------------------------------------------

# High AUC(T; C_1)=1, route-away when T=0, recoveries=destructions=0 ⇒ G=0.
AUC_POOL: Tuple[Row, ...] = (
    (1, 1, 1),
    (1, 1, 1),
    (1, 1, 0),
    (1, 1, 1),
    (1, 1, 0),
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
)

# Complementarity holds (one C2-only win) but advisors co-fail: E < 0.
REDUNDANT: Tuple[Row, ...] = (
    (0, 1, 1),
    (0, 1, 1),
    (0, 1, 1),
    (0, 1, 1),
    (1, 1, 0),  # disagree, primary wins
    (1, 0, 1),  # complementarity: competitor recovers
    (0, 0, 0),
    (0, 0, 0),
    (0, 1, 1),
    (0, 1, 1),
)

# Genuine-gain block: stay, stay, stay, recover, recover. G = 2/5.
GAIN_BLOCK: Tuple[Row, ...] = (
    (0, 1, 1),
    (0, 1, 0),
    (0, 0, 0),
    (1, 0, 1),
    (1, 0, 1),
)

# Matched null block: stay, stay, stay, recover, destroy. G = 0.
NULL_BLOCK: Tuple[Row, ...] = (
    (0, 1, 1),
    (0, 1, 0),
    (0, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
)

# Uninformative-enough gate: greedy still picks on T=3 noise.
UNINFORM: Tuple[Row, ...] = (
    (0, 1, 1),
    (0, 1, 1),
    (0, 1, 1),
    (0, 1, 0),
    (1, 1, 1),
    (1, 1, 1),
    (1, 1, 0),
    (1, 0, 0),
    (2, 1, 1),
    (2, 1, 0),
    (2, 0, 1),
    (2, 0, 0),
    (3, 1, 1),
    (3, 0, 1),
    (3, 0, 0),
    (3, 0, 0),
)


def repeat_block(block: Sequence[Row], times: int) -> Tuple[Row, ...]:
    out: List[Row] = []
    for k in range(times):
        for t, c1, c2 in block:
            out.append((t + 2 * k, c1, c2))
    return tuple(out)


def cluster_pool() -> Tuple[List[Row], List[int], List[List[int]]]:
    """8 cells × 10 prompts. Gain lives in 2 of 8 cells (all recoveries)."""
    rows: List[Row] = []
    route: List[int] = []
    cells: List[List[int]] = []
    for cell in range(8):
        z_cell: List[int] = []
        for _ in range(10):
            if cell < 2:
                rows.append((cell, 0, 1))  # recovery if we route away
                route.append(2)
                z_cell.append(1)
            else:
                rows.append((cell, 1, 1))
                route.append(1)
                z_cell.append(0)
        cells.append(z_cell)
    return rows, route, cells


def check(i: int, name: str, ok: bool) -> None:
    print(f"{i}. {name}: {'ok' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(f"assertion {i} failed: {name}")


def main() -> None:
    # 1. G = π Δ_E on a toy pool; AUC of the gate is 1 while G is 0.
    r1 = route_on_t(AUC_POOL, (0,))
    g1, pi1, de1, z1 = decomp(AUC_POOL, r1)
    auc1 = auc_gate_vs_primary(AUC_POOL)
    ph1 = phi(AUC_POOL)
    ok1 = (
        g1 == pi1 * de1
        and g1 == 0
        and pi1 == F(3, 8)
        and de1 == 0
        and auc1 == 1
        and ph1 == 0
        and all(zi == 0 for zi in z1)
    )
    check(1, f"G=πΔ_E={g1} with AUC={auc1} (Φ={ph1})", ok1)

    # 2. Complementarity does not imply G>0. Redundant advisors:
    #    oracle sits below the independence baseline ⇒ refuse.
    r2 = route_on_t(REDUNDANT, (1,))
    g2, pi2, de2, z2 = decomp(REDUNDANT, r2)
    p1, p2, a_star = marginals(REDUNDANT)
    exc = excess_over_independence(REDUNDANT)
    c1_holds = any(c1 == 0 and c2 == 1 for _, c1, c2 in REDUNDANT)
    disagree = sum(1 for _, c1, c2 in REDUNDANT if c1 != c2)
    sig2 = routing_sigma(z2, r2)
    refuse2 = not certifies(float(g2), sig2, len(REDUNDANT))
    ok2 = (
        g2 == pi2 * de2
        and g2 == 0
        and c1_holds
        and disagree == 2
        and a_star <= 1 - (1 - p1) * (1 - p2)
        and exc < 0
        and exc == F(-11, 100)
        and refuse2
        and p1 == p2 == F(7, 10)
        and a_star == F(4, 5)
    )
    check(2, f"redundant E={exc}; complementarity holds; protocol refuses", ok2)

    # 3. Genuine-gain control certifies once m ≥ m★; a true null does not.
    gain_hi = repeat_block(GAIN_BLOCK, 16)  # m = 80
    gain_lo = repeat_block(GAIN_BLOCK, 4)  # m = 20
    null_hi = repeat_block(NULL_BLOCK, 16)
    rg = greedy_route(gain_hi)
    rlo = greedy_route(gain_lo)
    rn = greedy_route(null_hi)
    gg, pgi, dgi, zg = decomp(gain_hi, rg)
    glo, _, _, zlo = decomp(gain_lo, rlo)
    gn, _, _, zn = decomp(null_hi, rn)
    sig_g = routing_sigma(zg, rg)
    sig_lo = routing_sigma(zlo, rlo)
    sig_n = routing_sigma(zn, rn)
    ms = m_star(float(gg), sig_g)
    cert_lo = certifies(float(glo), sig_lo, len(gain_lo))
    cert_hi = certifies(float(gg), sig_g, len(gain_hi))
    cert_null = certifies(float(gn), sig_n, len(null_hi))
    ok3 = (
        gg == pgi * dgi
        and gg == F(2, 5)
        and glo == F(2, 5)
        and gn == 0
        and ms is not None
        and len(gain_lo) < ms <= len(gain_hi)
        and (not cert_lo)
        and cert_hi
        and (not cert_null)
    )
    check(
        3,
        f"genuine certifies at m={len(gain_hi)}≥m★={ms}; null refuses",
        ok3,
    )

    # 4. Sampling unit: prompt-level certifies; cluster resampling withholds.
    crow, croute, cells = cluster_pool()
    gc, pic, dec, zc = decomp(crow, croute)
    sig_c = routing_sigma(zc, croute)
    m_prompt = len(crow)
    cert_prompt = certifies(float(gc), sig_c, m_prompt)
    icc = icc_oneway(cells)
    n_bar = len(cells[0])
    d_eff = deff(n_bar, icc)
    m_eff = m_prompt / d_eff
    cert_clu = certifies(float(gc), sig_c, int(round(m_eff)))
    gain_cells = sum(1 for c in cells if any(c))
    ok4 = (
        gc == pic * dec
        and gc == F(1, 4)
        and gain_cells == 2
        and len(cells) == 8
        and icc == 1.0
        and d_eff == 10.0
        and int(round(m_eff)) == 8
        and cert_prompt
        and (not cert_clu)
    )
    check(
        4,
        f"prompt certifies G={gc}; cluster (2/8 cells, m_eff=8) withholds",
        ok4,
    )

    # 5. Predictions hash-frozen before outcomes open; opening first is refused.
    choices = greedy_route(GAIN_BLOCK)
    proto_ok = FreezeProtocol()
    d_ok = proto_ok.freeze(choices)
    open_ok = proto_ok.open(choices)
    proto_bad = FreezeProtocol()
    mark_opened_first(proto_bad)
    d_bad = proto_bad.freeze(choices)
    ok5 = (
        len(d_ok) == 64
        and all(ch in "0123456789abcdef" for ch in d_ok)
        and open_ok == "opened"
        and proto_ok.opened
        and d_bad == "refuse:opened-before-freeze"
        and proto_bad.digest is None
    )
    check(5, "freeze-then-open is allowed; open-then-freeze is a protocol violation", ok5)

    # 6. Uninformative / redundant gate: protocol withholds even if greedy picks.
    r6 = greedy_route(UNINFORM)
    g6, pi6, de6, z6 = decomp(UNINFORM, r6)
    ph6 = phi(UNINFORM)
    exc6 = excess_over_independence(UNINFORM)
    greedy_picks = any(r != 1 for r in r6)
    sig6 = routing_sigma(z6, r6)
    refuse6 = not certifies(float(g6), sig6, len(UNINFORM))
    stay_g, _, _, _ = decomp(UNINFORM, stay_route(UNINFORM))
    ok6 = (
        g6 == pi6 * de6
        and g6 == F(1, 16)
        and ph6 == F(1, 16)
        and ph6 > 0
        and ph6 < F(1, 8)
        and greedy_picks
        and r6.count(2) == 4
        and exc6 < 0
        and stay_g == 0
        and refuse6
    )
    check(
        6,
        f"greedy still picks (Φ={ph6}) but protocol withholds; E={exc6}<0",
        ok6,
    )

    # 7. Declared freeze that does not match the later measurement fails.
    declared = greedy_route(GAIN_BLOCK)
    measured = list(declared)
    measured[0] = 2 if measured[0] == 1 else 1  # tamper after freeze
    proto7 = FreezeProtocol()
    d7 = proto7.freeze(declared)
    mismatch = proto7.open(measured)
    match = FreezeProtocol()
    match.freeze(declared)
    matched = match.open(declared)
    silent_pass = mismatch == "opened"
    ok7 = (
        d7 == freeze_digest(declared)
        and d7 != freeze_digest(measured)
        and mismatch == "failed-certificate"
        and matched == "opened"
        and not silent_pass
        and not (mismatch in ("opened", "certify"))
    )
    check(7, "mismatched freeze is a failed certificate, not a silent pass", ok7)

    print("7/7 passed")


if __name__ == "__main__":
    main()
