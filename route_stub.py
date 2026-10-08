#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Minimal RouteGuard stub for the trust-compile-gate join.

Faithful subset of Sun & Yang, arXiv:2608.07583 (refuse-to-certify):
G = π Δ_E; Bernstein lower bound γ̂_m; SHA-256 freeze before outcomes open.
Stdlib only. Not the full 7/7 RouteGuard pack — just the objects the gate needs.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

F = Fraction
Row = Tuple[int, int, int]  # (T, C_primary, C_competitor)

DELTA = 0.05
M_BOUND = 2

# High-AUC, G=0 pool (uncertified escalate refuse)
AUC_POOL: Tuple[Row, ...] = (
    (1, 1, 1),
    (1, 1, 1),
    (1, 1, 0),
    (1, 1, 0),
    (1, 1, 1),
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
)

# Genuine-gain block: G = 2/5 when routed on T=1
GAIN_BLOCK: Tuple[Row, ...] = (
    (0, 1, 1),
    (0, 1, 0),
    (0, 0, 0),
    (1, 0, 1),
    (1, 0, 1),
)


def mean_f(xs: Sequence[int]) -> F:
    if not xs:
        return F(0)
    return F(sum(xs), len(xs))


def decomp(rows: Sequence[Row], route: Sequence[int]) -> Tuple[F, F, F, Tuple[int, ...]]:
    assert len(rows) == len(route)
    z: List[int] = []
    routed: List[int] = []
    for (_t, c1, c2), r in zip(rows, route):
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


def route_on_t(rows: Sequence[Row], t_away) -> List[int]:
    away = set(t_away)
    return [2 if r[0] in away else 1 for r in rows]


def greedy_route(rows: Sequence[Row]) -> List[int]:
    buckets: dict = {}
    for row in rows:
        buckets.setdefault(row[0], []).append(row)
    pick = {}
    for t, group in buckets.items():
        eta1 = mean_f([r[1] for r in group])
        eta2 = mean_f([r[2] for r in group])
        pick[t] = 1 if eta1 >= eta2 else 2
    return [pick[r[0]] for r in rows]


def auc_gate_vs_primary(rows: Sequence[Row]) -> F:
    pos = [r[0] for r in rows if r[1] == 1]
    neg = [r[0] for r in rows if r[1] == 0]
    if not pos or not neg:
        return F(1, 2)
    wins = ties = 0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return F(2 * wins + ties, 2 * len(pos) * len(neg))


def routing_sigma(z: Sequence[int], route: Sequence[int]) -> float:
    m = len(z)
    pi = sum(1 for r in route if r != 1) / m
    routed = [zi for zi, r in zip(z, route) if r != 1]
    if not routed or pi == 0.0:
        return 0.0
    d_e = sum(routed) / len(routed)
    sig_e = sum((zi - d_e) ** 2 for zi in routed) / len(routed)
    return math.sqrt(pi * sig_e + pi * (1.0 - pi) * d_e * d_e)


def bernstein(sigma: float, m: int, delta: float = DELTA, bound: int = M_BOUND) -> float:
    if m <= 0:
        return float("inf")
    lg = math.log(1.0 / delta)
    return sigma * math.sqrt(2.0 * lg / m) + (2.0 * bound * lg) / (3.0 * m)


def gamma_hat(g: float, sigma: float, m: int, rho: float = 0.0) -> float:
    return g - bernstein(sigma, m) - 2.0 * rho


def certifies(g: float, sigma: float, m: int, rho: float = 0.0) -> bool:
    return gamma_hat(g, sigma, m, rho) > 0.0


def repeat_block(block: Sequence[Row], times: int) -> Tuple[Row, ...]:
    out: List[Row] = []
    for k in range(times):
        for t, c1, c2 in block:
            out.append((t + 2 * k, c1, c2))
    return tuple(out)


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
    proto.opened = True


def certify_escalate(
    rows: Sequence[Row],
    *,
    opened_before_freeze: bool = False,
    tamper_after_freeze: bool = False,
) -> dict:
    """Gate-facing escalate certificate: freeze → measure → Bernstein.

    Returns certified bool plus audit fields. Protocol violations fail.
    """
    route = greedy_route(rows)
    proto = FreezeProtocol()
    if opened_before_freeze:
        mark_opened_first(proto)
        digest = proto.freeze(route)
        return {
            "certified": False,
            "reason": "protocol:opened-before-freeze",
            "digest": digest,
            "G": None,
            "gamma": None,
            "auc": float(auc_gate_vs_primary(rows)),
        }
    digest = proto.freeze(route)
    measured = list(route)
    if tamper_after_freeze:
        measured[0] = 2 if measured[0] == 1 else 1
    open_status = proto.open(measured)
    if open_status != "opened":
        return {
            "certified": False,
            "reason": f"protocol:{open_status}",
            "digest": digest,
            "G": None,
            "gamma": None,
            "auc": float(auc_gate_vs_primary(rows)),
        }
    g, pi, d_e, z = decomp(rows, route)
    sig = routing_sigma(z, route)
    gamma = gamma_hat(float(g), sig, len(rows))
    ok = certifies(float(g), sig, len(rows))
    return {
        "certified": ok,
        "reason": "certify" if ok else "refuse:G_le_0",
        "digest": digest,
        "G": float(g),
        "pi": float(pi),
        "Delta_E": float(d_e),
        "gamma": gamma,
        "auc": float(auc_gate_vs_primary(rows)),
        "m": len(rows),
    }


__all__ = [
    "AUC_POOL",
    "GAIN_BLOCK",
    "certify_escalate",
    "repeat_block",
    "freeze_digest",
    "FreezeProtocol",
    "mark_opened_first",
    "decomp",
    "greedy_route",
    "route_on_t",
    "auc_gate_vs_primary",
    "certifies",
    "routing_sigma",
]
