#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Minimal check-then-lease stub for the trust-compile-gate join.

Faithful subset of Wang & Shi, arXiv:2608.17007 (check-then-lease):
tool program is proposal; independent checker rebuilds bounded IR;
atomic capacity lease; staged output; publish iff postcondition G holds.
Adapted to a CRM outbound draft tool (send / escalate), not the CSV table.
Stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Toy CRM contact list (immutable input). Units are exact integers.
# ---------------------------------------------------------------------------
Contact = Tuple[int, int, int]  # id, score, eligible
CONTACTS: Tuple[Contact, ...] = (
    (1, 4, 1),
    (2, 1, 0),
    (3, 5, 1),
    (4, 2, 1),
    (5, 3, 0),
    (6, 6, 1),
    (7, 1, 1),
    (8, 2, 0),
)
N_CONTACTS = len(CONTACTS)
SUM_ELIGIBLE = sum(c[1] for c in CONTACTS if c[2] == 1)  # 18

R_RUN = 1
R_IO = 1
STAGED = 1
WINDOW = 2
B = 6  # capacity: bounded scan fits; eager whole-list does not
D_WINDOW = (1, 2)

EAGER_SRC = (
    "df = read_contacts(rows)\n"
    "out = df.filter(eligible == 1).sum(score)\n"
)
STREAM_SRC = (
    "df = scan_contacts(rows, window=2)\n"
    "out = df.filter(eligible == 1).sum(score)\n"
)
ESCALATE_SRC = (
    "df = scan_contacts(rows, window=2)\n"
    "out = df.filter(eligible == 1).sum(score)\n"
    "# escalate-tool\n"
)


def _digest(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


PLATFORM = {
    "R_run": R_RUN,
    "R_io": R_IO,
    "hash": _digest({"R_run": R_RUN, "R_io": R_IO, "vm": "toy-crm-v1"}),
}


@dataclass(frozen=True)
class Theta:
    family: str
    filter_col: str
    filter_val: int
    agg: str
    agg_col: str


@dataclass(frozen=True)
class Facts:
    n_rows: int
    schema: Tuple[str, ...]
    sum_score: int
    n_eligible: int
    input_hash: str


IR = Tuple[Tuple[Any, ...], ...]


def S_crm(src: str) -> Optional[Theta]:
    if "read_contacts" not in src and "scan_contacts" not in src:
        return None
    if "filter" not in src or "eligible" not in src:
        return None
    if "sum" not in src or "score" not in src:
        return None
    return Theta("crm", "eligible", 1, "sum", "score")


def S_other(src: str) -> Optional[Theta]:
    if "fasta" in src.lower() or "SeqIO" in src:
        return Theta("fasta", "eligible", 1, "sum", "score")
    return None


def I_crm(rows: Sequence[Contact]) -> Facts:
    frozen = tuple(rows)
    return Facts(
        n_rows=len(frozen),
        schema=("id", "score", "eligible"),
        sum_score=sum(r[1] for r in frozen),
        n_eligible=sum(1 for r in frozen if r[2] == 1),
        input_hash=_digest(frozen),
    )


def L_crm(theta: Theta, facts: Facts, window: int, staged: int) -> IR:
    del facts
    return (
        ("scan", window),
        ("filter", theta.filter_col, theta.filter_val),
        ("sum", theta.agg_col),
        ("stage", staged),
    )


def A_of(ir: IR) -> int:
    head = ir[0]
    if head[0] in ("load", "scan"):
        return int(head[1])
    raise ValueError("unknown IR head")


def U_of(ir: IR, staged: int) -> int:
    return R_RUN + R_IO + A_of(ir) + staged


def G_crm(y: Optional[int], n_seen: int, facts: Facts) -> bool:
    if y is None or type(y) is not int:
        return False
    if n_seen != facts.n_rows:
        return False
    return 0 <= y <= facts.sum_score


@dataclass(frozen=True)
class Proposal:
    relation: str
    theta: Theta
    facts: Facts
    ir: IR
    U: int
    window: int
    staged: int
    binds: Tuple[str, ...]


@dataclass
class CheckResult:
    ok: bool
    reason: str
    rebuilt: Optional[IR] = None
    U: int = 0
    facts: Optional[Facts] = None
    theta: Optional[Theta] = None


@dataclass
class Ledger:
    cap: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    active: Dict[int, int] = field(default_factory=dict)
    nxt: int = 0

    def acquire(self, u: int) -> Optional[int]:
        with self.lock:
            if sum(self.active.values()) + u > self.cap:
                return None
            self.nxt += 1
            self.active[self.nxt] = u
            return self.nxt

    def release(self, lid: Optional[int]) -> None:
        if lid is None:
            return
        with self.lock:
            self.active.pop(lid, None)

    def charged(self) -> int:
        with self.lock:
            return sum(self.active.values())


def bind_hashes(src: str, facts: Facts) -> Tuple[str, ...]:
    return (
        _digest(src),
        facts.input_hash,
        PLATFORM["hash"],
        _digest("toy-crm-checker-v1"),
    )


def eager_load_ir(n: int) -> IR:
    return (("load", n), ("filter", "eligible", 1), ("sum", "score"), ("stage", STAGED))


def make_proposal(
    src: str,
    rows: Sequence[Contact],
    window: int,
    *,
    relation: str = "crm",
    ir: Optional[IR] = None,
    U: Optional[int] = None,
    theta: Optional[Theta] = None,
    facts: Optional[Facts] = None,
    staged: int = STAGED,
) -> Proposal:
    facts = facts if facts is not None else I_crm(rows)
    theta = theta if theta is not None else S_crm(src)
    assert theta is not None
    ir = ir if ir is not None else L_crm(theta, facts, window, staged)
    U = U if U is not None else U_of(ir, staged)
    return Proposal(relation, theta, facts, ir, U, window, staged, bind_hashes(src, facts))


def check(z: Proposal, src: str, rows: Sequence[Contact]) -> CheckResult:
    matches = []
    for name, rec in (("crm", S_crm), ("fasta", S_other)):
        th = rec(src)
        if th is not None:
            matches.append((name, th))
    if len(matches) != 1:
        return CheckResult(False, "no-unique-relation")
    name_hat, theta_hat = matches[0]
    if z.relation != name_hat:
        return CheckResult(False, "wrong-relation")
    facts_hat = I_crm(rows)
    if z.ir and z.ir[0][0] == "load":
        u_load = U_of(z.ir, z.staged)
        if u_load > B:
            return CheckResult(False, "over-cap", None, u_load, facts_hat, theta_hat)
    if z.window not in D_WINDOW or z.staged != STAGED:
        return CheckResult(False, "config-not-in-D")
    t_hat = L_crm(theta_hat, facts_hat, z.window, z.staged)
    u_hat = U_of(t_hat, z.staged)
    binds_hat = bind_hashes(src, facts_hat)
    if z.theta != theta_hat:
        return CheckResult(False, "theta-mismatch", t_hat, u_hat, facts_hat, theta_hat)
    if z.facts != facts_hat:
        return CheckResult(False, "facts-mismatch", t_hat, u_hat, facts_hat, theta_hat)
    if z.ir != t_hat:
        return CheckResult(False, "ir-mismatch", t_hat, u_hat, facts_hat, theta_hat)
    if z.U != u_hat:
        return CheckResult(False, "U-mismatch", t_hat, u_hat, facts_hat, theta_hat)
    if z.binds != binds_hat:
        return CheckResult(False, "bind-mismatch", t_hat, u_hat, facts_hat, theta_hat)
    if u_hat > B:
        return CheckResult(False, "over-cap", t_hat, u_hat, facts_hat, theta_hat)
    return CheckResult(True, "pass", t_hat, u_hat, facts_hat, theta_hat)


def run_vm(ir: IR, rows: Sequence[Contact]) -> Tuple[int, int, int]:
    peak = 0
    acc = 0
    n_seen = 0
    head = ir[0]
    filt = ir[1]
    if head[0] == "load":
        live = len(rows)
        peak = max(peak, live + STAGED)
        for r in rows:
            n_seen += 1
            if r[2] == filt[2]:
                acc += r[1]
        return peak, acc, n_seen
    if head[0] == "scan":
        w = int(head[1])
        for i in range(0, len(rows), w):
            chunk = rows[i : i + w]
            live = len(chunk) + STAGED
            peak = max(peak, live)
            for r in chunk:
                n_seen += 1
                if r[2] == filt[2]:
                    acc += r[1]
        return peak, acc, n_seen
    raise ValueError("vm: unknown IR")


@dataclass
class Runtime:
    ledger: Ledger = field(default_factory=lambda: Ledger(B))
    executed: bool = False
    published: Optional[int] = None
    staged: Optional[int] = None
    last_reason: str = ""
    hold_lease: bool = False  # if True, do not release (for concurrent test)

    def publish(self, y: Optional[int], n_seen: int, facts: Facts) -> bool:
        if not G_crm(y, n_seen, facts):
            self.staged = None
            return False
        self.published = y
        self.staged = None
        return True

    def dispatch(self, src: str, rows: Sequence[Contact], z: Proposal) -> str:
        self.executed = False
        self.published = None
        self.staged = None
        c = check(z, src, rows)
        if not c.ok:
            self.last_reason = c.reason
            return f"refuse:{c.reason}"
        assert c.rebuilt is not None and c.facts is not None
        lid = self.ledger.acquire(c.U)
        if lid is None:
            self.last_reason = "lease-refused"
            return "refuse:lease"
        try:
            peak, value, n_seen = run_vm(c.rebuilt, rows)
            self.executed = True
            if peak > c.U:
                self.last_reason = "peak-over-lease"
                return "refuse:peak"
            self.staged = value
            if not self.publish(value, n_seen, c.facts):
                self.last_reason = "postcondition"
                return "refuse:postcondition"
            self.last_reason = "commit"
            return "commit"
        finally:
            if not self.hold_lease:
                self.ledger.release(lid)


def lease_send(
    *,
    eager: bool = False,
    shared_ledger: Optional[Ledger] = None,
    hold_lease: bool = False,
    fail_postcondition: bool = False,
) -> dict:
    """Gate-facing send/escalate tool: check → lease → stage → postcondition."""
    src = EAGER_SRC if eager else STREAM_SRC
    rows = CONTACTS
    rt = Runtime()
    if shared_ledger is not None:
        rt.ledger = shared_ledger
    rt.hold_lease = hold_lease
    if eager:
        z = make_proposal(
            src,
            rows,
            window=N_CONTACTS,
            ir=eager_load_ir(N_CONTACTS),
            U=U_of(eager_load_ir(N_CONTACTS), STAGED),
        )
    else:
        z = make_proposal(src, rows, WINDOW)
    if fail_postcondition:
        # Force a bad publish path after a successful check+lease by
        # monkey-patching G via a dispatch that swaps publish.
        c = check(z, src, rows)
        if not c.ok:
            return {"admitted": False, "published": False, "reason": c.reason, "value": None}
        lid = rt.ledger.acquire(c.U)
        if lid is None:
            return {"admitted": False, "published": False, "reason": "lease-refused", "value": None}
        try:
            peak, value, n_seen = run_vm(c.rebuilt, rows)
            rt.staged = value
            # Broken postcondition: negative score
            ok = rt.publish(-1, n_seen, c.facts)
            return {
                "admitted": True,
                "published": ok,
                "reason": "postcondition" if not ok else "commit",
                "value": rt.published,
                "peak": peak,
            }
        finally:
            if not hold_lease:
                rt.ledger.release(lid)
    status = rt.dispatch(src, rows, z)
    return {
        "admitted": status == "commit",
        "published": rt.published is not None,
        "reason": status if status.startswith("refuse") else status,
        "value": rt.published,
        "charged": rt.ledger.charged(),
        "U_stream": U_of(L_crm(S_crm(STREAM_SRC), I_crm(rows), WINDOW, STAGED), STAGED),
    }


__all__ = [
    "CONTACTS",
    "N_CONTACTS",
    "SUM_ELIGIBLE",
    "B",
    "WINDOW",
    "EAGER_SRC",
    "STREAM_SRC",
    "ESCALATE_SRC",
    "Ledger",
    "Runtime",
    "make_proposal",
    "check",
    "lease_send",
    "U_of",
    "L_crm",
    "S_crm",
    "I_crm",
    "eager_load_ir",
    "STAGED",
]
