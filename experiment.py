#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Witness, don't self-attest.

recheck-after-forget: a re-check certifies the home it re-reads, not the fact; watermark +
home-aware leases leave 4 wrong sends (retention home reverted AND newer write unobserved).
refuse-to-certify: refuse until an independent, frozen certificate says the decision holds;
complementarity is not enough; a mismatched freeze is a failed certificate.

Joined: who may attest that a re-read is the newest write? Not the store's own counter if
it reverts with the data, not a second home that lags with it; an independent monotone
source closes the residual by refusing, without holding any correct draft.

Stdlib only. No network. No GPU. No vendor API. No product import.
Run: python3 experiment.py
"""
from __future__ import annotations

import contextlib
import io
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import certify_gate as cg
import recheck_experiment as rx
from attest_gate import MODES, attested_gate, freeze, head, witness_heads
from recheck_experiment import KINDS, correct, matrix, updated
from send_gate import LEASE, LONG_PATH, draft_from_live, live_window1, cross
from answer_gate import history_store
from store_gate import (SourceLog, Record, bump, lease_ok, certify, recheck_gate, view,
                        watermark_items, watermark_live)
from triage_stub import hard_cost, kitchen
from trust_gate import RISK_HIGH

RESIDUAL = [("revert", "before/unobs", "pre"), ("revert", "between/unobs", "pre")]


def _ok(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _tally(mode: Optional[str], at_send: bool = True) -> Tuple[dict, list]:
    cases, _ = matrix("pinned")
    s = defaultdict(int)
    wrong = set()
    for x in cases:
        a, ok = attested_gate(x, mode, at_send=at_send)["action"], correct(x)
        s["sent"] += a == "send"
        s["correct_sent"] += a == "send" and ok
        s["wrong_sends"] += a == "send" and not ok
        s["held_correct"] += a != "send" and ok
        s["refused"] += a == "refuse"
        s["refused_correct"] += a == "refuse" and ok
        if a == "send" and not ok:
            wrong.add((x["kind"], x["ev"], x["dn"]))
    return dict(s), sorted(wrong)


def scenario_1():
    """Parents bind in-process; the residual of recheck-after-forget is reproduced exactly."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for _, fn in rx.SCENARIOS:
            fn()
        cg.main()                                   # refuse-to-certify, raises on failure
    _ok(out.getvalue().count(": ok") == 7, "refuse-to-certify 7/7")
    cases, res = matrix("pinned")
    s, wrong = _tally(None)
    _ok(all(attested_gate(x, None)["action"] == g["wm+home"]["action"] for x, g in zip(cases, res)),
        "mode None is exactly the parent wm+home gate")
    _ok(len(cases) == 88 and s["wrong_sends"] == 4 and wrong == RESIDUAL, (s, wrong))
    return {"parents_7of7": ["recheck-after-forget (and its 4 parents)", "refuse-to-certify"],
            "cases": len(cases), "wm+home": s, "residual": wrong}


def scenario_2():
    """Self-attestation follows the snapshot: same reverted bytes, two counters."""
    cases, _ = matrix("pinned")
    res = [x for x in cases if (x["kind"], x["ev"], x["dn"]) in RESIDUAL]
    seen = {}
    for x in res:
        r = view(x["ls"], "revert", compile_seq=x["cs"])[LONG_PATH]
        snap = head("self/snapshot", x["ls"], "revert", LONG_PATH, x["cs"])
        mono = head("self/monotone", x["ls"], "revert", LONG_PATH, x["cs"])
        seen[(x["ev"], x["keys"])] = (r.seq, snap, mono,
                                      attested_gate(x, "self/snapshot")["action"],
                                      attested_gate(x, "self/monotone")["action"])
    _ok(len(res) == 4, len(res))
    _ok(all(v[0] == v[1] == 3 and v[2] == 7 for v in seen.values()),
        f"snapshot counter attests the reverted seq as newest; monotone counter says 7: {seen}")
    _ok(all(v[3] == "send" and v[4] == "refuse" for v in seen.values()), seen)
    return {"read_seq/snapshot_head/monotone_head/verdicts": {f"{e}/{k}keys": v for (e, k), v in seen.items()}}


def scenario_3():
    """NEW: 88-case matrix by attestation source (attested at certify and at send)."""
    t = {m: _tally(m) for m in (None,) + MODES}
    base = t[None][0]
    for m in ("self/snapshot", "cross/cartridge"):
        _ok(t[m][0] == base, f"{m} adds nothing: {t[m][0]}")
    for m in ("self/monotone", "cross/replica", "witness"):
        s, wrong = t[m]
        _ok(s["wrong_sends"] == 0 and wrong == [], f"{m}: {s}")
        _ok(s["refused"] == 4 and s["refused_correct"] == 0, f"{m} refuses exactly the residual")
        _ok(s["correct_sent"] == base["correct_sent"] == 14 and s["held_correct"] == base["held_correct"] == 2,
            f"{m} wins without holding a single extra correct draft")
    return {("wm+home" if m is None else m): {k: s[k] for k in ("sent", "correct_sent", "wrong_sends",
                                                                "held_correct", "refused")}
            for m, (s, _) in t.items()}


def scenario_4():
    """Second home as an advisor (refuse-to-certify math): redundant vs useful, and the bracket."""
    cases, res = matrix("pinned")
    out = {}
    for mode in ("cross/cartridge", "cross/replica"):
        rows = []
        for x, g in zip(cases, res):
            ok = correct(x)
            right = lambda a: int((a == "send") == ok)
            rows.append((KINDS.index(x["kind"]), right(g["wm+home"]["action"]),
                         right(attested_gate(x, mode)["action"])))
        route = [2] * len(rows)
        G, pi, dE, z = cg.decomp(rows, route)
        sig = cg.routing_sigma(z, route)
        cell = [r for r in rows if r[0] == KINDS.index("revert")]
        Gc, _, _, zc = cg.decomp(cell, [2] * len(cell))
        sc = cg.routing_sigma(zc, [2] * len(cell))
        out[mode] = {"G": str(G), "Phi": str(cg.phi(rows)), "E": str(cg.excess_over_independence(rows)),
                     "certifies@88": cg.certifies(float(G), sig, len(rows)),
                     "m*": cg.m_star(float(G), sig),
                     "revert_cell": {"m": len(cell), "G": str(Gc), "certifies": cg.certifies(float(Gc), sc, len(cell)),
                                     "m*": cg.m_star(float(Gc), sc)}}
    c, r = out["cross/cartridge"], out["cross/replica"]
    _ok(c["G"] == "0" and c["Phi"] == "0" and c["m*"] is None, "lagging second home: zero gain, never certifies")
    _ok(r["G"] == "1/22" and r["Phi"] == "1/22", "independent second home: gain 4/88, all in the revert cell")
    _ok(not r["certifies@88"] and r["m*"] > 88 and not r["revert_cell"]["certifies"]
        and r["revert_cell"]["m*"] < r["m*"], "a 0-wrong-send matrix is not a certificate at this m")
    _ok(cg.F(c["E"].split("/")[0]) < 0 and cg.F(r["E"].split("/")[0]) < 0,
        "excess over independence is negative for BOTH: it does not separate redundant from useful here")
    return out


def scenario_5():
    """NEW: the witness head is frozen at certify and re-opened at send."""
    cases, _ = matrix("pinned")
    moved = [x for x in cases if freeze(witness_heads(x["lc"])) != freeze(witness_heads(x["ls"]))]
    _ok(len(moved) > 0 and all(x["ev"].startswith("between") for x in moved)
        and all(x["ev"].startswith("between") is False or x in moved for x in cases),
        "the freeze moves exactly when a write lands between certify and send")
    once = {m: _tally(m, at_send=False) for m in ("self/monotone", "cross/replica", "witness")}
    for m, (s, wrong) in once.items():
        _ok(s["wrong_sends"] == 2 and wrong == [("revert", "between/unobs", "pre")],
            f"attested at certify only, {m} silently passes the between-write: {s}")
    labels = defaultdict(int)
    for x in moved:
        g = attested_gate(x, "witness")
        labels[(g.get("freeze"), g["action"])] += 1
    _ok(all(f == "failed-certificate" for f, _ in labels) and _tally("witness")[0]["wrong_sends"] == 0,
        "every moved freeze is reported as a failed certificate and re-certified, never a silent pass")
    return {"moved_freezes": len(moved), "certify_only_wrong_sends": {m: s["wrong_sends"] for m, (s, _) in once.items()},
            "reopened_wrong_sends": 0, "moved_freeze_verdicts": {f"{a}": n for (_, a), n in labels.items()}}


# ---------------------------------------------------------------------------
# Hold vs escalate over time: transient lag vs persistent revert
# ---------------------------------------------------------------------------

H, R, CATCH_UP = 8, 3, 2


def store_trace(kind: str) -> List[Tuple[int, int]]:
    """(read seq of LONG_PATH, store's self-reported head) per tick. Newest write is seq 7."""
    if kind == "lag":                        # replica catches up at CATCH_UP; counter never goes back
        return [(3, 3) if t < CATCH_UP else (7, 7) for t in range(H)]
    if kind == "revert":                     # restored from snapshot; counter restored too
        return [(3, 3)] * H
    raise ValueError(kind)


def run_policy(policy: str, trace: List[Tuple[int, int]], witness: int = 7,
               prior_head: int = 0) -> Tuple[str, int]:
    """prior_head: the store's own head as the agent last saw it before this window
    (7 if it read the store after the newer write and before the revert)."""
    prev_head, stale_for = prior_head, 0
    for t, (seq, self_head) in enumerate(trace):
        went_back = self_head < prev_head
        prev_head = max(prev_head, self_head)
        if seq >= witness:
            return "send", t
        stale_for += 1
        if policy == "escalate-now":
            return "escalate", t
        if policy == "monotone-aware" and went_back:
            return "escalate", t
        if policy in ("hold-then-escalate", "monotone-aware") and stale_for > R:
            return "escalate", t
    return "stuck", H


def scenario_6():
    """NEW: should a stale home hold or escalate? A lag never goes back; a revert does."""
    pols = ("hold-forever", "escalate-now", "hold-then-escalate", "monotone-aware")
    grid = {}
    for kind, sb in (("lag", False), ("revert", True), ("revert", False)):
        for p in pols:
            grid[(kind, sb, p)] = run_policy(p, store_trace(kind), prior_head=7 if sb else 0)
    _ok(grid[("lag", False, "hold-forever")] == ("send", CATCH_UP)
        and grid[("revert", True, "hold-forever")] == ("stuck", H), "hold-forever: fine on lag, stuck on revert")
    _ok(grid[("lag", False, "escalate-now")] == ("escalate", 0), "escalate-now wakes a human for a lag")
    _ok(grid[("lag", False, "monotone-aware")] == ("send", CATCH_UP)
        and grid[("revert", True, "monotone-aware")] == ("escalate", 0)
        and grid[("revert", True, "hold-then-escalate")] == ("escalate", R),
        "a counter that goes back is a revert: escalate at once; a lag is held until it catches up")
    _ok(grid[("revert", False, "monotone-aware")] == grid[("revert", False, "hold-then-escalate")] == ("escalate", R),
        "revert before the agent ever saw the newer head: no backward step, falls back to the retry budget")
    # The witness is what makes 'stale' visible at all: with only the store's own head,
    # the reverted read looks newest and every policy sends at tick 0.
    blind = run_policy("monotone-aware", store_trace("revert"), witness=3)
    _ok(blind == ("send", 0), blind)
    return {f"{k}{'/seen7' if sb else ''}:{p}": v for (k, sb, p), v in grid.items()} | {"no-witness": blind}


def scenario_7():
    """Trust stages and the horizon still bind; the pack is clean."""
    cases, _ = matrix("pinned")
    x = next(x for x in cases if x["kind"] == "faithful" and x["ev"] == "none" and x["dn"] == "pre"
             and correct(x))
    ok = attested_gate(x, "witness")
    log0 = x["lc"]
    w = watermark_live(log0, log0.head)
    fv = view(log0, "faithful", compile_seq=log0.head)
    live = live_window1()
    keys = tuple(x["d"].keys())
    c = cross(live, keys=keys, items=kitchen() + watermark_items(w, keys))
    risk = recheck_gate(x["d"], c, fv, fv, w_cert=w, w_send=w, scalars=RISK_HIGH)
    tight = cross(live, keys=keys, budget=hard_cost(kitchen()), items=kitchen() + watermark_items(w, keys))
    unsafe = recheck_gate(x["d"], tight, fv, fv, w_cert=w, w_send=w)
    _ok(ok["action"] == "send" and risk["action"] == "hold" and unsafe["action"] == "escalate",
        (ok["action"], risk["action"], unsafe["action"]))
    needles = ("cosmi" + "cat", "svor" + "onov")
    here = os.path.dirname(os.path.abspath(__file__))
    scanned = 0
    for fn in os.listdir(here):
        if fn.endswith((".py", ".md", ".txt")) or fn == "LICENSE":
            with open(os.path.join(here, fn), encoding="utf-8") as fh:
                body = fh.read().lower()
            _ok(not any(n in body for n in needles), f"product name found in {fn}")
            scanned += 1
    return {"witness_send": ok["action"], "risk_high": risk["action"], "tight_budget": unsafe["action"],
            "files_scanned_clean": scanned}


SCENARIOS = [
    ("1_parents_bind_residual_reproduced", scenario_1),
    ("2_self_attestation_follows_the_snapshot", scenario_2),
    ("3_NEW_matrix_by_attestation_source", scenario_3),
    ("4_second_home_as_advisor_certificate", scenario_4),
    ("5_NEW_witness_freeze_reopened_at_send", scenario_5),
    ("6_NEW_hold_vs_escalate_lag_vs_revert", scenario_6),
    ("7_trust_stages_horizon_and_clean", scenario_7),
]


def main() -> int:
    passed = 0
    for name, fn in SCENARIOS:
        try:
            detail = fn()
            passed += 1
            print(f"PASS  {name}: {detail}")
        except AssertionError as e:
            print(f"FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(SCENARIOS)} assertions passed")
    return 0 if passed == len(SCENARIOS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
