#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Recheck-after-forget: a re-check is only as good as the home it re-reads.

send-after-cliff re-checks claims against a 'retention store' it assumes is source.
cartridge-retrieves-forgets says the retrieval-winning home is also the forgetting home.
Joined: if the store a re-check reads is a lagging / compacting / reverted home, the
re-check can 'succeed' by re-reading an OLD value (the old fact returns), and a reverted
body has a valid digest of its own.

Hypothesis: leases need a monotone watermark (highest write seq seen for the fact) and
the gate must know which home it re-reads (retention vs retrieval).

Stdlib only. No network. No GPU. No vendor API. No product import.
Run: python3 experiment.py
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Dict, List

import answer_gate
import cartridge_gate
import send_gate
import trust_gate as trust_pack
from answer_gate import history_store
from cliff_stub import Msg, Session, apply_cliff
from horizon_gate import SYSTEM, TASK
from send_gate import (LEASE, LEASE_NOTICE, LONG_PATH, REFUND, REFUND_NOTICE, cross,
                       draft_from_live, draft_from_memory, live_window1, live_window2)
from store_gate import (HOME_TAG, SourceLog, WLease, _digest, bump, certify, home_view,
                        lease_ok, parent_gate, receipt, recheck_gate, view,
                        watermark_context, watermark_from_items, watermark_items,
                        watermark_live, wrong_claims)
from triage_stub import hard_cost, kitchen
from trust_gate import RISK_HIGH

KINDS = ("faithful", "cartridge", "compaction", "revert")
EVENTS = (("none", None, False), ("before/obs", "before", True), ("before/unobs", "before", False),
          ("between/obs", "between", True), ("between/unobs", "between", False))
UPDATE = {LEASE: "LX-4421", REFUND: "135"}   # the source moves: one new write of parser_2
GATES = ("trust", "digest", "home", "wm", "wm+home")


def _ok(cond, msg):
    if not cond:
        raise AssertionError(msg)


def updated(body: str) -> str:
    for k, v in UPDATE.items():
        body = re.sub(rf"\b{k}=[A-Za-z0-9\-]+", f"{k}={v}", body)
    return body


def base():
    live = live_window1()
    log0 = SourceLog.from_store(history_store(live))
    return live, log0


# ---------------------------------------------------------------------------
# Case matrix: 4 homes x 5 update events x 2 notice templates x drafts
# ---------------------------------------------------------------------------

def build_cases(wsource: str = "pinned") -> List[dict]:
    """wsource: 'pinned' = watermark written into the typed set before the cliff
    (survival decided by triage); 'context' = re-derived after the cliff from carried text."""
    live, log0 = base()
    compile_seq = log0.head          # cartridges compiled at the end of window 1
    out = []
    for keys in (LEASE_NOTICE, REFUND_NOTICE):
        plain = cross(live, keys=keys)
        if wsource == "pinned":
            c = cross(live, keys=keys,
                      items=kitchen() + watermark_items(watermark_live(log0, log0.head), keys))
            w0 = watermark_from_items(c.kept)
        else:
            c = plain
            w0 = watermark_context(plain, log0)
        for ev, when, obs in EVENTS:
            lc, wc = log0.copy(), dict(w0)
            if when == "before":
                r = lc.write(LONG_PATH, updated(lc.latest()[LONG_PATH].body))
                if obs:
                    for k in UPDATE:
                        wc = bump(wc, k, r.seq)
            ls, ws = lc.copy(), dict(wc)
            if when == "between":
                r = ls.write(LONG_PATH, updated(ls.latest()[LONG_PATH].body))
                if obs:
                    for k in UPDATE:
                        ws = bump(ws, k, r.seq)
            pre = draft_from_live(live, keys)
            drafts = {"pre": pre, "post": draft_from_memory(c, keys)}
            if when == "before" and obs:   # the agent redrafts from the receipt it saw
                drafts["redraft"] = {k: UPDATE.get(k, v) for k, v in pre.items()}
            for kind in KINDS:
                for dn, d in drafts.items():
                    out.append(dict(kind=kind, ev=ev, keys=len(keys), dn=dn, d=d, c=c,
                                    lc=lc, ls=ls, wc=wc, ws=ws, cs=compile_seq))
    return out


def run_gates(x: dict) -> Dict[str, dict]:
    k, cs, d, c = x["kind"], x["cs"], x["d"], x["c"]
    vc, vs = view(x["lc"], k, compile_seq=cs), view(x["ls"], k, compile_seq=cs)
    _, hc = home_view(x["lc"], k, home_aware=True, compile_seq=cs)
    _, hs = home_view(x["ls"], k, home_aware=True, compile_seq=cs)
    return {"trust": parent_gate(d, c, vc, vs, "none"),
            "digest": parent_gate(d, c, vc, vs, "path"),
            "home": parent_gate(d, c, hc, hs, "path"),
            "wm": recheck_gate(d, c, vc, vs, w_cert=x["wc"], w_send=x["ws"]),
            "wm+home": recheck_gate(d, c, hc, hs, w_cert=x["wc"], w_send=x["ws"]),
            "_reduced": recheck_gate(d, c, vc, vs, w_cert=None, w_send=None)}


def correct(x: dict) -> bool:
    return all(x["ls"].truth(k) == v for k, v in x["d"].items())


def tally(cases: List[dict], results: List[Dict[str, dict]]):
    t = defaultdict(lambda: defaultdict(int))
    for x, g in zip(cases, results):
        ok = correct(x)
        for name in GATES:
            a = g[name]["action"]
            for scope in ("all", x["kind"]):
                s = t[(scope, name)]
                s["sent"] += a == "send"
                s["correct_sent"] += a == "send" and ok
                s["wrong_sends"] += a == "send" and not ok
                s["wrong_claims"] += wrong_claims(x["d"], a, x["ls"])
                s["held_correct"] += a != "send" and ok
    return {k: dict(v) for k, v in t.items()}


_CACHE: dict = {}


def matrix(wsource: str = "pinned"):
    if wsource not in _CACHE:
        cases = build_cases(wsource)
        _CACHE[wsource] = (cases, [run_gates(x) for x in cases])
    return _CACHE[wsource]


# ---------------------------------------------------------------------------
# Seven join assertions
# ---------------------------------------------------------------------------

def scenario_1():
    """Parents bind inside the pack; the faithful home is exactly the parent's store."""
    for name, fn in send_gate.SCENARIOS + trust_pack.SCENARIOS + answer_gate.SCENARIOS:
        fn()
    for i in range(1, 8):
        getattr(cartridge_gate, f"scenario_{i}")()
    live, log0 = base()
    store = history_store(live)
    fv = view(log0, "faithful", compile_seq=log0.head)
    _ok({p: r.body for p, r in fv.items()} == store, "faithful home == parent retention store")
    c = cross(live, keys=LEASE_NOTICE)
    d = draft_from_memory(c, LEASE_NOTICE)
    _ok(parent_gate(d, c, fv, fv, "context")["action"]
        == send_gate.send_after_cliff(d, c, store)["action"] == "hold", "parent verdict reproduced")
    t2 = send_gate.scenario_2()
    _ok(t2["parent_only"]["wrong"] == 16 and t2["provenance"]["wrong"] == 0, t2)
    return {"parents_7of7": ["send-after-cliff", "trust-compile-gate", "answer-after-cliff",
                             "cartridge-retrieves-forgets"],
            "parent_matrix": {k: t2[k]["wrong"] for k in t2}, "store_paths": len(store)}


def scenario_2():
    """The old fact returns with a valid digest: compaction drop and snapshot revert."""
    live, log0 = base()
    w = watermark_live(log0, log0.head)
    cs = log0.head
    # (a) high-rate compaction keeps LEASE_ID of parser_2, drops its REFUND update;
    #     the re-check walks on to parser_1 and finds the older REFUND=90.
    c = cross(live, keys=REFUND_NOTICE)
    d = draft_from_memory(c, REFUND_NOTICE)
    vc = view(log0, "compaction", compile_seq=cs)
    dig = parent_gate(d, c, vc, vc, "path")
    wm = recheck_gate(d, c, vc, vc, w_cert=w, w_send=w)
    _, hv = home_view(log0, "compaction", home_aware=True, compile_seq=cs)
    home = parent_gate(d, c, hv, hv, "path")
    _ok(d == {REFUND: "90", "TICKET": "T-88"} and log0.truth(REFUND) == "120", d)
    _ok(REFUND not in vc[LONG_PATH].body and dig["action"] == "send"
        and wrong_claims(d, "send", log0) == 1, "compaction: digest-only re-check certifies 90")
    _ok(wm["action"] == "hold" and wm["claims"][REFUND].get("revert") == "seq 2 < W 3", wm["claims"])
    _ok(home["action"] == "hold" and home["claims"][REFUND]["seen"] == "120", "home-aware reads 120")
    # (b) snapshot revert after a certified lease: same bytes, same digest, older seq.
    d2 = draft_from_live(live, LEASE_NOTICE)
    c2 = cross(live, keys=LEASE_NOTICE)
    cert = certify(d2, c2, view(log0, "revert", compile_seq=cs), w)
    ls_log = log0.copy()
    new = ls_log.write(LONG_PATH, updated(ls_log.latest()[LONG_PATH].body))
    rv = view(ls_log, "revert", compile_seq=cs)
    lease: WLease = cert["leases"][LEASE]
    _ok(_digest(rv[LONG_PATH].body) == lease.digest == _digest(log0.versions(LONG_PATH)[0].body),
        "reverted body carries a valid digest equal to the lease's")
    _ok(lease_ok(lease, rv, None) and not lease_ok(lease, rv, bump(w, LEASE, new.seq)),
        "digest-only accepts the revert; a watermark that saw seq 7 refuses it")
    return {"compaction": {"reread": f"{REFUND}=90 @ seq 2 (parser_1)", "digest_only": dig["action"],
                           "wrong_claims": 1, "watermark": wm["action"], "home_aware": home["action"]},
            "revert": {"lease_seq": lease.seq, "send_seq": rv[LONG_PATH].seq, "newest_write": new.seq,
                       "digest_equal": True}}


def scenario_3():
    """NEW count: wrong sends over 88 cases (4 homes x 5 events x 2 notices x drafts)."""
    cases, res = matrix("pinned")
    t = tally(cases, res)
    al = {g: t[("all", g)] for g in GATES}
    _ok(len(cases) == 88 and sum(correct(x) for x in cases) == 16, len(cases))
    _ok(al["digest"]["wrong_sends"] > 0, "falsifier: digest-only never sent an old value")
    # Every wrong send is an OLD value that really was in the log, never a guess.
    for x, g in zip(cases, res):
        if g["digest"]["action"] == "send" and not correct(x):
            for k, v in x["d"].items():
                if x["ls"].truth(k) != v:
                    _ok(any(v in re.findall(rf"\b{k}=([A-Za-z0-9\-]+)", r.body)
                            for r in x["ls"].records), f"{k}={v} is not a past value")
    _ok(al["wm+home"]["wrong_sends"] < al["home"]["wrong_sends"] < al["digest"]["wrong_sends"]
        < al["trust"]["wrong_sends"], {g: al[g]["wrong_sends"] for g in GATES})
    _ok(al["wm+home"]["correct_sent"] > al["digest"]["correct_sent"]
        and al["wm+home"]["held_correct"] < al["digest"]["held_correct"],
        "and not by holding: it sends more correct drafts")
    resid = sorted({(x["kind"], x["ev"], x["dn"]) for x, g in zip(cases, res)
                    if g["wm+home"]["action"] == "send" and not correct(x)})
    _ok(resid == [("revert", "before/unobs", "pre"), ("revert", "between/unobs", "pre")],
        f"residual wrong sends: {resid}")
    return {g: {k: al[g][k] for k in ("sent", "correct_sent", "wrong_sends", "wrong_claims",
                                       "held_correct")} for g in GATES} | {"residual": resid}


def scenario_4():
    """Which piece fixes which home: home-awareness vs watermark (per-home breakdown)."""
    cases, res = matrix("pinned")
    t = tally(cases, res)
    ws = {kind: {g: t[(kind, g)]["wrong_sends"] for g in GATES} for kind in KINDS}
    hc = {kind: {g: t[(kind, g)]["held_correct"] for g in GATES} for kind in KINDS}
    _ok(all(g["_reduced"]["action"] == g["digest"]["action"] for g in res),
        "watermark gate with W off reduces to the parent digest gate on all cases")
    f = [g for x, g in zip(cases, res) if x["kind"] == "faithful"]
    _ok(all(len({g[n]["action"] for n in ("digest", "home", "wm", "wm+home")}) == 1 for g in f)
        and ws["faithful"]["digest"] == 0 and ws["faithful"]["trust"] > 0,
        "faithful home: every re-check gate agrees (no extra holds); trust-lease alone is wrong")
    _ok(ws["cartridge"]["digest"] == ws["cartridge"]["trust"] > 0
        and ws["revert"]["digest"] == ws["revert"]["trust"] > 0,
        "lagging/reverted home: digest re-check adds nothing over trusting the lease")
    _ok(ws["cartridge"]["home"] == ws["compaction"]["home"] == 0, "retrieval homes: home-aware fixes")
    _ok(ws["revert"]["home"] == ws["revert"]["digest"] and 0 < ws["revert"]["wm"] < ws["revert"]["digest"],
        "retention home itself reverted: only the watermark helps, and only partly")
    _ok(t[("compaction", "wm")]["sent"] == 0 and hc["compaction"]["wm"] == 4,
        "watermark alone on compaction wins only by holding every draft")
    return {"wrong_sends": ws, "held_correct": hc}


def scenario_5():
    """NEW: the watermark must cross the cliff pinned; re-derived from context it is lost."""
    live, log0 = base()
    c = cross(live, keys=LEASE_NOTICE)
    ctx_w = watermark_context(c, log0)
    cp = cross(live, keys=LEASE_NOTICE,
               items=kitchen() + watermark_items(watermark_live(log0, log0.head), LEASE_NOTICE))
    pin_w = watermark_from_items(cp.kept)
    tp = tally(*matrix("pinned"))
    tc = tally(*matrix("context"))
    wm_p, wm_c = tp[("all", "wm")]["wrong_sends"], tc[("all", "wm")]["wrong_sends"]
    wh_p, wh_c = tp[("all", "wm+home")]["wrong_sends"], tc[("all", "wm+home")]["wrong_sends"]
    _ok(LEASE not in ctx_w and ctx_w[REFUND] == 2, f"context watermark after the cliff: {ctx_w}")
    _ok(pin_w == {LEASE: 3, REFUND: 3, "TICKET": 6}, f"pinned watermark survives triage: {pin_w}")
    _ok(wm_c > wm_p, f"context watermark sends more old values: {wm_c} vs {wm_p}")
    _ok(wh_c >= wh_p, (wh_c, wh_p))
    b0 = hard_cost(kitchen())
    tight = cross(live, keys=LEASE_NOTICE, budget=b0,
                  items=kitchen() + watermark_items(watermark_live(log0, log0.head), LEASE_NOTICE))
    _ok(cp.b_min > b0 and tight.horizon == "unsafe", "the pinned watermark is not free")
    return {"context_W": ctx_w, "pinned_W": pin_w,
            "wm_wrong_sends": {"pinned": wm_p, "context": wm_c},
            "wm+home_wrong_sends": {"pinned": wh_p, "context": wh_c},
            "b_min": (b0, cp.b_min), "tight_budget": tight.horizon}


def scenario_6():
    """NEW second cliff x rotation x revert; same bytes re-read vs reverted (ABA)."""
    w1 = live_window1()
    log0 = SourceLog.from_store(history_store(w1))
    d = draft_from_live(w1, (LEASE,))
    c0 = cross(w1, keys=(LEASE,), token_pressure=False)          # certify while evidence is live
    v0 = view(log0, "faithful", compile_seq=log0.head)
    lease: WLease = certify(d, c0, v0, watermark_live(log0, log0.head))["leases"][LEASE]
    v1_body = log0.latest()[LONG_PATH].body
    grid, wrong, held_ok = {}, defaultdict(int), defaultdict(int)
    same_bytes = {}
    for hist in ("none", "rotate", "rotate+ABA"):
        log, live = log0.copy(), list(w1)
        if hist != "none":
            r = log.write(LONG_PATH, updated(v1_body))
            live += [Msg("tool_call", "notify", "notify", f"events/w{r.seq}"),
                     Msg("tool_result", receipt(LONG_PATH, r.seq, (LEASE,)))]
        if hist == "rotate+ABA":                                  # genuinely written back
            r = log.write(LONG_PATH, v1_body)
            live += [Msg("tool_call", "notify", "notify", f"events/w{r.seq}"),
                     Msg("tool_result", receipt(LONG_PATH, r.seq, (LEASE,)))]
        seen = watermark_live(log, log.head)                     # incl. receipts, pre-cliff
        wi = watermark_items({LEASE: max(seen[LEASE], lease.watermark)}, (LEASE,))
        c1 = cross(live, keys=(LEASE,), items=kitchen() + wi, epoch=1)
        prior = apply_cliff(Session(SYSTEM, TASK, prior_cliff=None, live=live)).prior_cliff
        c2 = cross(live_window2(), keys=(LEASE,), prior_cliff=prior, epoch=2,
                   items=kitchen() + [i for i in c1.kept if i.iid.startswith("wm_")])
        _ok(c2.horizon == "cliff" and not any("WRITE" in m.text for m in c2.carried),
            "cliff 2 discards C_{t-1}, receipts included")
        for home in ("faithful", "revert"):
            vs = view(log, home, compile_seq=log0.head)
            truth_ok = log.truth(LEASE) == d[LEASE]
            for ep, cc in ((1, c1), (2, c2)):
                acts = {"trust": True, "digest": lease_ok(lease, vs, None),
                        "wm-context": lease_ok(lease, vs, watermark_context(cc, log)),
                        "wm-pinned": lease_ok(lease, vs, watermark_from_items(cc.kept))}
                grid[(home, hist, ep)] = {g: ("send" if a else "hold") for g, a in acts.items()}
                for g, a in acts.items():
                    wrong[g] += a and not truth_ok
                    held_ok[g] += (not a) and truth_ok
            if (home, hist) in (("revert", "rotate"), ("faithful", "rotate+ABA")):
                same_bytes[(home, hist)] = (vs[LONG_PATH].seq, _digest(vs[LONG_PATH].body) == lease.digest,
                                            truth_ok, grid[(home, hist, 2)]["wm-pinned"])
    _ok(all(v == {"trust": "send", "digest": "send", "wm-context": "send", "wm-pinned": "send"}
            for k, v in grid.items() if k[1] == "none"), "no rotation: everyone sends (not holding all)")
    _ok(grid[("revert", "rotate", 1)]["wm-context"] == "hold"
        and grid[("revert", "rotate", 2)]["wm-context"] == "send",
        "context watermark catches the revert after cliff 1, loses it at cliff 2")
    _ok(wrong["trust"] > wrong["digest"] > wrong["wm-context"] > wrong["wm-pinned"] == 0, dict(wrong))
    _ok(same_bytes[("revert", "rotate")][1] and same_bytes[("faithful", "rotate+ABA")][1]
        and same_bytes[("revert", "rotate")][3] == "hold"
        and same_bytes[("faithful", "rotate+ABA")][3] == "send", same_bytes)
    return {"cells": len(grid), "wrong_sends": dict(wrong), "held_correct": dict(held_ok),
            "same_bytes(seq, digest==lease, draft_true, wm-pinned@e2)": {
                f"{h}/{s}": v for (h, s), v in same_bytes.items()}}


def scenario_7():
    """Trust stages and the horizon still bind after a watermark certify; pack is clean."""
    live, log0 = base()
    keys = LEASE_NOTICE
    w = watermark_live(log0, log0.head)
    c = cross(live, keys=keys, items=kitchen() + watermark_items(w, keys))
    fv = view(log0, "faithful", compile_seq=log0.head)
    d = draft_from_live(live, keys)
    ok = recheck_gate(d, c, fv, fv, w_cert=w, w_send=w)
    risk = recheck_gate(d, c, fv, fv, w_cert=w, w_send=w, scalars=RISK_HIGH)
    tight = cross(live, keys=keys, budget=hard_cost(kitchen()),
                  items=kitchen() + watermark_items(w, keys))
    unsafe = recheck_gate(d, tight, fv, fv, w_cert=w, w_send=w)
    _ok(ok["action"] == "send", ok.get("reason"))
    _ok(all(r["certified"] for r in risk["claims"].values()) and risk["action"] == "hold",
        "certified claims, high task-risk: the trust gate still holds")
    _ok(unsafe["action"] == "escalate", "pinned watermark over a tight budget: escalate")
    _ok(set(HOME_TAG.values()) == {"retention", "retrieval"}, HOME_TAG)
    needles = ("cosmi" + "cat", "svor" + "onov")
    here = os.path.dirname(os.path.abspath(__file__))
    scanned = 0
    for fn in os.listdir(here):
        if fn.endswith((".py", ".md", ".txt")) or fn == "LICENSE":
            with open(os.path.join(here, fn), encoding="utf-8") as fh:
                body = fh.read().lower()
            _ok(not any(n in body for n in needles), f"product name found in {fn}")
            scanned += 1
    return {"faithful_send": ok["action"], "risk_high": risk["action"],
            "tight_budget": unsafe["action"], "files_scanned_clean": scanned}


SCENARIOS = [
    ("1_parents_bind_faithful_home_is_parent_store", scenario_1),
    ("2_old_fact_returns_with_valid_digest", scenario_2),
    ("3_NEW_wrong_sends_88_case_matrix", scenario_3),
    ("4_which_piece_fixes_which_home", scenario_4),
    ("5_NEW_watermark_must_cross_the_cliff_pinned", scenario_5),
    ("6_NEW_second_cliff_rotation_revert_vs_reread", scenario_6),
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
