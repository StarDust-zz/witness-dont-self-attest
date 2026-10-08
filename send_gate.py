#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Send-after-cliff: can an outbound send be certified when its facts crossed a cliff?

Two published keepers, one decision:
  trust-compile-gate -> send | hold | escalate   (outbound draft is source; MDL + RouteGuard + lease)
  answer-after-cliff -> serve | hold | escalate  (faithful cliff != answerable; pin before, or REPLAY)

Joined object: a claim-level provenance check inserted in front of the trust gate.
Every claim the draft relies on is classed by HOW it survived the boundary:

  PINNED   committed into the pinned typed set BEFORE the cliff (from source)  -> certifiable
  SESSION  visible only as carried session text (cliff short result / recent
           window) or pinned AFTER the cliff from that text                    -> re-check first
  REPLAY   not in memory; reachable via an unrewritten cliff signature path    -> re-check first
  GENERIC  only a generic after-cliff summary / nothing                        -> not certifiable

Re-check = follow carried signature paths into the retention store, take the
latest value, compare to the draft, and lease (path, digest, cliff epoch).
Baselines: parent-only (trust gate alone, never looks at claims) and
text-match (a claim counts as supported iff "K=v" appears in post-cliff text).

Stdlib only. No network. No GPU. No vendor API. No product import.
Run: python3 experiment.py
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from answer_gate import history_store
from answerable_stub import Finding, answerable_finding, generic_finding
from cliff_stub import Msg, Session, apply_cliff, is_faithful_to_source
from horizon_gate import SYSTEM, TASK, _long_live, horizon_triage_gate
from route_stub import AUC_POOL
from triage_stub import Item, Type, hard_cost, kitchen
from trust_gate import GAIN_POOL, RISK_HIGH, SAFE
from trust_gate import gate as trust_gate

# ---------------------------------------------------------------------------
# Facts the outbound draft relies on, planted in a tool-heavy trajectory
# ---------------------------------------------------------------------------
LEASE, REFUND, TICKET = "LEASE_ID", "REFUND_EUR", "TICKET"
LEASE_NOTICE = (LEASE, REFUND, TICKET)
REFUND_NOTICE = (REFUND, TICKET)
GUESS = {LEASE: "LX-4400", REFUND: "100", TICKET: "T-00"}   # post-cliff drafter's fill-in
LONG_PATH = "src/parser_2.py"


def _ok(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def values_in(text: str, key: str) -> List[str]:
    return re.findall(rf"\b{key}=([A-Za-z0-9\-]+)", text)


def live_window1() -> List[Msg]:
    """Turn1 short result (kept by cliff): stale REFUND=90.
    Turn2 long result (dropped by cliff): LEASE=LX-4417 and the REFUND update to 120.
    Turn5 short result (recent window, verbatim): TICKET=T-88."""
    live = _long_live(6)
    live[5] = Msg("tool_result", live[5].text + f" {REFUND}=90")
    t = live[8].text
    live[8] = Msg("tool_result", t[:1200] + f" {LEASE}=LX-4417 {REFUND}=120 " + t[1200:])
    live[17] = Msg("tool_result", live[17].text + f" {TICKET}=T-88")
    return live


def live_window2() -> List[Msg]:
    """Fresh turns with no facts (forces a second cliff)."""
    out = []
    for m in _long_live(6):
        path = (m.path + ".w2") if m.path else m.path
        text = m.text if m.kind == "tool_result" and len(m.text) > 500 else m.text + " W2"
        out.append(Msg(m.kind, text, m.tool_name, path))
    return out


def rotate(store: Dict[str, str], path: str, key: str, new: str) -> Dict[str, str]:
    """Source moves after drafting: the fact at `path` is updated in the retention home."""
    s = dict(store)
    s[path] = re.sub(rf"\b{key}=[A-Za-z0-9\-]+", f"{key}={new}", s[path])
    return s


def latest_in_texts(texts: Sequence[str], key: str) -> Optional[str]:
    last = None
    for t in texts:
        v = values_in(t, key)
        if v:
            last = v[-1]
    return last


def truth(store: Dict[str, str], key: str) -> Optional[str]:
    """Ground truth at send time: latest value in the full retention store (source order)."""
    return latest_in_texts(list(store.values()), key)


# ---------------------------------------------------------------------------
# Crossing the boundary (reuses the horizon gate + answer-after-cliff vocabulary)
# ---------------------------------------------------------------------------

@dataclass
class Pin:
    iid: str
    key: str
    value: str
    path: str
    digest: Optional[str]   # digest of the source body at pin time (None if pinned from carried text)
    pre_cliff: bool         # written from source before the cliff vs from carried text after


def pin_item(p: Pin) -> Item:
    f = answerable_finding(p.path, f"{p.key}={p.value}")
    return Item(p.iid, f.text, Type.C, "pin:pre" if p.pre_cliff else "pin:post",
                scope=("wm",), tokens=len(f.text.split()), sim=0.30)


def source_of(live: Sequence[Msg], key: str) -> Optional[Tuple[str, str]]:
    """(provenance path, latest value) for key in the live trajectory."""
    found = None
    for j, m in enumerate(live):
        v = values_in(m.text, key)
        if v:
            path = next((live[k].path for k in range(j, -1, -1)
                         if live[k].kind == "tool_call" and live[k].path), f"msg{j}")
            found = (path, v[-1])
    return found


@dataclass
class Crossed:
    horizon: str
    carried: List[Msg]
    kept: List[Item]
    pins: Dict[str, Pin]
    memory: List[Finding]
    faithful: Optional[bool]
    b_min: int
    epoch: int


def cross(live: List[Msg], *, keys: Sequence[str], prior_cliff: Optional[List[Msg]] = None,
          items: Optional[List[Item]] = None, pins: Optional[Dict[str, Pin]] = None,
          write: str = "none", nest: bool = False, budget: Optional[int] = None,
          token_pressure: bool = True, epoch: int = 1) -> Crossed:
    """Run one horizon boundary. write in {first, after, none}."""
    items = list(items) if items is not None else kitchen()
    pins = dict(pins or {})
    if budget is None:
        budget = hard_cost(kitchen()) + 12
    store = history_store(live)
    if write == "first":
        for k in keys:
            src = source_of(live, k)
            if src is not None:
                path, val = src
                p = Pin(f"w_{k}", k, val, path, _digest(store.get(path, "")), True)
                pins[p.iid] = p
                items.append(pin_item(p))
    h = horizon_triage_gate(stop_policy="full_budget", items=items, budget=budget,
                            session=Session(SYSTEM, TASK, prior_cliff=prior_cliff, live=list(live)),
                            token_pressure=token_pressure, nest_summarize=nest)
    if h["action"] == "unsafe":
        return Crossed("unsafe", [], [], pins, [], None, hard_cost(items), epoch)
    carried = list(live) if h["action"] == "continue" else list(h["cliff_msgs"] or [])
    kept = [i for i in items if i.iid in h.get("kept_ids", set())]
    memory = [Finding(m.path or m.kind, m.text) for m in carried]
    memory += [Finding(i.topic, i.text) for i in kept]
    if write == "after":
        # Write from what survived: verbatim if the key is visible, else generic.
        text = " ".join(f.text for f in memory)
        for k in keys:
            v = latest_in_texts([text], k)
            if v is not None:
                p = Pin(f"a_{k}", k, v, "carried", None, False)
                pins[p.iid] = p
                kept.append(pin_item(p))
                memory.append(Finding("pin:post", pin_item(p).text))
            else:
                memory.append(generic_finding("carried"))
    pins = {iid: p for iid, p in pins.items() if iid in {i.iid for i in kept}}
    return Crossed(h["action"], carried, kept, pins, memory, h.get("faithful"),
                   hard_cost(items), epoch)


# ---------------------------------------------------------------------------
# Drafts (the outbound draft is source; where do its values come from?)
# ---------------------------------------------------------------------------

def draft_from_live(live: Sequence[Msg], keys: Sequence[str]) -> Dict[str, str]:
    """Drafted before the cliff with evidence live: correct at drafting time."""
    return {k: (source_of(live, k) or ("", GUESS[k]))[1] for k in keys}


def draft_from_memory(c: Crossed, keys: Sequence[str]) -> Dict[str, str]:
    """Drafted after the cliff from carried memory only; fills gaps with a guess."""
    texts = [f.text for f in c.memory]
    return {k: latest_in_texts(texts, k) or GUESS[k] for k in keys}


# ---------------------------------------------------------------------------
# Claim-level provenance check + lease
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClaimLease:
    key: str
    value: str
    path: str
    digest: str
    epoch: int


def replay_latest(c: Crossed, store: Dict[str, str], key: str) -> Optional[Tuple[str, str, int]]:
    """Follow carried, unrewritten signature paths into the store; latest value wins."""
    found = None
    for m in c.carried:
        if m.kind == "tool_call" and m.path in store and not m.rewritten:
            v = values_in(store[m.path], key)
            if v:
                found = (v[-1], m.path, len(store[m.path]))
    return found


def classify(c: Crossed, key: str) -> str:
    if any(p.key == key and p.pre_cliff for p in c.pins.values()):
        return "PINNED"
    visible = " ".join(f.text for f in c.memory)
    if values_in(visible, key):
        return "SESSION"   # includes after-cliff pins: provenance = carried text, not the home
    return "REPLAY?"


def certify_claims(draft: Dict[str, str], c: Crossed, store: Dict[str, str], *,
                   replay: bool = True, pinned_recheck: bool = False) -> dict:
    claims, leases, recheck_chars = {}, {}, 0
    for k, v in draft.items():
        cls = classify(c, k)
        r = {"cls": cls, "draft": v}
        if cls == "PINNED":
            p = next(p for p in c.pins.values() if p.key == k and p.pre_cliff)
            ok = p.value == v
            if ok and pinned_recheck:
                ok = p.path in store and _digest(store[p.path]) == p.digest
                r["pin_fresh"] = ok
            r.update(certified=ok, seen=p.value)
        else:
            hit = replay_latest(c, store, k) if (replay or cls == "SESSION") else None
            if cls == "REPLAY?":
                cls = "REPLAY" if hit else ("GENERIC" if any(
                    "trends" in f.text for f in c.memory) else "NONE")
                r["cls"] = cls
            if hit is None:
                r.update(certified=False, seen=None)
            else:
                val, path, n = hit
                recheck_chars += n
                ok = val == v
                r.update(certified=ok, seen=val)
                if ok:
                    leases[k] = ClaimLease(k, v, path, _digest(store[path]), c.epoch)
        claims[k] = r
    return {"claims": claims, "leases": leases, "recheck_chars": recheck_chars,
            "all_certified": all(r["certified"] for r in claims.values())}


def lease_valid(lease: ClaimLease, c_send: Crossed, store_send: Dict[str, str], recheck: str) -> bool:
    """Re-check a claim lease at send time.

    none    : trust the lease as issued (no re-check)
    context : the signature path must still be carried AND the store digest must match
    path    : the lease keeps its own path; the store digest must match
    """
    if recheck == "none":
        return True
    if lease.path not in store_send or _digest(store_send[lease.path]) != lease.digest:
        return False
    if recheck == "context":
        return any(m.kind == "tool_call" and m.path == lease.path and not m.rewritten
                   for m in c_send.carried)
    return True


# ---------------------------------------------------------------------------
# Joined gate and baselines
# ---------------------------------------------------------------------------

def send_after_cliff(draft: Dict[str, str], c: Crossed, store: Dict[str, str], *,
                     c_send: Optional[Crossed] = None, store_send: Optional[Dict[str, str]] = None,
                     recheck: str = "context", replay: bool = True, pinned_recheck: bool = False,
                     scalars: Optional[dict] = None, escalate_proposed: bool = False,
                     route_pool=None) -> dict:
    c_send = c_send or c
    store_send = store_send if store_send is not None else store
    if c.horizon == "unsafe" or c_send.horizon == "unsafe":
        return {"action": "escalate", "reason": "horizon:unsafe", "claims": {}}
    cert = certify_claims(draft, c, store, replay=replay, pinned_recheck=pinned_recheck)
    out = dict(cert)
    if c_send is not c:
        # A later boundary: pinned claims are re-read from what survived it.
        for k, r in cert["claims"].items():
            if r["cls"] == "PINNED":
                still = any(p.key == k and p.pre_cliff for p in c_send.pins.values())
                r["certified"] = r["certified"] and still
    # Every claim lease is re-checked at send time (unless recheck == "none").
    bad = [k for k, ls in cert["leases"].items()
           if not lease_valid(ls, c_send, store_send, recheck)]
    for k in bad:
        cert["claims"][k]["certified"] = False
        cert["claims"][k]["lease"] = "refused"
    out["all_certified"] = all(r["certified"] for r in cert["claims"].values())
    if not out["all_certified"]:
        out.update(action="hold", reason="claims:" + ",".join(
            k for k, r in cert["claims"].items() if not r["certified"]))
        return out
    t = trust_gate(**(scalars or SAFE), escalate_proposed=escalate_proposed, route_pool=route_pool)
    out.update(action=t["action"], reason=f"trust:{t['reason']}")
    return out


def parent_only(draft: Dict[str, str], c: Crossed, **_) -> dict:
    """trust-compile-gate alone: never looks at the claims."""
    return {"action": trust_gate(**SAFE)["action"]}


def text_match(draft: Dict[str, str], c: Crossed, **_) -> dict:
    """Certify on the post-cliff text alone: K=v must appear somewhere in it."""
    visible = " ".join(f.text for f in c.memory)
    ok = all(v in values_in(visible, k) for k, v in draft.items())
    return {"action": trust_gate(**SAFE)["action"] if ok else "hold"}


def wrong_sent(draft: Dict[str, str], action: str, store_send: Dict[str, str]) -> int:
    if action != "send":
        return 0
    return sum(1 for k, v in draft.items() if truth(store_send, k) != v)


# ---------------------------------------------------------------------------
# Seven join assertions
# ---------------------------------------------------------------------------

def _w1(**kw) -> Tuple[List[Msg], Dict[str, str], Crossed]:
    live = live_window1()
    return live, history_store(live), cross(live, keys=kw.pop("keys", LEASE_NOTICE), **kw)


def scenario_1():
    """Blind != gate: after a faithful cliff the post-cliff draft carries a guess and a stale value."""
    live, store, c = _w1()
    draft = draft_from_memory(c, LEASE_NOTICE)
    blind = parent_only(draft, c)
    gate = send_after_cliff(draft, c, store)
    _ok(c.horizon == "cliff" and c.faithful is True and is_faithful_to_source(c.carried, live),
        "the cliff is faithful")
    _ok(draft == {LEASE: "LX-4400", REFUND: "90", TICKET: "T-88"}, f"post-cliff draft {draft}")
    _ok(blind["action"] == "send" and wrong_sent(draft, "send", store) == 2,
        "parent gate alone sends 2 wrong claims")
    cls = {k: (r["cls"], r["certified"]) for k, r in gate["claims"].items()}
    _ok(gate["action"] == "hold", f"provenance gate holds; got {gate['action']}")
    _ok(cls == {LEASE: ("REPLAY", False), REFUND: ("SESSION", False), TICKET: ("SESSION", True)},
        f"per-claim classes {cls}")
    return {"draft": draft, "blind": blind["action"], "blind_wrong": 2,
            "gate": gate["action"], "claims": cls}


def scenario_2():
    """NEW count: wrong claims sent over a case matrix — blind vs text-match vs provenance gate."""
    live = live_window1()
    store = history_store(live)
    cases = []
    for write in ("none", "after", "first"):
        for nest in (False, True):
            for keys in (LEASE_NOTICE, REFUND_NOTICE):
                c = cross(live, keys=keys, write=write, nest=nest)
                for origin in ("pre", "post"):
                    d = draft_from_live(live, keys) if origin == "pre" else draft_from_memory(c, keys)
                    cases.append((f"{write}/{'nest' if nest else 'cliff'}/{len(keys)}/{origin}", d, c))
    tally = {}
    for name, fn in (("parent_only", parent_only), ("text_match", text_match),
                     ("provenance", lambda d, c: send_after_cliff(d, c, store))):
        acts = [(fn(d, c)["action"], d) for _, d, c in cases]
        correct = [all(truth(store, k) == v for k, v in d.items()) for _, d in acts]
        tally[name] = {"sent": sum(a == "send" for a, _ in acts),
                       "wrong": sum(wrong_sent(d, a, store) for a, d in acts),
                       "held_correct": sum(a != "send" and ok for (a, _), ok in zip(acts, correct))}
    _ok(len(cases) == 24, len(cases))
    _ok(tally["parent_only"]["wrong"] > tally["text_match"]["wrong"] > 0,
        f"blind > text-match > 0 wrong: {tally}")
    _ok(tally["provenance"]["wrong"] == 0, f"provenance gate sends 0 wrong: {tally}")
    _ok(tally["provenance"]["sent"] > tally["text_match"]["sent"]
        and tally["provenance"]["held_correct"] < tally["text_match"]["held_correct"],
        f"and it is not just conservative — it sends more correct drafts: {tally}")
    return tally


def scenario_3():
    """Faithful-but-stale: the cliff keeps a superseded short result and drops the update."""
    live, store, c = _w1(keys=REFUND_NOTICE)
    post = draft_from_memory(c, REFUND_NOTICE)
    pre = draft_from_live(live, REFUND_NOTICE)
    tm_post, gate_post = text_match(post, c), send_after_cliff(post, c, store)
    tm_pre, gate_pre = text_match(pre, c), send_after_cliff(pre, c, store)
    _ok(post[REFUND] == "90" and pre[REFUND] == "120" and truth(store, REFUND) == "120", (post, pre))
    _ok(tm_post["action"] == "send" and wrong_sent(post, "send", store) == 1,
        "text-match certifies the stale faithful value")
    _ok(gate_post["action"] == "hold" and gate_post["claims"][REFUND]["seen"] == "120",
        f"re-check via the signature path finds the update; {gate_post['reason']}")
    _ok(tm_pre["action"] == "hold" and gate_pre["action"] == "send",
        "the correct pre-cliff draft: text-match holds, re-check certifies and sends")
    return {"post_draft": post, "text_match_post": tm_post["action"],
            "gate_post": gate_post["action"], "pre_draft": pre,
            "text_match_pre": tm_pre["action"], "gate_pre": gate_pre["action"]}


def scenario_4():
    """Pinned before the cliff certifies; after-cliff pins are generic or launder stale text."""
    live, store, first = _w1(write="first")
    _, _, after = _w1(write="after")
    d_first = draft_from_memory(first, LEASE_NOTICE)
    d_after = draft_from_memory(after, LEASE_NOTICE)
    g_first = send_after_cliff(d_first, first, store, replay=False)
    g_after = send_after_cliff(d_after, after, store, replay=False)
    _ok(all(r["cls"] == "PINNED" for r in g_first["claims"].values()), g_first["claims"])
    _ok(g_first["action"] == "send" and wrong_sent(d_first, "send", store) == 0, g_first["reason"])
    _ok(g_first["recheck_chars"] == 0, "pinned claims need no store read")
    # After-cliff write: LEASE is generic, REFUND is pinned from the stale carried text.
    _ok(any(p.key == REFUND and not p.pre_cliff and p.value == "90" for p in after.pins.values()),
        "after-cliff pin copies the stale carried value into the pinned home")
    _ok(g_after["claims"][LEASE]["cls"] == "GENERIC" and g_after["claims"][REFUND]["cls"] == "SESSION",
        g_after["claims"])
    _ok(g_after["action"] == "hold", "location != provenance: the laundered pin is not certified")
    # A gate that keys on the pinned HOME (not write-epoch) would have certified REFUND=90.
    home_keyed = any(p.key == REFUND and p.value == d_after[REFUND] for p in after.pins.values())
    _ok(home_keyed and truth(store, REFUND) != d_after[REFUND], "home-keyed gate would send stale")
    return {"first": g_first["action"], "first_recheck_chars": g_first["recheck_chars"],
            "after": g_after["action"],
            "after_classes": {k: r["cls"] for k, r in g_after["claims"].items()}}


def scenario_5():
    """REPLAY is certifiable only after re-check: rotation between certify and send; nest."""
    live, store, c = _w1(keys=(LEASE,))
    d = draft_from_live(live, (LEASE,))
    ok = send_after_cliff(d, c, store)
    rot = rotate(store, LONG_PATH, LEASE, "LX-4421")
    chk = send_after_cliff(d, c, store, store_send=rot, recheck="context")
    blind_lease = send_after_cliff(d, c, store, store_send=rot, recheck="none")
    _, _, cn = _w1(keys=(LEASE,), nest=True)
    nst = send_after_cliff(d, cn, store)
    _ok(ok["claims"][LEASE]["cls"] == "REPLAY" and ok["action"] == "send", ok)
    _ok(ok["recheck_chars"] > 2000, "re-check pays the long body")
    _ok(chk["action"] == "hold" and chk["claims"][LEASE].get("lease") == "refused", chk)
    _ok(blind_lease["action"] == "send" and wrong_sent(d, "send", rot) == 1,
        "trusting the lease without re-check sends the rotated-away value")
    _ok(nst["action"] == "hold" and nst["claims"][LEASE]["cls"] == "NONE", "nest kills the path")
    return {"replay": ok["action"], "recheck_chars": ok["recheck_chars"],
            "rotated_recheck": chk["action"], "rotated_no_recheck": blind_lease["action"],
            "nest": nst["action"]}


def scenario_6():
    """NEW second cliff: a leased claim must be re-checked at the next epoch; pinned survives."""
    w1 = live_window1()
    s1 = history_store(w1)
    d = draft_from_live(w1, (LEASE,))
    c1 = cross(w1, keys=(LEASE,), epoch=1)                       # REPLAY-leased at epoch 1
    p1 = cross(w1, keys=(LEASE,), write="first", epoch=1)        # pinned before cliff 1
    prior = apply_cliff(Session(SYSTEM, TASK, prior_cliff=None, live=w1)).prior_cliff
    w2 = live_window2()
    s12 = {**s1, **history_store(w2)}                            # retention home keeps both windows
    c2 = cross(w2, keys=(LEASE,), prior_cliff=prior, epoch=2)
    pinned_items = kitchen() + [i for i in p1.kept if i.iid.startswith("w_")]
    p2 = cross(w2, keys=(LEASE,), prior_cliff=prior, items=pinned_items, pins=p1.pins, epoch=2)
    ctx = send_after_cliff(d, c1, s1, c_send=c2, store_send=s12, recheck="context")
    pth = send_after_cliff(d, c1, s1, c_send=c2, store_send=s12, recheck="path")
    s12r = rotate(s12, LONG_PATH, LEASE, "LX-4421")
    pth_rot = send_after_cliff(d, c1, s1, c_send=c2, store_send=s12r, recheck="path")
    none_rot = send_after_cliff(d, c1, s1, c_send=c2, store_send=s12r, recheck="none")
    pin = send_after_cliff(d, p1, s1, c_send=p2, store_send=s12)
    pin_rot = send_after_cliff(d, p1, s1, c_send=p2, store_send=s12r)
    pin_rot_chk = send_after_cliff(d, p1, s12r, c_send=p2, store_send=s12r, pinned_recheck=True)
    _ok(c2.horizon == "cliff" and not any(m.path == LONG_PATH for m in c2.carried),
        "cliff 2 discards C_{t-1}: the signature path is gone from context")
    _ok(ctx["action"] == "hold", "context re-check fails at epoch 2")
    _ok(pth["action"] == "send", "a lease that keeps its own path re-checks via the store")
    _ok(pth_rot["action"] == "hold" and none_rot["action"] == "send"
        and wrong_sent(d, "send", s12r) == 1, "only re-check catches rotation; trusting sends wrong")
    _ok(any(p.key == LEASE for p in p2.pins.values()) and pin["action"] == "send",
        "pinned claim survives cliff 2 and certifies")
    _ok(pin_rot["action"] == "send" and wrong_sent(d, "send", s12r) == 1
        and pin_rot_chk["action"] == "hold",
        "pinned is certifiable only against its pin-time binding (boundary)")
    return {"ctx_recheck": ctx["action"], "path_recheck": pth["action"],
            "path_recheck_rotated": pth_rot["action"], "no_recheck_rotated": none_rot["action"],
            "pinned": pin["action"], "pinned_rotated_no_check": pin_rot["action"],
            "pinned_rotated_digest_check": pin_rot_chk["action"]}


def scenario_7():
    """Trust stages and horizon still bind after certification; pack is clean."""
    live, store, c = _w1(write="first")
    d = draft_from_memory(c, LEASE_NOTICE)
    risk = send_after_cliff(d, c, store, scalars=RISK_HIGH)
    esc = send_after_cliff(d, c, store, escalate_proposed=True, route_pool=GAIN_POOL)
    unc = send_after_cliff(d, c, store, escalate_proposed=True, route_pool=AUC_POOL)
    tight = hard_cost(kitchen())
    _, _, ct = _w1(write="first", budget=tight)
    _, _, cn = _w1(write="none", budget=tight)
    unsafe = send_after_cliff(d, ct, store)
    nowrite = send_after_cliff(draft_from_live(live, LEASE_NOTICE), cn, store, replay=False)
    _ok(risk["all_certified"] and risk["action"] == "hold", "certified claims, high task-risk: MDL holds")
    _ok(esc["action"] == "escalate" and unc["action"] == "hold", (esc["reason"], unc["reason"]))
    _ok(ct.horizon == "unsafe" and ct.b_min > tight and unsafe["action"] == "escalate",
        "pinning the claims raises B_min: tight budget escalates instead of dropping")
    _ok(cn.horizon == "cliff" and nowrite["action"] == "hold", "no pin, no replay at tight budget: hold")
    needles = ("cosmi" + "cat", "svor" + "onov")
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in os.listdir(here):
        if fn.endswith((".py", ".md", ".txt")):
            with open(os.path.join(here, fn), encoding="utf-8") as fh:
                body = fh.read().lower()
                _ok(not any(n in body for n in needles), f"product name found in {fn}")
    return {"risk_high": risk["action"], "certified_escalate": esc["action"],
            "uncertified_escalate": unc["action"], "tight_pinned": unsafe["action"],
            "b_min": (tight, ct.b_min), "tight_no_pin": nowrite["action"]}


SCENARIOS = [
    ("1_blind_vs_provenance_gate", scenario_1),
    ("2_wrong_claims_sent_count", scenario_2),
    ("3_faithful_but_stale_needs_recheck", scenario_3),
    ("4_pinned_before_vs_laundered_after", scenario_4),
    ("5_replay_certifiable_only_after_recheck", scenario_5),
    ("6_second_cliff_lease_recheck", scenario_6),
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
