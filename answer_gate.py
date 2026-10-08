#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copied verbatim from answer-after-cliff/experiment.py (https://github.com/StarDust-zz/answer-after-cliff).
# Its own 7 parent assertions still run here: python3 answer_gate.py
"""Answer-after-cliff: join the horizon gate with the memory-access gate.

Two published keepers, one boundary:
  stop-triage-cliff     -> continue | cliff | stop | unsafe   (what crosses the horizon)
  memory-access-compile -> serve | hold                        (is what crossed answerable?)

Joined action in {serve, hold, escalate}. The question the join measures:
is a *faithful* cliff also an *answerable* one, and where must the
answerable write live so it survives the next cliff?

Stdlib only. No network. No GPU. No vendor API. No product import.
Run: python3 experiment.py
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

from answerable_stub import (
    Finding,
    Question,
    answer_from_evidence,
    answer_from_memory,
    answerable_finding,
    cell,
    generic_finding,
)
from cliff_stub import Msg, Session, apply_cliff, is_faithful_to_source, signature
from horizon_gate import SYSTEM, TASK, _long_live, horizon_triage_gate
from triage_stub import Item, Type, hard_cost, kitchen

SLOT = "LEASE_ID=LX-4417"
Q = Question(ask="Which lease id did the migration write?", gold="LX-4417", slot=SLOT)
LONG_PATH = "src/parser_2.py"   # slot lives in a long (>500 char) old tool result
SHORT_IDX = 5                   # turn 1 short grep result (kept by cliff)


def _ok(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def live_long_slot() -> List[Msg]:
    """Slot only inside an old, long tool_result: cliff drops it (faithfully)."""
    live = _long_live(6)
    i = 8  # turn 2 tool_result (write body, ~2400 chars), outside the recent 2K window
    live[i] = Msg("tool_result", live[i].text[:1200] + f" {SLOT} " + live[i].text[1200:])
    return live


def live_short_slot() -> List[Msg]:
    """Slot inside an old, short tool_result: cliff keeps it in the session."""
    live = _long_live(6)
    live[SHORT_IDX] = Msg("tool_result", live[SHORT_IDX].text + f" {SLOT}")
    return live


def live_next_window() -> List[Msg]:
    """Window 2: fresh turns with no slot (forces a second cliff)."""
    out = []
    for m in _long_live(6):
        path = (m.path + ".w2") if m.path else m.path
        text = m.text if m.kind == "tool_result" and len(m.text) > 500 else m.text + " W2"
        out.append(Msg(m.kind, text, m.tool_name, path))
    return out


def history_store(live: Sequence[Msg]) -> Dict[str, str]:
    """Retention home: provenance path -> full source body (JustMem-style store).

    Each tool_call path maps to the tool_call text plus the tool_result that follows it.
    """
    store: Dict[str, str] = {}
    for j, m in enumerate(live):
        if m.kind == "tool_call" and m.path:
            nxt = live[j + 1].text if j + 1 < len(live) else ""
            store[m.path] = m.text + "\n" + nxt
    return store


def finding_item(f: Finding, iid: str = "w1") -> Item:
    """Answerable finding committed into the typed set as a pinned constraint."""
    return Item(iid, f.text, Type.C, "wm", scope=("wm",), tokens=len(f.text.split()), sim=0.30)


def locate_slot_source(live: Sequence[Msg]) -> Optional[str]:
    """Which provenance path (or message) currently shows the slot."""
    for j, m in enumerate(live):
        if SLOT in m.text:
            for k in range(j, -1, -1):
                if live[k].kind == "tool_call" and live[k].path:
                    return live[k].path
            return f"msg{j}"
    return None


# ---------------------------------------------------------------------------
# Joined gate
# ---------------------------------------------------------------------------

def answer_after_cliff(
    *,
    live: List[Msg],
    prior_cliff: Optional[List[Msg]] = None,
    items: Optional[List[Item]] = None,
    budget: Optional[int] = None,
    stop_policy: str = "full_budget",
    token_pressure: bool = True,
    write: str = "first",          # first | after | generic | none
    replay: bool = True,
    uniform: bool = False,
    nest: bool = False,
    blind_serve: bool = False,     # serve on final-answer path without memory check
    store: Optional[Dict[str, str]] = None,
    stop_kwargs: Optional[dict] = None,
) -> dict:
    items = list(items) if items is not None else kitchen()
    base_bmin = hard_cost(items)
    if budget is None:
        budget = base_bmin + 12
    store = store if store is not None else history_store(live)

    # Final-answer path still sees the pre-boundary trajectory.
    ans_ok = answer_from_evidence([m.text for m in live], Q) == Q.gold

    wrote: Optional[Finding] = None
    if write == "first":
        src = locate_slot_source(live)
        if src is not None:
            wrote = answerable_finding(src, SLOT)
            items.append(finding_item(wrote))

    h = horizon_triage_gate(
        stop_policy=stop_policy,
        items=items,
        budget=budget,
        session=Session(SYSTEM, TASK, prior_cliff=prior_cliff, live=list(live)),
        token_pressure=token_pressure,
        uniform=uniform,
        nest_summarize=nest,
        **(stop_kwargs or {}),
    )
    out = {"horizon": h["action"], "ans_ok": ans_ok, "b_min": hard_cost(items),
           "budget": budget, "wrote": wrote is not None, "replayed": False}

    if h["action"] == "unsafe":
        out.update(action="escalate", cell=None, reason="horizon:unsafe", carried_tokens=None)
        return out

    # What crosses the boundary (the reader sees ONLY this, not the old trajectory).
    if h["action"] == "stop":
        carried_msgs: List[Msg] = []         # trajectory ends; only WM survives
        kept = list(items)
    elif h["action"] == "continue":
        carried_msgs = list(live)
        kept = [i for i in items if i.iid in h.get("kept_ids", set())]
    else:  # cliff (faithful) or nest (blind)
        carried_msgs = list(h["cliff_msgs"] or [])
        kept = [i for i in items if i.iid in h.get("kept_ids", set())]

    memory: List[Finding] = [Finding(m.path or m.kind, m.text) for m in carried_msgs]
    memory += [Finding(i.topic, i.text) for i in kept]

    if write == "after":
        # Write from what survived the boundary, not from the old trajectory.
        if any(SLOT in f.text for f in memory):
            memory.append(answerable_finding("carried", SLOT))
        else:
            memory.append(generic_finding("carried"))
    elif write == "generic":
        memory.append(generic_finding("wm"))

    mem_ok = answer_from_memory(memory, Q) == Q.gold
    replay_cost = 0
    if not mem_ok and replay:
        # REPLAY: follow a faithful signature's provenance path into the retention home.
        for m in carried_msgs:
            if m.kind == "tool_call" and m.path in store and not m.rewritten:
                body = store[m.path]
                if SLOT in body:
                    mem_ok = True
                    out["replayed"] = True
                    replay_cost = len(body)
                    break

    c = cell(ans_ok, mem_ok)
    carried_tokens = sum(len(f.text) for f in memory) + replay_cost
    if blind_serve:
        action = "serve" if ans_ok else "hold"
    else:
        action = "serve" if c == "MSC" else "hold"
    lost_c = {i.iid for i in items if i.typ == Type.C} - {i.iid for i in kept}
    out.update(action=action, cell=c, carried_tokens=carried_tokens,
               lost_constraints=sorted(lost_c), faithful=h.get("faithful"),
               cliff_msgs=h.get("cliff_msgs"), memory=memory)
    return out


# ---------------------------------------------------------------------------
# Join assertions
# ---------------------------------------------------------------------------

STOP = dict(seed=7, b=0.18, stop_budget=20, delta=0.05, alpha=0.12, peak_at=4)


def scenario_1():
    """Blind join (full-budget + uniform + nest + no write + serve-on-answer) != gate."""
    live = live_long_slot()
    blind = answer_after_cliff(live=live, uniform=True, nest=True, write="none",
                               blind_serve=True)
    gate = answer_after_cliff(live=live, write="first")
    blind_w = answer_after_cliff(live=live, uniform=True, nest=True, write="first")
    _ok(blind["action"] == "serve" and blind["cell"] == "MMC",
        f"blind serves an unlicensed answer; got {blind['action']}/{blind['cell']}")
    _ok(len(blind["lost_constraints"]) >= 1, "blind uniform drops a constraint")
    _ok(gate["action"] == "serve" and gate["cell"] == "MSC", f"gate MSC; got {gate['cell']}")
    _ok(gate["lost_constraints"] == [], "gate keeps every constraint incl. the finding")
    _ok("w1" in blind_w["lost_constraints"] and blind_w["action"] == "hold",
        "uniform drops even an honest write; nest leaves no replay path")
    return {"blind": (blind["action"], blind["cell"], blind["lost_constraints"]),
            "gate": (gate["action"], gate["cell"]),
            "blind_with_write": (blind_w["action"], blind_w["lost_constraints"])}


def scenario_2():
    """Faithful != answerable: cliff is a subset of source yet the slot is gone."""
    live = live_long_slot()
    r = answer_after_cliff(live=live, write="none", replay=False)
    _ok(r["horizon"] == "cliff", r["horizon"])
    _ok(r["faithful"] is True and is_faithful_to_source(r["cliff_msgs"], live), "cliff faithful")
    _ok(r["ans_ok"] is True, "final-answer path could still answer before the boundary")
    _ok(r["cell"] == "MMC" and r["action"] == "hold", f"faithful cliff is MMC; got {r['cell']}")
    return {"horizon": r["horizon"], "faithful": r["faithful"], "cell": r["cell"],
            "action": r["action"]}


def scenario_3():
    """Order and home: write-before-cliff serves and survives cliff 2; session-carried slot dies."""
    live = live_long_slot()
    first = answer_after_cliff(live=live, write="first", replay=False)
    after = answer_after_cliff(live=live, write="after", replay=False)
    _ok(first["cell"] == "MSC", f"write-first MSC; got {first['cell']}")
    _ok(after["cell"] == "MMC", f"write-after-cliff can only be generic; got {after['cell']}")

    # Two-cliff chain. (a) slot in short result: carried by cliff 1 in the SESSION only.
    short = live_short_slot()
    c1 = apply_cliff(Session(SYSTEM, TASK, prior_cliff=None, live=short)).prior_cliff
    _ok(any(SLOT in m.text for m in c1), "cliff 1 carries the short-result slot in session")
    w2 = live_next_window()
    sess_only = answer_after_cliff(live=w2, prior_cliff=c1, write="none", replay=False,
                                   store=history_store(w2))
    # (b) same slot, but committed into the typed pinned set before cliff 1.
    pinned_items = kitchen() + [finding_item(answerable_finding(locate_slot_source(short), SLOT))]
    pinned = answer_after_cliff(live=w2, prior_cliff=c1, items=pinned_items, write="none",
                                replay=False, store=history_store(w2))
    _ok(sess_only["horizon"] == "cliff" and pinned["horizon"] == "cliff", "both cliff twice")
    _ok(not any(SLOT in f.text for f in sess_only["memory"]), "cliff 2 discards C_{t-1} slot")
    # ans_ok is False on window 2 (slot not in L_t), so session-only is UE, not MMC.
    _ok(sess_only["action"] == "hold", f"session-carried finding lost; got {sess_only['cell']}")
    _ok(answer_from_memory(pinned["memory"], Q) == Q.gold, "pinned finding survives cliff 2")
    return {"write_first": first["cell"], "write_after": after["cell"],
            "cliff2_session_only": sess_only["cell"],
            "cliff2_pinned_mem_ok": True}


def scenario_4():
    """Signature is a provenance pointer: REPLAY recovers after cliff, not after nest."""
    live = live_long_slot()
    rep = answer_after_cliff(live=live, write="none", replay=True)
    nst = answer_after_cliff(live=live, write="none", replay=True, nest=True)
    wf = answer_after_cliff(live=live, write="first", replay=False)
    sig_paths = [m.path for m in rep["cliff_msgs"] if m.kind == "tool_call"]
    _ok(LONG_PATH in sig_paths, "cliff signature keeps the provenance path")
    _ok(rep["replayed"] and rep["action"] == "serve", "REPLAY via signature serves")
    _ok(not nst["replayed"] and nst["action"] == "hold", "nest loses the path; REPLAY fails")
    _ok(wf["carried_tokens"] < rep["carried_tokens"], "write-first is cheaper than replay")
    return {"replay": (rep["action"], rep["carried_tokens"]),
            "nest": nst["action"], "write_first_tokens": wf["carried_tokens"]}


def scenario_5():
    """Principled stop: trajectory ends, only WM crosses. MSC serves, generic holds."""
    live = live_long_slot()
    s_w = answer_after_cliff(live=live, stop_policy="principled", write="first",
                             stop_kwargs=STOP)
    s_g = answer_after_cliff(live=live, stop_policy="principled", write="generic",
                             stop_kwargs=STOP)
    _ok(s_w["horizon"] == "stop" and s_g["horizon"] == "stop", "principled stop wins")
    _ok(s_w["action"] == "serve" and s_w["cell"] == "MSC", f"stop+write MSC; {s_w['cell']}")
    _ok(s_g["action"] == "hold" and s_g["cell"] == "MMC", f"stop+generic MMC; {s_g['cell']}")
    return {"stop_write": s_w["cell"], "stop_generic": s_g["cell"]}


def scenario_6():
    """The write costs B_min: at a tight budget an honest write flips cliff -> escalate."""
    live = live_long_slot()
    tight = hard_cost(kitchen())  # exactly the old B_min
    no_w = answer_after_cliff(live=live, budget=tight, write="none", replay=False)
    yes_w = answer_after_cliff(live=live, budget=tight, write="first", replay=False)
    _ok(no_w["horizon"] == "cliff" and no_w["action"] == "hold", f"no write: {no_w['action']}")
    _ok(yes_w["b_min"] > tight, "the finding raises B_min")
    _ok(yes_w["horizon"] == "unsafe" and yes_w["action"] == "escalate",
        f"write at tight budget escalates; got {yes_w['action']}")
    # Stop precedence still beats unsafe: halting needs no compaction.
    st = answer_after_cliff(live=live, budget=tight, stop_policy="principled", write="first",
                            stop_kwargs=STOP)
    _ok(st["horizon"] == "stop" and st["action"] == "serve", f"stop beats unsafe; {st['action']}")
    return {"tight": tight, "no_write": no_w["action"], "write": yes_w["action"],
            "b_min_with_write": yes_w["b_min"], "stop_at_tight": st["action"]}


def scenario_7():
    """Coverage: every horizon action lands on the right joined action; pack is clean."""
    live = live_long_slot()
    table = {
        "continue+none": answer_after_cliff(live=live, token_pressure=False, write="none"),
        "stop+first": answer_after_cliff(live=live, stop_policy="principled", write="first",
                                         stop_kwargs=STOP),
        "cliff+first": answer_after_cliff(live=live, write="first", replay=False),
        "cliff+replay": answer_after_cliff(live=live, write="none", replay=True),
        "cliff+none": answer_after_cliff(live=live, write="none", replay=False),
        "nest+replay": answer_after_cliff(live=live, write="none", nest=True),
        "unsafe+first": answer_after_cliff(live=live, budget=hard_cost(kitchen()) - 1,
                                           write="first"),
    }
    want = {"continue+none": "serve", "stop+first": "serve", "cliff+first": "serve",
            "cliff+replay": "serve", "cliff+none": "hold", "nest+replay": "hold",
            "unsafe+first": "escalate"}
    got = {k: v["action"] for k, v in table.items()}
    _ok(got == want, f"coverage mismatch: {got}")
    _ok(table["cliff+first"]["carried_tokens"] < table["continue+none"]["carried_tokens"],
        "write-first cliff carries far less than continue")
    needle = "cosmi" + "cat"
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in os.listdir(here):
        if fn.endswith((".py", ".md", ".txt")):
            with open(os.path.join(here, fn), encoding="utf-8") as fh:
                _ok(needle not in fh.read().lower(), f"product name found in {fn}")
    return {"actions": got,
            "tokens": {k: table[k]["carried_tokens"] for k in ("continue+none", "cliff+first")}}


SCENARIOS = [
    ("1_blind_join_vs_gate", scenario_1),
    ("2_faithful_is_not_answerable", scenario_2),
    ("3_write_before_cliff_into_pinned_home", scenario_3),
    ("4_signature_is_provenance_for_replay", scenario_4),
    ("5_stop_carries_only_wm", scenario_5),
    ("6_write_costs_b_min_escalate", scenario_6),
    ("7_coverage_and_clean", scenario_7),
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
