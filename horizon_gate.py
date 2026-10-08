#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Horizon gate copied verbatim from stop-triage-cliff (its own 7 scenarios still run via main()).

Stages over a long-horizon evidence/session / typed working set (source):
  A. stop-dont-drift     — principled stop (KL + quality-peak); full-budget inject
  B. triage-dont-uniform — TypeCompact / TypeDecompose; below B_min → Unsafe
  C. cliff-dont-drift    — under token pressure, cliff truncate+signature;
                           discard prior cliff; never nest summarization

Final action ∈ {continue, cliff, stop, unsafe}.
Blind inject = full_budget stop policy AND uniform compact AND nest_summarize.

Stdlib only. No network. No GPU. No vendor API.
Run: python3 experiment.py
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from stop_stub import (
    DEFAULT_BUDGET,
    simulate_null_query,
)
from triage_stub import (
    Item,
    Type,
    CompactResult,
    constraints,
    hard_cost,
    kitchen,
    total_tokens,
    type_compact,
    type_decompose,
    uniform_compact,
    uniform_decompose_by_tokens,
)
from cliff_stub import (
    Msg,
    Session,
    TOOL_RESULT_KEEP,
    any_rewritten,
    apply_cliff,
    context_tokens,
    is_faithful_to_source,
    prior_cliff_discarded,
    summarize_nest,
)


# ---------------------------------------------------------------------------
# Session fixtures (coding-agent trajectory; tool-heavy)
# ---------------------------------------------------------------------------
SYSTEM = Msg("system", "You are a long-horizon agent. Use tools.")
TASK = Msg("task", "Complete the multi-step subgoal under a token budget.")
SHORT_GREP = "parser.py:42: assert tok == expected"


def _turn(i: int, write_body: bool = True) -> List[Msg]:
    thought = (
        f"Step {i}: inspect and apply a small patch. " + ("analysis " * 40)
    )
    long_body = f"FILE_BODY_{i}_" + ("x" * 2400)
    call_body = (
        f"write_file contents={long_body}"
        if write_body
        else f"grep -n assert src/parser.py #{i}"
    )
    path = f"src/parser_{i}.py"
    result = long_body if write_body else f"{SHORT_GREP} #{i}"
    return [
        Msg("assistant", thought),
        Msg(
            "tool_call",
            call_body,
            tool_name="file_editor" if write_body else "grep",
            path=path,
        ),
        Msg("tool_result", result),
    ]


def _long_live(n_turns: int = 6) -> List[Msg]:
    live: List[Msg] = []
    for i in range(n_turns):
        live.extend(_turn(i, write_body=(i % 2 == 0)))
    return live


def _mark_window(live: List[Msg], tag: str) -> List[Msg]:
    out: List[Msg] = []
    for m in live:
        text = m.text
        if not (m.kind == "tool_result" and len(m.text) > TOOL_RESULT_KEEP):
            text = m.text + f" {tag}"
        path = (m.path + f".{tag.lower()}") if m.path else m.path
        out.append(Msg(m.kind, text, m.tool_name, path, m.rewritten, m.is_prior_cliff))
    return out


def _ok(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Horizon triage gate: stop → triage → (optional) cliff
# ---------------------------------------------------------------------------

def horizon_triage_gate(
    *,
    # Stage A — stop evidence
    stop_policy: str = "principled",
    b: float = 0.18,
    stop_budget: int = DEFAULT_BUDGET,
    seed: int = 7,
    reports: Optional[Sequence[bool]] = None,
    delta: float = 0.05,
    alpha: float = 0.12,
    peak_at: int = 4,
    # Stage B — typed working set
    items: Optional[Sequence[Item]] = None,
    budget: Optional[int] = None,
    # Stage C — session / token pressure
    session: Optional[Session] = None,
    token_pressure: bool = False,
    # Blind knobs
    uniform: bool = False,
    nest_summarize: bool = False,
) -> dict:
    """Compile stop → type triage → (optional) cliff into continue|cliff|stop|unsafe.

    Precedence (explicit, tested):
      A. principled stop (stopped_at < stop_budget)  → stop
      B. TypeCompact.unsafe (budget < B_min)         → unsafe  (skip cliff)
      C. under typed budget ∧ token_pressure         → cliff
         under typed budget ∧ ¬token_pressure        → continue
      blind: full_budget ∧ uniform ∧ nest            → drifts (must differ)
    """
    items = list(items) if items is not None else kitchen()
    b_min = hard_cost(items)
    if budget is None:
        budget = b_min + 6

    audit: dict = {
        "stop_policy": stop_policy,
        "stop_budget": stop_budget,
        "budget": budget,
        "b_min": b_min,
        "token_pressure": token_pressure,
        "uniform": uniform,
        "nest_summarize": nest_summarize,
    }

    # --- Stage A: principled stop vs full-budget ---
    stop_out = simulate_null_query(
        b,
        stop_budget,
        seed=seed,
        delta=delta,
        alpha=alpha,
        policy=stop_policy,
        reports=reports,
        peak_at=peak_at,
    )
    audit["stopped_at"] = stop_out["stopped_at"]
    audit["stop_reason"] = stop_out["stop_reason"]
    audit["final_vote"] = stop_out["final_vote"]
    audit["drift_event"] = stop_out["drift_event"]

    if stop_out["stopped_at"] < stop_budget and stop_policy == "principled":
        audit["action"] = "stop"
        audit["reason"] = f"principled:{stop_out['stop_reason']}"
        audit["faithful"] = True
        audit["rewritten"] = False
        audit["lost_constraints"] = set()
        audit["kept_constraint_ids"] = set()
        audit["triage_unsafe"] = False
        audit["session"] = session
        audit["tokens"] = None
        audit["cliff_msgs"] = None
        return audit

    # --- Stage B: type triage vs uniform ---
    if uniform:
        compact: CompactResult = uniform_compact(items, budget)
    else:
        compact = type_compact(items, budget)

    audit["lost_constraints"] = set(compact.lost_constraint_ids)
    audit["kept_ids"] = {i.iid for i in compact.kept}
    audit["kept_constraint_ids"] = {
        i.iid for i in compact.kept if i.typ == Type.C
    }
    audit["triage_unsafe"] = compact.unsafe

    if compact.unsafe and not uniform:
        # Gate path: below B_min escalate — do not cliff-rewrite over a drop
        audit["action"] = "unsafe"
        audit["reason"] = "below_B_min:escalate"
        audit["faithful"] = None
        audit["rewritten"] = False
        audit["session"] = session
        audit["tokens"] = None
        audit["cliff_msgs"] = None
        return audit

    # --- Stage C: cliff vs nest (only when continuing under pressure) ---
    sess = session
    if sess is None:
        sess = Session(SYSTEM, TASK, prior_cliff=None, live=_long_live(6))

    if nest_summarize and token_pressure:
        after = summarize_nest(sess)
        audit["action"] = "cliff"  # pretends to compact, but injects
        audit["reason"] = "blind:full_budget+uniform+nest"
        audit["rewritten"] = any_rewritten(after)
        live_src = list(sess.live)
        cliff_msgs = list(after.prior_cliff or [])
        audit["faithful"] = is_faithful_to_source(cliff_msgs, live_src)
        audit["session"] = after
        audit["cliff_msgs"] = cliff_msgs
        audit["tokens"] = context_tokens(after)
        audit["drift"] = True
        return audit

    if token_pressure:
        live_src = list(sess.live)
        after = apply_cliff(sess)
        cliff_msgs = list(after.prior_cliff or [])
        audit["action"] = "cliff"
        audit["reason"] = "token_pressure:typed_cliff"
        audit["rewritten"] = any_rewritten(after)
        audit["faithful"] = is_faithful_to_source(cliff_msgs, live_src)
        audit["session"] = after
        audit["cliff_msgs"] = cliff_msgs
        audit["tokens"] = context_tokens(after)
        audit["drift"] = False
        return audit

    # No pressure: typed keep; session untouched
    audit["action"] = "continue"
    audit["reason"] = "under_budget:type_compact"
    audit["rewritten"] = False
    audit["faithful"] = True
    audit["session"] = sess
    audit["tokens"] = context_tokens(sess)
    audit["cliff_msgs"] = None
    audit["drift"] = bool(compact.lost_constraint_ids)
    return audit


def blind_inject(
    *,
    items: Optional[Sequence[Item]] = None,
    budget: Optional[int] = None,
    session: Optional[Session] = None,
    stop_budget: int = DEFAULT_BUDGET,
    seed: int = 7,
    b: float = 0.18,
) -> dict:
    """Blind inject = full_budget stop + uniform compact + nest under pressure."""
    return horizon_triage_gate(
        stop_policy="full_budget",
        items=items,
        budget=budget,
        session=session,
        token_pressure=True,
        uniform=True,
        nest_summarize=True,
        stop_budget=stop_budget,
        seed=seed,
        b=b,
    )


# ---------------------------------------------------------------------------
# Join assertions (measure the JOIN, not each pack alone)
# ---------------------------------------------------------------------------

def scenario_1():
    """Blind full-budget+uniform+nest ≠ gate under pressure (gate→cliff faithful)."""
    items = kitchen()
    budget = hard_cost(items) + 6
    live = _long_live(6)
    # Force past Stage A stop so we exercise typed cliff under pressure
    gate = horizon_triage_gate(
        stop_policy="full_budget",
        items=items,
        budget=budget,
        session=Session(SYSTEM, TASK, prior_cliff=None, live=list(live)),
        token_pressure=True,
    )
    blind = blind_inject(
        items=items,
        budget=budget,
        session=Session(SYSTEM, TASK, prior_cliff=None, live=list(live)),
    )
    _ok(gate["action"] == "cliff", f"gate under pressure → cliff; got {gate['action']}")
    _ok(gate["faithful"] is True, "gate cliff must be faithful")
    _ok(gate["rewritten"] is False, "gate cliff must not rewrite")
    _ok(len(gate["lost_constraints"]) == 0, "gate TypeCompact pins all C")
    differs = (
        blind["rewritten"] is True
        or blind["faithful"] is False
        or len(blind["lost_constraints"]) >= 1
        or blind["action"] != gate["action"]
    )
    _ok(differs, "blind must differ from gate on fidelity/constraints/action")
    _ok(blind["rewritten"] is True, "blind nest rewrites")
    _ok(blind["faithful"] is False, "blind nest unfaithful")
    _ok(blind["stopped_at"] == blind["stop_budget"], "blind full-budget never stops early")
    _ok(len(blind["lost_constraints"]) >= 1, "blind uniform loses ≥1 C")
    return {
        "gate": gate["action"],
        "gate_lost": sorted(gate["lost_constraints"]),
        "blind_rewritten": blind["rewritten"],
        "blind_lost": sorted(blind["lost_constraints"]),
        "blind_stopped_at": blind["stopped_at"],
    }


def scenario_2():
    """Principled stop wins before cliff/continue even with pressure + typed budget."""
    items = kitchen()
    budget = hard_cost(items) + 6
    live = _long_live(6)
    gate = horizon_triage_gate(
        stop_policy="principled",
        items=items,
        budget=budget,
        session=Session(SYSTEM, TASK, prior_cliff=None, live=list(live)),
        token_pressure=True,  # would cliff if stop did not win
        seed=7,
        b=0.18,
        stop_budget=20,
        delta=0.05,
        alpha=0.12,
        peak_at=4,
    )
    _ok(gate["action"] == "stop", f"principled stop wins; got {gate['action']}")
    _ok(gate["stopped_at"] < gate["stop_budget"], f"stop before B; t={gate['stopped_at']}")
    _ok(gate["stop_reason"] in ("kl", "decline", "stop"), gate["stop_reason"])
    _ok(gate["cliff_msgs"] is None, "stop path skips cliff")
    _ok(gate["rewritten"] is False, "stop never rewrites")
    # Contrast: full_budget under same pressure would cliff
    cont = horizon_triage_gate(
        stop_policy="full_budget",
        items=items,
        budget=budget,
        session=Session(SYSTEM, TASK, prior_cliff=None, live=list(live)),
        token_pressure=True,
    )
    _ok(cont["action"] == "cliff", f"full_budget under pressure → cliff; got {cont['action']}")
    return {
        "stop_t": gate["stopped_at"],
        "stop_reason": gate["stop_reason"],
        "full_budget_action": cont["action"],
        "B": gate["stop_budget"],
    }


def scenario_3():
    """Below B_min → unsafe; uniform silently drops C; unsafe skips cliff_msgs."""
    items = kitchen()
    b_min = hard_cost(items)
    tight = b_min - 1
    # continue-eligible stop (full_budget) + pressure — still escalate unsafe
    gate = horizon_triage_gate(
        stop_policy="full_budget",
        items=items,
        budget=tight,
        token_pressure=True,
    )
    _ok(gate["action"] == "unsafe", f"below B_min → unsafe; got {gate['action']}")
    _ok(gate["triage_unsafe"] is True, "triage_unsafe flag set")
    _ok(gate.get("cliff_msgs") is None, "unsafe path skips cliff rewrite")
    uni = uniform_compact(items, tight)
    _ok(uni.unsafe is False, "uniform does not escalate Unsafe")
    _ok(len(uni.lost_constraint_ids) >= 1, "uniform silently loses constraints")
    return {
        "b_min": b_min,
        "tight": tight,
        "gate": gate["action"],
        "uniform_lost": sorted(uni.lost_constraint_ids),
    }


def scenario_4():
    """Under budget + no pressure + no stop → continue; TypeCompact pins where uniform drops."""
    items = kitchen()
    budget = hard_cost(items) + 6
    gate = horizon_triage_gate(
        stop_policy="full_budget",
        items=items,
        budget=budget,
        token_pressure=False,
    )
    uni = horizon_triage_gate(
        stop_policy="full_budget",
        items=items,
        budget=budget,
        token_pressure=False,
        uniform=True,
    )
    _ok(gate["action"] == "continue", f"gate continue; got {gate['action']}")
    _ok(len(gate["lost_constraints"]) == 0, "TypeCompact pins all constraints")
    _ok(len(uni["lost_constraints"]) >= 1, "uniform at same budget loses ≥1 C")
    _ok(
        gate["kept_constraint_ids"] == {c.iid for c in constraints(items)},
        "all C in gate kept_ids",
    )
    return {
        "gate": gate["action"],
        "gate_kept_c": sorted(gate["kept_constraint_ids"]),
        "uniform_lost": sorted(uni["lost_constraints"]),
        "budget": budget,
    }


def scenario_5():
    """Under budget + pressure + no stop → cliff faithful; nest path unfaithful."""
    items = kitchen()
    budget = hard_cost(items) + 6
    live = _long_live(6)
    gate = horizon_triage_gate(
        stop_policy="full_budget",
        items=items,
        budget=budget,
        session=Session(SYSTEM, TASK, prior_cliff=None, live=list(live)),
        token_pressure=True,
    )
    nest = horizon_triage_gate(
        stop_policy="full_budget",
        items=items,
        budget=budget,
        session=Session(SYSTEM, TASK, prior_cliff=None, live=list(live)),
        token_pressure=True,
        nest_summarize=True,
        uniform=False,  # isolate nest inject even with typed Stage B
    )
    _ok(gate["action"] == "cliff", gate["action"])
    _ok(gate["faithful"] is True, "cliff ⊆ source")
    _ok(gate["rewritten"] is False, "cliff never rewrite")
    _ok(len(gate["lost_constraints"]) == 0, "typed cliff pins C")
    _ok(is_faithful_to_source(gate["cliff_msgs"], live), "direct faithfulness")
    _ok(nest["rewritten"] is True, "nest rewrites")
    _ok(nest["faithful"] is False, "nest ⊈ source")
    return {
        "cliff_n": len(gate["cliff_msgs"]),
        "nest_rewritten": nest["rewritten"],
        "gate_lost": sorted(gate["lost_constraints"]),
    }


def scenario_6():
    """Prior cliffs discarded across successive cliffs; no nest-grow."""
    items = kitchen()
    budget = hard_cost(items) + 6
    live1 = _mark_window(_long_live(6), "WINDOW1_UNIQUE")
    s0 = Session(SYSTEM, TASK, prior_cliff=None, live=live1)
    g1 = horizon_triage_gate(
        stop_policy="full_budget",
        items=items,
        budget=budget,
        session=s0,
        token_pressure=True,
    )
    _ok(g1["action"] == "cliff", g1["action"])
    c1 = list(g1["cliff_msgs"])
    t1 = g1["tokens"]

    live2 = _mark_window(_long_live(5), "WINDOW2_ONLY")
    s1 = Session(SYSTEM, TASK, prior_cliff=c1, live=live2)
    g2 = horizon_triage_gate(
        stop_policy="full_budget",
        items=items,
        budget=budget,
        session=s1,
        token_pressure=True,
    )
    _ok(g2["action"] == "cliff", g2["action"])
    c2 = list(g2["cliff_msgs"])
    t2 = g2["tokens"]
    _ok(prior_cliff_discarded(c1, live2, c2), "C_{t-1} discarded; C_t from L_t")
    retained = " ".join(m.text for m in c2)
    _ok("WINDOW1_UNIQUE" not in retained, "window-1 marker absent from C_t")
    _ok("WINDOW2_ONLY" in retained, "window-2 drives C_t")

    sn0 = Session(SYSTEM, TASK, prior_cliff=None, live=list(live1))
    sn1 = summarize_nest(sn0)
    sn1.live = list(live2)
    t_nest1 = context_tokens(sn1)
    sn2 = summarize_nest(sn1)
    t_nest2 = context_tokens(sn2)
    _ok(t2 is not None and t1 is not None, "token counts present")
    _ok(any_rewritten(sn2), "nest-of-nest rewritten")
    _ok(g2["faithful"] is True and g2["rewritten"] is False, "typed cliff stays faithful")
    raw2 = context_tokens(Session(SYSTEM, TASK, prior_cliff=None, live=live2))
    _ok(t2 < raw2 + len(" ".join(m.text for m in c1)), "no nest-grow of prior cliff into next")
    return {
        "t1": t1,
        "t2": t2,
        "t_nest1": t_nest1,
        "t_nest2": t_nest2,
        "discarded": True,
    }


def scenario_7():
    """Safe path: gate can stop|cliff|continue|unsafe; blind drifts; TypeDecompose locality."""
    items = kitchen()
    typed_budget = hard_cost(items) + 6
    live = _long_live(6)

    # stop
    stop_path = horizon_triage_gate(
        stop_policy="principled",
        items=items,
        budget=typed_budget,
        session=Session(SYSTEM, TASK, prior_cliff=None, live=list(live)),
        token_pressure=True,
        seed=7,
        b=0.18,
        stop_budget=20,
    )
    _ok(stop_path["action"] == "stop", stop_path["action"])

    # cliff
    cliff_path = horizon_triage_gate(
        stop_policy="full_budget",
        items=items,
        budget=typed_budget,
        session=Session(SYSTEM, TASK, prior_cliff=None, live=list(live)),
        token_pressure=True,
    )
    _ok(cliff_path["action"] == "cliff", cliff_path["action"])
    _ok(cliff_path["faithful"] is True and cliff_path["rewritten"] is False, "cliff faithful")
    _ok(len(cliff_path["lost_constraints"]) == 0, "typed cliff pins C")

    # continue
    cont_path = horizon_triage_gate(
        stop_policy="full_budget",
        items=items,
        budget=typed_budget,
        token_pressure=False,
    )
    _ok(cont_path["action"] == "continue", cont_path["action"])
    _ok(len(cont_path["lost_constraints"]) == 0, "typed continue pins C")

    # unsafe
    unsafe_path = horizon_triage_gate(
        stop_policy="full_budget",
        items=items,
        budget=hard_cost(items) - 1,
        token_pressure=True,
    )
    _ok(unsafe_path["action"] == "unsafe", unsafe_path["action"])
    _ok(unsafe_path.get("cliff_msgs") is None, "unsafe skips cliff")

    # TypeDecompose locality vs uniform chunk
    per_b = max(10, total_tokens(items) // 5)
    _, viol_td = type_decompose(items, per_b)
    viol_u = uniform_decompose_by_tokens(items, per_b)
    _ok(viol_td == 0, f"type_decompose violations == 0; got {viol_td}")
    _ok(viol_u >= 1, f"uniform token-chunk violations >= 1; got {viol_u}")

    # Blind drifts
    drift = blind_inject(
        items=items,
        budget=typed_budget,
        session=Session(SYSTEM, TASK, prior_cliff=None, live=list(live)),
    )
    _ok(drift["rewritten"] is True, "blind nest rewrites")
    _ok(drift["faithful"] is False, "blind nest unfaithful")
    _ok(
        len(drift["lost_constraints"]) >= 1 or drift.get("drift") is True,
        "blind uniform loses constraints / drifts",
    )
    actions = {
        stop_path["action"],
        cliff_path["action"],
        cont_path["action"],
        unsafe_path["action"],
    }
    _ok(
        actions == {"stop", "cliff", "continue", "unsafe"},
        f"gate covers all four actions; got {actions}",
    )
    return {
        "actions": sorted(actions),
        "type_violations": viol_td,
        "uniform_violations": viol_u,
        "blind_lost": sorted(drift["lost_constraints"]),
        "blind_rewritten": drift["rewritten"],
        "cliff_tokens": cliff_path["tokens"],
    }


SCENARIOS = [
    ("1_blind_full_uniform_nest_ne_gate", scenario_1),
    ("2_principled_stop_wins_before_cliff", scenario_2),
    ("3_below_bmin_escalates_unsafe", scenario_3),
    ("4_continue_typecompact_pins_uniform_drops", scenario_4),
    ("5_cliff_faithful_nest_unfaithful", scenario_5),
    ("6_prior_cliff_discarded_no_nest_grow", scenario_6),
    ("7_safe_four_actions_vs_blind_drift", scenario_7),
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
        except Exception as e:
            print(f"ERROR {name}: {e}")
    total = len(SCENARIOS)
    print(f"\n{passed}/{total} assertions passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
