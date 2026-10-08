#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run cartridge-retrieves-forgets assertions. python3 experiment.py

Discrete pack of Rakotonirina et al., arXiv:2609.17346:
Cartridges match ICL on multi-doc retrieval; they also forget.
"""

from __future__ import annotations

from homes import (
    CartridgeBank,
    Compaction,
    ContextWindow,
    Document,
    ParametricAdapter,
    Query,
    control_queries,
    homes_of,
    knowledge_queries,
    score,
    toy_corpus,
)


def _ok(cond, msg):
    if not cond:
        raise AssertionError(msg)


def scenario_1():
    """Same document bytes, four distinct homes (lowerings)."""
    docs = toy_corpus()
    ctx, cart, comp, param = homes_of(docs)
    _ok(isinstance(ctx, ContextWindow), type(ctx))
    _ok(isinstance(cart, CartridgeBank), type(cart))
    _ok(isinstance(comp, Compaction), type(comp))
    _ok(isinstance(param, ParametricAdapter), type(param))
    # Cartridges are one unit per knowledge doc; compaction keeps fewer facts.
    _ok(set(cart.cartridges) == {"leases", "payroll", "inventory"}, cart.cartridges.keys())
    lease_cart = dict(cart.cartridges["leases"].kv)
    lease_comp = dict(comp.blobs["leases"])
    _ok(len(lease_cart) == 3, lease_cart)
    _ok(len(lease_comp) == 1, lease_comp)
    _ok(lease_comp.items() <= lease_cart.items(), (lease_comp, lease_cart))
    return {
        "homes": ["context", "cartridge", "compaction", "parametric"],
        "cartridge_facts": len(lease_cart),
        "compaction_facts": len(lease_comp),
    }


def scenario_2():
    """Oracle single-doc: cartridge matches context; both beat high-compression."""
    docs = toy_corpus()
    ctx, cart, comp, _param = homes_of(docs)
    # Only leases in play (oracle).
    q = Query("tenant", "leases", "Ada")
    _ok(ctx.answer(q) == "Ada", ctx.answer(q))
    _ok(cart.answer(q) == "Ada", cart.answer(q))
    # Compaction kept only first key alphabetically ("due"), so tenant is gone.
    _ok(comp.answer(q) is None, comp.answer(q))
    ctx_s = score(ctx, [q])
    cart_s = score(cart, [q])
    comp_s = score(comp, [q])
    _ok(ctx_s == cart_s == 1.0, (ctx_s, cart_s))
    _ok(comp_s == 0.0, comp_s)
    return {"context": ctx_s, "cartridge": cart_s, "compaction": comp_s}


def scenario_3():
    """Multi-doc retrieval: cartridge matches context (ICL); parametric lags."""
    docs = toy_corpus()
    ctx, cart, _comp, param = homes_of(docs)
    qs = knowledge_queries()
    ctx_s = score(ctx, qs)
    cart_s = score(cart, qs)
    param_s = score(param, qs)
    _ok(ctx_s == 1.0, ctx_s)
    _ok(cart_s == 1.0, cart_s)
    # Capacity-3 parametric cannot hold all 9 knowledge facts with routing intact.
    _ok(param_s < cart_s, (param_s, cart_s))
    _ok(param_s <= 1 / 3 + 1e-9, param_s)  # at most one doc's worth under collision/cap
    return {"context": ctx_s, "cartridge": cart_s, "parametric": param_s}


def scenario_4():
    """High-compression compaction lags cartridge on multi-doc (paper: ~15 pts)."""
    docs = toy_corpus()
    _ctx, cart, comp, _param = homes_of(docs)
    qs = knowledge_queries()
    cart_s = score(cart, qs)
    comp_s = score(comp, qs)
    _ok(cart_s == 1.0, cart_s)
    # keep=1: each query key is usually not the kept key.
    _ok(comp_s < cart_s, (comp_s, cart_s))
    _ok(comp_s < 0.5, comp_s)
    return {"cartridge": cart_s, "compaction": comp_s}


def scenario_5():
    """Cartridges forget: after install, control (coding) queries fail."""
    docs = toy_corpus()
    _ctx, cart, _comp, _param = homes_of(docs)
    cq = control_queries()
    _ok(cart.forgetting >= 0.5, cart.forgetting)
    cart_ctrl = score(cart, cq)
    _ok(cart_ctrl == 0.0, cart_ctrl)
    # Sanity: knowledge still works.
    _ok(score(cart, knowledge_queries()) == 1.0, "knowledge should hold")
    return {"forgetting": cart.forgetting, "control_score": cart_ctrl}


def scenario_6():
    """Small parametric adapter does NOT forget control; context neither."""
    docs = toy_corpus()
    ctx, _cart, _comp, param = homes_of(docs)
    cq = control_queries()
    _ok(param.forgetting == 0.0, param.forgetting)
    _ok(score(param, cq) == 1.0, score(param, cq))
    _ok(score(ctx, cq) == 1.0, score(ctx, cq))
    return {
        "parametric_control": score(param, cq),
        "context_control": score(ctx, cq),
        "parametric_forget": param.forgetting,
    }


def scenario_7():
    """Dual object: retrieval home ≠ retention home for the same bytes."""
    docs = toy_corpus()
    ctx, cart, comp, param = homes_of(docs)
    kq, cq = knowledge_queries(), control_queries()
    # Retrieval ranking (multi-doc): context ≈ cartridge > compaction, parametric
    _ok(score(ctx, kq) == score(cart, kq) == 1.0, "retrieval winners")
    _ok(score(comp, kq) < 1.0 and score(param, kq) < 1.0, "retrieval losers")
    # Retention: context and small parametric keep control; cartridge does not.
    _ok(score(ctx, cq) == 1.0 and score(param, cq) == 1.0, "retainers")
    _ok(score(cart, cq) == 0.0, "cartridge forgets")
    # Same leases document: three executables from one source.
    leases = next(d for d in docs if d.doc_id == "leases")
    _ok(isinstance(leases, Document), leases)
    q = Query("rent", "leases", "1200")
    _ok(cart.answer(q) == "1200", "cartridge has rent")
    _ok(ctx.answer(q) == "1200", "context has rent")
    # Compaction dropped "rent" (kept only first alphabetical key).
    _ok(comp.answer(q) is None, "compaction lost rent")
    return {
        "retrieval_winners": ["context", "cartridge"],
        "retention_winners": ["context", "parametric"],
        "forgetting_home": "cartridge",
    }


def main():
    scenarios = [
        ("1 same bytes, four homes", scenario_1),
        ("2 oracle: cartridge matches context", scenario_2),
        ("3 multi-doc: cartridge matches ICL, parametric lags", scenario_3),
        ("4 high compaction lags cartridge", scenario_4),
        ("5 cartridges forget control", scenario_5),
        ("6 small parametric does not forget", scenario_6),
        ("7 retrieval home ≠ retention home", scenario_7),
    ]
    failed = 0
    for name, fn in scenarios:
        try:
            info = fn()
            print(f"PASS  {name}")
            for k, v in info.items():
                print(f"      {k}: {v}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(scenarios) - failed}/{len(scenarios)} passed")
    raise SystemExit(failed)


if __name__ == "__main__":
    main()
