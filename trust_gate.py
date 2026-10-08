#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copied verbatim from trust-compile-gate/experiment.py (https://github.com/StarDust-zz/trust-compile-gate).
# Its own 7 parent assertions still run here: python3 trust_gate.py
"""Trust-compile gate: CRM/outbound draft gate joining three keepers.

Stages before outbound contact:
  1. MDL trust on CRM/memory signals (M, R, A) → adopt vs abstain
  2. RouteGuard certify when escalate is proposed (G=πΔ_E, freeze, Bernstein)
  3. Check-then-lease when send/escalate tool is proposed

Final action ∈ {send, hold, escalate}.
Blind RAG inject (high M alone → send) is the AMMI-like lowering.

Stdlib only. No network. No GPU. No vendor API.
Run: python3 experiment.py
"""

from __future__ import annotations

from typing import Optional

from mdl_stub import blind_rag_inject, decide, reliability
from route_stub import (
    AUC_POOL,
    GAIN_BLOCK,
    certify_escalate,
    repeat_block,
)
from lease_stub import (
    B,
    Ledger,
    STREAM_SRC,
    U_of,
    L_crm,
    S_crm,
    I_crm,
    CONTACTS,
    WINDOW,
    STAGED,
    lease_send,
    make_proposal,
)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def gate(
    M: float,
    R: float,
    A: float,
    *,
    escalate_proposed: bool = False,
    route_pool=None,
    opened_before_freeze: bool = False,
    tamper_after_freeze: bool = False,
    eager_send: bool = False,
    shared_ledger: Optional[Ledger] = None,
    hold_lease: bool = False,
    fail_postcondition: bool = False,
) -> dict:
    """Compile MDL → (optional) RouteGuard → check-then-lease into send|hold|escalate.

    Mapping:
      MDL Active → candidate send/escalate
      MDL Supp/Silent/Opt-Out → hold
      escalate path requires RouteGuard certify + escalate-tool lease
      send path requires no escalate needed + send-tool lease + postcondition
    """
    mdl = decide(M, R, A)
    audit = {
        "mdl_action": mdl["action"],
        "C_final": mdl["C_final"],
        "M": M,
        "R": R,
        "A": A,
        "escalate_proposed": escalate_proposed,
        "blind_would_send": blind_rag_inject(M),
    }

    if not mdl["adopt"]:
        audit["action"] = "hold"
        audit["reason"] = f"mdl:{mdl['action']}"
        return audit

    # Active: candidate for send or escalate
    if escalate_proposed:
        pool = route_pool if route_pool is not None else AUC_POOL
        cert = certify_escalate(
            pool,
            opened_before_freeze=opened_before_freeze,
            tamper_after_freeze=tamper_after_freeze,
        )
        audit["cert"] = {
            "certified": cert["certified"],
            "reason": cert["reason"],
            "G": cert.get("G"),
            "gamma": cert.get("gamma"),
            "auc": cert.get("auc"),
        }
        if not cert["certified"]:
            audit["action"] = "hold"
            audit["reason"] = f"routeguard:{cert['reason']}"
            return audit
        # Escalate tool still needs a lease
        lease = lease_send(
            eager=eager_send,
            shared_ledger=shared_ledger,
            hold_lease=hold_lease,
            fail_postcondition=fail_postcondition,
        )
        audit["lease"] = lease
        if not lease["admitted"] or not lease["published"]:
            audit["action"] = "hold"
            audit["reason"] = f"lease:{lease['reason']}"
            return audit
        audit["action"] = "escalate"
        audit["reason"] = "certified+leased"
        return audit

    # Send path: no escalate needed
    lease = lease_send(
        eager=eager_send,
        shared_ledger=shared_ledger,
        hold_lease=hold_lease,
        fail_postcondition=fail_postcondition,
    )
    audit["lease"] = lease
    if not lease["admitted"] or not lease["published"]:
        audit["action"] = "hold"
        audit["reason"] = f"lease:{lease['reason']}"
        return audit
    audit["action"] = "send"
    audit["reason"] = "adopt+leased"
    return audit


def _ok(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Shared CRM draft fixtures (reuse decide-dont-inject scalar patterns)
# ---------------------------------------------------------------------------
SAFE = dict(M=0.92, R=reliability(0.90, 0.05), A=0.10)
BLIND = dict(M=0.88, R=reliability(0.75, 0.90), A=0.75)
RISK_BASE_M, RISK_BASE_R = 0.90, reliability(0.85, 0.10)
RISK_HIGH = dict(M=RISK_BASE_M, R=RISK_BASE_R, A=0.90)
RISK_LOW = dict(M=RISK_BASE_M, R=RISK_BASE_R, A=0.10)
GAIN_POOL = repeat_block(GAIN_BLOCK, 16)  # m=80; certifies at m★≈25


# ---------------------------------------------------------------------------
# Seven join assertions
# ---------------------------------------------------------------------------

def scenario_1():
    """Blind inject ≠ gate: high-M conflict+risk → blind would send; gate holds."""
    g = gate(**BLIND)
    _ok(g["blind_would_send"] is True, "blind RAG should send on high M")
    _ok(g["action"] == "hold", f"gate must hold under conflict+risk; got {g['action']}")
    _ok(g["mdl_action"] != "Active", g["mdl_action"])
    return {
        "blind_would_send": g["blind_would_send"],
        "gate": g["action"],
        "mdl": g["mdl_action"],
        "Cf": round(g["C_final"], 4),
    }


def scenario_2():
    """Risk inversion at gate: identical M,R; high A → hold; low A can be send-candidate."""
    hi = gate(**RISK_HIGH)
    lo = gate(**RISK_LOW)
    _ok(hi["action"] == "hold", f"high-A must hold; got {hi['action']}")
    _ok(
        lo["action"] == "send",
        f"low-A send-candidate should send (lease ok); got {lo['action']} reason={lo.get('reason')}",
    )
    _ok(hi["C_final"] < lo["C_final"], f"Cf hi={hi['C_final']:.3f} vs lo={lo['C_final']:.3f}")
    _ok(RISK_HIGH["A"] >= 0.85 and RISK_LOW["A"] <= 0.15, "A twin fixture")
    return {
        "high_A": hi["action"],
        "low_A": lo["action"],
        "Cf_high": round(hi["C_final"], 4),
        "Cf_low": round(lo["C_final"], 4),
    }


def scenario_3():
    """Uncertified escalate refused: escalate proposed, high-AUC but G=0 → hold."""
    g = gate(**SAFE, escalate_proposed=True, route_pool=AUC_POOL)
    _ok(g["mdl_action"] == "Active", g["mdl_action"])
    _ok(g["action"] == "hold", f"uncertified escalate must hold; got {g['action']}")
    cert = g.get("cert", {})
    _ok(cert.get("certified") is False, cert)
    _ok(cert.get("auc", 0) >= 0.99, f"AUC should be ~1; got {cert.get('auc')}")
    _ok(
        cert.get("G") == 0.0 or cert.get("G") is None or abs(cert.get("G", 1)) < 1e-12,
        f"G should be 0; got {cert.get('G')}",
    )
    return {
        "gate": g["action"],
        "cert_reason": cert.get("reason"),
        "auc": cert.get("auc"),
        "G": cert.get("G"),
    }


def scenario_4():
    """Certified escalate path: genuine-gain stub pool with freeze-then-measure → escalate."""
    g = gate(**SAFE, escalate_proposed=True, route_pool=GAIN_POOL)
    _ok(g["mdl_action"] == "Active", g["mdl_action"])
    _ok(g["action"] == "escalate", f"genuine gain should escalate; got {g['action']} {g.get('reason')}")
    cert = g.get("cert", {})
    _ok(cert.get("certified") is True, cert)
    _ok(cert.get("G", 0) > 0, cert.get("G"))
    _ok(g.get("lease", {}).get("admitted") is True, g.get("lease"))
    return {
        "gate": g["action"],
        "G": cert.get("G"),
        "gamma": round(cert.get("gamma", 0), 4) if cert.get("gamma") is not None else None,
        "lease": g.get("lease", {}).get("reason"),
    }


def scenario_5():
    """Eager send without lease refused; bounded checked lowering + postcondition → send."""
    eager = gate(**SAFE, eager_send=True)
    bounded = gate(**SAFE, eager_send=False)
    _ok(eager["action"] == "hold", f"eager over-cap must hold; got {eager['action']}")
    _ok(
        "lease" in eager["reason"] or "over-cap" in eager.get("reason", ""),
        eager.get("reason"),
    )
    _ok(bounded["action"] == "send", f"bounded lease should send; got {bounded['action']}")
    _ok(bounded.get("lease", {}).get("published") is True, bounded.get("lease"))
    # Sanity: stream U ≤ B < eager U
    facts = I_crm(CONTACTS)
    u_stream = U_of(L_crm(S_crm(STREAM_SRC), facts, WINDOW, STAGED), STAGED)
    _ok(u_stream <= B, f"stream U={u_stream} should be ≤ B={B}")
    return {
        "eager": eager["action"],
        "eager_reason": eager.get("reason"),
        "bounded": bounded["action"],
        "U_stream": u_stream,
        "B": B,
    }


def scenario_6():
    """Protocol: open-before-freeze fails escalate; matching freeze passes on genuine-gain."""
    bad = gate(
        **SAFE,
        escalate_proposed=True,
        route_pool=GAIN_POOL,
        opened_before_freeze=True,
    )
    good = gate(**SAFE, escalate_proposed=True, route_pool=GAIN_POOL)
    _ok(bad["action"] == "hold", f"unfrozen escalate must hold; got {bad['action']}")
    _ok(
        "opened-before-freeze" in bad.get("reason", "")
        or "protocol" in bad.get("reason", ""),
        bad.get("reason"),
    )
    _ok(good["action"] == "escalate", f"matching freeze should escalate; got {good['action']}")
    # Mismatch after freeze also holds
    tamper = gate(
        **SAFE,
        escalate_proposed=True,
        route_pool=GAIN_POOL,
        tamper_after_freeze=True,
    )
    _ok(tamper["action"] == "hold", f"mismatched freeze must hold; got {tamper['action']}")
    return {
        "unfrozen": bad["action"],
        "unfrozen_reason": bad.get("reason"),
        "matched": good["action"],
        "tampered": tamper["action"],
    }


def scenario_7():
    """Safe full pipeline → send; concurrent/over-cap lease cannot sneak a second send."""
    first = gate(**SAFE)
    _ok(first["action"] == "send", f"safe pipeline should send; got {first['action']}")
    _ok(first["mdl_action"] == "Active", first["mdl_action"])
    _ok(not first["escalate_proposed"], "no escalate needed")
    _ok(first.get("lease", {}).get("admitted") is True, first.get("lease"))

    # Concurrent: hold first lease so second cannot acquire over cap
    led = Ledger(B)
    held = gate(**SAFE, shared_ledger=led, hold_lease=True)
    _ok(held["action"] == "send", f"first concurrent should send; got {held['action']}")
    sneaky = gate(**SAFE, shared_ledger=led)
    _ok(
        sneaky["action"] == "hold",
        f"over-cap second send must hold; got {sneaky['action']} reason={sneaky.get('reason')}",
    )
    _ok(
        "lease" in sneaky.get("reason", ""),
        sneaky.get("reason"),
    )
    _ok(led.charged() > 0 and led.charged() <= B, f"charged={led.charged()} B={B}")
    return {
        "safe": first["action"],
        "held_first": held["action"],
        "sneaky_second": sneaky["action"],
        "charged": led.charged(),
        "B": B,
    }


SCENARIOS = [
    ("1_blind_inject_ne_gate", scenario_1),
    ("2_risk_inversion_at_gate", scenario_2),
    ("3_uncertified_escalate_refused", scenario_3),
    ("4_certified_escalate_path", scenario_4),
    ("5_eager_refused_bounded_send", scenario_5),
    ("6_freeze_protocol", scenario_6),
    ("7_safe_pipeline_and_lease_cap", scenario_7),
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
