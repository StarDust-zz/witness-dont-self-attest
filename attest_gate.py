# SPDX-License-Identifier: MIT
"""Witness, don't self-attest: who may vouch that a re-read is the newest write?

recheck-after-forget (store_gate.py, recheck_experiment.py) left a residual: a watermark
is a lower bound on what the agent has SEEN, not on what EXISTS. When the retention home
itself reverts and the newer write was never observed, every gate there sends the old value.

refuse-to-certify (certify_gate.py) says a decision is refused until a frozen certificate
from an independent measurement says it holds; complementarity alone is not enough, and a
freeze that does not match the later measurement is a failed certificate, not a silent pass.

Joined: the residual needs a head attestation, a per-path claim of the newest write seq,
and the question is who may make it.

  self/snapshot   the store's own counter, restored with the same snapshot as its data
  self/monotone   the store's own counter, kept outside the snapshot (never goes back)
  cross/cartridge a second home that lags (the KV cartridge compiled at window 1)
  cross/replica   a second, independent home that keeps the newest version
  witness         an append-only write witness; its head is frozen (hashed) at certify and
                  re-opened at send: a moved head is a failed certificate -> re-certify

Verdicts: send | hold (a fresh read disagrees with the draft: redraft) |
          refuse (the home is older than the attested head: do not certify from it).

Stdlib only. No network, no GPU, no live model, no vendor API, no product import.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Optional

from store_gate import SourceLog, View, home_view, recheck_gate, view

MODES = ("self/snapshot", "self/monotone", "cross/cartridge", "cross/replica", "witness")


def head(mode: str, log: SourceLog, kind: str, path: str, cs: int) -> Optional[int]:
    """Per-path newest-write seq as claimed by an attestation source."""
    read, v = home_view(log, kind, home_aware=True, compile_seq=cs)
    if mode == "self/snapshot":           # counter index restored with the data
        r = v.get(path)
        return r.seq if r else None
    if mode in ("self/monotone", "cross/replica", "witness"):
        vs = log.versions(path)           # same number here; differs in what can revert
        return vs[-1].seq if vs else None
    if mode == "cross/cartridge":
        r = view(log, "cartridge", compile_seq=cs).get(path)
        return r.seq if r else None
    raise ValueError(mode)


def freeze(heads: Dict[str, int]) -> str:
    return hashlib.sha256(json.dumps(sorted(heads.items()), separators=(",", ":")).encode()).hexdigest()


def witness_heads(log: SourceLog) -> Dict[str, int]:
    return {p: log.versions(p)[-1].seq for p in log.paths()}


def attested_gate(x: dict, mode: Optional[str], *, at_send: bool = True) -> dict:
    """wm+home gate (recheck-after-forget) plus a head attestation on every leased path.

    mode None reproduces wm+home exactly. at_send=False checks the attestation only at
    certify (the silent-pass variant for the witness freeze)."""
    k, cs, d, c = x["kind"], x["cs"], x["d"], x["c"]
    _, hc = home_view(x["lc"], k, home_aware=True, compile_seq=cs)
    _, hs = home_view(x["ls"], k, home_aware=True, compile_seq=cs)
    g = recheck_gate(d, c, hc, hs, w_cert=x["wc"], w_send=x["ws"])
    if mode == "witness":
        moved = freeze(witness_heads(x["lc"])) != freeze(witness_heads(x["ls"]))
        g = {**g, "freeze": "failed-certificate" if moved else "opened"}
    if mode is None or g["action"] != "send":
        return g
    stale = []
    for key, ls in g.get("leases", {}).items():
        checks = [(x["lc"], hc)] + ([(x["ls"], hs)] if at_send else [])
        for log, v in checks:
            h = head(mode, log, k, ls.path, cs)
            if h is not None and v[ls.path].seq < h:
                stale.append(f"{key}@{ls.path}: read seq {v[ls.path].seq} < head {h}")
    if mode == "witness" and g["freeze"] == "failed-certificate" and at_send:
        # a moved witness head is not a silent pass: re-certify against the send store
        r = recheck_gate(d, c, hs, hs, w_cert=x["ws"], w_send=x["ws"])
        if r["action"] != "send":
            return {**r, "freeze": "failed-certificate", "recertified": True}
        for key, ls in r.get("leases", {}).items():
            if hs[ls.path].seq < witness_heads(x["ls"])[ls.path]:
                stale.append(f"{key}@{ls.path}: recertify read stale")
    if stale:
        return {**g, "action": "refuse", "reason": "stale-home:" + "; ".join(stale)}
    return g
