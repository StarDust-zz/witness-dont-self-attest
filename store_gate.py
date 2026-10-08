# SPDX-License-Identifier: MIT
"""Recheck-after-forget: is a re-check only as good as the home it re-reads?

Join of two published packs (both copied in unchanged):
  send-after-cliff            (send_gate.py)      per-claim provenance gate; re-check leases
                                                  (key, value, path, digest, epoch) against a
                                                  'retention store' assumed to be source
  cartridge-retrieves-forgets (cartridge_gate.py, homes.py)
                                                  the retrieval-winning home (KV cartridge) is
                                                  also the forgetting home; high-rate
                                                  compaction drops facts. Retrieval home !=
                                                  retention home.

New object: a versioned source log (monotone write seq) under four homes, and a
lease that carries a monotone watermark (highest write seq seen for the fact).

  RE-READ   the home returns the current version of the record
  REVERT    the home returns an OLDER version; its body is a real past body, so its
            digest is self-consistent (digest equality cannot tell it from a re-read)

Homes (what a re-check reads back for a path):
  faithful    retention home: latest version                        (tag: retention)
  cartridge   KV cartridges compiled at the end of window 1 and not
              recompiled on update: exact facts, lagging version     (tag: retrieval)
  compaction  high-rate compaction (homes.Compaction keep=1) of the
              latest version: drops facts                           (tag: retrieval)
  revert      the retention home restored from its previous snapshot:
              per path, the previous version if one exists          (tag: retention)

Gates:
  trust      parent gate, lease trusted at send (recheck="none")
  digest     parent gate, path-bound lease, digest re-check at send (recheck="path")
  home       parent digest gate, but re-reads from the retention home when the
             configured home is tagged retrieval
  wm         digest gate + monotone watermark: a read with seq < W[key] is a revert
  wm+home    both

Stdlib only. No network, no GPU, no live model, no vendor API.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from homes import CartridgeBank, Compaction, Document, Query
from send_gate import (Crossed, _digest, send_after_cliff, values_in, classify)
from trust_gate import SAFE
from trust_gate import gate as trust_gate
from triage_stub import Item, Type

KEYS = ("LEASE_ID", "REFUND_EUR", "TICKET")

# ---------------------------------------------------------------------------
# Versioned source log (append-only, monotone write seq)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    seq: int
    path: str
    body: str


def facts_of(body: str) -> Dict[str, str]:
    out = {}
    for k in KEYS:
        v = values_in(body, k)
        if v:
            out[k] = v[-1]
    return out


@dataclass
class SourceLog:
    records: List[Record] = field(default_factory=list)

    @classmethod
    def from_store(cls, store: Dict[str, str]) -> "SourceLog":
        """Each history_store path becomes one record, seq in trajectory order."""
        return cls([Record(i + 1, p, b) for i, (p, b) in enumerate(store.items())])

    def copy(self) -> "SourceLog":
        return SourceLog(list(self.records))

    @property
    def head(self) -> int:
        return max((r.seq for r in self.records), default=0)

    def write(self, path: str, body: str) -> Record:
        r = Record(self.head + 1, path, body)
        self.records.append(r)
        return r

    def versions(self, path: str, upto: Optional[int] = None) -> List[Record]:
        return [r for r in self.records if r.path == path and (upto is None or r.seq <= upto)]

    def paths(self) -> List[str]:
        seen: List[str] = []
        for r in self.records:
            if r.path not in seen:
                seen.append(r.path)
        return seen

    def latest(self, upto: Optional[int] = None) -> Dict[str, Record]:
        out = {}
        for p in self.paths():
            v = self.versions(p, upto)
            if v:
                out[p] = v[-1]
        return out

    def truth(self, key: str) -> Optional[str]:
        """Ground truth: the value of the latest write (by seq) that carries the key."""
        best = None
        for r in self.latest().values():
            v = values_in(r.body, key)
            if v and (best is None or r.seq > best[0]):
                best = (r.seq, v[-1])
        return best[1] if best else None


# ---------------------------------------------------------------------------
# Homes (lowerings of the same log; what a re-check reads back)
# ---------------------------------------------------------------------------

HOME_TAG = {"faithful": "retention", "cartridge": "retrieval",
            "compaction": "retrieval", "revert": "retention"}
RETENTION_OF = {"faithful": "faithful", "cartridge": "faithful",
                "compaction": "faithful", "revert": "revert"}

View = Dict[str, Record]


def _docs(recs: Dict[str, Record]) -> List[Document]:
    return [Document(p, frozenset(facts_of(r.body).items())) for p, r in recs.items()]


def view(log: SourceLog, kind: str, *, compile_seq: int) -> View:
    latest = log.latest()
    if kind == "faithful":
        return latest
    if kind == "cartridge":
        compiled = log.latest(upto=compile_seq)
        bank = CartridgeBank()
        bank.install(_docs(compiled), forget=True)
        out = {}
        for p, r in compiled.items():
            # Retrieval winner: every compiled fact is answered exactly from the cartridge.
            assert all(bank.answer(Query(k, p, v)) == v for k, v in facts_of(r.body).items())
            out[p] = r                     # the compiled version's exact bytes
        return out
    if kind == "compaction":
        comp = Compaction(keep=1)
        comp.install(_docs(latest))
        return {p: Record(r.seq, p, " ".join(f"{k}={v}" for k, v in sorted(comp.blobs[p])))
                for p, r in latest.items()}
    if kind == "revert":
        out = {}
        for p in log.paths():
            v = log.versions(p)
            out[p] = v[-2] if len(v) > 1 else v[-1]
        return out
    raise ValueError(kind)


def bodies(v: View) -> Dict[str, str]:
    return {p: r.body for p, r in v.items()}


def home_view(log: SourceLog, kind: str, *, home_aware: bool, compile_seq: int) -> Tuple[str, View]:
    read = RETENTION_OF[kind] if (home_aware and HOME_TAG[kind] == "retrieval") else kind
    return read, view(log, read, compile_seq=compile_seq)


# ---------------------------------------------------------------------------
# Watermark: monotone register (highest write seq seen per fact)
# ---------------------------------------------------------------------------

def bump(w: Dict[str, int], key: str, seq: int) -> Dict[str, int]:
    w = dict(w)
    w[key] = max(w.get(key, 0), seq)
    return w


def watermark_live(log: SourceLog, upto: int) -> Dict[str, int]:
    """Everything the agent read while the evidence was live (before the cliff)."""
    w: Dict[str, int] = {}
    for r in log.records:
        if r.seq <= upto:
            for k in facts_of(r.body):
                w = bump(w, k, r.seq)
    return w


def watermark_context(c: Crossed, log: SourceLog) -> Dict[str, int]:
    """Watermark re-derived after the cliff from what the post-cliff agent can still see:
    a fact counts only if its K=v text is still carried (short results, recent window)."""
    seq_of = {p: r.seq for p, r in log.latest().items()}
    w: Dict[str, int] = {}
    carried = c.carried
    for j, m in enumerate(carried):
        if m.kind == "tool_result" and j > 0 and carried[j - 1].kind == "tool_call":
            p = carried[j - 1].path
            for k in facts_of(m.text):
                if p in seq_of:
                    w = bump(w, k, seq_of[p])
    for m in carried:
        for p, s, k in RECEIPT_RE.findall(m.text):
            w = bump(w, k, int(s))
    return w


RECEIPT_RE = re.compile(r"WRITE (\S+) seq=(\d+) key (\w+)")


def receipt(path: str, seq: int, keys: Sequence[str]) -> str:
    """A short write receipt (no K=v text: it raises the watermark, it is not a value)."""
    return " ".join(f"WRITE {path} seq={seq} key {k}" for k in keys)


def watermark_items(w: Dict[str, int], keys: Sequence[str]) -> List[Item]:
    """Watermark pinned into the typed set before the cliff (one small constraint per fact)."""
    out = []
    for k in keys:
        if k in w:
            text = f"WATERMARK {k} seq={w[k]}"
            out.append(Item(f"wm_{k}", text, Type.C, "pin:wm", scope=("wm",),
                            tokens=len(text.split()), sim=0.30))
    return out


def watermark_from_items(kept: Sequence[Item]) -> Dict[str, int]:
    w: Dict[str, int] = {}
    for i in kept:
        m = re.fullmatch(r"WATERMARK (\w+) seq=(\d+)", i.text)
        if m:
            w = bump(w, m.group(1), int(m.group(2)))
    return w


# ---------------------------------------------------------------------------
# Watermark-aware re-check gate (reduces to the parent digest gate when W is None)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WLease:
    key: str
    value: str
    path: str
    digest: str
    seq: int
    watermark: int
    epoch: int


def reread(c: Crossed, v: View, key: str) -> Optional[Tuple[str, str, int, str]]:
    """Parent replay route: follow carried, unrewritten signature paths; latest wins."""
    found = None
    for m in c.carried:
        if m.kind == "tool_call" and m.path in v and not m.rewritten:
            r = v[m.path]
            vals = values_in(r.body, key)
            if vals:
                found = (vals[-1], m.path, r.seq, _digest(r.body))
    return found


def certify(draft: Dict[str, str], c: Crossed, v: View, w: Optional[Dict[str, int]]) -> dict:
    claims, leases = {}, {}
    for k, val in draft.items():
        hit = reread(c, v, k)
        r = {"cls": classify(c, k), "draft": val, "seen": None, "seq": None}
        if hit is None:
            r["certified"] = False
        else:
            seen, path, seq, dg = hit
            r.update(seen=seen, seq=seq)
            floor = (w or {}).get(k, 0)
            if w is not None and seq < floor:
                r.update(certified=False, revert=f"seq {seq} < W {floor}")
            else:
                r["certified"] = seen == val
                if r["certified"]:
                    leases[k] = WLease(k, val, path, dg, seq, max(floor, seq), c.epoch)
        claims[k] = r
    return {"claims": claims, "leases": leases}


def lease_ok(ls: WLease, v_send: View, w_send: Optional[Dict[str, int]]) -> bool:
    r = v_send.get(ls.path)
    if r is None:
        return False
    if w_send is None:                                   # parent: digest only
        return _digest(r.body) == ls.digest
    floor = max(ls.watermark, w_send.get(ls.key, 0))
    if r.seq < floor:
        return False                                     # revert: older than seen
    if r.seq == ls.seq:
        return _digest(r.body) == ls.digest              # same version: bytes must match
    vals = values_in(r.body, ls.key)                     # newer version: value must still hold
    return bool(vals) and vals[-1] == ls.value


def recheck_gate(draft: Dict[str, str], c: Crossed, v_cert: View, v_send: View, *,
                 w_cert: Optional[Dict[str, int]], w_send: Optional[Dict[str, int]],
                 scalars: Optional[dict] = None) -> dict:
    if c.horizon == "unsafe":
        return {"action": "escalate", "reason": "horizon:unsafe", "claims": {}}
    cert = certify(draft, c, v_cert, w_cert)
    for k, ls in cert["leases"].items():
        if not lease_ok(ls, v_send, w_send):
            cert["claims"][k]["certified"] = False
            cert["claims"][k]["lease"] = "refused"
    bad = [k for k, r in cert["claims"].items() if not r["certified"]]
    if bad:
        return {**cert, "action": "hold", "reason": "claims:" + ",".join(bad)}
    t = trust_gate(**(scalars or SAFE))
    return {**cert, "action": t["action"], "reason": f"trust:{t['reason']}"}


def parent_gate(draft, c, v_cert: View, v_send: View, recheck: str) -> dict:
    """The copied send-after-cliff gate, fed with a home's bodies."""
    return send_after_cliff(draft, c, bodies(v_cert), store_send=bodies(v_send), recheck=recheck)


def wrong_claims(draft: Dict[str, str], action: str, log: SourceLog) -> int:
    if action != "send":
        return 0
    return sum(1 for k, v in draft.items() if log.truth(k) != v)
