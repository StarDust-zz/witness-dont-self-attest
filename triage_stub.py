#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Minimal triage-dont-uniform stub for the stop-triage-cliff join.

Faithful subset of Zerhoudi, Mitrović, Granitzer, arXiv:2608.22752
(triage-dont-uniform): Knowledge Triage classifies items by type and
routes TypeCompact / TypeDecompose / TypeRetrieve. Uniform rate is inject.

Copied from triage-cliff-compile / triage-dont-uniform so the join is
self-contained for GitHub.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence, Set, Tuple


class Type(Enum):
    C = "constraint"   # zero distortion
    P = "procedural"   # behavior-preserving rewrite ok
    B = "belief"       # compressible
    F = "preference"   # high distortion ok
    E = "episodic"     # free to drop / placeholder


@dataclass(frozen=True)
class Item:
    iid: str
    text: str
    typ: Type
    topic: str
    # constraint scope: leaf topics it covers; empty => global
    scope: Tuple[str, ...] = ()
    tokens: int = 1
    # type-blind salience used by uniform compact / relevance retrieve
    sim: float = 0.0


def kitchen() -> List[Item]:
    """Heterogeneous working set. Constraints sit at mid salience so
    the first uniform 50% cut keeps some of them; later rounds of the
    same policy drive the Compaction Cliff. Scoped non-constraint
    items live under med/db so locality can fail without replication.
    """
    return [
        Item("c1", "Patient is allergic to penicillin.", Type.C, "med",
             scope=("med", "rx"), tokens=6, sim=0.72),
        Item("c2", "Never modify production schema without a migration.", Type.C,
             "db", scope=("db",), tokens=8, sim=0.68),
        Item("p1", "Run pytest -x --tb=short before merging.", Type.P, "ci",
             tokens=7, sim=0.50),
        Item("b1", "Backend exposes a REST API over HTTPS.", Type.B, "svc",
             tokens=7, sim=0.70),
        Item("b2", "WidgetX ships from Osaka.", Type.B, "ops",
             tokens=5, sim=0.60),
        Item("b3", "Chart notes live in the med chart store.", Type.B, "med",
             tokens=6, sim=0.45),
        Item("b4", "Schema migrations land in db/migrate.", Type.B, "db",
             tokens=5, sim=0.48),
        Item("f1", "Prefer 2-space indentation.", Type.F, "style",
             tokens=4, sim=0.40),
        Item("e1", "Refactored auth module yesterday.", Type.E, "log",
             tokens=5, sim=0.85),
        Item("e2", "Cafeteria soup was lentil.", Type.E, "log",
             tokens=5, sim=0.90),
        Item("e3", "Trains ran on time this morning.", Type.E, "log",
             tokens=6, sim=0.88),
        Item("e4", "Meeting fluff: agenda postponed.", Type.E, "log",
             tokens=5, sim=0.80),
        Item("e5", "Slack ping about snacks.", Type.E, "log",
             tokens=4, sim=0.78),
        Item("e6", "Printer jammed twice.", Type.E, "log",
             tokens=4, sim=0.75),
        Item("b5", "Amoxicillin is a penicillin-class drug.", Type.B, "rx",
             tokens=6, sim=0.35),
    ]


def constraints(items: Sequence[Item]) -> List[Item]:
    return [i for i in items if i.typ == Type.C]


def total_tokens(items: Sequence[Item]) -> int:
    return sum(i.tokens for i in items)


def hard_cost(items: Sequence[Item]) -> int:
    return total_tokens([i for i in items if i.typ in (Type.C, Type.P)])


@dataclass
class CompactResult:
    kept: List[Item]
    lost_constraint_ids: Set[str]
    unsafe: bool = False


def uniform_compact(items: Sequence[Item], budget: int) -> CompactResult:
    """Type-blind: keep highest-sim items until budget. No type channel."""
    ordered = sorted(items, key=lambda i: (-i.sim, i.iid))
    kept: List[Item] = []
    used = 0
    for it in ordered:
        if used + it.tokens <= budget:
            kept.append(it)
            used += it.tokens
    kept_ids = {i.iid for i in kept}
    lost = {c.iid for c in constraints(items) if c.iid not in kept_ids}
    return CompactResult(kept=kept, lost_constraint_ids=lost)


def type_compact(items: Sequence[Item], budget: int) -> CompactResult:
    """Pin C+P; compress B/F; placeholder E. Escalates Unsafe below B_min."""
    hard = [i for i in items if i.typ in (Type.C, Type.P)]
    soft = [i for i in items if i.typ not in (Type.C, Type.P)]
    hc = total_tokens(hard)
    if hc > budget:
        return CompactResult(
            kept=[],
            lost_constraint_ids={c.iid for c in constraints(items)},
            unsafe=True,
        )

    residual = budget - hc
    soft_out: List[Item] = []
    bf = sorted(
        [i for i in soft if i.typ in (Type.B, Type.F)],
        key=lambda i: (-i.sim, i.iid),
    )
    ep = sorted(
        [i for i in soft if i.typ == Type.E],
        key=lambda i: (-i.sim, i.iid),
    )

    used = 0
    for it in bf:
        cost = max(1, it.tokens // 2)
        if used + cost <= residual:
            soft_out.append(
                Item(it.iid, it.text, it.typ, it.topic, it.scope, cost, it.sim)
            )
            used += cost
    for it in ep:
        cost = 1  # placeholder
        if used + cost <= residual:
            soft_out.append(
                Item(it.iid, "[episodic]", Type.E, it.topic, (), cost, it.sim)
            )
            used += cost

    kept = list(hard) + soft_out
    kept_ids = {i.iid for i in kept}
    lost = {c.iid for c in constraints(items) if c.iid not in kept_ids}
    if lost:
        restore_cost = sum(c.tokens for c in constraints(items) if c.iid in lost)
        if total_tokens(kept) + restore_cost <= budget:
            for c in constraints(items):
                if c.iid in lost:
                    kept.append(c)
            lost = set()
        else:
            return CompactResult(kept=kept, lost_constraint_ids=lost, unsafe=True)
    return CompactResult(kept=kept, lost_constraint_ids=lost, unsafe=False)


def leaves(items: Sequence[Item]) -> Set[str]:
    return {i.topic for i in items}


def type_decompose(
    items: Sequence[Item], per_partition_budget: int
) -> Tuple[List[List[Item]], int]:
    """Partition by topic; replicate constraints into every intersecting scope."""
    by_topic: Dict[str, List[Item]] = {}
    for it in items:
        by_topic.setdefault(it.topic, []).append(it)

    partitions: List[List[Item]] = []
    for _topic, group in sorted(by_topic.items()):
        non_c = [i for i in group if i.typ != Type.C]
        chunk: List[Item] = []
        used = 0
        for it in non_c:
            if used + it.tokens > per_partition_budget and chunk:
                partitions.append(chunk)
                chunk = []
                used = 0
            chunk.append(it)
            used += it.tokens
        if chunk:
            partitions.append(chunk)

    all_leaves = leaves(items)
    for c in constraints(items):
        scope = set(c.scope) if c.scope else set(all_leaves)
        for part in partitions:
            part_topics = {i.topic for i in part}
            if part_topics & scope and c not in part:
                part.append(c)

    violations = 0
    for part in partitions:
        part_topics = {i.topic for i in part}
        part_c_ids = {i.iid for i in part if i.typ == Type.C}
        for c in constraints(items):
            scope = set(c.scope) if c.scope else set(all_leaves)
            if part_topics & scope and c.iid not in part_c_ids:
                violations += 1
    return partitions, violations


def uniform_decompose_by_tokens(
    items: Sequence[Item], per_partition_budget: int
) -> int:
    """Token-chunk without scope replication → locality violations."""
    non_c = sorted(
        [i for i in items if i.typ != Type.C],
        key=lambda i: (i.topic, i.iid),
    )
    cs = constraints(items)
    partitions: List[List[Item]] = []
    chunk: List[Item] = []
    used = 0
    for it in non_c:
        if used + it.tokens > per_partition_budget and chunk:
            partitions.append(chunk)
            chunk = []
            used = 0
        chunk.append(it)
        used += it.tokens
    if chunk:
        partitions.append(chunk)
    if partitions:
        partitions[-1].extend(cs)
    else:
        partitions.append(list(cs))

    all_leaves = leaves(items)
    violations = 0
    for part in partitions:
        part_topics = {i.topic for i in part}
        part_c_ids = {i.iid for i in part if i.typ == Type.C}
        for c in cs:
            scope = set(c.scope) if c.scope else set(all_leaves)
            if part_topics & scope and c.iid not in part_c_ids:
                violations += 1
    return violations
