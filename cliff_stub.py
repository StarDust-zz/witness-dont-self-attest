#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Minimal cliff-dont-drift stub for the stop-triage-cliff join.

Faithful subset of Nguyen et al., arXiv:2609.26779 (cliff-dont-drift):
live turns are source; cliff truncate+signature never rewrite/nest;
nested LLM summarization is the inject that drifts.

Copied from triage-cliff-compile / cliff-dont-drift so the join is
self-contained for GitHub.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

Kind = Literal["system", "task", "assistant", "tool_call", "tool_result", "compaction"]

TOOL_RESULT_KEEP = 500
THINK_TRUNCATE = 300
SIGNATURE_CAP = 150
RECENT_TURN_PAIRS = 2  # K in paper Algorithm 1 (2K messages = K turn pairs)


@dataclass
class Msg:
    kind: Kind
    text: str
    tool_name: str = ""
    path: str = ""
    # True iff this string was produced by rewrite/summary (inject)
    rewritten: bool = False
    # True iff this block came from a CliffCompaction output
    is_prior_cliff: bool = False


@dataclass
class Session:
    """S_t = C_{t-1} ⊕ L_t  (paper §2.3)."""

    system: Msg
    task: Msg
    prior_cliff: Optional[List[Msg]] = None  # C_{t-1}
    live: List[Msg] = field(default_factory=list)  # L_t


def signature(tool_name: str, path: str, body: str, cap: int = SIGNATURE_CAP) -> str:
    """Reduce a tool call to name + path + capped head — never paraphrase."""
    head = body[:cap] if body else ""
    parts = [f"[{tool_name}]"]
    if path:
        parts.append(path)
    if head:
        parts.append(head)
    return " ".join(parts)


def truncate(text: str, n: int) -> str:
    """Prefix truncate only — never rewrite."""
    return text[:n]


def cliff_compact(live: List[Msg], k: int = RECENT_TURN_PAIRS) -> List[Msg]:
    """Algorithm 1: CliffCompaction(turns) over the live session only.

    - Discard any prior compaction messages that somehow appear in live
    - Truncate assistant thoughts; reduce tool calls to signatures
    - Keep short tool results; drop long ones
    - Keep the most recent 2K messages verbatim
    - Never invent summary text (rewritten stays False)
    """
    if not live:
        return []

    cut = 2 * k
    if len(live) <= cut:
        return [
            Msg(m.kind, m.text, m.tool_name, m.path, rewritten=False, is_prior_cliff=True)
            for m in live
        ]

    old, recent = live[:-cut], live[-cut:]
    parts: List[Msg] = []

    for m in old:
        if m.kind == "compaction" or m.is_prior_cliff:
            continue
        if m.kind == "assistant":
            parts.append(
                Msg(
                    "assistant",
                    truncate(m.text, THINK_TRUNCATE),
                    rewritten=False,
                    is_prior_cliff=True,
                )
            )
        elif m.kind == "tool_call":
            sig = signature(m.tool_name, m.path, m.text)
            parts.append(
                Msg(
                    "tool_call",
                    sig,
                    tool_name=m.tool_name,
                    path=m.path,
                    rewritten=False,
                    is_prior_cliff=True,
                )
            )
        elif m.kind == "tool_result":
            if len(m.text) <= TOOL_RESULT_KEEP:
                parts.append(
                    Msg(
                        "tool_result",
                        m.text,
                        rewritten=False,
                        is_prior_cliff=True,
                    )
                )

    for m in recent:
        parts.append(
            Msg(
                m.kind,
                m.text,
                m.tool_name,
                m.path,
                rewritten=False,
                is_prior_cliff=True,
            )
        )
    return parts


def apply_cliff(session: Session) -> Session:
    """C_t = CliffCompaction(L_t); discard C_{t-1}; S_{t+1} = C_t ⊕ empty live."""
    new_cliff = cliff_compact(session.live)
    return Session(
        system=session.system,
        task=session.task,
        prior_cliff=new_cliff,
        live=[],
    )


def summarize_nest(session: Session) -> Session:
    """Inject failure: fold prior cliff + live into a nested natural-language summary."""
    chunks = []
    if session.prior_cliff:
        for m in session.prior_cliff:
            chunks.append(m.text)
    for m in session.live:
        chunks.append(m.text)
    summary_text = "SUMMARY: " + " | ".join(c[:40] for c in chunks if c)
    summary = Msg("compaction", summary_text, rewritten=True, is_prior_cliff=True)
    return Session(
        system=session.system,
        task=session.task,
        prior_cliff=[summary],
        live=[],
    )


def context_tokens(session: Session) -> int:
    n = len(session.system.text) + len(session.task.text)
    if session.prior_cliff:
        n += sum(len(m.text) for m in session.prior_cliff)
    n += sum(len(m.text) for m in session.live)
    return n


def any_rewritten(session: Session) -> bool:
    if session.prior_cliff and any(m.rewritten for m in session.prior_cliff):
        return True
    return any(m.rewritten for m in session.live)


def is_faithful_to_source(cliff_msgs: List[Msg], source_live: List[Msg]) -> bool:
    """Every cliff fragment is a prefix/substring of some source message text
    or a signature built from source tool_call fields — never a novel rewrite.
    """
    source_texts = [m.text for m in source_live]
    source_sigs = [
        signature(m.tool_name, m.path, m.text)
        for m in source_live
        if m.kind == "tool_call"
    ]
    for c in cliff_msgs:
        if c.rewritten:
            return False
        if c.kind == "tool_call":
            if c.text in source_sigs:
                continue
            ok = any(
                c.tool_name == m.tool_name and c.path == m.path
                for m in source_live
                if m.kind == "tool_call"
            )
            if not ok:
                return False
            continue
        if not any(s.startswith(c.text) or c.text in s for s in source_texts):
            return False
    return True


def prior_cliff_discarded(c_prev: List[Msg], live_next: List[Msg], c_next: List[Msg]) -> bool:
    """C_t was compiled from L_t only: every C_t fragment is faithful to L_t,
    and unique C_{t-1} content (not regenerable from L_t) is absent from C_t.
    """
    if not is_faithful_to_source(c_next, live_next):
        return False
    # unique marker texts from C_{t-1} that cannot be produced from live_next
    live_union = " ".join(m.text for m in live_next)
    live_sigs = {
        signature(m.tool_name, m.path, m.text)
        for m in live_next
        if m.kind == "tool_call"
    }
    next_union = " ".join(m.text for m in c_next)
    for m in c_prev:
        # skip fragments that live_next could regenerate (overlap)
        if m.text in live_union or m.text in live_sigs:
            continue
        if len(m.text) < 20:
            continue  # too short to be a unique marker
        if m.text in next_union:
            return False
    return True
