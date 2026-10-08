# SPDX-License-Identifier: MIT
"""Answerable working-memory algebra (from answerable-memory / AWM).

Terminal working memory must be memory-only answerable.
Empty or generic findings with page/trajectory still visible → MMC.
Question-conditioned commit of the binding slot → MSC.

Discrete of Zhou et al., AWM, arXiv:2608.25618.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class Finding:
    source: str
    text: str


@dataclass(frozen=True)
class Question:
    ask: str
    gold: str
    slot: str


# AWM reward defaults: (r11, r10, r01, r00) = (2, 0, -0.1, -1)
REWARD = {"MSC": 2.0, "MMC": 0.0, "AE": -0.1, "UE": -1.0}


def empty_memory() -> Tuple[Finding, ...]:
    return ()


def generic_finding(source: str) -> Finding:
    return Finding(source, "the chart shows trends")


def answerable_finding(source: str, slot: str) -> Finding:
    return Finding(source, f"{slot} (from {source})")


def answer_from_memory(memory: Sequence[Finding], q: Question) -> Optional[str]:
    """Memory-only reader: question + terminal M, no trajectory/pages."""
    for finding in memory:
        if q.slot in finding.text:
            return q.gold
    return None


def answer_from_evidence(texts: Sequence[str], q: Question) -> Optional[str]:
    """Final-answer path that may still see the retrieved evidence."""
    for text in texts:
        if q.slot in text:
            return q.gold
    return None


def cell(ans_ok: bool, mem_ok: bool) -> str:
    if ans_ok and mem_ok:
        return "MSC"
    if ans_ok and not mem_ok:
        return "MMC"
    if not ans_ok and mem_ok:
        return "AE"
    return "UE"


def write_wm(
    *,
    mode: str,
    source: str,
    slot: str,
    force_empty: bool = False,
    force_generic: bool = False,
) -> Tuple[Finding, ...]:
    """Compile terminal WM after an access step.

    mode in {answerable, empty, generic}. force_* override for inject paths.
    """
    if force_empty or mode == "empty":
        return empty_memory()
    if force_generic or mode == "generic":
        return (generic_finding(source),)
    return (answerable_finding(source, slot),)
