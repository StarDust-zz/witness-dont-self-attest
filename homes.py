# SPDX-License-Identifier: MIT
"""Document homes: context, cartridge, compaction, parametric.

Discrete algebra of Rakotonirina et al., arXiv:2609.17346 —
Where Should a Document Live: Context, Representations, or Parameters?

Same document bytes, four lowerings. Retrieval home ≠ retention home.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Document:
    """A document is a named bag of facts (key -> value)."""

    doc_id: str
    facts: FrozenSet[Tuple[str, str]]  # (key, value)


@dataclass(frozen=True)
class Query:
    """Ask for a fact key; ground-truth lives in exactly one document."""

    key: str
    target_doc: str
    expected: str


def toy_corpus() -> List[Document]:
    """Three knowledge docs + one unrelated control doc (coding skill)."""
    return [
        Document(
            "leases",
            frozenset(
                {
                    ("tenant", "Ada"),
                    ("rent", "1200"),
                    ("due", "first"),
                }
            ),
        ),
        Document(
            "payroll",
            frozenset(
                {
                    ("employee", "Bo"),
                    ("salary", "90k"),
                    ("role", "eng"),
                }
            ),
        ),
        Document(
            "inventory",
            frozenset(
                {
                    ("sku", "W-41"),
                    ("stock", "17"),
                    ("bin", "A3"),
                }
            ),
        ),
        Document(
            "coding-control",
            frozenset(
                {
                    ("sort", "timsort"),
                    ("hash", "siphash"),
                    ("gc", "refcount"),
                }
            ),
        ),
    ]


def knowledge_queries() -> List[Query]:
    return [
        Query("tenant", "leases", "Ada"),
        Query("salary", "payroll", "90k"),
        Query("stock", "inventory", "17"),
    ]


def control_queries() -> List[Query]:
    return [
        Query("sort", "coding-control", "timsort"),
        Query("hash", "coding-control", "siphash"),
        Query("gc", "coding-control", "refcount"),
    ]


# ---------------------------------------------------------------------------
# Homes (lowerings of the same documents)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cartridge:
    """KV-style prefix for one document: exact facts, loadable as a unit."""

    doc_id: str
    kv: FrozenSet[Tuple[str, str]]


@dataclass
class CartridgeBank:
    """Installed cartridges. Loading new ones can displace base control knowledge."""

    cartridges: Dict[str, Cartridge] = field(default_factory=dict)
    base_control: Dict[str, str] = field(default_factory=dict)
    forgetting: float = 0.0  # 0 = intact; 1 = fully overwritten

    def install(self, docs: Sequence[Document], *, forget: bool = True) -> None:
        for d in docs:
            if d.doc_id == "coding-control":
                # Control stays in "base" unless overwritten by forgetting.
                self.base_control = dict(d.facts)
            else:
                self.cartridges[d.doc_id] = Cartridge(d.doc_id, d.facts)
        if forget and self.cartridges:
            # Paper: cartridges (like large adapters / full FT) degrade control.
            self.forgetting = 0.6  # ~6–13% regime as a discrete drop flag

    def answer(self, q: Query) -> Optional[str]:
        # Multi-doc: route by target_doc id (oracle router / dense retrieval).
        if q.target_doc == "coding-control":
            if self.forgetting >= 0.5:
                return None  # catastrophic forgetting of control
            return self.base_control.get(q.key)
        cart = self.cartridges.get(q.target_doc)
        if cart is None:
            return None
        return dict(cart.kv).get(q.key)


@dataclass
class ContextWindow:
    """Raw documents in the prompt. Exact if present; no parametric overwrite."""

    present: Dict[str, FrozenSet[Tuple[str, str]]] = field(default_factory=dict)
    base_control: Dict[str, str] = field(default_factory=dict)

    def load(self, docs: Sequence[Document]) -> None:
        for d in docs:
            if d.doc_id == "coding-control":
                self.base_control = dict(d.facts)
            else:
                self.present[d.doc_id] = d.facts

    def answer(self, q: Query) -> Optional[str]:
        if q.target_doc == "coding-control":
            return self.base_control.get(q.key)
        facts = self.present.get(q.target_doc)
        if facts is None:
            return None
        return dict(facts).get(q.key)


@dataclass
class Compaction:
    """Compressed KV: keeps only first `keep` facts per doc (high compression loss)."""

    keep: int
    blobs: Dict[str, FrozenSet[Tuple[str, str]]] = field(default_factory=dict)
    base_control: Dict[str, str] = field(default_factory=dict)

    def install(self, docs: Sequence[Document]) -> None:
        for d in docs:
            if d.doc_id == "coding-control":
                self.base_control = dict(d.facts)
            else:
                # Deterministic lossy keep: sort keys, take first `keep`.
                ordered = sorted(d.facts, key=lambda kv: kv[0])
                self.blobs[d.doc_id] = frozenset(ordered[: self.keep])

    def answer(self, q: Query) -> Optional[str]:
        if q.target_doc == "coding-control":
            return self.base_control.get(q.key)
        facts = self.blobs.get(q.target_doc)
        if facts is None:
            return None
        return dict(facts).get(q.key)


@dataclass
class ParametricAdapter:
    """Small adapter bake: partial multi-doc recall, no catastrophic forgetting.

    Discrete stand-in for LoRA / small MLP: stores a flat fact table that
    does not route as well as a per-doc cartridge (misses under collision /
    capacity), but leaves base control intact.
    """

    capacity: int
    table: Dict[str, str] = field(default_factory=dict)  # key -> value (last write wins)
    owners: Dict[str, str] = field(default_factory=dict)  # key -> doc_id
    base_control: Dict[str, str] = field(default_factory=dict)
    forgetting: float = 0.0

    def install(self, docs: Sequence[Document]) -> None:
        n = 0
        for d in docs:
            if d.doc_id == "coding-control":
                self.base_control = dict(d.facts)
                continue
            for k, v in sorted(d.facts):
                if n >= self.capacity:
                    break
                # Collision: same key from later doc overwrites (weak multi-doc).
                self.table[k] = v
                self.owners[k] = d.doc_id
                n += 1
        # Small adapter: no catastrophic forgetting.
        self.forgetting = 0.0

    def answer(self, q: Query) -> Optional[str]:
        if q.target_doc == "coding-control":
            return self.base_control.get(q.key)
        # Only hit if key is in table AND still owned by the target doc.
        if q.key in self.table and self.owners.get(q.key) == q.target_doc:
            return self.table[q.key]
        return None


def score(answerer, queries: Sequence[Query]) -> float:
    """Fraction of queries answered exactly."""
    if not queries:
        return 0.0
    hits = sum(1 for q in queries if answerer.answer(q) == q.expected)
    return hits / len(queries)


def homes_of(docs: Sequence[Document]) -> Tuple[ContextWindow, CartridgeBank, Compaction, ParametricAdapter]:
    """Lower the same corpus into four homes."""
    ctx = ContextWindow()
    ctx.load(docs)
    cart = CartridgeBank()
    cart.install(docs, forget=True)
    comp = Compaction(keep=1)  # high compression: one fact per doc
    comp.install(docs)
    param = ParametricAdapter(capacity=3)  # tight capacity across all docs
    param.install(docs)
    return ctx, cart, comp, param
