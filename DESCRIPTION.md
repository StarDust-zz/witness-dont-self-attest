# Witness, don't self-attest

A synergy deep-dive joining two published packs:

- [recheck-after-forget](https://github.com/StarDust-zz/recheck-after-forget)
  — a re-check certifies the home it re-reads, not the fact. Watermark +
  home-aware leases cut wrong sends 21 → 4 on an 88-case matrix; the 4 left are
  a reverted retention home with a newer write the agent never observed. Its
  refined claim: *a watermark is a lower bound on what you have seen, not on
  what exists*. Its open hunch: a store that attests its own write seq would
  close the residual, one that can revert its own counter would not; whether
  the revert-prone case should escalate rather than hold.
- [refuse-to-certify](https://github.com/StarDust-zz/refuse-to-certify)
  — a decision is refused until a frozen, finite-sample certificate says it
  holds; complementarity is not enough (redundant advisors co-fail); a freeze
  that does not match the later measurement is a failed certificate, not a
  silent pass. (Sun, Yang, *RouteGuard*,
  [arXiv:2608.07583](https://arxiv.org/abs/2608.07583).)

Not a fresh arXiv pack: no new papers. Both parents are copied in unchanged
(`recheck_experiment.py` is recheck-after-forget's `experiment.py` with all its
modules; `certify_gate.py` is refuse-to-certify's `experiment.py`, without its
optional live residual). New code is `attest_gate.py` + `experiment.py`.

```
python3 experiment.py
```

Stdlib only. No network, no GPU, no live model, no vendor API, no product import.

## Question

The parent residual needs someone to say "the newest write to this path is seq
N" (a head attestation). Who may say it? The store itself, a second home, or an
independent write witness? And when a home is found stale, should the gate hold
and re-check later, or escalate?

## The object

```
attestation head(path)   per-path newest write seq, as claimed by
  self/snapshot          the store's own counter, restored with the same snapshot as its data
  self/monotone          the store's own counter, kept outside the snapshot (never goes back)
  cross/cartridge        a second home that lags (KV cartridge compiled at window 1)
  cross/replica          a second, independent home holding the newest version
  witness                append-only write witness; head-per-path hashed (frozen) at certify,
                         re-opened at send: a moved head is a failed certificate -> re-certify

gate = recheck-after-forget's wm+home gate, then for every leased path:
       read seq < head(path) at certify or at send  ->  refuse (stale home: do not certify)
verdicts: send | hold (a fresh read disagrees with the draft) | refuse | escalate (horizon)
```

**Claim.** A store cannot vouch for itself across its own revert: a counter
restored with the data attests the old version as newest, and a second home
that fails with it (a lagging cartridge) agrees with the old value. Only an
attestation that does not go back with the store closes the residual, and it
does so by refusing, without holding any correct draft. The attestation must be
re-opened at send, not only checked at certify. A clean matrix is not a
certificate: the gain is real but small and concentrated in one cell.

**Falsifier.** If a snapshot-restored counter or a lagging second home caught
any residual case; if an independent attestation removed the residual only by
holding correct drafts; if attesting at certify only were enough; if the
88-case gain certified at δ = 0.05 without more cases.

## What this pack tests (7/7)

1. **Parents bind.** recheck-after-forget (with its four parents) and
   refuse-to-certify run 7/7 in-process. With no attestation the gate is
   exactly wm+home on all 88 cases: 18 sent, 14 correct, 4 wrong, 2 correct
   held. Residual = `revert × {before, between}/unobserved × pre-cliff draft`.
2. **Self-attestation follows the snapshot.** On the 4 residual cases the
   reverted read is seq 3; a snapshot-restored counter attests head 3 (send,
   wrong); a monotone counter on the same bytes attests head 7 (refuse).
3. **NEW — matrix by attestation source** (attested at certify and at send):

   | attestation | sent | correct sent | wrong sends | correct held | refused |
   |---|---:|---:|---:|---:|---:|
   | none (wm+home) | 18 | 14 | 4 | 2 | 0 |
   | self/snapshot | 18 | 14 | 4 | 2 | 0 |
   | cross/cartridge | 18 | 14 | 4 | 2 | 0 |
   | self/monotone | 14 | 14 | **0** | 2 | 4 |
   | cross/replica | 14 | 14 | **0** | 2 | 4 |
   | witness | 14 | 14 | **0** | 2 | 4 |

   The three independent sources refuse exactly the 4 residual cases and no
   correct draft.
4. **Second home as an advisor (refuse-to-certify's math).** Rows = (home,
   wm+home decision right, attested decision right) over 88 cases. Lagging
   cartridge: G = Φ = 0, never certifies. Replica: G = Φ = 1/22, all of it in
   the revert cell (G = 2/11 there). The Bernstein bracket (δ = 0.05) refuses
   at m = 88 (m★ = 274) and in the 22-case revert cell (m★ = 64). Excess over
   independence is negative for **both** (−123/1936, −41/1936): it does not
   separate a redundant second home from a useful one here; G and Φ do.
5. **NEW — witness freeze re-opened at send.** The frozen witness head moves on
   32 cases, exactly the between-certify-and-send writes; all 32 are labelled a
   failed certificate (30 hold, 2 refuse), never a silent pass. Attested at
   certify only, every independent source leaves 2 wrong sends
   (`revert × between/unobserved × pre`).
6. **NEW — hold vs escalate (stylized: catch-up after 2 ticks, retry budget 3,
   horizon 8).** A transient lag: hold-forever, hold-then-escalate and
   monotone-aware all send at tick 2; escalate-now wakes a human at tick 0. A
   persistent revert: hold-forever is stuck for the whole horizon;
   hold-then-escalate escalates at tick 3; monotone-aware (escalate when the
   store's own counter went back below a head the agent saw) escalates at tick 0.
   If the agent never saw the newer head from the store, there is no backward
   step and monotone-aware falls back to the retry budget (tick 3). Without a
   witness head at all, the reverted read looks newest and the gate sends at tick 0.
7. **Trust stages + horizon still bind.** Witness-attested faithful draft →
   send; RISK_HIGH → hold; tight budget → escalate. Self-scan: no product names.

## Synergy

recheck-after-forget, in plain words: when you re-read a fact, you only learn
what that particular place has, and a place that rolled back hands you an old
answer that looks perfectly valid. refuse-to-certify, in plain words: don't
accept a decision because two sources look different or because things went
well so far; accept it when an independent, frozen measurement says so, and
treat a measurement that changed under you as a failure.

Together: the "is this the newest version?" question needs a witness that does
not roll back with the store. The store's own counter rolls back with it, and a
second place that is stale in the same way just agrees. An independent
monotone record catches exactly the leftover wrong sends without blocking a
single correct one, but only if it is checked again at send time. And the
pack's own clean result is too small a sample to certify that gain.

## Keepers

See `KEEPERS.txt`.
