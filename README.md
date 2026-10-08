# witness-dont-self-attest

Synergy deep-dive joining
[recheck-after-forget](https://github.com/StarDust-zz/recheck-after-forget)
(a re-check certifies the home it re-reads; watermark + home-aware leases leave
4 wrong sends when the retention home reverts and the newer write was unseen)
with [refuse-to-certify](https://github.com/StarDust-zz/refuse-to-certify)
(refuse until an independent, frozen certificate says the decision holds).

Keeper: a store cannot vouch for itself across its own revert. A counter
restored with the snapshot, or a second home that lags with it, adds nothing
(4 → 4). An attestation that does not go back with the store takes the residual
to 0 by refusing exactly those 4 cases, with no correct draft held. It has to
be re-checked at send (certify-only leaves 2). The pack's clean result is
itself too small to certify (gain 1/22, m★ = 274).

## Run

```bash
python3 experiment.py           # 7/7 join assertions
python3 recheck_experiment.py   # parent recheck-after-forget 7/7
python3 certify_gate.py         # parent refuse-to-certify 7/7
```

Stdlib only. No network, GPU, or vendor API.

## Layout

- `attest_gate.py`, `experiment.py`: new code (attestation sources, witness freeze, matrix, hold vs escalate)
- `recheck_experiment.py`, `store_gate.py`, `send_gate.py`, `trust_gate.py`, `answer_gate.py`,
  `cartridge_gate.py`, `homes.py`, `*_stub.py`, `horizon_gate.py`: recheck-after-forget, unchanged
- `certify_gate.py`: refuse-to-certify's experiment, unchanged

MIT. See `DESCRIPTION.md` and `KEEPERS.txt`.
