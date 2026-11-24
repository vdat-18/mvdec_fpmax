# MvDEC protocol choice and DEKM_F import limits (FP-Max study)

Date: 2026-11-24. Scope: how MvDEC appears in the FP-Max study. MvDEC is **not
reproduced** (see `MVDEC_HYPOTHESIS_TESTS_2026-10-24.md`), so it is appendix
material, never main evidence.

## Chosen protocol

`mvdec_dekm_consistent_v1`: it is the protocol the private pipeline already consumes
(`configs/private_experiments.json`, CLI default in `MVDEC_dense.py`). It is a
DEKM-style hybrid, not literal Eq. 11 of Kassem et al. (L3 equals L2 for orthonormal
V, so literal Eq. 11 counts the same distance twice). The paper must say so.

## Not used, and why

- `mvdec_2025_paper_informed_full_batch_v2` (README "canonical candidate"): not
  reproduced either; no evidence it is closer to the authors' code.
- S1 / S2 arms (frozen-DEKM_F View1 or scratch "min500"): REUTERS research arms for
  hypothesis tests, not general-purpose protocols.

## DEKM_F View1 import is single-checkpoint by design

`src/pipeline/dekm_checkpoint.py` pins one checkpoint (REUTERS, seed 42) by manifest
SHA-256, weights SHA-256 and config hash. A manifest written by the current
`DEKM_dense.py` also has different keys, so it fails the schema check as well, but
widening the key set alone would not make it importable: the hash pins would still
reject it. Multi-seed View1 import therefore needs a new authorization mechanism
(for example a signed registry of checkpoints), which is a security-relevant change.

Decision: not done now. The main study does not need it: the `dekm` backbone uses
DEKM embeddings directly, and MvDEC runs in the appendix train View1 from scratch.

## Deferred

- Refactor of the module-level globals in `MVDEC_dense.py` (dimensions, k,
  tolerance, `assignment_change_tolerance` read by the artifact writer). Importing
  the module silently runs with air-pollution settings. Do it in the Step 3 plan
  before any MvDEC run for the appendix.
