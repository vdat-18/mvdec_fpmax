# DEKM Canonical Baseline

Status: **FROZEN** on 2026-10-14.

This document is the single repository-level decision record for the DEKM 2021
baseline used by subsequent MvDEC experiments. The baseline is a verified
replay of the released DEKM behavior on a modern runtime. It is not a claim of
bitwise reproduction because the historical environment and all random-state
details are not fully specified by the release.

## Canonical results

All values are the mean and sample standard deviation across the preregistered
seeds 42, 43, and 44. Deltas are baseline mean minus the DEKM Table III value.

| Dataset | Canonical protocol | ACC mean +/- std | Paper ACC | Delta | NMI mean +/- std | Paper NMI | Delta |
|---|---|---:|---:|---:|---:|---:|---:|
| REUTERS | `dekm_2021_release_seeded_modern_runtime_v1` | 0.770900 +/- 0.035248 | 0.7628 | +0.810 pp | 0.579669 +/- 0.014124 | 0.5906 | -1.093 pp |
| 20NEWS | `dekm_2021_20news_sklearn18846_seeded_modern_runtime_v1` | 0.402260 +/- 0.022039 | 0.4108 | -0.854 pp | 0.434869 +/- 0.009363 | 0.4027 | +3.217 pp |
| RCV1 | `dekm_2021_release_seeded_modern_runtime_v1` | 0.704567 +/- 0.087224 | 0.6715 | +3.307 pp | 0.459421 +/- 0.105966 | 0.4618 | -0.238 pp |

These are the comparison values to use for MvDEC. Do not replace an aggregate
with a favorable individual seed or tune a protocol against the paper labels.

## Dataset and protocol provenance

- REUTERS and RCV1 use the paper-era source revision
  `129b7d08b545f02d920ccff0948ec44b654b3158` and protocol
  `dekm_2021_release_seeded_modern_runtime_v1`.
- REUTERS uses 10,000 rows, 2,000 features, and source SHA-256
  `427993db6b0544dfce1b8e3985e63e248d4d006de122cb202c504489d951706b`.
- RCV1 uses 10,000 rows, 2,000 features, and source SHA-256
  `083201cff76ee4952531472fcd630bd6366ca9b2a34ec08a3092c6f4f1aef90c`.
- 20NEWS uses the official executable loader path
  `fetch_20newsgroups(subset="all")` followed by
  `TfidfVectorizer(max_features=2000)`. It has 18,846 rows, 2,000 features,
  source SHA-256
  `fe15f52bffcfc3601a783d3c6343441620e93afb45c7ec50cdaba2b3fe0b2b50`,
  and protocol `dekm_2021_20news_sklearn18846_seeded_modern_runtime_v1`.
- The earlier 18,057-row 20NEWS NPZ matrix is retained only as an audit
  artifact. It is not the canonical 20NEWS baseline.

The release-derived protocols retain paper-era choices such as uniform
initialization, `np.linalg.eig`, global-RNG K-Means, historical batching, and
the released assignment-change threshold. The wrapper records seeds and
artifacts, but global RNG use and TensorFlow runtime behavior prevent a numeric
seed from guaranteeing bit-identical reruns.

## Verification and artifacts

Independent verification recomputed ACC and NMI from every assignment file,
checked all manifest-declared hashes, checked run counts and aggregate
statistics with `ddof=1`, confirmed empty failure lists, and validated the ZIP
contents and checksums.

- REUTERS and RCV1 evidence:
  `artifacts/kaggle/dekm-2021-three-datasets-seeds42-44/version-3`
- REUTERS/RCV1 result ZIP SHA-256:
  `05443c7992348c9546406695c8f41ea4ff95743ae1dedb45b280bd1297bd211e`
- Corrected 20NEWS evidence:
  `artifacts/kaggle/dekm-2021-20news-official-source-seeds42-44/version-1`
- Corrected 20NEWS result ZIP SHA-256:
  `b28f5a91bf971bc3820eeab5329b32577800f05d5ae3fde4a25dbb214f9b1e87`
- Full investigation record: `AUDIT_DEKM_MVDEC_2026-10-13.md`

## Freeze policy

DEKM experimentation is closed for the current MvDEC study. Reopen this
baseline only if one of the following becomes available:

1. New author-provided code, data, seeds, or environment information that
   changes the executable contract.
2. A demonstrable provenance or verification defect in the frozen artifacts.

Normal runtime variation, a more favorable rerun, or a hyperparameter setting
that moves a metric closer to the paper is not sufficient reason to reopen the
baseline.

## Next stage

All new benchmark work now belongs to MvDEC. First reconcile the repository's
canonical MvDEC protocol and status documents. Then run a bounded seed-42 gate
on corrected official-source 20NEWS using the frozen 18,846-row input. Run
seeds 43-44 only after the gate and its artifacts pass independent validation.
RCV1 follows after 20NEWS, with full-batch memory feasibility checked explicitly
and any OOM reported instead of silently changing the protocol.