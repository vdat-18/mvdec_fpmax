# Official Intuitive-K-prototypes code (from the authors)

Received by email on 2026-11-25 from Hongli Wang and Jusheng Mi (Hebei Normal
University), the authors of *Intuitive-K-prototypes: A mixed data clustering algorithm
with intuitionistic distribution centroid*, Pattern Recognition 158 (2025) 111062. The
authors asked to be cited in related publications.

| File | What it is | SHA-256 |
| --- | --- | --- |
| `Code.py` | The authors' implementation, unchanged, with a Cleveland Heart-2 demo | `9e97a35c52964e165dad8876109d3baca45f5e177699e664af91ce184d933de4` |
| `processed.cleveland.txt` | UCI Heart Disease, Cleveland subset (303 rows, `?` = missing) | `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8` |
| `audit/` | Scripts behind [AUDIT.md](AUDIT.md) | |

The authors' stated preprocessing: mode imputation for categorical attributes, median for
numerical ones, label-encoded categories, min-max scaling of numerical attributes to
[0, 1].

`Code.py` is kept byte-for-byte as received; do not edit it. Read [AUDIT.md](AUDIT.md)
before relying on it: its demo is affected by a pandas 3 change, its F1 is not
permutation-invariant, and it silently continues with empty clusters.

## Reproducing the audit

The scripts need the paper datasets from `scripts/fetch_paper_uci_datasets.py` (written
to the untracked `data/uci/`). From the repository root:

```powershell
uv run --python 3.12 python -I paper_replication/intuitive_kprototypes/official/audit/equivalence.py heart_cleveland_2 select 0 1 2
uv run --python 3.12 python -I paper_replication/intuitive_kprototypes/official/audit/beta_sweep.py heart_cleveland_2 random 2 6 20
uv run --python 3.12 python -I paper_replication/intuitive_kprototypes/official/audit/edge_cases.py
```
