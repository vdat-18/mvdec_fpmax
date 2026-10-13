# DEKM 2021 / MvDEC 2025 Audit (2026-10-13)

## Scope

The next cloud runs will exclude 20NEWS. The audit scope is REUTERS-10K and
RCV1-10K only, with the DEKM 2021 released protocol used as the control and the
MvDEC 2025 protocol evaluated only after the control is verified.

This report is a static code/paper audit. It does not claim that the current
Kaggle numbers are a faithful MvDEC reproduction.

## Executive verdict

The audit found one confirmed preprocessing bug in the reproduction contract:
the so-called `paper-text-full-batch-2021` diagnostics still passed every text
row through the released source's unit-L2 normalization. The DEKM paper instead
states `(1/d)||x_i||_2^2 approximately 1`, which requires row norm `sqrt(d)`.
The released REUTERS array already satisfies the paper formula before the loader
renormalizes it. Therefore the completed REUTERS/RCV1 "paper-text" diagnostics
did not actually test the paper preprocessing and cannot close that branch.

The low MvDEC Kaggle result is also not evidence that more seeds will fix the
problem. Its protocol changes several unresolved choices at once, while MvDEC
provides no executable source or complete text-dataset View2 contract. The next
valid experiment is a bounded seed-42 DEKM_F control with the corrected paper
normalization, not another multi-seed MvDEC run.

## Findings

### F0 - Confirmed: the paper-era source and the current source use different REUTERS batches

The initial audit inspected only the repository's 2025 head and therefore drew
an incomplete conclusion. Git history shows that the paper-era dense runner at
commit `129b7d08b545f02d920ccff0948ec44b654b3158` explicitly set
`batch_size=10000` for REUTERS. Its byte-exact `DEKM_dense.py` SHA-256 is
`1298eff9d4de23f75dccc8a341945f98e908521cf100c440b43a6521d78e5185`.
On 2024-06-04, commit `d15eecec87ebd663fbad17c11ad38dc675120e47`
commented out only that override, causing REUTERS to inherit the global
256-row batch. The later audited head
`e58f506922137abe2905aa0e4eb04a6bf952e80d` retains the 2024 behavior; its
`DEKM_dense.py` SHA-256 is
`8536630e35b05e5ad8cd0f9235c2012537f31e5b3b2a5b3313ee1c246cb3dec9`.

Consequently, the historical `release-2021` wrapper is not merely a local
mislabelling: its dataset-specific batch policy reproduces the paper-era source
state. The `release-minibatch-2021` protocol represents the later 2024+ source
state and the paper's general mini-batch prose, while `full-batch-2021` is the
explicit DEKM_F interpretation. For REUTERS, paper-era DEKM and DEKM_F are
operationally indistinguishable at the batch-policy level; for RCV1 they remain
separate because the paper-era source uses 256-row refinement.

### F1 - The current MvDEC run is a compound interpretation, not the paper

Evidence:

- `src/representation_learning/MVDEC_dense.py` defines
  `PAPER_RESOLVED_FULL_BATCH_PROTOCOL` with `smallest`,
  `selected_dimension_only`, and `full_dataset`.
- MvDEC 2025 says the last dimension is changed, while another paragraph says
  the smallest-eigenvalue direction is least informative. Those statements
  conflict.
- DEKM 2021 Algorithm 1 and the release replace only the last transformed
  coordinate and use a frozen target.
- MvDEC Table 2 says View 1 equals `DEKM_F`, but that does not prove the
  complete multi-view refinement schedule or loss implementation.

Consequence: the 71% REUTERS result cannot be used to conclude that the
architecture itself fails. It mainly tells us that this particular compound
interpretation is not close to Table 2.

### F2 - DEKM and MvDEC are not currently separated by a clean control ladder

The repository contains both implementations, but the correct order must be:

1. Reproduce DEKM 2021 REUTERS and RCV1 with the release contract.
2. Reproduce DEKM_F (full-batch) with only the batch policy changed.
3. Add the MvDEC View 2 and latent average, keeping the DEKM control schedule.
4. Only then test Eq. 11 loss terms and the MvDEC-specific ambiguities one at a
   time.

Without this ladder, a failed MvDEC run mixes data, architecture, objective,
refinement schedule, and model-selection differences.

### F3 - MvDEC Eq. 11 has a paper-level scaling hazard, not a hidden mean/sum bug

The current literal Eq. 11 path consistently sums squared dimensions and then
averages over the batch for L1/L2/L3/L4. The batch mean is a common scalar and
does not change their relative weighting. However, the paper assigns unit weight
to terms with very different dimensionalities: L1 contains two 2,000-dimensional
reconstructions, while L2/L3/L4 operate in a 10-dimensional latent space. L3 is
also mathematically equal to L2 when V is orthonormal, so the literal objective
double-counts the same cluster distance.

Thus the implementation is literal, but the objective can still be dominated by
reconstruction. Any normalized or reweighted objective is an ablation, not an
exact implementation of Eq. 11, and must use a separate protocol identity.

### F4 - The View 2 definition is under-specified and is the largest MvDEC risk

The paper says the two views use different attribute sets, but its equations
feed the same `x_i` to both encoders and the figure only gives a 13-feature air
pollution example. For REUTERS/RCV1 there is no published feature partition,
View 2 latent head, or source code.

The current implementation uses the same 2,000-dimensional X for both views and
chooses a latent/reconstruction split for the U-Net branch. That is a documented
assumption, not an established paper fact. Existing diagnostics show View 1 is
near the paper while View 2 is far below it, so this ambiguity is a stronger
candidate than random seed variation.

### F5 - The primary MvDEC metric is not the paper's undocumented "best" rule

The paper reports ?best performance metrics? but gives no seed list,
checkpoint-selection rule, or run count. The repository reports all explicit
seeds and uses final fused metrics as the fair primary result; best-of-run values
are diagnostics. This is methodologically correct for a reproducible study, but
it means exact Table 2 matching cannot be claimed without the authors' selection
rule.

### F6 - Confirmed: the paper-text protocol used the wrong normalization

The released source divides every text row by its L2 norm, producing
`||x_i||_2 = 1`. The DEKM paper visually and unambiguously states
`(1/d)||x_i||_2^2 approximately 1`, producing `||x_i||_2 = sqrt(d)`. For
2,000-dimensional text, the two contracts differ by a factor of `sqrt(2000) =
44.721...` in input magnitude.

The released REUTERS `10k_feature.npy` already has row norm `44.721...` and mean
squared feature norm exactly `1.0`. The old loader renormalized it to row norm
`1.0` and mean squared norm `0.0005`. The completed paper-text artifacts use hash
`427993db...`, proving that they used the released unit-L2 contract. The corrected
paper-normalized REUTERS fingerprint is
`7eb3f41aca38ffb93c104eef261adc3273d1da57c3750e04c92a67e335d423a1`.

This is a confirmed protocol bug because `paper-text-full-batch-2021` changed the
initializer, eigensolver, batch size, tolerance, and seeding but silently kept the
source normalization. The loader and protocol contracts now expose separate
`release_l2_unit_norm` and `paper_mean_squared_norm_one` modes. Existing
paper-text results remain valid artifacts, but they must be relabeled as hybrid
training diagnostics rather than paper-faithful preprocessing tests.

### F7 - DEKM protocol variants must not be mixed

`src/representation_learning/DEKM_dense.py` now exposes separate modern,
official-source mini-batch, DEKM_F full-batch, and legacy contracts. For paper
comparison, use `--protocol release-minibatch-2021` for DEKM and
`--protocol full-batch-2021` for DEKM_F. The legacy `release-2021` identity is
retained only so existing artifacts are not silently reinterpreted; the modern
contract remains a deterministic engineering regression test.

### F8 - MvDEC View1 is not a DEKM_F parity check after joint training

`model_view1` has the DEKM-style architecture, but the reported View1 embedding
is evaluated after the joint MvDEC optimizer has updated View1 together with
View2 and the fused objective. Therefore a final MvDEC `view1_acc` is not the
same experiment as a frozen standalone DEKM_F run. Table 2 duplicates the DEKM_F
values exactly in the Proposed View1 columns, but the paper does not state that
it reran or jointly refined View1.

Before judging fusion, the code must preserve a standalone DEKM_F artifact as
the View1 control and report the post-joint View1 result under a different name.
Exact equality between the current final View1 metric and DEKM_F is not a valid
invariant.

### F9 - The checked-in `external_repos/DEKM` tree is a local fork, not pristine source

Git history shows local 2026 changes from `uniform` to `glorot_uniform`, from
`np.linalg.eig` to symmetric `eigh`, from tolerance `0.005` to `0.001`, and the
addition of deterministic seeding. The reproducible runner correctly pins the
actual upstream revision and SHA-256 separately, but code review must not cite
`external_repos/DEKM/DEKM_dense.py` as the untouched official implementation.


### F10 - MvDEC Table 1 conflicts with its dataset names and inherited controls

MvDEC Table 1 prints public sample counts of `210`, `2000`, and `10 000` for
REUTERS-10K, 20NEWS, and RCV1-10K respectively. The first two counts conflict
with both the dataset names and DEKM 2021 (`10,000`, `18,846`, `10,000`). At the
same time, MvDEC Table 2 copies the DEKM_F values exactly into Proposed View1.
The paper gives no subset indices or sampling rule.

This creates two non-equivalent interpretations: Table 1 contains typographical
errors, or MvDEC used undisclosed reduced subsets for at least REUTERS/20NEWS.
The current 10,000-row REUTERS contract follows the dataset name, DEKM source,
and copied View1 baseline. Any reduced-sample run must be labeled a sensitivity
experiment, because choosing a favorable 210-row subset by labels would be
invalid test-set tuning.

### F11 - The paper defines incompatible fusion tensors

Equation 4 averages equal-width encoder embeddings. Figure 1 instead draws an
`AVG` after two 23-wide air-pollution outputs, while Figure 2 draws a final
concatenation of two 23-wide outputs into 46 dimensions. For text input width
2,000, the analogous full outputs would be 2,010 dimensions per view, not the
10-dimensional embeddings used in Equation 4.

The existing latent-average implementation is equation-faithful. The completed
20-dimensional latent-concatenation ablation tested a different hypothesis and
cannot close the Figure 2 full-output-concatenation branch. That branch remains
low priority because the paper never says the 46-wide tensor is sent to K-means
and it mixes reconstruction coordinates into the clustering representation.

### F12 - View-scale imbalance is the strongest remaining bounded hypothesis

The direct-head REUTERS diagnostic found mean latent row norms around `0.705`
for View1 versus `0.046` for View2, with View2 total variance over 400 times
smaller. The fused assignments therefore nearly duplicate View1. This explains
why changing average to latent concatenation did not approach Table 2.

A single preregistered scale-sensitivity protocol is justified before broader
search: per-sample L2-normalize each 10-dimensional latent and then apply the
paper's Equation 4 average. It changes only fusion geometry, uses no labels, and
keeps dataset, architecture, objective, schedule, seed, and evaluator fixed. It
is an ablation, not an exact-paper claim.


## Required implementation order

1. Freeze the dataset scope to REUTERS and RCV1 in the cloud notebook only;
   leave the loader capable of handling 20NEWS but do not launch it.
2. Run no MvDEC training yet.
3. Preserve the completed released-source controls, but do not reinterpret them
   as paper-preprocessing runs.
4. Run one corrected paper-normalized DEKM_F seed-42 diagnostic per dataset. Only
   expand to seeds 43-44 if the correction materially improves paper agreement.
5. Add a two-view model with the DEKM release objective and schedule. Compare
   View 1, View 2, and fused results separately.
6. Add MvDEC Eq. 11 terms with an explicit reduction/weight contract and test
   each term in isolation. Do not silently reuse the release MSE scaling.
7. Evaluate the eigen-direction and target choices as a 2x2 ablation, not as a
   single "paper-resolved" run.
8. Only after these controls pass, launch the two-dataset multi-seed Kaggle run.

## Acceptance criteria for "match"

A result should be called a match only if:

- DEKM REUTERS/RCV1 controls reproduce the 2021 reference range under the
  release contract.
- The exact dataset fingerprints and preprocessing are recorded.
- View 1 reproduces the MvDEC paper's DEKM_F reference within a declared
  tolerance, before judging View 2 or fusion.
- Every MvDEC ambiguity is isolated in a named protocol.
- Metrics are reported per seed and as mean plus sample standard deviation.
- "Best" values are labeled sensitivity diagnostics, not silently substituted
  for the primary aggregate.

## Files reviewed

- `docs/2021_Deep Embedded K-Means Clustering.pdf`
- `docs/2025_Multi-view Deep Embedded Clustering- Exploring a new dimension of air pollution.pdf`
- `external_repos/DEKM/DEKM_dense.py`
- `src/representation_learning/DEKM_dense.py`
- `src/representation_learning/MVDEC_dense.py`
- `src/pipeline/public_data.py`
- `src/representation_learning/utils.py`
- `MVDEC_PAPER_CODE_AUDIT.md`
- `MVDEC_REUTERS_ARCHITECTURE_DIAGNOSIS.md`


## Post-run RCV1 provenance audit

The completed Kaggle DEKM_F run was independently checked against a freshly
built local scikit-learn RCV1 cache. The split is structurally valid: 9,000 test
rows plus 1,000 validation rows, 10,000 unique rows, no overlap, and exactly
2,500 samples in each of CCAT/ECAT/GCAT/MCAT. All three saved assignment files
contain the canonical true labels, and their ACC/NMI values recompute exactly.
There is no evidence of label leakage, incorrect metric calculation, or a wrong
RCV1 subset.

The byte-level source hash is runtime-sensitive. Under NumPy 2.1.3 and
scikit-learn 1.6.1, the local loader reproduces the Kaggle hash exactly:
`083201cff76ee4952531472fcd630bd6366ca9b2a34ec08a3092c6f4f1aef90c`.
Under NumPy 2.4.6 and scikit-learn 1.9.0, it produces
`e03069e5dd5841ede5915267228c7d8222b63620e8946d7b3eba2fd667692743`.
The selected top-2,000 original feature IDs are identical in both environments,
but `numpy.argpartition` returns them in a different column order; only 74 of
2,000 positions coincide. Row normalization also differs at floating-point
roundoff scale (maximum absolute difference `1.1920929e-07`). Thus the hash
mismatch does not mean different documents, labels, or selected vocabulary, but
it proves that the released preprocessing is not byte-reproducible across
NumPy versions. The neural network can still be seed-sensitive to this column
permutation because fixed initialized weights are attached to different input
coordinates.

A raw-data K-Means control with `n_init=10` gives the same result under both
runtime families for seed 42: ACC `0.6100`, NMI `0.409672`. Seeds 43 and 44 give
ACC `0.6103`/`0.6106` and NMI `0.410413`/`0.416054`. These are already well above
DEKM Table III's K-Means reference of ACC `0.5196`, NMI `0.3135`. Therefore the
paper table cannot be treated as an exact executable oracle under the released
loader without the authors' missing run-selection/randomness details and the
historical dependency stack.

The next experiment is intentionally bounded to RCV1 seed 42. It uses the
paper-text settings (`glorot_uniform`, symmetric `eigh`, full-dataset
refinement, assignment tolerance `0.001`) while retaining deterministic seeding
for auditability. Seeds 43-44 will not be launched unless this one diagnostic is
materially closer to the DEKM_F paper row than the released-source seed-42 run.


## RCV1 paper-text diagnostic result

Kaggle Version 1 (`scriptVersionId=358003532`) completed successfully on two
T4 GPUs using training commit `69a91371f7e7f11ad6956a7daf3fc2255f4b7843`.
The bounded `paper-text-full-batch-2021` run stopped by assignment tolerance
after 15 refreshes / 140 refinement steps and 112.430485 seconds. RCV1 seed 42
produced ACC `0.6917` and NMI `0.4958711118536992`.

Against the DEKM_F paper row (ACC `0.6212`, NMI `0.3598`), the diagnostic is
still higher by `+7.05` percentage points ACC and `+13.607111` percentage
points NMI. Against the released-source seed-42 control (ACC `0.7404`, NMI
`0.5058763446541975`), the paper-text changes lower ACC by `4.87` percentage
points and NMI by `1.000523` percentage points. The ACC gap therefore moves
materially toward the paper, but the NMI gap barely moves and remains too large
to call this a reproduction.

The artifact audit passed. The source SHA-256 is the canonical Kaggle-runtime
RCV1 value
`083201cff76ee4952531472fcd630bd6366ca9b2a34ec08a3092c6f4f1aef90c`.
The 10,000 saved assignments independently recompute to ACC `0.6917` and NMI
`0.4958711118536992`; `diagnostic.csv`, `diagnostic.json`, and the manifest
agree; and every manifest-declared output hash matches. The result ZIP SHA-256
is `825093151b5c092ab7dfc13958e11a7e81e0ceb50bf9e5dd834c277af1baadc5`.
The downloaded audit bundle is stored under
`artifacts/kaggle/dekm-f-2021-rcv1-paper-text-seed42/version-1-358003532`.

Correction after the full audit: this result did not isolate all paper-text
differences because the loader still applied released unit-L2 normalization.
It is a valid hybrid diagnostic, but it cannot decide whether the paper's RMS-like
text scaling closes the RCV1 gap. Seeds 43-44 remain unjustified until one
corrected paper-normalized seed-42 run is evaluated.


## REUTERS paper-text diagnostic result

Kaggle Version 1 (`scriptVersionId=358020636`) completed successfully on two
T4 GPUs using training commit `69a91371f7e7f11ad6956a7daf3fc2255f4b7843`.
The bounded `paper-text-full-batch-2021` run stopped by assignment tolerance
after 13 refreshes / 120 refinement steps and 111.284053 seconds. REUTERS seed
42 produced ACC `0.7563` and NMI `0.5845111843402694`.

Against the DEKM_F paper row (ACC `0.7642`, NMI `0.5999`), the diagnostic is
lower by `0.79` percentage points ACC and `1.538882` percentage points NMI.
Against the released-source seed-42 control (ACC `0.7557`, NMI `0.593243`), it
is higher by only `0.06` percentage points ACC but lower by `0.873182`
percentage points NMI. The paper-text settings therefore do not provide a
consistent improvement over the released-source contract on REUTERS.

The artifact audit passed. The REUTERS source SHA-256 is
`427993db6b0544dfce1b8e3985e63e248d4d006de122cb202c504489d951706b`.
The 10,000 saved assignments independently recompute to ACC `0.7563` and NMI
`0.5845111843402694`; `diagnostic.csv`, `diagnostic.json`, and the manifest
agree; and every manifest-declared output hash matches. The result ZIP SHA-256
is `158cdd9a27a714a8a764a79b527bd6988a79996740745eb8041b11c480f4303d`.
The downloaded audit bundle is stored under
`artifacts/kaggle/dekm-f-2021-reuters-paper-text-seed42/version-1-358020636`.

Correction after the full audit: this result also used released unit-L2
normalization, not the paper formula, so it does not close the paper-text branch.
The next useful experiment is the corrected paper-normalized seed-42 DEKM_F
control. Only after that control should MvDEC preserve standalone DEKM_F as its
View1 baseline and isolate View2, fusion, and Equation 11 as named ablations.

## Corrected paper-normalized seed-42 diagnostics

The corrected loader now implements the paper statement `(1/d)||x_i||^2 = 1`
as row norm `sqrt(d)` and records
`input_normalization=paper_mean_squared_norm_one`. This is a materially
different experiment from the earlier hybrid diagnostics, which accidentally
retained released unit-L2 normalization.

REUTERS Kaggle Version 2 completed on two T4 GPUs using commit
`27549ca943d27819bcf8247bb49cef85c9987255`. The exact source SHA-256 is
`7eb3f41aca38ffb93c104eef261adc3273d1da57c3750e04c92a67e335d423a1`.
Seed 42 stopped by assignment tolerance after 14 refreshes / 130 refinement
steps and 108.702019 seconds, with ACC `0.5101` and NMI
`0.21382318689551078`. Independent recomputation from all 10,000 saved
assignments matches exactly, and every manifest-declared artifact hash passes.
The ZIP SHA-256 is
`6ad1dac99a72f0a8b060dc2b837bd05167951e72898ac2c63d2c9bd7cbdf86d5`;
the downloaded bundle is under
`artifacts/kaggle/dekm-f-2021-reuters-paper-text-seed42/version-2-corrected`.

This corrected result is *farther* from the DEKM_F paper row (ACC `0.7642`, NMI
`0.5999`) by `-25.41` and `-38.607681` percentage points, respectively. It is
also far below the released-source seed-42 control (ACC `0.7557`, NMI
`0.593243`). Therefore the paper's normalization sentence cannot be combined
literally with the released training implementation to reproduce Table III.
The earlier unit-L2 hybrid result was much closer, which is strong evidence
that the paper prose and released executable protocol are inconsistent or
incomplete. Seeds 43-44 are not justified for this corrected branch.

RCV1 corrected Version 3 failed before training because Google Drive throttled
the official `test` split-index file; this was an input retrieval failure, not
an OOM or model failure. Commit `d2a39fcd312b27c00866fa8cd6a84300f3c18b16`
adds byte-exact bundled fallbacks for the two small official RCV1 split files,
verifies their canonical SHA-256 values, and records the retrieval path.
Commit `550d6c8` pins the diagnostic notebook to that fix. Ruff passed and the
full CPU test suite reports `157 passed, 72 skipped`; the skips require a local
TensorFlow installation. Kaggle RCV1 Version 4 was launched with GPU and
Internet enabled and is being monitored.

RCV1 Kaggle Version 5 completed successfully using the corrected literal paper
normalization. The Kaggle-runtime source SHA-256 is
`305877a57be6f4984b4f890d1b4809bf631582116e7a1c23045ae93a540574d6`;
the previously recorded local-runtime hash differs because the selected RCV1
columns are ordered differently across NumPy/scikit-learn versions. Seed 42
stopped by assignment tolerance after 17 refreshes / 160 refinement steps and
121.690796 seconds, with ACC `0.4174` and NMI `0.15222196301955926`.

Against the paper row (ACC `0.6212`, NMI `0.3598`), this is lower by `20.38`
and `20.757804` percentage points. Against released-source seed 42 (ACC
`0.7404`, NMI `0.5058763446541975`), it is lower by `32.30` and `35.365438`
percentage points. Independent recomputation from all 10,000 assignments
matches exactly; the normalization identity, source hash, and every
manifest-declared output hash pass. The ZIP is valid and has SHA-256
`56178e997be7f6f526865c16906b06511f225e5254bca57f723a3e07f4a53505`.
The downloaded bundle is stored under
`artifacts/kaggle/dekm-f-2021-rcv1-paper-text-seed42/version-5-corrected`.

The corrected REUTERS and RCV1 results agree on the audit conclusion: applying
the paper's normalization sentence literally to the released DEKM training
implementation makes performance dramatically worse, not closer to the paper.
Therefore the paper prose and executable release are inconsistent or omit a
material compensating detail. Running seeds 43-44 on this literal branch is not
justified. Future MvDEC work should keep the released-source-compatible DEKM_F
protocol as a separately named baseline and treat literal paper normalization
as a failed diagnostic ablation, not as the canonical reproduction protocol.

## Paper-era source-history correction and next DEKM gate

A complete text extraction of the nine-page paper confirms the following
explicit statements: DEKM uses mini-batch representation updates; DEKM_F changes
the representation update to full batch; text inputs satisfy the stated RMS
normalization; stopping occurs below 0.1% assignment changes; and Table III is
the average of three runs. The executable 2021 source conflicts with two of
those statements: it uses unit-L2 text normalization and a 0.5% stopping
threshold. It also special-cases REUTERS to a 10,000-row refinement batch.

This source history changes the experiment order. REUTERS has already passed the
paper-era source gate: the three-seed `release-2021` mean is ACC `0.760433` and
NMI `0.591668`, close to the paper's DEKM row ACC `0.7628` and NMI `0.5906`.
The next non-redundant diagnostic is therefore RCV1 seed 42 under the same
paper-era `release-2021` contract, where the source uses the intended 256-row
mini-batch. Its target is the DEKM row ACC `0.6715`, NMI `0.4618`; it must not be
compared to the DEKM_F row ACC `0.6212`, NMI `0.3598`. Only after that control
should a one-factor 0.1%-stopping variant be considered.
